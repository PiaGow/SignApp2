import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm
import warnings
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
from openai import OpenAI
import json
import re
# --- 1. SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = FastAPI(title="Sign Language Recognition API with GPT-4", version="4.3-advanced-gpt")

# --- 2. LOAD ASSETS ---
ASSETS_DIR = 'final_assets'
MODEL_PATH = os.path.join(ASSETS_DIR, 'final_model.h5')
SCALER_PATH = os.path.join(ASSETS_DIR, 'final_scaler.pkl')
ACTIONS_PATH = os.path.join(ASSETS_DIR, 'final_actions.npy')
ACTIONS_MAPPING_PATH = 'actions_mapping.txt'

model: tf.keras.Model
scaler: any
action_labels: np.ndarray
actions_mapping: dict = {}
gpt_client: OpenAI = None
SEQUENCE_LENGTH: int
EXPECTED_FEATURES: int


@app.on_event("startup")
def load_all_models_and_assets():
    """Tải tất cả tài sản và khởi tạo client OpenAI để dùng với proxy ChatAnywhere."""
    global model, scaler, action_labels, actions_mapping, gpt_client, SEQUENCE_LENGTH, EXPECTED_FEATURES
    try:
        # --- Phần tải model, scaler, actions, mapping file giữ nguyên ---
        logger.info("Loading Sign Recognition assets...")
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        action_labels = np.load(ACTIONS_PATH, allow_pickle=True)

        SEQUENCE_LENGTH = model.input_shape[1]
        EXPECTED_FEATURES = model.input_shape[2]

        logger.info("Loading Actions-to-Meaning mapping file...")
        if os.path.exists(ACTIONS_MAPPING_PATH):
            with open(ACTIONS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        actions_mapping[key.strip()] = value.strip()
            logger.info(f"{len(actions_mapping)} mappings loaded.")
        else:
            logger.warning(f"Mapping file not found at {ACTIONS_MAPPING_PATH}.")

        # --- PHẦN SỬA LỖI: Cấu hình key và URL cho ChatAnywhere ---

        # Dán key của bạn từ trang ChatAnywhere vào đây
        chatanywhere_api_key = "sk-d6jrYkqQUjh9RDQVTydBrpGnm987yzCwCCBbVEJNqmi86878"

        # Khai báo Base URL của dịch vụ proxy
        chatanywhere_base_url = "https://api.chatanywhere.com.cn/v1"

        # Điều kiện IF đã được sửa lại cho đúng logic:
        # Chỉ cần kiểm tra xem key có phải là một chuỗi hợp lệ hay không.
        if chatanywhere_api_key:
            logger.warning("SECURITY_RISK: API key is hardcoded. Use for local testing only.")
            logger.info(f"Initializing GPT client with ChatAnywhere proxy at {chatanywhere_base_url}")

            # Khởi tạo client OpenAI với cả api_key và base_url
            gpt_client = OpenAI(
                api_key=chatanywhere_api_key,
                base_url=chatanywhere_base_url
            )
        else:
            # Điều kiện này sẽ xảy ra nếu bạn để chuỗi key rỗng ""
            logger.warning("GPT feature is disabled. API key is empty.")
            gpt_client = None

        logger.info(
            f"Assets loaded successfully. Model expects {EXPECTED_FEATURES} features. Total actions: {len(action_labels)}.")
    except Exception as e:
        logger.error(f"FATAL: Could not load assets: {e}", exc_info=True)
        model = None

# --- 3. CORE LOGIC - SYNCHRONIZED 100% WITH ORIGINAL TRAINING CODE ---
mp_holistic = mp.solutions.holistic
POSE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
FACE_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                159, 160, 161, 246, 46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 55, 65, 70, 63, 105, 66, 107]


def extract_keypoints(results):
    pose_lm = results.pose_landmarks.landmark if results.pose_landmarks else None
    pose_arr = np.array([(pose_lm[i].x, pose_lm[i].y, pose_lm[i].z, pose_lm[i].visibility) for i in
                         POSE_INDICES]).flatten() if pose_lm else np.zeros(len(POSE_INDICES) * 4)
    lh_arr = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    rh_arr = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    face_lm = results.face_landmarks.landmark if results.face_landmarks else None
    face_arr = np.array(
        [(face_lm[i].x, face_lm[i].y, face_lm[i].z) for i in FACE_INDICES]).flatten() if face_lm and len(face_lm) > max(
        FACE_INDICES) else np.zeros(len(FACE_INDICES) * 3)
    return np.concatenate([pose_arr, lh_arr, rh_arr, face_arr])


def calculate_hand_features(hand_landmarks):
    if not hand_landmarks: return np.zeros(10)
    lm = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
    if lm.shape[0] < 21: return np.zeros(10)
    tips = [lm[i] for i in [4, 8, 12, 16, 20]];
    wrist = lm[0]
    features = [np.linalg.norm(tip - wrist) for tip in tips]
    features.extend([np.linalg.norm(tips[i] - tips[i + 1]) for i in range(4)])
    features.append(np.arctan2(lm[9][1] - wrist[1], lm[9][0] - wrist[0]))
    return np.array(features)


def get_raw_features_from_results(results):
    keypoints = extract_keypoints(results)
    left_hand_features = calculate_hand_features(results.left_hand_landmarks)
    right_hand_features = calculate_hand_features(results.right_hand_landmarks)
    return np.concatenate([keypoints, left_hand_features, right_hand_features])


def normalize_landmarks_advanced(data_array):
    pose_end = 96;
    lh_end = 159;
    rh_end = 222;
    face_end = 351
    normalized_data = data_array.copy()
    for f in range(data_array.shape[0]):
        frame = normalized_data[f, :];
        if np.sum(np.abs(frame)) < 1e-6: continue

        frame_keypoints = frame[:face_end]
        custom_features = frame[face_end:]

        pose_data = frame_keypoints[:pose_end].reshape(-1, 4)
        if np.sum(np.abs(pose_data)) > 1e-6 and pose_data.shape[0] > 12:
            left_s, right_s = pose_data[11, :3], pose_data[12, :3]
            if np.sum(np.abs(left_s)) > 1e-6 and np.sum(np.abs(right_s)) > 1e-6:
                center = (left_s + right_s) / 2;
                width = np.linalg.norm(right_s - left_s)
                if width > 1e-6:
                    pose_data[:, :3] = (pose_data[:, :3] - center) / width
                    frame_keypoints[:pose_end] = pose_data.flatten()

        lh_data = frame_keypoints[pose_end:lh_end].reshape(21, 3)
        if np.sum(np.abs(lh_data)) > 1e-6:
            wrist = lh_data[0].copy();
            lh_data -= wrist;
            scale = np.linalg.norm(lh_data[9]);
            if scale > 1e-6: lh_data /= scale;
            frame_keypoints[pose_end:lh_end] = lh_data.flatten()

        rh_data = frame_keypoints[lh_end:rh_end].reshape(21, 3)
        if np.sum(np.abs(rh_data)) > 1e-6:
            wrist = rh_data[0].copy();
            rh_data -= wrist;
            scale = np.linalg.norm(rh_data[9]);
            if scale > 1e-6: rh_data /= scale;
            frame_keypoints[lh_end:rh_end] = rh_data.flatten()

        face_data = frame_keypoints[rh_end:face_end].reshape(-1, 3)
        if np.sum(np.abs(face_data)) > 1e-6:
            center = np.mean(face_data, axis=0);
            face_data -= center;
            size = np.std(face_data)
            if size > 1e-6: face_data /= size;
            frame_keypoints[rh_end:face_end] = face_data.flatten()

        normalized_data[f, :] = np.concatenate([frame_keypoints, custom_features])

    return normalized_data
# --- PHẦN 4: HÀM DỰ ĐOÁN (ĐÃ CHỈNH SỬA ĐỂ TRẢ VỀ LIST[STR]) ---
def predict_single_action(scaled_sequence) -> List[str]:
    """Xử lý video ngắn. Logic không đổi."""
    input_data = np.zeros((SEQUENCE_LENGTH, EXPECTED_FEATURES))
    sequence_to_use = scaled_sequence[-SEQUENCE_LENGTH:]
    input_data[-len(sequence_to_use):] = sequence_to_use
    input_for_model = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_for_model, verbose=0)[0]
    # === SỬA LỖI: Sử dụng biến toàn cục 'action_labels' ===
    action = action_labels[np.argmax(prediction)]
    return [action]


def predict_continuous_actions(scaled_sequence, stride, confidence_threshold) -> List[str]:
    """Xử lý video dài với logic cửa sổ trượt và look-ahead."""
    recognized_actions = []
    last_action_added = None
    current_pos = 0

    while current_pos <= len(scaled_sequence) - SEQUENCE_LENGTH:
        window = scaled_sequence[current_pos: current_pos + SEQUENCE_LENGTH]
        prediction = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
        confidence = np.max(prediction)

        if confidence < confidence_threshold:
            current_pos += stride
            continue

        # === SỬA LỖI: Sử dụng biến toàn cục 'action_labels' ===
        action = action_labels[np.argmax(prediction)]

        if action != last_action_added:
            recognized_actions.append(action)
            last_action_added = action
        # Luôn trượt cửa sổ về phía trước, bất kể có lặp lại hay không,
        # để tránh bỏ sót các hành động ngắn.
        # Logic look-ahead phức tạp có thể được thêm lại nếu cần.
        current_pos += stride

    # Xử lý phần cuối của video bằng cách đệm
    if current_pos < len(scaled_sequence):
        final_segment_data = scaled_sequence[current_pos:]
        if len(final_segment_data) > 10:
            input_data = np.zeros((SEQUENCE_LENGTH, EXPECTED_FEATURES))
            input_data[-len(final_segment_data):] = final_segment_data
            prediction = model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
            confidence = np.max(prediction)
            if confidence >= confidence_threshold:
                # === SỬA LỖI: Sử dụng biến toàn cục 'action_labels' ===
                action = action_labels[np.argmax(prediction)]
                if action != last_action_added:
                    recognized_actions.append(action)

    return recognized_actions if recognized_actions else []


# def formulate_sentence_with_gpt(raw_actions: List[str]) -> str:
#     """Sử dụng một prompt siêu chi tiết, buộc GPT phải thực hiện các bước trung gian."""
#     if not gpt_client:
#         logger.warning("GPT client not available.")
#         action_meanings = [actions_mapping.get(action, action).split('] ')[-1] for action in raw_actions]
#         return ' '.join(action_meanings)
#
#     tagged_meanings = [actions_mapping.get(action, action) for action in raw_actions]
#
#     # Đây là bước debug quan trọng. Hãy kiểm tra log server của bạn để đảm bảo
#     # prompt này thực sự được gửi đi khi bạn test.
#     logger.info(f"DEBUG: Input to GPT (tagged): {tagged_meanings}")
#
#     prompt = f"""
#     Bạn là một nhà biên tập ngôn ngữ tiếng Việt. Nhiệm vụ của bạn là nhận một danh sách TỪ KHÓA BỊ XÁO TRỘN và tái cấu trúc nó thành một CÂU HOÀN CHỈNH.
#
#     **### QUY TRÌNH BẮT BUỘC (LÀM THEO ĐÚNG 3 BƯỚC):**
#
#     **1. SẮP XẾP LẠI (Reorder):**
#     Phân tích danh sách từ khóa đầu vào và sắp xếp chúng lại theo đúng trật tự ngữ pháp tiếng Việt: [Thời gian] -> [Chủ ngữ] -> [Động từ] -> [Bổ ngữ/Trạng từ]. VIẾT RA KẾT QUẢ ĐÃ SẮP XẾP.
#
#     **2. LÀM GIÀU CÂU (Enrich):**
#     Dựa trên chuỗi ĐÃ SẮP XẾP, thêm các từ nối, giới từ, trợ từ cần thiết để câu văn trở nên tự nhiên.
#
#     **3. TẠO JSON CUỐI CÙNG (Final JSON):**
#     Tạo ra đối tượng JSON cuối cùng chứa câu văn hoàn chỉnh.
#
#     **--- VÍ DỤ MẪU ĐỂ LÀM THEO TUYỆT ĐỐI ---**
#
#     **Ví dụ 1:**
#     *   **Input:** `['[TT] Bên phải', '[ĐV] Chạy', '[CN] Ông', '[TG] Hôm qua']`
#     *   **Quá trình xử lý của bạn:**
#         *   **Sắp xếp lại:** `[TG] Hôm qua [CN] Ông [ĐV] Chạy [TT] Bên phải`
#         *   **Làm giàu câu:** Thêm giới từ "về phía" hoặc "ở". Thêm dấu chấm câu.
#         *   **Tạo JSON cuối cùng:** `{{"sentence": "Hôm qua ông chạy về phía bên phải."}}`
#
#     **Ví dụ 2:**
#     *   **Input:** `['[DT] Cơm', '[ĐV] Ăn']`
#     *   **Quá trình xử lý của bạn:**
#         *   **Sắp xếp lại:** `[CN] Tôi [ĐV] Ăn [DT] Cơm` (Tự động thêm chủ ngữ "Tôi")
#         *   **Làm giàu câu:** Giữ nguyên, thêm dấu chấm câu.
#         *   **Tạo JSON cuối cùng:** `{{"sentence": "Tôi ăn cơm."}}`
#
#     ---
#
#     **BÂY GIỜ, HÃY XỬ LÝ DỮ LIỆU SAU THEO ĐÚNG QUY TRÌNH TRÊN:**
#
#     **Input:** `{tagged_meanings}`
#
#     Chỉ trả về đối tượng JSON ở bước cuối cùng.
#     """
#
#     try:
#         response = gpt_client.chat.completions.create(
#             model="gpt-4o",  # <-- THỬ NÂNG CẤP LÊN gpt-4o để có sức mạnh suy luận tốt hơn
#             response_format={"type": "json_object"},
#             messages=[
#                 {"role": "system",
#                  "content": "Bạn là nhà biên tập ngôn ngữ. Hãy tuân thủ nghiêm ngặt quy trình được giao và chỉ trả về JSON cuối cùng."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.2,  # Giảm nhiệt độ tối đa để bám sát logic
#             max_tokens=300
#         )
#
#         response_content = response.choices[0].message.content
#         logger.info(f"Raw GPT response: {response_content}")
#
#         json_response = json.loads(response_content)
#         un_tagged_meanings = [meaning.split('] ')[-1] for meaning in tagged_meanings]
#         final_sentence = json_response.get("sentence", ' '.join(un_tagged_meanings))
#
#         logger.info(f"Parsed sentence from GPT: '{final_sentence}'")
#         return final_sentence
#     except Exception as e:
#         logger.error(f"Error during GPT call: {e}", exc_info=True)
#         un_tagged_meanings = [meaning.split('] ')[-1] for meaning in tagged_meanings]
#         return ' '.join(un_tagged_meanings)


def reorder_keywords_with_gpt(tagged_meanings: List[str]) -> List[str]:
    """Lần gọi API 1: CHỈ để sắp xếp lại các từ khóa một cách máy móc."""
    prompt = f"""
    NHIỆM VỤ: Sắp xếp lại danh sách từ khóa sau đây theo trật tự ngữ pháp tiếng Việt: [Thời gian] -> [Chủ ngữ] -> [Động từ] -> [Bổ ngữ].
    QUY TẮC:
    1.  KHÔNG ĐƯỢC thay đổi, thêm, hoặc bớt bất kỳ từ nào.
    2.  Giữ nguyên các thẻ `[CN]`, `[ĐV]`, v.v.
    3.  Chỉ trả về một danh sách Python.

    Input: {tagged_meanings}
    Output:
    """
    try:
        logger.info(f"Step 1: Reordering keywords: {tagged_meanings}")
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Nhiệt độ = 0 để đảm bảo tính logic, không sáng tạo
        )
        reordered_string = response.choices[0].message.content
        logger.info(f"GPT reorder response (raw string): {reordered_string}")

        reordered_list = eval(reordered_string)

        if isinstance(reordered_list, list):
            logger.info(f"Successfully reordered list: {reordered_list}")
            return reordered_list
        else:
            logger.warning("Reordering did not return a list. Falling back to original.")
            return tagged_meanings

    except Exception as e:
        logger.error(f"Failed to reorder keywords with GPT: {e}")
        return tagged_meanings


def build_sentence_with_gpt(ordered_meanings: List[str]) -> str:
    """Lần gọi API 2: Nhận danh sách ĐÃ SẮP XẾP và làm giàu câu, có khả năng hợp nhất các động từ liên quan."""
    un_tagged_meanings = [s.split('] ')[-1] for s in ordered_meanings]

    # --- PROMPT NÂNG CẤP LẦN CUỐI CÙNG ---
    prompt = f"""
    Bạn là một biên tập viên ngôn ngữ tiếng Việt bậc thầy. Nhiệm vụ của bạn là nhận một chuỗi từ khóa ĐÃ ĐƯỢC SẮP XẾP và tái cấu trúc nó thành một câu văn tự nhiên và logic nhất.

    **### CÁC QUY TẮC VÀNG ĐỂ TUÂN THỦ:**

    1.  **GIỮ NGUYÊN Ý NGHĨA:** Câu cuối cùng phải chứa ĐẦY ĐỦ ý nghĩa của các từ khóa gốc. Không được phép bịa đặt thông tin mới.

    2.  **XỬ LÝ ĐA ĐỘNG TỪ (QUAN TRỌNG NHẤT):** Khi có nhiều hành động (động từ) trong câu, hãy tìm cách **kết hợp chúng một cách logic** bằng cách sử dụng các từ nối như "và", "để", hoặc biến một động từ thành danh từ/bổ ngữ cho động từ chính. **Không liệt kê các động từ một cách máy móc.**

    3.  **TỰ ĐỘNG THÊM CHỦ NGỮ:** Nếu câu chỉ có hành động (ví dụ: 'Chạy'), hãy mặc định thêm chủ ngữ "Tôi" hoặc "Bạn" tùy theo ngữ cảnh.

    4.  **CHỈ THÊM TỪ CẦN THIẾT:** Chỉ được thêm các từ chức năng để câu đúng ngữ pháp và trôi chảy.

    **### VÍ DỤ CỤ THỂ ĐỂ HỌC THEO:**

    - **Input:** `['Mẹ', 'Tập thể dục', 'Chạy', 'Buổi sáng']`
      **Tư duy của bạn:** "Tập thể dục" và "Chạy" là hai hành động liên quan. "Chạy" có thể là một hình thức của "Tập thể dục". Cần hợp nhất chúng. "Chạy bộ" là một cụm từ tự nhiên.
      **Câu cuối cùng:** `"Mẹ chạy bộ tập thể dục vào buổi sáng."` (Đã hợp nhất và thêm "bộ", "vào")

    - **Input:** `['Hôm qua', 'Ông', 'Chạy', 'Bên phải']`
      **Tư duy của bạn:** Các từ khóa đã đầy đủ. Cần thêm giới từ "về phía" để làm rõ hướng.
      **Câu cuối cùng:** `"Hôm qua ông chạy về phía bên phải."`

    - **Input:** `['Uống', 'Nước']`  (Giả sử 'Nước' là một từ khóa)
      **Tư duy của bạn:** Thiếu chủ ngữ. Thêm "Tôi".
      **Câu cuối cùng:** `"Tôi uống nước."`

    **---**
    **BÂY GIỜ, HÃY XỬ LÝ CHUỖI TỪ KHÓA SAU ĐÂY:**

    **Từ khóa có sẵn:** `{un_tagged_meanings}`

    **Câu hoàn chỉnh của bạn là:**
    """
    try:
        logger.info(f"Step 2 (Advanced): Building sentence from: {un_tagged_meanings}")
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Bạn là một biên tập viên ngôn ngữ thông minh, có khả năng hợp nhất các hành động một cách logic và tự nhiên."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4  # Tăng nhẹ nhiệt độ để cho phép sự linh hoạt trong việc kết hợp từ
        )
        final_sentence = response.choices[0].message.content.strip().strip('"')
        logger.info(f"GPT built sentence (Advanced): {final_sentence}")
        return final_sentence
    except Exception as e:
        logger.error(f"Failed to build sentence with GPT: {e}")
        return ' '.join(un_tagged_meanings)
# @app.post("/predict")
# async def create_prediction(video: UploadFile = File(...)):
#     if not model:
#         raise HTTPException(status_code=503, detail="Server not ready: Model is not loaded.")
#
#     video_path = f"temp_{video.filename}"
#     try:
#         with open(video_path, "wb") as buffer:
#             shutil.copyfileobj(video.file, buffer)
#
#         # Trích xuất và xử lý video
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         all_keypoints = []
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             for _ in range(total_frames):
#                 ret, frame = cap.read()
#                 if not ret: break
#                 results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 all_keypoints.append(get_raw_features_from_results(results))
#         cap.release()
#
#         if not all_keypoints:
#             raise HTTPException(status_code=400, detail="Could not extract any keypoints.")
#
#         raw_sequence = np.array(all_keypoints)
#         if raw_sequence.shape[1] != EXPECTED_FEATURES:
#             raise HTTPException(status_code=500,
#                                 detail=f"Feature mismatch: Extracted {raw_sequence.shape[1]}, but model expects {EXPECTED_FEATURES}.")
#
#         # Tiền xử lý và dự đoán
#         scaled_sequence = scaler.transform(normalize_landmarks_advanced(raw_sequence))
#
#         if total_frames > SEQUENCE_LENGTH:
#             result_actions = predict_continuous_actions(scaled_sequence, stride=30, confidence_threshold=0.8)
#         else:
#             result_actions = predict_single_action(scaled_sequence)
#
#         logger.info(f"Raw recognized actions: {result_actions}")
#
#         # Gọi GPT để hoàn thiện câu
#         final_sentence = "Could not generate sentence."
#         if result_actions:  # Chỉ gọi GPT nếu có kết quả
#             final_sentence = formulate_sentence_with_gpt(result_actions)
#         else:
#             logger.info("No actions recognized, skipping GPT call.")
#             final_sentence = "Không nhận dạng được hành động nào."
#
#         return JSONResponse(content={
#             "raw_actions": result_actions,
#             "final_sentence": final_sentence
#         })
#     except Exception as e:
#         logger.error(f"Error during video processing: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if os.path.exists(video_path):
#             os.remove(video_path)
@app.post("/predict")
async def create_prediction(video: UploadFile = File(...)):
    # --- Phần 1: Kiểm tra và chuẩn bị ---
    if not model or not scaler or action_labels is None:
        raise HTTPException(status_code=503, detail="Server not ready: Core assets are not loaded.")

    video_path = f"temp_{video.filename}"
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # --- Phần 2: Trích xuất Keypoints từ Video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_keypoints = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                all_keypoints.append(get_raw_features_from_results(results))
        cap.release()

        if not all_keypoints:
            raise HTTPException(status_code=400, detail="Could not extract any keypoints from the video.")

        # --- Phần 3: Tiền xử lý và Dự đoán bằng Model ---
        raw_sequence = np.array(all_keypoints)
        if raw_sequence.shape[1] != EXPECTED_FEATURES:
            raise HTTPException(status_code=500,
                                detail=f"Feature mismatch: Extracted {raw_sequence.shape[1]}, but model expects {EXPECTED_FEATURES}.")

        scaled_sequence = scaler.transform(normalize_landmarks_advanced(raw_sequence))

        result_actions = []
        if total_frames > 0:
            if total_frames > SEQUENCE_LENGTH:
                result_actions = predict_continuous_actions(scaled_sequence, stride=40, confidence_threshold=0.8)
            else:
                result_actions = predict_single_action(scaled_sequence)

        # Log này là của bạn, chúng ta sẽ so sánh với log debug của tôi
        logger.info(f"Raw recognized actions from model: {result_actions}")

        # --- Phần 4: Pipeline gọi GPT 2 bước và Chuẩn bị Debug ---
        final_sentence = "Không thể tạo câu hoàn chỉnh."

        # Chuẩn bị các biến để debug, gán giá trị mặc định
        tagged_meanings = []
        ordered_keywords = []
        gpt_skipped = True

        if result_actions and gpt_client:
            gpt_skipped = False
            tagged_meanings = [actions_mapping.get(action, action) for action in result_actions]
            ordered_keywords = reorder_keywords_with_gpt(tagged_meanings)
            final_sentence = build_sentence_with_gpt(ordered_keywords)
        elif not result_actions:
            final_sentence = "Không nhận dạng được hành động nào."
        elif not gpt_client:
            logger.warning("GPT client not configured. Returning joined raw actions.")
            un_tagged_meanings = [actions_mapping.get(action, action).split('] ')[-1] for action in result_actions]
            final_sentence = ' '.join(un_tagged_meanings)

        # --- KHỐI DEBUG ĐÃ ĐƯỢC TÍCH HỢP SẴN ---
        print("\n" + "=" * 20 + " DEBUGGING START " + "=" * 20)
        print(f"1. Model Predicted Actions (result_actions): {result_actions}")

        if not gpt_skipped:
            print(f"2. Tagged Meanings (Input for Step 1)   : {tagged_meanings}")
            print(f"3. Reordered Keywords (Output of Step 1) : {ordered_keywords}")
            print(f"4. Final Sentence (Output of Step 2)   : '{final_sentence}'")
        else:
            print("--> GPT Pipeline was SKIPPED.")
            if not result_actions:
                print("    Reason: No actions were recognized by the model.")
            elif not gpt_client:
                print("    Reason: GPT client is not configured.")

        print("=" * 22 + " DEBUGGING END " + "=" * 23 + "\n")
        # --- KẾT THÚC KHỐI DEBUG ---

        # --- Phần 5: Trả về kết quả cuối cùng ---
        return JSONResponse(content={
            "raw_actions": result_actions,
            "final_sentence": final_sentence
        })

    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
