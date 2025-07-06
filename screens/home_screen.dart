import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:app/api/api_service.dart'; // Đảm bảo tên package 'app' là đúng
import 'package:app/api/prediction_response.dart'; // Import lớp response mới

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  final ApiService _apiService = ApiService();

  // Biến trạng thái mới để lưu trữ kết quả
  String _displayText = "Hãy chọn một video để bắt đầu nhận dạng.";
  bool _isLoading = false;

  Future<void> _pickAndUploadVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(source: ImageSource.gallery);

      if (video == null) return; // Người dùng không chọn video

      // --- Bắt đầu xử lý: Reset giao diện ---
      setState(() {
        _isLoading = true;
        _displayText = "Đang tải và xử lý video...";
      });

      File videoFile = File(video.path);
      // Gọi API và nhận về đối tượng PredictionResponse đầy đủ
      final PredictionResponse? response = await _apiService.uploadVideo(videoFile);

      // --- Xử lý hoàn tất: Cập nhật kết quả mới ---
      setState(() {
        if (response != null) {
          // Ưu tiên hiển thị câu đã được GPT xử lý
          if (response.finalSentence.isNotEmpty && !response.finalSentence.contains("failed")) {
            _displayText = response.finalSentence;
          }
          // Nếu không có câu từ GPT, hiển thị các hành động thô
          else if (response.rawActions.isNotEmpty) {
            _displayText = response.rawActions.join(' ');
          }
          // Nếu cả hai đều rỗng
          else {
            _displayText = "Không nhận dạng được hành động nào.";
          }
        } else {
          _displayText = "Không nhận được phản hồi từ server hoặc có lỗi kết nối.";
        }
        _isLoading = false;
      });

    } catch (e) {
      // Xử lý các lỗi ngoại lệ (ví dụ: không có quyền truy cập file)
      setState(() {
        _displayText = "Đã xảy ra lỗi trong ứng dụng: $e";
        _isLoading = false;
      });
    }
  }

  // Widget helper để hiển thị kết quả một cách linh hoạt
  Widget _buildResultView() {
    if (_isLoading) {
      // Khi đang tải, hiển thị vòng quay và tin nhắn
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(
            _displayText, // Hiển thị trạng thái "Đang xử lý..."
            style: Theme.of(context).textTheme.titleMedium,
            textAlign: TextAlign.center,
          ),
        ],
      );
    }

    // Khi đã có kết quả hoặc ở trạng thái ban đầu
    return Text(
      _displayText,
      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
        color: Colors.black87,
        fontWeight: FontWeight.w500,
      ),
      textAlign: TextAlign.center,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dịch Ngôn Ngữ Ký Hiệu'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            const Spacer(flex: 2),
            Text(
              "Kết quả nhận dạng",
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w300),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            Container(
              height: 180, // Tăng chiều cao một chút
              padding: const EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(12.0),
                  border: Border.all(color: Colors.grey.shade300)
              ),
              child: Center(
                  child: SingleChildScrollView(
                    child: _buildResultView(),
                  )
              ),
            ),
            const Spacer(flex: 3),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _pickAndUploadVideo,
              icon: const Icon(Icons.video_library_rounded),
              label: const Text('Chọn Video'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}