import 'package:dio/dio.dart';
import 'dart:io';
import 'package:http_parser/http_parser.dart';
import 'package:app/api/prediction_response.dart'; // Đảm bảo import này đúng

class ApiService {
  final Dio _dio = Dio();

  static const String _baseUrl = 'http://192.168.88.121:8000';

  Future<PredictionResponse?> uploadVideo(File videoFile) async {
    try {
      String fileName = videoFile.path.split('/').last;
      FormData formData = FormData.fromMap({
        'video': await MultipartFile.fromFile(
          videoFile.path,
          filename: fileName,
          contentType: MediaType('video', 'mp4'),
        ),
      });

      // Thêm timeout để tránh chờ đợi quá lâu
      final response = await _dio.post(
        '$_baseUrl/predict',
        data: formData,
        options: Options(
          receiveTimeout: const Duration(minutes: 2), // Chờ tối đa 2 phút
        ),
      );

      if (response.statusCode == 200 && response.data != null) {
        // === SỬA LỖI 2: TRẢ VỀ TOÀN BỘ ĐỐI TƯỢNG RESPONSE ===
        // Không chỉ trả về một phần, mà trả về cả đối tượng đã được parse
        return PredictionResponse.fromJson(response.data);
      } else {
        // Nếu có lỗi từ server, trả về null để bên ngoài xử lý
        print('Server returned error code: ${response.statusCode}');
        return null;
      }
    } on DioException catch (e) {
      // Xử lý các lỗi kết nối hoặc từ server
      if (e.response != null) {
        print('Server Error: ${e.response?.data}');
      } else {
        print('Network Error: ${e.message}');
      }
      return null; // Trả về null khi có lỗi
    } catch (e) {
      print('Unexpected Error: $e');
      return null; // Trả về null khi có lỗi
    }
  }
}