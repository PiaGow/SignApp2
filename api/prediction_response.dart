class PredictionResponse {
  // Thêm thuộc tính để chứa câu đã hoàn thiện
  final List<String> rawActions;
  final String finalSentence;

  PredictionResponse({
    required this.rawActions,
    required this.finalSentence,
  });

  // Viết lại factory constructor để đọc đúng các key từ JSON
  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    // Đọc danh sách các hành động thô từ key "raw_actions"
    final List<dynamic> rawActionList = json['raw_actions'] as List<dynamic>? ?? [];
    final List<String> parsedRawActions = rawActionList.map((item) => item.toString()).toList();

    // Đọc câu hoàn chỉnh từ key "final_sentence"
    final String parsedSentence = json['final_sentence'] as String? ?? "Lỗi: Không nhận được câu hoàn chỉnh.";

    return PredictionResponse(
      rawActions: parsedRawActions,
      finalSentence: parsedSentence,
    );
  }
}