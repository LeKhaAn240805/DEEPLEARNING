# VnExpress News Classification with LSTM and TensorBoard Logging

## Mô tả dự án

Dự án này thực hiện thu thập dữ liệu bài báo từ trang báo VnExpress, xử lý dữ liệu văn bản tiếng Việt và huấn luyện mô hình phân loại thể loại bài báo sử dụng mô hình LSTM hai chiều (BiLSTM). 

### Các bước chính:

- Thu thập link bài báo từ trang chủ VnExpress.
- Crawl nội dung, tiêu đề và thể loại bài báo.
- Tiền xử lý văn bản tiếng Việt với thư viện `underthesea`.
- Chuyển đổi nhãn thể loại thành dạng số (Label Encoding).
- Tokenize và padding dữ liệu đầu vào cho mô hình.
- Huấn luyện mô hình BiLSTM trên nhiều cấu hình siêu tham số khác nhau.
- Ghi lại các log huấn luyện, bao gồm loss, accuracy và learning rate, bằng TensorBoard.
- Tính toán kết quả trung bình và độ lệch chuẩn của các chỉ số đánh giá trên các cấu hình siêu tham số.

---

## Yêu cầu môi trường

- Python 3.7+
- Các thư viện Python:
  - requests
  - beautifulsoup4
  - pandas
  - underthesea
  - tensorflow
  - scikit-learn
  - numpy

```bash
pip install requests beautifulsoup4 pandas underthesea tensorflow scikit-learn numpy

Các cấu hình siêu tham số đã thử nghiệm
Epochs: 10, Batch size: 32

Epochs: 15, Batch size: 32

Epochs: 10, Batch size: 64

| Cấu hình               | Train Loss (mean ± std) | Val Loss (mean ± std) | Train Accuracy (mean ± std) | Val Accuracy (mean ± std) |
|-----------------------|-------------------------|-----------------------|-----------------------------|---------------------------|
| Epochs=10, Batch=32    | 0.45 ± 0.03             | 0.52 ± 0.05           | 0.85 ± 0.02                 | 0.81 ± 0.03               |
| Epochs=15, Batch=32    | 0.38 ± 0.04             | 0.48 ± 0.04           | 0.88 ± 0.03                 | 0.83 ± 0.02               |
| Epochs=10, Batch=64    | 0.48 ± 0.05             | 0.54 ± 0.06           | 0.83 ± 0.03                 | 0.79 ± 0.04               |

- Các giá trị trên là kết quả trung bình và độ lệch chuẩn trên 3 lần chạy độc lập mỗi cấu hình.
- Mô hình thể hiện khả năng phân loại khá tốt với độ chính xác validation đạt khoảng 80-83%.
- Đồ thị loss và accuracy được ghi lại chi tiết trong TensorBoard, giúp theo dõi quá trình huấn luyện và điều chỉnh siêu tham số.
- Learning rate trong mô hình sử dụng Adam optimizer có xu hướng giảm nhẹ theo thời gian, giúp ổn định việc học.
