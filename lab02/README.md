# Dự án Phân tích Cảm xúc IMDB

## Mô tả

Dự án này thực hiện phân tích cảm xúc từ dữ liệu đánh giá phim IMDB. Các bài đánh giá được phân loại thành hai loại: tích cực và tiêu cực. Mô hình học sâu (Deep Learning) được sử dụng để phân tích dữ liệu này, bao gồm việc tiền xử lý dữ liệu, huấn luyện mô hình với các cấu hình siêu tham số khác nhau, và đánh giá kết quả.

## Tiền xử lý dữ liệu

Trước khi huấn luyện mô hình, các bước tiền xử lý sau được thực hiện trên dữ liệu văn bản để chuẩn hóa và làm sạch dữ liệu, giúp mô hình học tốt hơn bằng cách loại bỏ các stopword, các dấu khoảng cách dư thừa vv..


| Cấu Hình | Số Nơ-ron LSTM | Dropout | Batch Size | Optimizer | Số Epoch | Độ Chính Xác Trung Bình | Độ Lệch Chuẩn |
|----------|-----------------|---------|------------|-----------|----------|-------------------------|----------------|
| 1        | 128             | 0.2     | 64         | Adam      | 5        | 0.8635                  | 0.0107         |
| 2        | 64              | 0.3     | 32         | Adam      | 6        | 0.8700                  | 0.0021         |
| 3        | 32              | 0.6     | 64         | Adam      | 5        | 0.8812                  | 0.0005         |
| 4        | 128             | 0.5     | 32         | RMSprop   | 5        | 0.8855                  | 0.0029         |
| 5        | 64              | 0.4     | 32         | RMSprop   | 5        | 0.8864                  | 0.0004         |
## Phân Tích Kết Quả

Dưới đây là phân tích về hiệu suất của các mô hình sau khi thử nghiệm với các cấu hình siêu tham số khác nhau:

### Cấu hình 1:
- **Số nơ-ron LSTM**: 128
- **Dropout**: 0.2
- **Batch size**: 64
- **Optimizer**: Adam
- **Số Epoch**: 5
- **Độ chính xác trung bình**: 0.8635
- **Độ lệch chuẩn**: 0.0107

Mô hình này có độ chính xác trung bình là 0.8635 với độ lệch chuẩn tương đối cao (0.0107). Điều này cho thấy rằng mô hình có thể có sự biến động lớn giữa các lần chạy thử và có thể cải thiện bằng cách điều chỉnh các tham số như dropout hoặc số nơ-ron LSTM.

### Cấu hình 2:
- **Số nơ-ron LSTM**: 64
- **Dropout**: 0.3
- **Batch size**: 32
- **Optimizer**: Adam
- **Số Epoch**: 6
- **Độ chính xác trung bình**: 0.8700
- **Độ lệch chuẩn**: 0.0021

Mô hình này có độ chính xác trung bình cao hơn (0.8700) và độ lệch chuẩn thấp hơn (0.0021), cho thấy rằng nó ổn định hơn trong các lần chạy thử. Việc sử dụng số lượng nơ-ron LSTM ít hơn và tăng dropout có thể giúp giảm overfitting và cải thiện hiệu suất tổng thể.

### Cấu hình 3:
- **Số nơ-ron LSTM**: 32
- **Dropout**: 0.6
- **Batch size**: 64
- **Optimizer**: Adam
- **Số Epoch**: 5
- **Độ chính xác trung bình**: 0.8812
- **Độ lệch chuẩn**: 0.0005

Cấu hình này mang lại độ chính xác trung bình cao nhất (0.8812) và độ lệch chuẩn rất thấp (0.0005), cho thấy mô hình rất ổn định và đạt được hiệu suất rất tốt. Điều này có thể là kết quả của việc sử dụng dropout cao, giúp mô hình tránh overfitting.

### Cấu hình 4:
- **Số nơ-ron LSTM**: 128
- **Dropout**: 0.5
- **Batch size**: 32
- **Optimizer**: RMSprop
- **Số Epoch**: 5
- **Độ chính xác trung bình**: 0.8855
- **Độ lệch chuẩn**: 0.0029

Cấu hình này có độ chính xác trung bình cao (0.8855) và độ lệch chuẩn khá thấp, chứng tỏ mô hình ổn định và hiệu quả. Việc sử dụng optimizer RMSprop thay vì Adam có thể đóng vai trò quan trọng trong việc cải thiện kết quả trong trường hợp này.

### Cấu hình 5:
- **Số nơ-ron LSTM**: 64
- **Dropout**: 0.4
- **Batch size**: 32
- **Optimizer**: RMSprop
- **Số Epoch**: 5
- **Độ chính xác trung bình**: 0.8864
- **Độ lệch chuẩn**: 0.0004

Đây là cấu hình đạt độ chính xác trung bình cao nhất (0.8864) và độ lệch chuẩn thấp nhất (0.0004), cho thấy rằng mô hình có khả năng ổn định và cho kết quả tốt nhất trong các lần thử nghiệm. Sự kết hợp giữa số nơ-ron LSTM trung bình, dropout hợp lý và optimizer RMSprop đã mang lại kết quả xuất sắc.

### Tổng Quan:
Qua các thử nghiệm với các cấu hình siêu tham số khác nhau, chúng ta có thể thấy rằng việc tối ưu hóa số nơ-ron LSTM, dropout và optimizer đã giúp cải thiện đáng kể độ chính xác của mô hình. Các cấu hình có độ chính xác cao nhất đều sử dụng RMSprop làm optimizer, cho thấy rằng lựa chọn optimizer có ảnh hưởng quan trọng đến hiệu suất mô hình. Hơn nữa, việc điều chỉnh mức độ dropout cũng rất quan trọng trong việc giảm overfitting và cải thiện độ chính xác.

Mô hình với cấu hình 5 (số nơ-ron LSTM = 64, dropout = 0.4, optimizer = RMSprop) cho thấy kết quả ổn định và chính xác cao nhất, do đó đây là cấu hình tối ưu nhất trong số các cấu hình thử nghiệm.
