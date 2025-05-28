<<<<<<< HEAD
deeplearing lab
=======
# 📘 Báo cáo bài tập: Mạng Neural đơn giản bằng NumPy trên MNIST

## 1. 📝 Đề bài

Xây dựng và huấn luyện một mạng neural đơn giản **bằng thư viện NumPy**, không sử dụng các framework huấn luyện cao cấp như Keras, PyTorch,... để phân loại chữ số viết tay từ **tập dữ liệu MNIST**.

Yêu cầu:
- Tự cài đặt mô hình (fully-connected feedforward neural network)
- Huấn luyện bằng thuật toán lan truyền ngược (backpropagation)
- Thử nghiệm với nhiều cấu hình siêu tham số (hyperparameters)
- Trình bày kết quả và phân tích

---

## 2. 🧠 Mô hình đề xuất

- Mạng neural gồm:
  - **1 lớp ẩn** (kích thước thay đổi)
  - Hàm kích hoạt: `ReLU` hoặc `Sigmoid`
  - Lớp output dùng `Softmax`
- Dữ liệu đầu vào:
  - Ảnh MNIST kích thước 28×28 → flatten thành vector 784 chiều
- Hàm mất mát:
  - `Cross-entropy`
- Huấn luyện:
  - Dựa trên lan truyền xuôi (forward), lan truyền ngược (backward), và cập nhật bằng `SGD` đơn giản.

---

## 3. ⚙️ Các siêu tham số thử nghiệm

Project chạy thử nghiệm trên 5 cấu hình khác nhau của:
- `batch_size`
- `learning_rate`
- `số lượng neuron lớp ẩn`
- `activation function`

| Batch Size | Learning Rate | Hidden Size | Activation | Mean Acc (%) | Std Acc (%) |
|------------|----------------|-------------|-------------|---------------|---------------|
| 32         | 0.1            | 16          | ReLU        | 94.97         | 0.30          |
| 16         | 0.2            | 64          | Sigmoid     | 97.21         | 0.11          |
| 64         | 0.3            | 32          | ReLU        | 96.45         | 0.18          |
| 32         | 0.4            | 128         | Sigmoid     | 97.60         | 0.06          |
| 16         | 0.5            | 32          | ReLU        | 94.46         | 0.41          |

> Kết quả là **trung bình accuracy** và **độ lệch chuẩn** trên 5 lần chạy cho mỗi cấu hình.

---

## 4. 📊 Nhận xét

- Mô hình hoạt động khá tốt trên MNIST, đạt đến **97.6% độ chính xác** dù chỉ dùng NumPy.
- Các mô hình sử dụng **Sigmoid với hidden layer lớn** (như 128) và **learning rate vừa phải (0.4)** cho kết quả tốt nhất.
- `ReLU` cho tốc độ hội tụ nhanh, nhưng với learning rate cao (`0.5`) thì dễ bị mất ổn định, nhất là với mạng nhỏ.
- Độ lệch chuẩn nhỏ (tầm 0.1–0.4%) cho thấy kết quả khá ổn định qua nhiều lần chạy.

---

## 5. 📁 File nộp

- `src.py`: mã nguồn hoàn chỉnh (training + evaluation)
- `README.md`: báo cáo tổng hợp kết quả

---

## 6. 🔧 Cách chạy

### Cài đặt thư viện
```bash
pip install numpy tensorflow tabulate
>>>>>>> e401e39 (Upload folder LAB_01)
