# Sơ Đồ Tuần Tự (Sequence Diagram)

Tài liệu này mô tả các luồng hoạt động chính trong dự án MLOps - Hyperparameter Tuning thông qua các sơ đồ tuần tự. Các sơ đồ này giúp hiểu rõ hơn về cách các thành phần tương tác với nhau và luồng dữ liệu giữa chúng.

## 1. Luồng Tuning Siêu Tham Số Đơn Giản

```
┌──────────┐      ┌──────────┐      ┌──────────────────────┐      ┌───────────┐      ┌──────────────┐
│          │      │          │      │                      │      │           │      │              │
│ Người    │      │ Flask    │      │ simple_hyperparam_   │      │ MLflow    │      │ Filesystem   │
│ dùng     │      │ (app.py) │      │ tuning.py           │      │ Server    │      │              │
│          │      │          │      │                      │      │           │      │              │
└────┬─────┘      └────┬─────┘      └──────────┬───────────┘      └─────┬─────┘      └──────┬───────┘
     │                 │                        │                        │                   │
     │ Gửi form với    │                        │                        │                   │
     │ tham số tuning  │                        │                        │                   │
     │────────────────>│                        │                        │                   │
     │                 │                        │                        │                   │
     │                 │ Gọi subprocess với     │                        │                   │
     │                 │ tham số đầu vào        │                        │                   │
     │                 │───────────────────────>│                        │                   │
     │                 │                        │                        │                   │
     │                 │                        │ Thiết lập MLflow       │                   │
     │                 │                        │ tracking               │                   │
     │                 │                        │───────────────────────>│                   │
     │                 │                        │                        │                   │
     │                 │                        │ Tạo dữ liệu mẫu        │                   │
     │                 │                        │─────────────┐          │                   │
     │                 │                        │             │          │                   │
     │                 │                        │<────────────┘          │                   │
     │                 │                        │                        │                   │
     │                 │                        │ Thực hiện Grid Search  │                   │
     │                 │                        │─────────────┐          │                   │
     │                 │                        │             │          │                   │
     │                 │                        │<────────────┘          │                   │
     │                 │                        │                        │                   │
     │                 │                        │ Log thông số và metrics│                   │
     │                 │                        │───────────────────────>│                   │
     │                 │                        │                        │                   │
     │                 │                        │ Log mô hình            │                   │
     │                 │                        │───────────────────────>│                   │
     │                 │                        │                        │                   │
     │                 │                        │ Lưu kết quả vào file   │                   │
     │                 │                        │─────────────────────────────────────────>  │
     │                 │                        │                        │                   │
     │                 │ Trả về kết quả tuning  │                        │                   │
     │                 │<───────────────────────│                        │                   │
     │                 │                        │                        │                   │
     │ Hiển thị kết quả│                        │                        │                   │
     │ và link MLflow  │                        │                        │                   │
     │<────────────────│                        │                        │                   │
     │                 │                        │                        │                   │
```

## 2. Luồng Tuning Siêu Tham Số Tùy Chỉnh

```
┌──────────┐      ┌──────────┐      ┌──────────────────────┐      ┌───────────┐      ┌──────────────┐
│          │      │          │      │                      │      │           │      │              │
│ Người    │      │ Flask    │      │ custom_hyperparam_   │      │ MLflow    │      │ Filesystem   │
│ dùng     │      │ (app.py) │      │ tuning.py           │      │ Server    │      │              │
│          │      │          │      │                      │      │           │      │              │
└────┬─────┘      └────┬─────┘      └──────────┬───────────┘      └─────┬─────┘      └──────┬───────┘
     │                 │                        │                        │                   │
     │ Gửi form với    │                        │                        │                   │
     │ tham số tuning  │                        │                        │                   │
     │ tùy chỉnh       │                        │                        │                   │
     │────────────────>│                        │                        │                   │
     │                 │                        │                        │                   │
     │                 │ Tạo tạm file JSON với  │                        │                   │
     │                 │ tham số tùy chỉnh      │                        │                   │
     │                 │─────────────────────────────────────────────────────────────────>  │
     │                 │                        │                        │                   │
     │                 │ Gọi subprocess với     │                        │                   │
     │                 │ đường dẫn file JSON    │                        │                   │
     │                 │───────────────────────>│                        │                   │
     │                 │                        │                        │                   │
     │                 │                        │ Đọc file tham số JSON  │                   │
     │                 │                        │<────────────────────────────────────────>  │
     │                 │                        │                        │                   │
     │                 │                        │ Thiết lập MLflow       │                   │
     │                 │                        │ tracking               │                   │
     │                 │                        │───────────────────────>│                   │
     │                 │                        │                        │                   │
     │                 │                        │ Tạo dữ liệu mẫu        │                   │
     │                 │                        │─────────────┐          │                   │
     │                 │                        │             │          │                   │
     │                 │                        │<────────────┘          │                   │
     │                 │                        │                        │                   │
     │                 │                        │ Thực hiện Grid Search  │                   │
     │                 │                        │ với tham số tùy chỉnh  │                   │
     │                 │                        │─────────────┐          │                   │
     │                 │                        │             │          │                   │
     │                 │                        │<────────────┘          │                   │
     │                 │                        │                        │                   │
     │                 │                        │ Log thông số và metrics│                   │
     │                 │                        │───────────────────────>│                   │
     │                 │                        │                        │                   │
     │                 │                        │ Log mô hình            │                   │
     │                 │                        │───────────────────────>│                   │
     │                 │                        │                        │                   │
     │                 │                        │ Lưu kết quả vào file   │                   │
     │                 │                        │─────────────────────────────────────────>  │
     │                 │                        │                        │                   │
     │                 │ Trả về kết quả tuning  │                        │                   │
     │                 │<───────────────────────│                        │                   │
     │                 │                        │                        │                   │
     │ Hiển thị kết quả│                        │                        │                   │
     │ và link MLflow  │                        │                        │                   │
     │<────────────────│                        │                        │                   │
     │                 │                        │                        │                   │
```

## 3. Luồng Lưu Mô Hình Tốt Nhất

```
┌──────────┐      ┌──────────┐      ┌──────────────────┐      ┌───────────┐      ┌──────────────┐
│          │      │          │      │                  │      │           │      │              │
│ Người    │      │ Flask    │      │ save_best_model  │      │ MLflow    │      │ Filesystem   │
│ dùng     │      │ (app.py) │      │ .py              │      │ Server    │      │              │
│          │      │          │      │                  │      │           │      │              │
└────┬─────┘      └────┬─────┘      └────────┬─────────┘      └─────┬─────┘      └──────┬───────┘
     │                 │                      │                      │                   │
     │ Nhấn nút lưu    │                      │                      │                   │
     │ mô hình tốt nhất│                      │                      │                   │
     │────────────────>│                      │                      │                   │
     │                 │                      │                      │                   │
     │                 │ Gọi subprocess      │                      │                   │
     │                 │ save_best_model.py   │                      │                   │
     │                 │─────────────────────>│                      │                   │
     │                 │                      │                      │                   │
     │                 │                      │ Thiết lập MLflow     │                   │
     │                 │                      │ tracking            │                   │
     │                 │                      │─────────────────────>│                   │
     │                 │                      │                      │                   │
     │                 │                      │ Tìm experiment       │                   │
     │                 │                      │ tuning_experiment    │                   │
     │                 │                      │─────────────────────>│                   │
     │                 │                      │                      │                   │
     │                 │                      │ Tìm run với          │                   │
     │                 │                      │ F1-score cao nhất    │                   │
     │                 │                      │─────────────────────>│                   │
     │                 │                      │                      │                   │
     │                 │                      │ Lấy thông tin run    │                   │
     │                 │                      │<─────────────────────│                   │
     │                 │                      │                      │                   │
     │                 │                      │ Đăng ký mô hình vào  │                   │
     │                 │                      │ Model Registry       │                   │
     │                 │                      │─────────────────────>│                   │
     │                 │                      │                      │                   │
     │                 │                      │ Tải mô hình          │                   │
     │                 │                      │<─────────────────────│                   │
     │                 │                      │                      │                   │
     │                 │                      │ Lưu mô hình vào      │                   │
     │                 │                      │ models/              │                   │
     │                 │                      │─────────────────────────────────────────>│
     │                 │                      │                      │                   │
     │                 │                      │ Lưu thông tin chi    │                   │
     │                 │                      │ tiết vào JSON        │                   │
     │                 │                      │─────────────────────────────────────────>│
     │                 │                      │                      │                   │
     │                 │ Trả về kết quả       │                      │                   │
     │                 │<─────────────────────│                      │                   │
     │                 │                      │                      │                   │
     │ Hiển thị thông  │                      │                      │                   │
     │ báo thành công  │                      │                      │                   │
     │<────────────────│                      │                      │                   │
     │                 │                      │                      │                   │
```

## 4. Luồng Dự Đoán Thông Qua API

```
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────────┐
│          │      │          │      │          │      │              │
│ Client   │      │ Flask    │      │ ML Model │      │ Filesystem   │
│ (API)    │      │ (app.py) │      │ (mlib.py)│      │              │
│          │      │          │      │          │      │              │
└────┬─────┘      └────┬─────┘      └────┬─────┘      └──────┬───────┘
     │                 │                  │                   │
     │ POST /predict   │                  │                   │
     │ (JSON data)     │                  │                   │
     │────────────────>│                  │                   │
     │                 │                  │                   │
     │                 │ Validate dữ liệu │                   │
     │                 │─────────┐        │                   │
     │                 │         │        │                   │
     │                 │<────────┘        │                   │
     │                 │                  │                   │
     │                 │ Tải mô hình      │                   │
     │                 │ nếu cần          │                   │
     │                 │─────────────────────────────────────>│
     │                 │                  │                   │
     │                 │ Trả về mô hình   │                   │
     │                 │<─────────────────────────────────────│
     │                 │                  │                   │
     │                 │ Gọi predict()    │                   │
     │                 │─────────────────>│                   │
     │                 │                  │                   │
     │                 │                  │ Tiền xử lý dữ liệu│
     │                 │                  │─────────┐         │
     │                 │                  │         │         │
     │                 │                  │<────────┘         │
     │                 │                  │                   │
     │                 │                  │ Thực hiện dự đoán │
     │                 │                  │─────────┐         │
     │                 │                  │         │         │
     │                 │                  │<────────┘         │
     │                 │                  │                   │
     │                 │                  │ Tính probabilities│
     │                 │                  │─────────┐         │
     │                 │                  │         │         │
     │                 │                  │<────────┘         │
     │                 │                  │                   │
     │                 │ Trả về kết quả   │                   │
     │                 │<─────────────────│                   │
     │                 │                  │                   │
     │ JSON response   │                  │                   │
     │ với predictions │                  │                   │
     │<────────────────│                  │                   │
     │                 │                  │                   │
```

## 5. Luồng Khởi Động MLflow Server

```
┌──────────┐      ┌────────────────┐      ┌────────────────────┐      ┌──────────────┐
│          │      │                │      │                    │      │              │
│ Người    │      │ run_mlflow_    │      │ mlflow_utils.py    │      │ Filesystem   │
│ dùng     │      │ server.py      │      │                    │      │              │
│          │      │                │      │                    │      │              │
└────┬─────┘      └───────┬────────┘      └──────────┬─────────┘      └──────┬───────┘
     │                    │                           │                       │
     │ make start-mlflow  │                           │                       │
     │───────────────────>│                           │                       │
     │                    │                           │                       │
     │                    │ Gọi setup_mlflow_tracking │                       │
     │                    │──────────────────────────>│                       │
     │                    │                           │                       │
     │                    │                           │ Kiểm tra MLflow đã    │
     │                    │                           │ chạy chưa             │
     │                    │                           │─────────┐             │
     │                    │                           │         │             │
     │                    │                           │<────────┘             │
     │                    │                           │                       │
     │                    │                           │ Kiểm tra cổng khả dụng│
     │                    │                           │─────────┐             │
     │                    │                           │         │             │
     │                    │                           │<────────┘             │
     │                    │                           │                       │
     │                    │                           │ Thiết lập SQLite      │
     │                    │                           │ backend               │
     │                    │                           │─────────────────────────>
     │                    │                           │                       │
     │                    │                           │ Tạo experiment mặc    │
     │                    │                           │ định nếu cần         │
     │                    │                           │─────────┐             │
     │                    │                           │         │             │
     │                    │                           │<────────┘             │
     │                    │                           │                       │
     │                    │                           │ Khởi động MLflow      │
     │                    │                           │ server với subprocess │
     │                    │                           │─────────┐             │
     │                    │                           │         │             │
     │                    │                           │<────────┘             │
     │                    │                           │                       │
     │                    │                           │ Đợi server khởi động  │
     │                    │                           │─────────┐             │
     │                    │                           │         │             │
     │                    │                           │<────────┘             │
     │                    │                           │                       │
     │                    │ Trả về trạng thái         │                       │
     │                    │<─────────────────────────│                       │
     │                    │                           │                       │
     │ Hiển thị thông báo │                           │                       │
     │ MLflow đang chạy   │                           │                       │
     │<──────────────────│                           │                       │
     │                    │                           │                       │
```

## 6. Luồng Xem Kết Quả Tuning

```
┌──────────┐      ┌──────────┐      ┌──────────────┐
│          │      │          │      │              │
│ Người    │      │ Flask    │      │ Filesystem   │
│ dùng     │      │ (app.py) │      │              │
│          │      │          │      │              │
└────┬─────┘      └────┬─────┘      └──────┬───────┘
     │                 │                   │
     │ GET /view_result│                   │
     │ /<filename>     │                   │
     │────────────────>│                   │
     │                 │                   │
     │                 │ Đọc file JSON     │
     │                 │ từ tuning_results/│
     │                 │──────────────────>│
     │                 │                   │
     │                 │ Trả về nội dung   │
     │                 │ file              │
     │                 │<──────────────────│
     │                 │                   │
     │                 │ Xử lý dữ liệu     │
     │                 │─────────┐         │
     │                 │         │         │
     │                 │<────────┘         │
     │                 │                   │
     │                 │ Render template   │
     │                 │ result_detail.html│
     │                 │─────────┐         │
     │                 │         │         │
     │                 │<────────┘         │
     │                 │                   │
     │ Hiển thị trang  │                   │
     │ chi tiết kết quả│                   │
     │<────────────────│                   │
     │                 │                   │
```

## 7. Luồng Phân Loại Với Mô Hình Tốt Nhất

```
┌──────────┐      ┌──────────┐      ┌──────────────┐
│          │      │          │      │              │
│ Người    │      │ Flask    │      │ Filesystem   │
│ dùng     │      │ (app.py) │      │              │
│          │      │          │      │              │
└────┬─────┘      └────┬─────┘      └──────┬───────┘
     │                 │                   │
     │ Gửi form với    │                   │
     │ dữ liệu cần     │                   │
     │ phân loại       │                   │
     │────────────────>│                   │
     │                 │                   │
     │                 │ Kiểm tra mô hình  │                  
     │                 │ đã tồn tại chưa   │                  
     │                 │──────────────────>│                  
     │                 │                   │                  
     │                 │ Trả về trạng thái │                  
     │                 │ tồn tại của mô hình│                 
     │                 │<──────────────────│                  
     │                 │                   │                  
     │                 │ Tải mô hình       │                  
     │                 │ best_model.joblib │                  
     │                 │──────────────────>│                  
     │                 │                   │                  
     │                 │ Trả về mô hình    │                  
     │                 │<──────────────────│                  
     │                 │                   │                  
     │                 │ Tải thông tin     │                  
     │                 │ model_info.json   │                  
     │                 │──────────────────>│                  
     │                 │                   │                  
     │                 │ Trả về thông tin  │                  
     │                 │<──────────────────│                  
     │                 │                   │                  
     │                 │ Chuyển đổi dữ liệu│                  
     │                 │ đầu vào           │                  
     │                 │─────────┐         │                  
     │                 │         │         │                  
     │                 │<────────┘         │                  
     │                 │                   │                  
     │                 │ Thực hiện dự đoán │                  
     │                 │ với mô hình       │                  
     │                 │─────────┐         │                  
     │                 │         │         │                  
     │                 │<────────┘         │                  
     │                 │                   │                  
     │                 │ Tính xác suất     │                  
     │                 │ các lớp           │                  
     │                 │─────────┐         │                  
     │                 │         │         │                  
     │                 │<────────┘         │                  
     │                 │                   │                  
     │                 │ Chuẩn bị kết quả  │                  
     │                 │ phân loại         │                  
     │                 │─────────┐         │                  
     │                 │         │         │                  
     │                 │<────────┘         │                  
     │                 │                   │                  
     │                 │ Render trang với  │                  
     │                 │ kết quả phân loại │                  
     │                 │─────────┐         │                  
     │                 │         │         │                  
     │                 │<────────┘         │                  
     │                 │                   │                  
     │                 │ Hiển thị trang    │                  
     │                 │ với kết quả       │                  
     │                 │<──────────────────│                  
     │                 │                   │                  
```

## 8. Luồng Phân Loại Dữ Liệu Mới

```
┌──────────┐      ┌──────────┐      ┌────────────┐      ┌──────────────┐
│          │      │          │      │            │      │              │
│ Người    │      │ Flask    │      │ ML Model   │      │ Filesystem   │
│ dùng     │      │ (app.py) │      │            │      │              │
│          │      │          │      │            │      │              │
└────┬─────┘      └────┬─────┘      └─────┬──────┘      └──────┬───────┘
     │                 │                   │                    │
     │ Nhập dữ liệu    │                   │                    │
     │ hoặc tạo dữ liệu│                   │                    │
     │ ngẫu nhiên     │                   │                    │
     │────────────────>│                   │                    │
     │                 │                   │                    │
     │                 │ Gửi form POST     │                    │
     │                 │ đến /predict      │                    │
     │                 │─────────┐         │                    │
     │                 │         │         │                    │
     │                 │<────────┘         │                    │
     │                 │                   │                    │
     │                 │ Kiểm tra mô hình  │                    │
     │                 │ có tồn tại không  │                    │
     │                 │────────────────────────────────────────>│
     │                 │                   │                    │
     │                 │ Tải model_info.json│                   │
     │                 │<───────────────────────────────────────│
     │                 │                   │                    │
     │                 │ Tải best_model.joblib                 │
     │                 │<───────────────────────────────────────│
     │                 │                   │                    │
     │                 │ Chuẩn bị dữ liệu  │                    │
     │                 │ đầu vào           │                    │
     │                 │─────────┐         │                    │
     │                 │         │         │                    │
     │                 │<────────┘         │                    │
     │                 │                   │                    │
     │                 │ Gọi model.predict() và                │
     │                 │ predict_proba()   │                    │
     │                 │─────────────────>│                    │
     │                 │                   │                    │
     │                 │                   │ Tiền xử lý dữ liệu │
     │                 │                   │─────────┐          │
     │                 │                   │         │          │
     │                 │                   │<────────┘          │
     │                 │                   │                    │
     │                 │                   │ Thực hiện dự đoán  │
     │                 │                   │─────────┐          │
     │                 │                   │         │          │
     │                 │                   │<────────┘          │
     │                 │                   │                    │
     │                 │                   │ Tính xác suất      │
     │                 │                   │─────────┐          │
     │                 │                   │         │          │
     │                 │                   │<────────┘          │
     │                 │                   │                    │
     │                 │ Trả về kết quả    │                    │
     │                 │<─────────────────│                    │
     │                 │                   │                    │
     │                 │ Định dạng kết quả │                    │
     │                 │ cho template      │                    │
     │                 │─────────┐         │                    │
     │                 │         │         │                    │
     │                 │<────────┘         │                    │
     │                 │                   │                    │
     │ Hiển thị kết quả│                   │                    │
     │ phân loại với   │                   │                    │
     │ các xác suất    │                   │                    │
     │<────────────────│                   │                    │
     │                 │                   │                    │
```

## Tổng Quan Các Luồng Hoạt Động

Các sơ đồ tuần tự trên cho thấy luồng hoạt động chính của hệ thống MLOps - Hyperparameter Tuning:

1. **Luồng Tuning Siêu Tham Số**: Bắt đầu từ người dùng, qua Flask API, thực hiện tuning và lưu kết quả vào MLflow
2. **Luồng Lưu Mô Hình Tốt Nhất**: Tìm mô hình có F1-score cao nhất từ MLflow và lưu vào thư mục models/
3. **Luồng Dự Đoán Thông Qua API**: Tiếp nhận request, tải mô hình, thực hiện dự đoán và trả về kết quả
4. **Luồng Khởi Động MLflow Server**: Thiết lập và khởi động MLflow server
5. **Luồng Xem Kết Quả Tuning**: Hiển thị kết quả chi tiết của một lần tuning cụ thể
6. **Luồng Phân Loại Với Mô Hình Tốt Nhất**: Sử dụng mô hình tốt nhất đã lưu để phân loại dữ liệu mới thông qua giao diện người dùng
7. **Luồng Phân Loại Dữ Liệu Mới**: Thực hiện phân loại dữ liệu mới bằng mô hình tốt nhất vào cuối tài liệu

Các luồng này hoạt động độc lập nhưng liên kết với nhau thông qua dữ liệu được lưu trữ trong MLflow và hệ thống file. 

ID Lỗi: BUG-01
Mô tả: [Mô tả ngắn gọn về lỗi]
Bước tái hiện:
1. [Bước 1]
2. [Bước 2]
3. [Bước 3]

Kết quả thực tế: [Mô tả những gì xảy ra]
Kết quả mong đợi: [Mô tả những gì nên xảy ra]
Mức độ nghiêm trọng: [Thấp/Trung bình/Cao/Nghiêm trọng] 