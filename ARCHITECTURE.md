# Kiến Trúc Dự Án MLOps - Hyperparameter Tuning

## Tổng Quan Kiến Trúc

Dự án MLOps - Hyperparameter Tuning là một hệ thống hoàn chỉnh để phát triển, đánh giá và triển khai các mô hình machine learning với khả năng tự động hóa quá trình tuning siêu tham số. Hệ thống được thiết kế theo nguyên tắc MLOps hiện đại, cho phép theo dõi đầy đủ vòng đời của mô hình từ thử nghiệm đến sản xuất.

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│   Flask API     │────▶│  Tuning Scripts   │────▶│    MLflow Server  │
│   (app.py)      │     │                   │     │                   │
│                 │     │                   │     │                   │
└────────┬────────┘     └────────┬──────────┘     └────────┬──────────┘
         │                       │                         │
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│  Model Library  │     │  Saved Models     │     │  MLflow Tracking  │
│    (mlib.py)    │     │                   │     │                   │
│                 │     │                   │     │                   │
└─────────────────┘     └───────────────────┘     └───────────────────┘
```

## Thành Phần Chính

### 1. Flask API (app.py)
- **Vai trò**: Giao diện người dùng web và REST API
- **Chức năng**:
  - Cung cấp giao diện người dùng trực quan
  - Xử lý yêu cầu tuning từ người dùng
  - Hiển thị kết quả tuning
  - Cung cấp endpoint API cho train và predict
  - **Mới**: Phân loại dữ liệu sử dụng mô hình tốt nhất

### 2. Thư Viện Machine Learning (mlib.py)
- **Vai trò**: Core library cho các chức năng machine learning
- **Chức năng**:
  - Tạo và xử lý dữ liệu
  - Training mô hình
  - Đánh giá hiệu suất
  - Lưu và tải mô hình

### 3. Tuning Scripts
- **Vai trò**: Thực hiện quá trình tuning siêu tham số
- **Thành phần**:
  - `simple_hyperparam_tuning.py`: Tuning đơn giản với không gian tham số định nghĩa sẵn
  - `custom_hyperparam_tuning.py`: Tuning với không gian tham số tùy chỉnh
  - `save_best_model.py`: Lưu mô hình tốt nhất từ kết quả tuning và thông tin chi tiết

### 4. MLflow Scripts
- **Vai trò**: Quản lý và theo dõi thử nghiệm
- **Thành phần**:
  - `mlflow_config.py`: Cấu hình MLflow
  - `mlflow_utils.py`: Tiện ích để làm việc với MLflow
  - `run_mlflow_server.py`: Khởi động MLflow server
  - `restart_mlflow.sh`: Script để khởi động lại MLflow

### 5. Thành Phần Khác
- **Tests**: Bộ test tự động cho các thành phần của hệ thống
- **Templates**: HTML templates cho giao diện web
- **Models**: Lưu trữ mô hình đã huấn luyện
- **Tuning Results**: Lưu kết quả của các lần tuning

## Luồng Dữ Liệu

1. **Luồng Tuning Siêu Tham Số**:
   ```
   Người dùng → Flask API → Tuning Scripts → MLflow Tracking → Lưu kết quả tuning
   ```

2. **Luồng Lưu Mô Hình Tốt Nhất**:
   ```
   Người dùng → Flask API → save_best_model.py → MLflow → Lưu mô hình tốt nhất
   ```

3. **Luồng Dự Đoán API**:
   ```
   Client → Flask API → Tải mô hình → Thực hiện dự đoán → Trả về kết quả
   ```

4. **Luồng Phân Loại Web UI**:
   ```
   Người dùng → Flask UI → Tải mô hình tốt nhất → Phân loại dữ liệu → Hiển thị kết quả
   ```

5. **Luồng Thử Nghiệm**:
   ```
   Tuning Script → Tạo dữ liệu → Thực hiện CV → Log kết quả vào MLflow → Lưu kết quả
   ```

## Cấu Trúc Thư Mục

```
MLOps-Final-Project/
├── app.py                         # Flask API chính
├── mlib.py                        # Thư viện ML core
├── tuning_scripts/                # Thư mục chứa các script tuning
│   ├── __init__.py                    # Đánh dấu là Python package
│   ├── simple_hyperparam_tuning.py    # Script tuning siêu tham số đơn giản
│   ├── custom_hyperparam_tuning.py    # Script tuning siêu tham số tùy chỉnh
│   ├── save_best_model.py             # Script lưu mô hình tốt nhất từ kết quả tuning
│   └── test_save_best_model.py        # Test script cho save_best_model
├── mlflow_scripts/                # Thư mục chứa các script MLflow
│   ├── __init__.py                    # Đánh dấu là Python package
│   ├── mlflow_utils.py                # Tiện ích làm việc với MLflow
│   ├── mlflow_config.py               # Cấu hình MLflow
│   ├── run_mlflow_server.py           # Script khởi động MLflow server
│   ├── restart_mlflow.sh              # Script khởi động lại MLflow server
│   ├── reset_mlflow.py                # Reset và xóa dữ liệu MLflow
│   └── restore_experiments.py         # Khôi phục experiments từ backup
├── models/                        # Thư mục chứa mô hình đã lưu
│   ├── best_model.joblib          # Mô hình tốt nhất đã lưu
│   └── model_info.json            # Thông tin chi tiết về mô hình tốt nhất
├── templates/                     # Templates HTML cho Flask UI
│   ├── index.html                 # Trang chủ ứng dụng
│   └── result_detail.html         # Trang chi tiết kết quả tuning
├── tuning_results/                # Thư mục chứa kết quả tuning
├── tests/                         # Unit tests và Integration tests
│   ├── __init__.py                    # Đánh dấu là Python package
│   ├── test_app_frontend.py           # Test cho giao diện người dùng
│   ├── test_error_fix.py              # Test xử lý lỗi
│   ├── test_integration.py            # Integration tests
│   ├── test_template_validation.py    # Validation tests cho templates
│   └── run_tests.py                   # Script chạy tất cả tests
├── mlflow_data/                   # Dữ liệu MLflow 
├── requirements.txt               # Các thư viện cần thiết
├── Dockerfile                     # Cấu hình container
└── Makefile                       # Các lệnh build và utility
```

## Các Công Nghệ Chính

1. **Backend**: Python, Flask
2. **Machine Learning**: Scikit-learn, Pandas, NumPy, Joblib
3. **Experiment Tracking**: MLflow
4. **Giao Diện Người Dùng**: HTML, Bootstrap, JavaScript
5. **Containerization**: Docker
6. **Testing**: Unittest
7. **Automation**: Make

## Tính Năng Mới: Phân Loại Với Mô Hình Tốt Nhất

Hệ thống đã được bổ sung tính năng mới cho phép người dùng thực hiện phân loại dữ liệu trực tiếp từ giao diện người dùng web, sử dụng mô hình tốt nhất đã được lưu:

1. **Tab Phân Loại trong Giao Diện**: Tab mới trong UI cho phép người dùng nhập hoặc tạo ngẫu nhiên dữ liệu cần phân loại
2. **Endpoint /predict**: Xử lý yêu cầu phân loại, tải mô hình và thực hiện dự đoán
3. **Thông Tin Mô Hình Tốt Nhất**: Hiển thị thông tin chi tiết về mô hình tốt nhất (loại, độ chính xác, F1-score)
4. **Kết Quả Phân Loại Trực Quan**: Hiển thị kết quả phân loại với các biểu đồ xác suất và thông tin chi tiết
5. **Tạo Dữ Liệu Ngẫu Nhiên**: Tính năng tạo dữ liệu mẫu ngẫu nhiên cho việc thử nghiệm nhanh
6. **Lưu Thông Tin Mô Hình Phong Phú**: File `model_info.json` lưu thông tin chi tiết về mô hình tốt nhất
7. **Đăng Ký Mô Hình vào Model Registry**: Mô hình tốt nhất được đăng ký vào MLflow Model Registry để quản lý phiên bản
8. **Hiển Thị Xác Suất Phân Loại**: Hiển thị xác suất chi tiết cho từng lớp bằng thanh tiến trình trực quan
9. **Phân Loại Hàng Loạt**: Hỗ trợ phân loại nhiều mẫu dữ liệu cùng lúc

### Quy Trình Phân Loại Dữ Liệu Mới

1. **Chuẩn Bị Dữ Liệu**:
   - Người dùng nhập dữ liệu JSON theo định dạng yêu cầu
   - Hoặc sử dụng tính năng "Tạo dữ liệu ngẫu nhiên" để tạo dữ liệu mẫu

2. **Gửi Yêu Cầu Phân Loại**:
   - Dữ liệu được gửi đến endpoint `/predict` thông qua form submission
   - Server xác thực dữ liệu đầu vào

3. **Tải và Sử Dụng Mô Hình**:
   - Hệ thống tải mô hình tốt nhất từ `models/best_model.joblib`
   - Đọc thông tin mô hình từ `models/model_info.json`
   - Thực hiện dự đoán và tính toán xác suất

4. **Hiển Thị Kết Quả**:
   - Kết quả phân loại được hiển thị trong giao diện người dùng
   - Mỗi mẫu dữ liệu hiển thị lớp được dự đoán, xác suất, và các thông số khác
   - Thanh tiến trình trực quan hiển thị xác suất cho từng lớp

### Cải Tiến Model Registry

Script `save_best_model.py` đã được nâng cấp để thực hiện:

1. **Tìm Mô Hình Tốt Nhất**: Tự động quét các runs trong MLflow để tìm mô hình có F1-score cao nhất
2. **Lưu Mô Hình và Thông Tin**: Lưu mô hình và thông tin chi tiết vào thư mục `models/`
3. **Đăng Ký vào Model Registry**: Tự động đăng ký mô hình vào MLflow Model Registry với phiên bản
4. **Cập Nhật Metadata**: Lưu thông tin phong phú về mô hình để hỗ trợ việc dự đoán và hiểu mô hình

Tính năng này khép kín toàn bộ vòng đời của một dự án machine learning: từ thử nghiệm, tuning, đến triển khai và sử dụng mô hình cho dự đoán.

## Sự Kết Hợp MLOps

Dự án triển khai các nguyên tắc MLOps thông qua:

1. **Tích hợp liên tục**: Tự động hóa quá trình test và build
2. **Theo dõi thử nghiệm**: Sử dụng MLflow để lưu trữ và theo dõi tất cả các thử nghiệm
3. **Model Registry**: Quản lý và phiên bản hóa các mô hình
4. **Tự động hóa quy trình**: Makefile để tự động hóa các tác vụ thường xuyên
5. **Reproducibility**: Lưu trữ tất cả các tham số và kết quả để đảm bảo khả năng tái tạo
6. **Serving**: Phân loại dữ liệu mới thông qua web UI và API