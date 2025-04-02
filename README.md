# MLOps Final Project - Hyperparameter Tuning

Dự án MLOps với Flask API, MLflow và Hyperparameter Tuning để quản lý vòng đời của model machine learning.

## Cấu trúc dự án

```
MLOps-Final-Project/
├── app.py                         # Flask API chính
├── mlib.py                        # Thư viện ML core
├── tuning_scripts/                # Thư mục chứa các script tuning
│   ├── simple_hyperparam_tuning.py    # Script tuning siêu tham số đơn giản
│   ├── custom_hyperparam_tuning.py    # Script tuning siêu tham số tùy chỉnh
│   ├── save_best_model.py             # Script lưu mô hình tốt nhất từ kết quả tuning
│   └── test_save_best_model.py        # Test script cho save_best_model
├── mlflow_scripts/                # Thư mục chứa các script MLflow
│   ├── mlflow_utils.py                # Tiện ích làm việc với MLflow
│   ├── mlflow_config.py               # Cấu hình MLflow
│   ├── run_mlflow_server.py           # Script khởi động MLflow server
│   ├── restart_mlflow.sh              # Script khởi động lại MLflow server
│   ├── reset_mlflow.py                # Reset và xóa dữ liệu MLflow
│   └── restore_experiments.py         # Khôi phục experiments từ backup
├── models/                        # Thư mục chứa mô hình đã lưu
│   └── best_model.joblib          # Mô hình tốt nhất đã lưu
├── templates/                     # Templates HTML cho Flask UI
│   ├── index.html                 # Trang chủ ứng dụng
│   └── result_detail.html         # Trang chi tiết kết quả tuning
├── tuning_results/                # Thư mục chứa kết quả tuning
├── tests/                         # Unit tests và Integration tests
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

## Yêu cầu hệ thống

- Python 3.9+
- Thư viện: flask, scikit-learn, pandas, numpy, joblib, mlflow, matplotlib

## Cài đặt

1. Tạo môi trường ảo:
```bash
make venv
```

2. Kích hoạt môi trường ảo:
```bash
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Cài đặt dependencies:
```bash
make install
```

## Chạy ứng dụng

1. Khởi động MLflow server:
```bash
make start-mlflow
```

2. Chạy ứng dụng Flask:
```bash
python app.py
```

Mở trình duyệt và truy cập: http://localhost:5001

### Sử dụng Docker

```bash
make docker-build
make docker-run
```

## API Endpoints

### Health Check
- URL: `/health`
- Method: GET
- Response: Trạng thái của ứng dụng và model

### Train Model
- URL: `/train`
- Method: POST
- Body: JSON với trường "data" chứa dữ liệu training
- Response: Metrics sau khi training

### Predict
- URL: `/predict`
- Method: POST
- Body: JSON với trường "data" chứa features
- Response: Predictions và probabilities

### Get Metrics
- URL: `/metrics`
- Method: GET
- Response: Metrics hiện tại của model

## Giao diện người dùng

Ứng dụng cung cấp giao diện người dùng web để:

1. **Chạy Tuning Đơn Giản**:
   - Lựa chọn loại mô hình (Random Forest, Gradient Boosting)
   - Chọn không gian tham số (Tiny, Small, Medium)
   - Chỉ định số lượng mẫu, features và folds

2. **Chạy Tuning Tùy chỉnh**:
   - Tùy chọn chi tiết các siêu tham số cho từng loại mô hình
   - Điều chỉnh quy mô và cấu hình thử nghiệm

3. **Xem kết quả trong MLflow**:
   - Trực tiếp xem báo cáo và phân tích trong MLflow UI
   - So sánh kết quả giữa các thử nghiệm

4. **Phân Loại với Mô Hình Tốt Nhất**:
   - Sử dụng mô hình đã được huấn luyện tốt nhất để phân loại dữ liệu mới
   - Tạo dữ liệu ngẫu nhiên để thử nghiệm nhanh
   - Xem kết quả phân loại và xác suất chi tiết

## Tuning Siêu Tham Số

### Khởi động MLflow Server

```bash
# Khởi động MLflow Server
make start-mlflow

# Khởi động lại MLflow (xóa dữ liệu cũ và khởi động lại)
make reset-mlflow
```

### Thực hiện Tuning từ Command Line

```bash
# Tuning đơn giản với tùy chọn mặc định (Random Forest, không gian small)
python simple_hyperparam_tuning.py
# hoặc
make simple-tuning

# Tuning với không gian tham số nhỏ (nhanh hơn)
python simple_hyperparam_tuning.py --space tiny
# hoặc
make simple-tuning-tiny

# Tuning với Gradient Boosting
python simple_hyperparam_tuning.py --model gradient_boosting
# hoặc
make simple-tuning-gb

# Tuning với nhiều mẫu dữ liệu hơn
python simple_hyperparam_tuning.py --samples 2000 --features 30
# hoặc
make simple-tuning-large

# Tuning tùy chỉnh với file tham số 
python custom_hyperparam_tuning.py --model random_forest --params-file my_params.json
```

Các tùy chọn có sẵn cho simple_hyperparam_tuning.py:
- `--model`: Loại mô hình (`random_forest`, `gradient_boosting`)
- `--space`: Kích thước không gian tham số (`tiny`, `small`, `medium`)
- `--samples`: Số lượng mẫu dữ liệu
- `--features`: Số lượng features
- `--cv`: Số fold cross-validation
- `--no-mlflow`: Không sử dụng MLflow tracking

### Xem Kết Quả

#### MLflow UI

Truy cập: http://localhost:5002

#### Kết Quả Cục Bộ

Tất cả kết quả tuning đều được lưu tự động trong thư mục `tuning_results/`.

## Lưu và Quản lý Mô hình Tốt nhất

Sau khi thực hiện tuning, bạn có thể lưu mô hình tốt nhất vào Model Registry của MLflow và ổ đĩa cục bộ:

```bash
# Lưu mô hình tốt nhất từ kết quả tuning
make save-best-model
```

Script `save_best_model.py` sẽ:
1. Tìm kiếm trong MLflow experiment `tuning_experiment` để xác định run có F1-score cao nhất
2. Đăng ký mô hình này vào MLflow Model Registry
3. Lưu mô hình vào thư mục `models/` dưới dạng file joblib
4. Lưu thông tin chi tiết về mô hình tốt nhất vào file `model_info.json`

## Phân Loại Dữ Liệu với Mô Hình Tốt Nhất

Sau khi lưu mô hình tốt nhất, bạn có thể sử dụng mô hình này để phân loại dữ liệu mới:

1. **Thông qua giao diện web**:
   - Truy cập tab "Phân Loại" trên giao diện người dùng
   - Nhập dữ liệu đầu vào dưới dạng JSON hoặc sử dụng nút "Tạo Dữ Liệu Ngẫu Nhiên"
   - Nhấn "Phân Loại" để xem kết quả

2. **Thông qua API**:
   ```bash
   curl -X POST http://localhost:5001/predict \
     -d "feature_data=[{\"feature_1\": 0.5, \"feature_2\": 0.3, ...}]"
   ```

3. **Sử dụng lệnh test**:
   ```bash
   make test-predict
   ```

Kết quả phân loại bao gồm:
- Lớp được dự đoán (0 hoặc 1)
- Xác suất của dự đoán
- Thông tin chi tiết về các đặc trưng đầu vào
- Biểu đồ xác suất cho từng lớp

## Development

### Testing
```bash
make test
```

### Linting
```bash
make lint
```

### Format Code
```bash
make format
```

### Clean Up
```bash
make clean  # Dọn dẹp cache, __pycache__, v.v.
make clean-mlflow  # Xóa dữ liệu MLflow và kết quả tuning
```

## MLflow Integration

MLflow được tích hợp chặt chẽ vào dự án để:
- Theo dõi thử nghiệm (Experiment Tracking)
- Quản lý mô hình (Model Registry)
- Lưu trữ artifacts
- So sánh kết quả thử nghiệm

Mỗi lần chạy tuning, kết quả sẽ được:
1. Tự động lưu vào MLflow Tracking Server
2. Lưu trữ dưới dạng JSON trong thư mục `tuning_results/`
3. Hiển thị trên giao diện web của ứng dụng

MLflow UI có thể truy cập tại: http://localhost:5002

## License

MIT