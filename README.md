# MLOps Lab 02

Dự án MLOps với Flask API và MLflow để quản lý vòng đời của model machine learning.

## Cấu trúc dự án

```
MLOps-Lab02/
├── app.py                       # Flask API
├── mlib.py                      # Thư viện ML
├── simple_hyperparam_tuning.py  # Script tuning siêu tham số đơn giản
├── save_best_model.py           # Script lưu mô hình tốt nhất từ kết quả tuning
├── models/                      # Thư mục chứa mô hình đã lưu
├── mlflow_data/                 # Dữ liệu MLflow
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container configuration
└── Makefile                     # Build và utility commands
```

## Yêu cầu hệ thống

- Python 3.12
- Docker (tùy chọn)

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

Hoặc sử dụng mô hình tốt nhất đã được lưu:
```bash
# Lưu mô hình tốt nhất từ kết quả tuning trước
make save-best-model

# Chạy ứng dụng với mô hình tốt nhất
python app.py --use-best-model
```

Hoặc sử dụng Docker:
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

## Tuning Siêu Tham Số

Script `simple_hyperparam_tuning.py` cung cấp giải pháp tuning đơn giản, dễ sử dụng và hiệu quả với nhiều tùy chọn linh hoạt.

### 1. Khởi động MLflow Server

```bash
# Khởi động MLflow Server
make start-mlflow

# Khởi động lại MLflow (xóa dữ liệu cũ và khởi động lại)
make reset-mlflow
```

### 2. Thực hiện Tuning

```bash
# Chạy với tùy chọn mặc định (RandomForest, không gian small)
python simple_hyperparam_tuning.py
# hoặc
make simple-tuning

# Tuning với không gian tham số nhỏ hơn (nhanh hơn)
python simple_hyperparam_tuning.py --space tiny
# hoặc
make simple-tuning-tiny

# Tuning với GradientBoosting
python simple_hyperparam_tuning.py --model gradient_boosting
# hoặc
make simple-tuning-gb

# Tuning với nhiều mẫu dữ liệu hơn
python simple_hyperparam_tuning.py --samples 2000 --features 30
# hoặc
make simple-tuning-large
```

Các tùy chọn có sẵn:
- `--model`: Loại mô hình (`random_forest`, `gradient_boosting`)
- `--space`: Kích thước không gian tham số (`tiny`, `small`, `medium`)
- `--samples`: Số lượng mẫu dữ liệu
- `--features`: Số lượng features
- `--cv`: Số fold cross-validation
- `--no-mlflow`: Không sử dụng MLflow tracking

Tất cả kết quả tuning được lưu trong:
1. MLflow UI: http://localhost:5002
2. Thư mục `tuning_results/` dưới dạng file JSON

### 3. Xem Kết Quả

#### MLflow UI

Truy cập: http://localhost:5002

#### Kết Quả Cục Bộ

Tất cả kết quả tuning đều được lưu tự động trong thư mục `tuning_results/`.

### 4. Dọn Dẹp Dữ Liệu

```bash
# Xóa dữ liệu MLflow và kết quả tuning
make clean-mlflow
```

## Lưu và Quản lý Mô hình Tốt nhất

Sau khi thực hiện tuning, bạn có thể lưu mô hình tốt nhất vào Model Registry của MLflow và ổ đĩa cục bộ bằng cách sử dụng:

```bash
# Lưu mô hình tốt nhất từ kết quả tuning
make save-best-model
```

Script `save_best_model.py` sẽ:
1. Tìm kiếm trong MLflow experiment `tuning_experiment` để xác định run có F1-score cao nhất
2. Đăng ký mô hình này vào MLflow Model Registry
3. Lưu mô hình vào thư mục `models/` dưới dạng file joblib
4. Lưu thông tin chi tiết về mô hình tốt nhất vào `models/best_model_info.json`

Thông tin mô hình bao gồm:
- Run ID
- Loại mô hình (random_forest, gradient_boosting, v.v.)
- Các metrics (F1-score, accuracy, precision, recall)
- Các siêu tham số tối ưu
- Thông tin về phiên bản mô hình trong Model Registry

Để kiểm tra chức năng này, bạn có thể chạy:

```bash
# Chạy unit test cho save_best_model.py
make test-save-model
```

Các unit test sẽ kiểm tra:
- Tìm và lưu mô hình thành công
- Xử lý trường hợp không tìm thấy experiment
- Xử lý trường hợp không có runs nào
- Xử lý trường hợp thiếu metrics
- Đăng ký mô hình vào Model Registry

## Development

### Code Style
- Sử dụng Black cho code formatting
- Sử dụng Pylint cho linting

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

## MLflow

MLflow được cấu hình để theo dõi:
- Metrics (accuracy, precision, recall, f1)
- Model parameters
- Model artifacts

MLflow UI có thể truy cập tại: http://localhost:5002

## License

MIT