# MLOps Project - Hyperparameter Tuning

Dự án MLOps với Flask API, MLflow và Hyperparameter Tuning để huấn luyện và triển khai mô hình machine learning.

## Cấu trúc dự án

```
MLOps-Lab02/
├── app.py                         # Flask API chính
├── mlib.py                        # Thư viện ML core
├── tuning_scripts/                # Thư mục chứa các script tuning
│   ├── simple_hyperparam_tuning.py    # Script tuning siêu tham số đơn giản
│   ├── custom_hyperparam_tuning.py    # Script tuning siêu tham số tùy chỉnh
│   ├── save_best_model.py             # Script lưu mô hình tốt nhất từ kết quả tuning
│   └── register_model.py              # Script đăng ký mô hình
├── mlflow_scripts/                # Thư mục chứa các script MLflow
│   ├── mlflow_utils.py                # Tiện ích làm việc với MLflow
│   ├── mlflow_config.py               # Cấu hình MLflow
│   ├── run_mlflow_server.py           # Script khởi động MLflow server
│   └── restart_mlflow.sh              # Script khởi động lại MLflow server
├── models/                        # Thư mục chứa mô hình đã lưu
│   └── best_model.joblib          # Mô hình tốt nhất đã lưu
├── templates/                     # Templates HTML cho Flask UI
│   ├── index.html                 # Trang chủ ứng dụng (3 tabs)
│   └── result_detail.html         # Trang chi tiết kết quả tuning
├── tuning_results/                # Thư mục chứa kết quả tuning
├── tests/                         # Unit tests và Integration tests
├── mlflow_data/                   # Dữ liệu MLflow 
├── requirements.txt               # Các thư viện cần thiết
└── Makefile                       # Các lệnh build và tiện ích
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

Nếu cần cập nhật các tệp trong container Docker:
```bash
make docker-cp-files
```

### Docker Hub Image

Ứng dụng đã được đóng gói và đẩy lên Docker Hub, bạn có thể sử dụng trực tiếp thông qua lệnh:

```bash
# Kéo image từ Docker Hub
docker pull fistdat/mlops-flask:v1.6

# Chạy container từ image
docker run -d -p 5001:5001 --name mlops-flask fistdat/mlops-flask:v1.6
```

Image Docker có sẵn tại:
- `fistdat/mlops-flask:latest` - phiên bản mới nhất
- `fistdat/mlops-flask:v1.6` - phiên bản ổn định v1.6

## Giao diện người dùng

Ứng dụng cung cấp giao diện người dùng web với 3 tab chính:

1. **Tuning Đơn Giản**:
   - Lựa chọn loại mô hình (Random Forest, Gradient Boosting)
   - Chọn không gian tham số (Tiny, Small, Medium)
   - Chỉ định số lượng mẫu, features và folds
   - Theo dõi tiến trình và kết quả từ MLflow

2. **Tuning Tùy chỉnh**:
   - Tùy chọn chi tiết các siêu tham số cho từng loại mô hình
   - Điều chỉnh quy mô và cấu hình thử nghiệm
   - Kiểm soát tối ưu hóa siêu tham số

3. **Phân Loại**:
   - Sử dụng mô hình đã được huấn luyện tốt nhất để phân loại dữ liệu mới
   - Tạo dữ liệu ngẫu nhiên để thử nghiệm nhanh
   - Xem kết quả phân loại và xác suất chi tiết

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
- Body: Form data với trường "feature_data" chứa features dưới dạng JSON
- Response: Kết quả phân loại và xác suất

### Get Metrics
- URL: `/metrics`
- Method: GET
- Response: Metrics hiện tại của model

## Tuning Siêu Tham Số

### Thực hiện Tuning từ Command Line

```bash
# Tuning đơn giản với tùy chọn mặc định (Random Forest, không gian small)
python tuning_scripts/simple_hyperparam_tuning.py
# hoặc
make simple-tuning

# Tuning với không gian tham số nhỏ (nhanh hơn)
python tuning_scripts/simple_hyperparam_tuning.py --space tiny
# hoặc
make simple-tuning-tiny

# Tuning với Gradient Boosting
python tuning_scripts/simple_hyperparam_tuning.py --model gradient_boosting
# hoặc
make simple-tuning-gb

# Tuning với nhiều mẫu dữ liệu hơn
python tuning_scripts/simple_hyperparam_tuning.py --samples 2000 --features 30
# hoặc
make simple-tuning-large
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

## Lưu Mô hình Tốt nhất

Sau khi thực hiện tuning, bạn có thể lưu mô hình tốt nhất:

```bash
# Lưu mô hình tốt nhất từ kết quả tuning
make save-best-model
```

Script `save_best_model.py` sẽ:
1. Tìm kiếm trong MLflow experiment `tuning_experiment` để xác định run có F1-score cao nhất
2. Lưu mô hình vào thư mục `models/` dưới dạng file joblib
3. Lưu thông tin chi tiết về mô hình tốt nhất vào file `model_info.json`

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
make clean          # Dọn dẹp cache, __pycache__, v.v.
make clean-mlflow   # Xóa dữ liệu MLflow và kết quả tuning
make clean-all      # Xóa tất cả, kể cả models
```