# Thiết Kế Kiến Trúc Tổng Thể Của Hệ Thống MLOps-Lab02

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT                                     │
│  (Web Browser, Công cụ API như Postman, Curl, Ứng dụng khác)        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP Requests (GET, POST)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FLASK API (app.py)                           │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  ┌──────────┐  │
│  │  /health    │   │   /train    │   │  /predict   │  │ /metrics │  │
│  │  Endpoint   │   │  Endpoint   │   │  Endpoint   │  │ Endpoint │  │
│  └─────────────┘   └─────────────┘   └─────────────┘  └──────────┘  │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Gọi các phương thức ML
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ML LIBRARY (mlib.py)                             │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  ┌──────────┐  │
│  │  MLModel    │   │ Data        │   │ Model       │  │ Model    │  │
│  │  Class      │   │ Processing  │   │ Training    │  │Evaluation│  │
│  └─────────────┘   └─────────────┘   └─────────────┘  └──────────┘  │
│                                                                      │
└─────────────┬─────────────────────────────────────┬─────────────────┘
              │                                     │
              ▼                                     ▼
┌─────────────────────────┐             ┌─────────────────────────────┐
│     SCIKIT-LEARN        │             │         MLFLOW               │
│                         │             │                              │
│  ┌─────────────────┐    │             │  ┌─────────────────────┐    │
│  │RandomForest     │    │             │  │Experiment Tracking  │    │
│  │Classifier       │    │  Logging    │  │                     │    │
│  └─────────────────┘    │  Metrics    │  └─────────────────────┘    │
│                         │◄────────────┤                              │
│  ┌─────────────────┐    │             │  ┌─────────────────────┐    │
│  │StandardScaler   │    │             │  │Model Registry       │    │
│  │                 │    │             │  │                     │    │
│  └─────────────────┘    │             │  └─────────────────────┘    │
│                         │             │                              │
└─────────────────────────┘             └─────────────────────────────┘
                                                    │
                                                    ▼
                                        ┌─────────────────────────────┐
                                        │    MLFLOW UI (Web)          │
                                        │    localhost:5002           │
                                        └─────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                         DOCKER CONTAINER (Tùy chọn)                   │
│                                                                        │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐  │
│   │  Flask API  │   │ ML Library  │   │Scikit-learn │   │ Python   │  │
│   │  (app.py)   │   │  (mlib.py)  │   │             │   │          │  │
│   └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘  │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

## Mô tả các thành phần

### 1. Client
- **Web Browser**: Có thể truy cập MLflow UI
- **Công cụ API**: Như Postman, Curl để gọi các API endpoint
- **Ứng dụng khác**: Có thể tích hợp với API của hệ thống

### 2. Flask API (app.py)
- **Endpoint /health**: Kiểm tra trạng thái của ứng dụng và model
- **Endpoint /train**: Huấn luyện model với dữ liệu được cung cấp
- **Endpoint /predict**: Thực hiện dự đoán với model đã huấn luyện
- **Endpoint /metrics**: Lấy các metrics hiện tại của model

### 3. ML Library (mlib.py)
- **MLModel Class**: Lớp chính quản lý vòng đời của model
- **Data Processing**: Chuẩn hóa dữ liệu với StandardScaler
- **Model Training**: Huấn luyện model RandomForestClassifier
- **Model Evaluation**: Tính toán các metrics (accuracy, precision, recall, F1)

### 4. Scikit-learn
- **RandomForestClassifier**: Thuật toán phân loại được sử dụng
- **StandardScaler**: Chuẩn hóa dữ liệu đầu vào
- **Metrics Calculation**: Tính toán độ chính xác, độ chính xác, độ nhớ, F1

### 5. MLflow
- **Experiment Tracking**: Theo dõi và so sánh các phiên huấn luyện
- **Model Registry**: Quản lý và phân phối models
- **MLflow UI**: Giao diện web để xem và phân tích metrics (http://localhost:5002)

### 6. Docker Container (Tùy chọn)
- Đóng gói tất cả các thành phần và dependencies
- Giảm thiểu vấn đề "works on my machine"
- Triển khai đơn giản và nhất quán

## Luồng dữ liệu
1. Client gửi request đến Flask API
2. Flask API xử lý request và gọi các phương thức từ ML Library
3. ML Library sử dụng Scikit-learn để xử lý dữ liệu và huấn luyện/dự đoán
4. Kết quả và metrics được ghi lại vào MLflow
5. Flask API trả về kết quả cho Client
6. MLflow UI hiển thị metrics và thông tin về các phiên huấn luyện

## Ưu điểm của kiến trúc
1. **Modular**: Các thành phần tách biệt và có thể tái sử dụng
2. **Scalable**: Có thể dễ dàng mở rộng với nhiều models và datasets
3. **Reproducible**: Theo dõi và ghi lại tất cả các thí nghiệm
4. **Portable**: Có thể triển khai bằng Docker ở bất kỳ đâu 