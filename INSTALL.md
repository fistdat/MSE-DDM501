# Hướng dẫn cài đặt MLOps-Lab02

Tài liệu này hướng dẫn chi tiết cách cài đặt và chạy dự án MLOps-Lab02.

## Yêu cầu hệ thống

- Python 3.12 trở lên
- pip (công cụ quản lý gói của Python)
- Hệ điều hành: Windows, macOS hoặc Linux

## Các bước cài đặt

### 1. Tạo môi trường ảo

Môi trường ảo (virtual environment) giúp cô lập các dependency của dự án, tránh xung đột với các dự án khác.

```bash
# Di chuyển đến thư mục dự án
cd MLOps-Lab02

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### 2. Cài đặt các dependency

Sau khi kích hoạt môi trường ảo, cài đặt các gói cần thiết:

```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt setuptools (cần thiết cho một số gói)
pip install "setuptools>=69.0.0"

# Cài đặt các dependency từ file requirements.txt
pip install -r requirements.txt
```

### 3. Xử lý các vấn đề thường gặp

#### Lỗi với setuptools và distutils

Nếu gặp lỗi liên quan đến việc thiếu module `distutils`:

```bash
pip install "setuptools>=69.0.0"
```

#### Lỗi với wrapt và deprecated

Python 3.12 không còn hỗ trợ hàm `formatargspec` trong module `inspect` mà thư viện `wrapt` sử dụng. Nếu gặp lỗi tương tự, hãy đảm bảo đang sử dụng phiên bản tương thích:

```bash
pip uninstall -y wrapt deprecated
pip install wrapt==1.12.1 deprecated<=1.2.13
```

#### Lỗi với MLflow

Nếu gặp lỗi khi cài đặt MLflow với Python 3.12, hãy sử dụng phiên bản MLflow ≤ 2.8.0:

```bash
pip uninstall -y mlflow
pip install "mlflow<=2.8.0"
```

#### Lỗi với port 5000 đã được sử dụng

Trên macOS, port 5000 thường được sử dụng bởi dịch vụ AirPlay Receiver. Chúng ta đã thay đổi cấu hình để sử dụng port 5002 cho MLflow.

## Chạy ứng dụng

### 1. Khởi động MLflow server

```bash
# Trong thư mục dự án, với môi trường ảo đã được kích hoạt
mlflow server --host 0.0.0.0 --port 5002
```

MLflow UI sẽ có thể truy cập tại: http://localhost:5002

### 2. Khởi động ứng dụng Flask

Mở một terminal mới, kích hoạt môi trường ảo, và chạy:

```bash
# Di chuyển đến thư mục dự án
cd MLOps-Lab02

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate

# Chạy ứng dụng
python app.py
```

Ứng dụng Flask API sẽ chạy tại: http://localhost:5001

## Kiểm tra API

Bạn có thể sử dụng `curl` hoặc công cụ như Postman để kiểm tra API:

### 1. Kiểm tra health check

```bash
curl -X GET http://localhost:5001/health
```

### 2. Train model

```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9]], "target": [0, 0, 1, 1, 2, 2]}' http://localhost:5001/train
```

### 3. Predict

```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [[5.0, 3.5, 1.3, 0.2], [6.3, 3.3, 6.0, 2.5]]}' http://localhost:5001/predict
```

### 4. Get metrics

```bash
curl -X GET http://localhost:5001/metrics
```

## Sử dụng Docker (tùy chọn)

Nếu bạn muốn sử dụng Docker, chúng ta đã cung cấp Dockerfile và có thể chạy:

```bash
# Build Docker image
docker build -t mlops-lab02 .

# Chạy container
docker run -p 5001:5001 mlops-lab02
```

Lưu ý: Khi chạy trong Docker, MLflow sẽ không thể kết nối với máy chủ bên ngoài container. Bạn cần cấu hình thêm volume và network nếu muốn lưu trữ dữ liệu MLflow.

## Giải thích chi tiết mã nguồn

### 1. File `mlib.py`

File `mlib.py` chứa code chính cho xử lý machine learning. Dưới đây là giải thích chi tiết từng phần:

#### 1.1 Import các thư viện cần thiết
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
```

- `numpy` và `pandas`: Thư viện xử lý dữ liệu
- `sklearn`: Cung cấp các công cụ ML như RandomForest, metrics, và xử lý dữ liệu
- `mlflow`: Theo dõi và quản lý experiment ML
- `joblib`: Lưu và tải model
- `logging`: Ghi log
- `typing`: Hỗ trợ type hint

#### 1.2 Class MLModel

Class chính quản lý toàn bộ vòng đời của model:

```python
class MLModel:
    """Main ML model class that handles training, prediction and model management"""
    
    def __init__(self, model_type: str = "random_forest"):
        """Khởi tạo model"""
        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
```

- Khởi tạo với model mặc định là RandomForest
- Tạo đối tượng StandardScaler để chuẩn hóa dữ liệu
- Biến `is_trained` theo dõi trạng thái của model
- Dictionary `metrics` lưu trữ các metric đánh giá model

#### 1.3 Phương thức _create_model
```python
def _create_model(self) -> Any:
    """Create the appropriate model based on model_type"""
    if self.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {self.model_type}")
```

- Khởi tạo model RandomForestClassifier với các tham số mặc định
- Có thể mở rộng để hỗ trợ nhiều loại model khác

#### 1.4 Phương thức prepare_data
```python
def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training
    
    Args:
        data: Input DataFrame
            
    Returns:
        Tuple of (features, labels)
    """
    # Assuming the last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y
```

- Tách dữ liệu thành features và label
- Giả định cột cuối cùng là nhãn (target)

#### 1.5 Phương thức train
```python
def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Train the model and return metrics
    
    Args:
        X: Training features
        y: Training labels
            
    Returns:
        Dictionary containing training metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    
    # Train model
    self.model.fit(X_train_scaled, y_train)
    self.is_trained = True
    
    # Get predictions
    y_pred = self.model.predict(X_test_scaled)
    
    # Calculate metrics
    self.metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    # Log metrics with MLflow
    mlflow.log_metrics(self.metrics)
    
    return self.metrics
```

- Chia dữ liệu thành training (80%) và test (20%)
- Chuẩn hóa dữ liệu với StandardScaler
- Huấn luyện model RandomForest
- Tính toán các metrics: accuracy, precision, recall, f1
- Ghi nhận các metrics vào MLflow
- Trả về metrics để hiển thị

#### 1.6 Phương thức predict và predict_proba
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions using the trained model"""
    if not self.is_trained:
        raise ValueError("Model must be trained before making predictions")
        
    # Scale features if needed
    if not hasattr(X, 'shape') or X.shape[1] != self.scaler.n_features_in_:
        X = self.scaler.transform(X)
        
    return self.model.predict(X)

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Get probability predictions"""
    if not self.is_trained:
        raise ValueError("Model must be trained before making predictions")
        
    # Scale features if needed
    if not hasattr(X, 'shape') or X.shape[1] != self.scaler.n_features_in_:
        X = self.scaler.transform(X)
        
    return self.model.predict_proba(X)
```

- `predict`: Trả về nhãn dự đoán
- `predict_proba`: Trả về xác suất dự đoán cho từng lớp
- Kiểm tra xem model đã được huấn luyện chưa
- Chuẩn hóa dữ liệu đầu vào nếu cần

#### 1.7 Phương thức save_model và load_model
```python
def save_model(self, path: str) -> None:
    """Save the trained model to disk"""
    if not self.is_trained:
        raise ValueError("Model must be trained before saving")
        
    # Save both model and scaler
    model_data = {
        'model': self.model,
        'scaler': self.scaler,
        'metrics': self.metrics
    }
    joblib.dump(model_data, path)
    logger.info(f"Model saved to {path}")

def load_model(self, path: str) -> None:
    """Load a trained model from disk"""
    model_data = joblib.load(path)
    self.model = model_data['model']
    self.scaler = model_data['scaler']
    self.metrics = model_data.get('metrics', {})
    self.is_trained = True
    logger.info(f"Model loaded from {path}")
```

- `save_model`: Lưu model, scaler và metrics vào file
- `load_model`: Tải model, scaler và metrics từ file

#### 1.8 Hàm generate_sample_data
```python
def generate_sample_data(n_samples: int = 1000, n_features: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data for testing"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=random_state
    )
    return X, y
```

- Hàm tiện ích để tạo dữ liệu mẫu cho testing
- Sử dụng `make_classification` từ scikit-learn

### 2. File `app.py`

File `app.py` chứa code cho Flask API, kết nối phía front-end với model machine learning.

#### 2.1 Import và cấu hình cơ bản
```python
"""MLflow Lab API"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from mlib import MLModel
import mlflow
import logging
from typing import Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = MLModel()
```

- Import các thư viện cần thiết
- Cấu hình logging để theo dõi hoạt động của API
- Khởi tạo Flask app và model MLModel

#### 2.2 Hàm validate_data
```python
def validate_data(data: Dict[str, Any]) -> bool:
    """
    Validate input data
    
    Args:
        data: Input data dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not data or "data" not in data:
        return False
    return True
```

- Kiểm tra tính hợp lệ của dữ liệu đầu vào
- Đảm bảo dữ liệu có trường "data"

#### 2.3 Endpoint /health
```python
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": model.is_trained
    }), 200
```

- Endpoint kiểm tra trạng thái của API
- Trả về trạng thái khỏe mạnh và thông tin về việc model đã được huấn luyện chưa

#### 2.4 Endpoint /train
```python
@app.route("/train", methods=["POST"])
def train():
    """Train the model with provided data"""
    try:
        data = request.get_json()
        if not validate_data(data):
            return jsonify({
                "error": "Invalid input data",
                "message": "Data must contain 'data' field with training data"
            }), 400
            
        # Check if target is provided
        if "target" not in data:
            return jsonify({
                "error": "Invalid input data",
                "message": "Data must contain 'target' field with labels"
            }), 400
            
        # Convert data to DataFrame and numpy array
        X = np.array(data["data"])
        y = np.array(data["target"], dtype=int)
        
        # Train model
        metrics = model.train(X, y)
        
        return jsonify({
            "message": "Model trained successfully",
            "metrics": metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Training failed",
            "message": str(e)
        }), 500
```

- Endpoint để huấn luyện model
- Nhận dữ liệu JSON với trường "data" và "target"
- Chuyển đổi dữ liệu thành numpy array
- Gọi phương thức `train` của model
- Xử lý các ngoại lệ và trả về thông báo lỗi phù hợp

#### 2.5 Endpoint /predict
```python
@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the trained model"""
    try:
        data = request.get_json()
        if not validate_data(data):
            return jsonify({
                "error": "Invalid input data",
                "message": "Data must contain 'data' field with features"
            }), 400
            
        # Convert data to numpy array
        X = np.array(data["data"])
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500
```

- Endpoint để dự đoán với model đã huấn luyện
- Nhận dữ liệu JSON với trường "data"
- Chuyển đổi dữ liệu thành numpy array
- Gọi phương thức `predict` và `predict_proba` của model
- Trả về kết quả dự đoán và xác suất

#### 2.6 Endpoint /metrics
```python
@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get current model metrics"""
    try:
        metrics = model.get_metrics()
        if not metrics:
            return jsonify({
                "error": "No metrics available",
                "message": "Model has not been trained yet"
            }), 404
            
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to get metrics",
            "message": str(e)
        }), 500
```

- Endpoint để lấy metrics của model
- Gọi phương thức `get_metrics` của model
- Kiểm tra nếu metrics trống (chưa huấn luyện model)

#### 2.7 Xử lý lỗi
```python
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested URL was not found on the server"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error has occurred"
    }), 500
```

- Xử lý lỗi 404 (trang không tìm thấy)
- Xử lý lỗi 500 (lỗi server)
- Trả về thông báo lỗi dạng JSON

#### 2.8 Khởi chạy ứng dụng
```python
if __name__ == "__main__":
    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("default")
    
    # Start Flask app
    app.run(host="0.0.0.0", port=5001)
```

- Cấu hình MLflow với URI tracking và tên experiment
- Khởi chạy Flask app trên host 0.0.0.0 (tất cả interface) và port 5001 