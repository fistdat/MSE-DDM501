"""MLflow Lab Library"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModel:
    """Main ML model class that handles training, prediction and model management"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the ML model
        
        Args:
            model_type: Type of model to use ("random_forest" or "logistic_regression")
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
        self.best_params: Dict[str, Any] = {}

    def _create_model(self, **kwargs) -> Any:
        """Create the appropriate model based on model_type with optional parameters"""
        if self.model_type == "random_forest":
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            }
            # Cập nhật parameters từ kwargs
            params.update(kwargs)
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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
        
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, param_grid: Dict[str, List], cv: int = 5) -> Dict[str, Any]:
        """
        Tối ưu hóa siêu tham số sử dụng GridSearchCV và MLflow tracking
        
        Args:
            X: Features
            y: Labels
            param_grid: Dictionary các siêu tham số và giá trị cần tìm kiếm
            cv: Số fold cho cross-validation
            
        Returns:
            Dictionary chứa siêu tham số tốt nhất
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Tạo base model
        base_model = self._create_model()
        
        # Sử dụng MLflow để tracking
        try:
            # Khởi tạo GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            # Thực hiện tìm kiếm
            logger.info(f"Bắt đầu grid search với {param_grid}")
            grid_search.fit(X_train_scaled, y_train)
            logger.info(f"Hoàn tất grid search")
            
            # Lấy thông tin về cấu hình tốt nhất
            self.best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Lấy mô hình tốt nhất
            best_model = grid_search.best_estimator_
            
            # Đánh giá trên tập test
            y_pred = best_model.predict(X_test_scaled)
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }
            
            # Cập nhật model và metrics
            self.model = best_model
            self.metrics = test_metrics
            self.is_trained = True
            
            # Thử log với MLflow nếu có thể - KHÔNG tạo run mới
            try:
                import mlflow
                # Kiểm tra xem đã có active run chưa
                active_run = mlflow.active_run()
                
                if active_run: 
                    # Nếu có active run, log trực tiếp vào run đó
                    logger.info(f"Logging vào MLflow run hiện tại: {active_run.info.run_id}")
                    
                    # Log tham số và metrics
                    mlflow.log_params(self.best_params)
                    mlflow.log_metric("best_cv_f1", best_score)
                    mlflow.log_metrics(test_metrics)
                    
                    # Log model nếu có thể
                    try:
                        mlflow.sklearn.log_model(best_model, "model")
                    except Exception as e:
                        logger.warning(f"Không thể log model vào MLflow: {e}")
                else:
                    # Không có active run, chỉ log thông tin cục bộ
                    logger.info("Không có MLflow run hiện tại, bỏ qua logging MLflow")
            except ImportError:
                logger.info("MLflow không có sẵn, bỏ qua logging")
            except Exception as e:
                logger.warning(f"Không thể sử dụng MLflow tracking: {e}")
            
            logger.info(f"Siêu tham số tốt nhất: {self.best_params}")
            logger.info(f"Metrics trên tập test: {test_metrics}")
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tuning: {str(e)}")
            # Đảm bảo vẫn lưu lại thông tin dù có lỗi
            if 'grid_search' in locals() and hasattr(grid_search, 'best_params_'):
                self.best_params = grid_search.best_params_
                self.model = grid_search.best_estimator_
                self.is_trained = True
                
        return self.best_params
    
    def _check_mlflow_server(self, uri):
        """Kiểm tra xem MLflow server có hoạt động không"""
        try:
            import requests
            response = requests.get(f"{uri}/api/2.0/mlflow/experiments/list")
            return response.status_code == 200
        except:
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Scale features if needed
        X_scaled = self.scaler.transform(X)
            
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Scale features if needed
        X_scaled = self.scaler.transform(X)
            
        return self.model.predict_proba(X_scaled)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        # Save both model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'best_params': self.best_params
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.metrics = model_data.get('metrics', {})
        self.best_params = model_data.get('best_params', {})
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current model metrics
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
        
    def get_best_params(self) -> Dict[str, Any]:
        """
        Lấy siêu tham số tốt nhất từ quá trình tuning
        
        Returns:
            Dictionary chứa siêu tham số tốt nhất
        """
        return self.best_params

def generate_sample_data(n_samples: int = 1000, n_features: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=random_state
    )
    return X, y 