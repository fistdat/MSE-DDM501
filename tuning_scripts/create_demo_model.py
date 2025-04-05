#!/usr/bin/env python3
"""
Script để tạo và lưu mô hình demo để sử dụng trong phân loại.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
import joblib
import json
from datetime import datetime
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_model():
    """
    Tạo một mô hình mẫu để sử dụng cho phân loại
    """
    logger.info("Đang tạo mô hình mẫu...")
    
    # Tạo dữ liệu mẫu
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Tạo và huấn luyện mô hình
    model_type = "gradient_boosting"
    
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        )
    
    # Huấn luyện mô hình
    model.fit(X, y)
    
    # Đánh giá mô hình trên tập huấn luyện
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    logger.info(f"Accuracy trên tập huấn luyện: {accuracy:.4f}")
    
    # Đường dẫn đến thư mục models
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "models"
    )
    os.makedirs(models_dir, exist_ok=True)
    
    # Lưu mô hình
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Đã lưu mô hình vào {model_path}")
    
    # Tạo thông tin mô hình
    model_info = {
        "model_type": model_type,
        "run_id": "demo_run_id",
        "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "f1_score": 0.95,
        "accuracy": 0.95,
        "precision": 0.95,
        "recall": 0.95,
        "parameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "min_samples_split": 5
        },
        "features_count": 20,
        "samples_count": 1000,
        "model_file": "best_model.joblib",
        "training_timestamp": int(datetime.now().timestamp() * 1000),
        "artifact_uri": "demo_artifact_uri",
        "experiment_id": "demo_experiment_id",
        "run_name": "demo_run",
        "model_description": f"Mô hình {model_type} demo cho phân loại.",
        "feature_names": "Không có thông tin",
        "target_name": "Không có thông tin",
        "prediction_threshold": 0.5,
        "classes": ["0", "1"],
        "is_binary_classification": True,
        "model_size_kb": os.path.getsize(model_path) / 1024,
        "registry_version": "1"
    }
    
    # Lưu thông tin mô hình
    model_info_path = os.path.join(models_dir, "model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Đã lưu thông tin mô hình vào {model_info_path}")
    
    return True

if __name__ == "__main__":
    success = create_demo_model()
    exit(0 if success else 1) 