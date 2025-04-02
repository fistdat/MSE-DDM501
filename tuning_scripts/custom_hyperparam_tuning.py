"""
Custom Hyperparameter Tuning
---------------------------
Script để thực hiện tuning siêu tham số với các tham số do người dùng tự định nghĩa
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import json
import argparse
import time
import logging
from datetime import datetime
import sys

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_scripts.mlflow_config import setup_mlflow as init_mlflow, DEFAULT_EXPERIMENT_NAME

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """
    Thiết lập MLflow tracking
    
    Returns:
        bool: True nếu thành công, False nếu không
    """
    try:
        # Sử dụng cấu hình từ mlflow_config
        if init_mlflow():
            mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
            logger.info(f"Đã thiết lập MLflow tracking với experiment: {DEFAULT_EXPERIMENT_NAME}")
            return True
        else:
            logger.warning("Không thể thiết lập MLflow, sẽ chạy mà không có tracking")
            return False
    except Exception as e:
        logger.warning(f"Lỗi khi thiết lập MLflow: {str(e)}")
        return False

def generate_data(n_samples=1000, n_features=20, test_size=0.2, random_state=42):
    """
    Tạo dữ liệu mẫu cho binary classification
    
    Parameters:
        n_samples (int): Số lượng mẫu
        n_features (int): Số lượng features
        test_size (float): Tỷ lệ tập test
        random_state (int): Seed cho quá trình tạo dữ liệu ngẫu nhiên
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - các tập dữ liệu đã chia
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.2),
        random_state=random_state
    )
    
    # Chia dữ liệu thành train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Đã tạo dữ liệu: {X_train.shape[0]} mẫu train, {X_test.shape[0]} mẫu test, {X_train.shape[1]} features")
    
    return X_train, X_test, y_train, y_test

def create_pipeline(model_type="random_forest"):
    """
    Tạo pipeline cho model
    
    Parameters:
        model_type (str): Loại mô hình ("random_forest" hoặc "gradient_boosting")
    
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}")
    
    # Tạo pipeline
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def load_param_grid(params_file):
    """
    Tải không gian tham số từ file JSON
    
    Parameters:
        params_file (str): Đường dẫn tới file JSON chứa tham số
    
    Returns:
        dict: Không gian tham số
    """
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Thêm tiền tố 'model__' vào tên tham số
        param_grid = {}
        for param, values in params.items():
            param_grid[f"model__{param}"] = values
        
        return param_grid
    except Exception as e:
        logger.error(f"Lỗi khi đọc file tham số {params_file}: {str(e)}")
        raise ValueError(f"Không thể đọc file tham số: {str(e)}")

def tune_hyperparameters(X_train, y_train, X_test, y_test, 
                        model_type="random_forest", 
                        param_grid=None,
                        cv=5,
                        use_mlflow=True):
    """
    Thực hiện tuning siêu tham số và evaluate model
    
    Parameters:
        X_train, y_train: Dữ liệu huấn luyện
        X_test, y_test: Dữ liệu kiểm tra
        model_type (str): Loại mô hình
        param_grid (dict): Không gian tham số tùy chỉnh
        cv (int): Số fold cross-validation
        use_mlflow (bool): Có sử dụng MLflow không
    
    Returns:
        tuple: (best_params, best_model, metrics) - kết quả tối ưu
    """
    # Tạo pipeline
    pipeline = create_pipeline(model_type)
    
    # Kiểm tra không gian tham số
    if not param_grid:
        raise ValueError("Không gian tham số không thể rỗng")
    
    # Tính tổng số tổ hợp tham số
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)
    
    logger.info(f"Bắt đầu tuning với {total_combinations} tổ hợp tham số")
    start_time = time.time()
    
    # Thiết lập MLflow run
    if use_mlflow:
        mlflow_active = setup_mlflow()
        if mlflow_active:
            run_name = f"{model_type}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = mlflow.start_run(run_name=run_name)
            logger.info(f"Đã bắt đầu MLflow run với ID: {run.info.run_id}")
            
            # Log thông tin cơ bản
            mlflow.log_params({
                "model_type": model_type,
                "param_space": "custom",
                "n_samples_train": X_train.shape[0],
                "n_samples_test": X_test.shape[0],
                "n_features": X_train.shape[1],
                "cv": cv,
                "total_combinations": total_combinations
            })
    
    try:
        # Thực hiện grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Lấy thông tin về cấu hình tốt nhất
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        # Đánh giá trên tập test
        y_pred = best_model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        # Đo thời gian tuning
        tuning_time = time.time() - start_time
        
        # Log với MLflow
        if use_mlflow and mlflow_active:
            # Log best params (loại bỏ tiền tố 'model__')
            clean_params = {k.replace('model__', ''): v for k, v in best_params.items()}
            mlflow.log_params(clean_params)
            
            # Log metrics
            mlflow.log_metric("best_cv_f1", best_score)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("tuning_time_seconds", tuning_time)
            
            # Log mô hình
            mlflow.sklearn.log_model(best_model, "model")
        
        # Lưu kết quả vào thư mục
        results_dir = "tuning_results"
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = os.path.join(
            results_dir, 
            f"{model_type}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Chuẩn bị kết quả để lưu
        result_data = {
            "model_type": model_type,
            "param_space": "custom",
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "n_features": X_train.shape[1],
            "cv": cv,
            "best_params": {k.replace('model__', ''): v for k, v in best_params.items()},
            "best_cv_f1": best_score,
            "metrics": metrics,
            "tuning_time_seconds": tuning_time,
            "total_combinations": total_combinations,
            "param_grid": {k.replace('model__', ''): v for k, v in param_grid.items()},
        }
        
        # Thêm thông tin MLflow nếu có
        if use_mlflow and mlflow_active:
            result_data["mlflow_run_id"] = run.info.run_id
            result_data["mlflow_experiment_id"] = run.info.experiment_id
            result_data["mlflow_experiment_name"] = DEFAULT_EXPERIMENT_NAME
        
        # Lưu kết quả
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Đã lưu kết quả vào file: {result_file}")
        
        # Kết thúc MLflow run
        if use_mlflow and mlflow_active:
            mlflow.end_run()
        
        return best_params, best_model, metrics
    
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tuning: {str(e)}")
        # Đảm bảo kết thúc MLflow run
        if use_mlflow and mlflow_active:
            mlflow.end_run()
        raise

def main():
    """Hàm chính để chạy custom hyperparameter tuning"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Custom Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="random_forest", 
                        choices=["random_forest", "gradient_boosting"],
                        help="Loại mô hình")
    parser.add_argument("--params-file", type=str, required=True,
                       help="File JSON chứa các tham số tùy chỉnh")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Số lượng mẫu")
    parser.add_argument("--features", type=int, default=20,
                       help="Số lượng features")
    parser.add_argument("--cv", type=int, default=5,
                       help="Số fold cross-validation")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Không sử dụng MLflow tracking")
    
    args = parser.parse_args()
    
    # In thông tin
    logger.info("=== CUSTOM HYPERPARAMETER TUNING ===")
    logger.info(f"Mô hình: {args.model}")
    logger.info(f"File tham số: {args.params_file}")
    logger.info(f"Số mẫu: {args.samples}")
    logger.info(f"Số features: {args.features}")
    logger.info(f"CV folds: {args.cv}")
    logger.info(f"Sử dụng MLflow: {not args.no_mlflow}")
    
    # Tạo dữ liệu
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=args.samples,
        n_features=args.features
    )
    
    # Tải không gian tham số
    param_grid = load_param_grid(args.params_file)
    logger.info(f"Không gian tham số: {param_grid}")
    
    # Thực hiện tuning
    try:
        best_params, best_model, metrics = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=args.model,
            param_grid=param_grid,
            cv=args.cv,
            use_mlflow=not args.no_mlflow
        )
        
        # In kết quả
        logger.info("\n=== KẾT QUẢ TUNING ===")
        logger.info(f"Tham số tốt nhất: {best_params}")
        logger.info(f"Metrics:\n" + "\n".join([f"  - {k}: {v:.4f}" for k, v in metrics.items()]))
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện tuning: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 