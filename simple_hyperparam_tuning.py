"""
Simple Hyperparameter Tuning
----------------------------
Script đơn giản để thử nghiệm tuning siêu tham số với các model Machine Learning
và theo dõi kết quả bằng MLflow.
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

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Định nghĩa các không gian tham số với kích cỡ khác nhau
PARAM_SPACES = {
    "tiny": {
        "random_forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [5, 10],
        },
        "gradient_boosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5],
        }
    },
    "small": {
        "random_forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 15],
            "model__min_samples_split": [2, 5],
        },
        "gradient_boosting": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
        }
    },
    "medium": {
        "random_forest": {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, 9],
            "model__min_samples_split": [2, 5, 10],
        }
    }
}

def setup_mlflow(experiment_name="tuning_experiment"):
    """
    Thiết lập MLflow tracking
    
    Parameters:
        experiment_name (str): Tên cho MLflow experiment
    
    Returns:
        bool: True nếu thiết lập thành công, False nếu thất bại
    """
    try:
        # Thiết lập MLflow tracking URI
        mlflow_uri = "http://localhost:5002"
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Kiểm tra xem experiment đã tồn tại chưa
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Tạo experiment mới nếu chưa tồn tại
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Đã tạo experiment mới '{experiment_name}' với ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Sử dụng experiment đã tồn tại '{experiment_name}' với ID: {experiment_id}")
        
        # Thiết lập experiment hiện tại
        mlflow.set_experiment(experiment_name)
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập MLflow: {e}")
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

def tune_hyperparameters(X_train, y_train, X_test, y_test, 
                        model_type="random_forest", 
                        param_space_size="small",
                        cv=5,
                        use_mlflow=True):
    """
    Thực hiện tuning siêu tham số và evaluate model
    
    Parameters:
        X_train, y_train: Dữ liệu huấn luyện
        X_test, y_test: Dữ liệu kiểm tra
        model_type (str): Loại mô hình
        param_space_size (str): Kích thước không gian tham số
        cv (int): Số fold cross-validation
        use_mlflow (bool): Có sử dụng MLflow không
    
    Returns:
        tuple: (best_params, best_model, metrics) - kết quả tối ưu
    """
    # Tạo pipeline
    pipeline = create_pipeline(model_type)
    
    # Lấy không gian tham số
    try:
        param_grid = PARAM_SPACES[param_space_size][model_type]
    except KeyError:
        logger.error(f"Không tìm thấy không gian tham số cho {model_type} với kích thước {param_space_size}")
        raise ValueError(f"Không gian tham số không hợp lệ: {param_space_size} cho {model_type}")
    
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
            run_name = f"{model_type}_{param_space_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = mlflow.start_run(run_name=run_name)
            logger.info(f"Đã bắt đầu MLflow run với ID: {run.info.run_id}")
            
            # Log thông tin cơ bản
            mlflow.log_params({
                "model_type": model_type,
                "param_space": param_space_size,
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
        
        # Tạo tên file kết quả
        result_filename = f"{model_type}_{param_space_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(results_dir, result_filename)
        
        # Tạo dictionary kết quả để lưu
        result_data = {
            "model_type": model_type,
            "param_space": param_space_size,
            "best_params": {k.replace('model__', ''): v for k, v in best_params.items()},
            "best_cv_score": best_score,
            "test_metrics": metrics,
            "tuning_time_seconds": tuning_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Lưu kết quả vào file JSON
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Đã lưu kết quả vào: {result_path}")
        
        return best_params, best_model, metrics
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tuning: {e}")
        raise
    
    finally:
        # Đảm bảo kết thúc MLflow run
        if use_mlflow and mlflow_active and mlflow.active_run():
            mlflow.end_run()
            logger.info("Đã kết thúc MLflow run")

def main():
    """Hàm chính thực hiện tuning siêu tham số"""
    # Định nghĩa parser
    parser = argparse.ArgumentParser(description="Thử nghiệm tuning siêu tham số đơn giản")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Loại mô hình để tuning"
    )
    
    parser.add_argument(
        "--space", 
        type=str, 
        default="small",
        choices=["tiny", "small", "medium"],
        help="Kích thước không gian tham số"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Số lượng mẫu dữ liệu"
    )
    
    parser.add_argument(
        "--features", 
        type=int, 
        default=20,
        help="Số lượng features"
    )
    
    parser.add_argument(
        "--cv", 
        type=int, 
        default=5,
        help="Số fold cross-validation"
    )
    
    parser.add_argument(
        "--no-mlflow", 
        action="store_true",
        help="Không sử dụng MLflow tracking"
    )
    
    # Parse tham số
    args = parser.parse_args()
    
    # In thông tin
    print("\n" + "=" * 60)
    print(f"TUNING SIÊU THAM SỐ CHO {args.model.upper()}")
    print("=" * 60)
    print(f"Không gian tham số: {args.space}")
    print(f"Số mẫu dữ liệu: {args.samples}")
    print(f"Số features: {args.features}")
    print(f"Cross-validation: {args.cv} folds")
    print(f"Sử dụng MLflow: {not args.no_mlflow}")
    print("=" * 60 + "\n")
    
    # Tạo dữ liệu
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=args.samples, 
        n_features=args.features
    )
    
    # Thực hiện tuning
    try:
        start_time = time.time()
        best_params, best_model, metrics = tune_hyperparameters(
            X_train, y_train, X_test, y_test,
            model_type=args.model,
            param_space_size=args.space,
            cv=args.cv,
            use_mlflow=not args.no_mlflow
        )
        
        # In kết quả
        print("\n" + "=" * 60)
        print("KẾT QUẢ TUNING SIÊU THAM SỐ")
        print("=" * 60)
        print(f"Thời gian tuning: {time.time() - start_time:.2f} giây")
        
        # In siêu tham số tốt nhất
        print("\nSiêu tham số tốt nhất:")
        for param, value in best_params.items():
            print(f"  {param.replace('model__', '')}: {value}")
        
        # In metrics trên tập test
        print("\nMetrics trên tập test:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Hiển thị MLflow UI link
        if not args.no_mlflow:
            print("\nXem kết quả chi tiết tại MLflow UI: http://localhost:5002")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\nLỗi: {str(e)}")
        print("Quá trình tuning thất bại!")

if __name__ == "__main__":
    main() 