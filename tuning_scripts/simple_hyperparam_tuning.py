"""
Simple Hyperparameter Tuning
----------------------------
Script đơn giản để thử nghiệm tuning siêu tham số với các model Machine Learning
và theo dõi kết quả bằng MLflow.
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import numpy as np
import mlflow
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
import traceback
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Only import DEFAULT_EXPERIMENT_NAME since we're defining our own setup_mlflow
from mlflow_scripts.mlflow_config import DEFAULT_EXPERIMENT_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define parameter spaces for different model types
PARAM_SPACES = {
    "tiny": {
        "random_forest": {
            "model__n_estimators": [10, 20],
            "model__max_depth": [3, 5]
        },
        "gradient_boosting": {
            "model__n_estimators": [10, 20],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.1]
        }
    },
    "small": {
        "random_forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5]
        },
        "gradient_boosting": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1, 0.2]
        }
    },
    "medium": {
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        }
    }
}

def setup_mlflow():
    """
    Thiết lập MLflow tracking
    
    Returns:
        bool: True nếu thành công, False nếu không
    """
    try:
        # First verify that MLFLOW_TRACKING_URI is set properly
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5002')
        logger.info(f"Setting up MLflow with tracking URI: {tracking_uri}")
        
        # Set tracking URI and verify
        mlflow.set_tracking_uri(tracking_uri)
        
        # Test connection to MLflow
        try:
            # Try to list experiments to see if connection works
            mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
            logger.info(f"Successfully set experiment to: {DEFAULT_EXPERIMENT_NAME}")
            return True
        except Exception as conn_error:
            logger.error(f"Failed to connect to MLflow server: {str(conn_error)}")
            logger.error(traceback.format_exc())
            return False
    except Exception as e:
        logger.error(f"Error in setup_mlflow: {str(e)}")
        logger.error(traceback.format_exc())
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

def tune_hyperparameters(X_train, X_test, y_train, y_test, model_type="random_forest", 
                        param_space_size="small", n_cv=5, use_mlflow=True, random_state=42):
    """
    Tune hyperparameters for a given model
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_type: Type of model to tune (random_forest, gradient_boosting)
        param_space_size: Size of parameter space (tiny, small, medium)
        n_cv: Number of cross-validation folds
        use_mlflow: Whether to use MLflow for tracking
        random_state: Random state for reproducibility
    
    Returns:
        best_params: Best hyperparameters
        best_estimator: Best model
        metrics: Dictionary of metrics
    """
    logger.info(f"Đang tinh chỉnh siêu tham số cho {model_type}...")
    
    # Create pipeline
    pipeline = create_pipeline(model_type)
    
    # Get parameter space
    param_grid = PARAM_SPACES[param_space_size][model_type]
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=n_cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Start MLflow run
    if use_mlflow:
        # Format current date and time for run name
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create a descriptive run name
        run_name = f"{model_type.capitalize()} {param_space_size} space - {current_time}"
        
        with mlflow.start_run(run_name=run_name):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            logger.info(f"MLflow run name: {run_name}")
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("param_space_size", param_space_size)
            mlflow.log_param("n_cv", n_cv)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_samples", len(X_train) + len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("space", param_space_size)
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Get the best parameters and model
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_
            
            # Log best parameters
            for param, value in best_params.items():
                clean_param = param.replace('model__', '')
                mlflow.log_param(clean_param, value)
            
            # Make predictions and calculate metrics
            y_pred = best_estimator.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            # Log model
            mlflow.sklearn.log_model(best_estimator, "model")
            
            # Create metrics dictionary
            metrics = {
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
            
            # Print results
            logger.info(f"Kết quả tốt nhất cho {model_type}:")
            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"Tham số tốt nhất: {best_params}")
            
            return best_params, best_estimator, metrics
    else:
        # Run without MLflow tracking
        grid_search.fit(X_train, y_train)
        
        # Get the best parameters and model
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        
        # Make predictions and calculate metrics
        y_pred = best_estimator.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Create metrics dictionary
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
        
        # Print results
        logger.info(f"Kết quả tốt nhất cho {model_type}:")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Tham số tốt nhất: {best_params}")
        
        return best_params, best_estimator, metrics

def parse_args():
    """Parse command line arguments"""
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
    
    # In thông tin
    args = parser.parse_args()
    logger.info("\n" + "=" * 60)
    logger.info(f"TUNING SIÊU THAM SỐ CHO {args.model.upper()}")
    logger.info("=" * 60)
    logger.info(f"Không gian tham số: {args.space}")
    logger.info(f"Số mẫu dữ liệu: {args.samples}")
    logger.info(f"Số features: {args.features}")
    logger.info(f"Cross-validation: {args.cv} folds")
    logger.info("=" * 60 + "\n")
    
    return args

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Thiết lập MLflow chỉ khi không có tùy chọn --no-mlflow
    use_mlflow = not args.no_mlflow
    
    if use_mlflow:
        # Sử dụng cấu hình từ mlflow_config
        if setup_mlflow():
            mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
            logger.info(f"Đã thiết lập MLflow tracking với experiment: {DEFAULT_EXPERIMENT_NAME}")
        else:
            logger.warning("Không thể thiết lập MLflow, sẽ chạy mà không có tracking")
            use_mlflow = False
    else:
        logger.info("Chạy mà không sử dụng MLflow tracking theo yêu cầu")
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=args.samples,
        n_features=args.features,
        random_state=42
    )
    
    # Tune hyperparameters
    best_params, best_model, metrics = tune_hyperparameters(
        X_train, X_test, y_train, y_test,
        model_type=args.model,
        param_space_size=args.space,
        n_cv=args.cv,
        use_mlflow=use_mlflow
    )
    
    # Lưu model tốt nhất trực tiếp vào thư mục models
    models_dir = "/app/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Lưu model vào file best_model.joblib
    model_path = os.path.join(models_dir, "best_model.joblib")
    try:
        # Lưu model trực tiếp từ kết quả tuning
        import joblib
        joblib.dump(best_model, model_path)
        logger.info(f"Đã lưu model trực tiếp vào {model_path}")
        
        # Cập nhật file model_info.json
        model_info_path = os.path.join(models_dir, "model_info.json")
        model_info = {
            "model_type": args.model,
            "run_id": mlflow.active_run().info.run_id if mlflow.active_run() else "local_tuning",
            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "f1_score": metrics.get("f1_score", 0),
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "parameters": best_params,
            "features_count": args.features,
            "samples_count": args.samples,
            "model_file": "best_model.joblib",
            "training_timestamp": int(datetime.now().timestamp() * 1000),
            "model_description": f"Mô hình {args.model} với f1_score={metrics.get('f1_score', 0):.4f}, được huấn luyện trên {args.samples} mẫu.",
            "prediction_threshold": 0.5,
            "classes": ["0", "1"],
            "is_binary_classification": True,
        }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Đã lưu thông tin mô hình vào {model_info_path}")
        
        # Check if auto-register is enabled
        auto_register = os.environ.get("AUTO_REGISTER_MODEL", "0") == "1"
        
        # Register model to MLflow Model Registry if auto-register is enabled
        if auto_register and use_mlflow and mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            logger.info(f"AUTO_REGISTER_MODEL is enabled. Đăng ký model từ run {run_id} vào Model Registry")
            
            try:
                # Đường dẫn đến script register_model.py - use absolute path for Docker
                register_script_path = "/app/tuning_scripts/register_model.py"
                if not os.path.exists(register_script_path):
                    logger.error(f"register_model.py script not found at {register_script_path}")
                    # Try alternative path
                    register_script_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "register_model.py"
                    )
                    logger.info(f"Trying alternative path: {register_script_path}")
                    if not os.path.exists(register_script_path):
                        logger.error(f"register_model.py script not found at alternative path either")
                        raise FileNotFoundError(f"register_model.py script not found")
                
                logger.info(f"Found register_model.py at: {register_script_path}")
                
                # Set environment variables for subprocess
                env = os.environ.copy()
                # Set NO_NEW_RUNS=1 to prevent creating a new run
                env["NO_NEW_RUNS"] = "1"
                # Ensure MLFLOW_TRACKING_URI is set
                if "MLFLOW_TRACKING_URI" not in env:
                    env["MLFLOW_TRACKING_URI"] = "http://mlflow:5002"
                
                logger.info(f"Environment variables: MLFLOW_TRACKING_URI={env.get('MLFLOW_TRACKING_URI')}, NO_NEW_RUNS={env.get('NO_NEW_RUNS')}")
                
                # Create model name
                model_name = f"{args.model}_model"
                
                logger.info(f"Executing command: python {register_script_path} {run_id} {model_name}")
                
                # Call register_model script
                process = subprocess.Popen(
                    ["python", register_script_path, run_id, model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    universal_newlines=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"Đã đăng ký model thành công: {stdout}")
                else:
                    logger.warning(f"Không thể đăng ký model. Return code: {process.returncode}")
                    logger.warning(f"stdout: {stdout}")
                    logger.warning(f"stderr: {stderr}")
                    
                    # Try a direct approach instead of subprocess
                    try:
                        logger.info("Trying direct model registration...")
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient(env.get("MLFLOW_TRACKING_URI"))
                        
                        # Create the model if it doesn't exist
                        try:
                            client.get_registered_model(model_name)
                        except:
                            client.create_registered_model(model_name)
                            logger.info(f"Created model {model_name}")
                        
                        # Register the model
                        model_uri = f"runs:/{run_id}/model"
                        model_details = mlflow.register_model(model_uri, model_name)
                        logger.info(f"Successfully registered model with version {model_details.version}")
                    except Exception as e:
                        logger.error(f"Direct registration failed: {str(e)}")
                        logger.error(traceback.format_exc())
                    
            except Exception as e:
                logger.error(f"Lỗi khi đăng ký model: {str(e)}")
                logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu model trực tiếp: {str(e)}")
    
    return 0

if __name__ == "__main__":
    main() 