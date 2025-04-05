"""
Script để đăng ký model vào MLflow Model Registry từ run ID
"""

import os
import sys
import json
import logging
import traceback
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if we're supposed to avoid creating new runs
NO_NEW_RUNS = os.environ.get('NO_NEW_RUNS', '0') == '1'

def register_model_from_run(run_id, model_name, description=None, tags=None):
    """
    Đăng ký model vào MLflow Model Registry từ run ID
    
    Args:
        run_id: Run ID của MLflow run
        model_name: Tên model trong registry
        description: Mô tả model
        tags: Tags của model (dict)
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Get MLflow tracking URI from environment
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5002')
        logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
        
        # Prepare model URI
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Model URI: {model_uri}")
        
        # Step 1: Check if the model exists in registry
        try:
            response = requests.get(
                f"{mlflow_tracking_uri}/api/2.0/mlflow/registered-models/get",
                params={"name": model_name}
            )
            model_exists = response.status_code == 200
            logger.info(f"Model exists in registry: {model_exists}")
        except Exception as e:
            logger.warning(f"Error checking if model exists: {str(e)}")
            model_exists = False
        
        # Step 2: Create model if it doesn't exist
        if not model_exists:
            try:
                create_model_payload = {
                    "name": model_name,
                    "description": description or ""
                }
                response = requests.post(
                    f"{mlflow_tracking_uri}/api/2.0/mlflow/registered-models/create",
                    json=create_model_payload
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully created model {model_name} in registry")
                else:
                    logger.warning(f"Failed to create model in registry: {response.text}")
            except Exception as e:
                logger.warning(f"Error creating model in registry: {str(e)}")
        
        # Step 3: Create a new model version
        try:
            create_version_payload = {
                "name": model_name,
                "source": model_uri
            }
            response = requests.post(
                f"{mlflow_tracking_uri}/api/2.0/mlflow/model-versions/create",
                json=create_version_payload
            )
            
            if response.status_code == 200:
                model_version = response.json()["model_version"]
                version = model_version["version"]
                logger.info(f"Successfully created model version {version}")
                
                # Update description if provided
                if description:
                    try:
                        update_payload = {
                            "name": model_name,
                            "version": version,
                            "description": description
                        }
                        response = requests.patch(
                            f"{mlflow_tracking_uri}/api/2.0/mlflow/model-versions/update",
                            json=update_payload
                        )
                        if response.status_code == 200:
                            logger.info(f"Updated description for model version {version}")
                        else:
                            logger.warning(f"Failed to update description: {response.text}")
                    except Exception as e:
                        logger.warning(f"Error updating description: {str(e)}")
                
                # Add tags if provided
                if tags and isinstance(tags, dict):
                    for key, value in tags.items():
                        try:
                            tag_payload = {
                                "name": model_name,
                                "version": version,
                                "key": key,
                                "value": str(value)
                            }
                            response = requests.post(
                                f"{mlflow_tracking_uri}/api/2.0/mlflow/model-versions/set-tag",
                                json=tag_payload
                            )
                            if response.status_code == 200:
                                logger.info(f"Added tag {key}={value} to model version {version}")
                            else:
                                logger.warning(f"Failed to add tag {key}: {response.text}")
                        except Exception as e:
                            logger.warning(f"Error adding tag {key}: {str(e)}")
                
                return True
            else:
                logger.error(f"Failed to create model version: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error creating model version: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def save_model_from_run(run_id):
    """
    Lưu model từ run ID vào local directory
    
    Args:
        run_id: Run ID của MLflow run
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Get MLflow tracking URI from environment
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5002')
        logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
        logger.info(f"NO_NEW_RUNS mode: {NO_NEW_RUNS}")
        
        # Get run info
        try:
            response = requests.post(
                f"{mlflow_tracking_uri}/api/2.0/mlflow/runs/get",
                json={"run_id": run_id}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get run info: {response.text}")
                return False
            
            run_data = response.json()["run"]
            run_info = run_data["info"]
            
            # Extract metrics
            metrics = {}
            if "metrics" in run_data["data"]:
                for metric in run_data["data"]["metrics"]:
                    metrics[metric["key"]] = metric["value"]
            
            # Extract params
            params = {}
            if "params" in run_data["data"]:
                for param in run_data["data"]["params"]:
                    params[param["key"]] = param["value"]
            
            # Extract tags
            tags_data = {}
            if "tags" in run_data["data"]:
                for tag in run_data["data"]["tags"]:
                    tags_data[tag["key"]] = tag["value"]
            
            # Create model_info
            model_info = {
                "model_type": params.get("model_type", "unknown"),
                "run_id": run_id,
                "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "f1_score": metrics.get("f1_score", 0),
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "parameters": params,
                "features_count": int(params.get("n_features", 20)),
                "samples_count": int(params.get("n_samples", 1000)),
                "model_file": "best_model.joblib",
                "training_timestamp": run_info.get("start_time", 0),
                "artifact_uri": run_info.get("artifact_uri", ""),
                "experiment_id": run_info.get("experiment_id", ""),
                "run_name": tags_data.get("mlflow.runName", "mlflow_run"),
                "model_description": f"Model from run {run_id} with f1_score={metrics.get('f1_score', 0):.4f}",
                "is_binary_classification": True,
                "classes": ["0", "1"],
            }
            
            # Prepare paths
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            os.makedirs(models_dir, exist_ok=True)
            model_info_path = os.path.join(models_dir, "model_info.json")
            
            # Save model_info
            with open(model_info_path, "w") as f:
                json.dump(model_info, f, indent=2)
            logger.info(f"Saved model info to {model_info_path}")
            
            # Register model
            model_name = f"{model_info['model_type']}_model"
            success = register_model_from_run(
                run_id=run_id,
                model_name=model_name,
                description=model_info["model_description"],
                tags={
                    "f1_score": str(model_info["f1_score"]),
                    "accuracy": str(model_info["accuracy"]),
                    "saved_date": model_info["saved_date"]
                }
            )
            
            if success:
                logger.info(f"Successfully registered model {model_name} from run {run_id}")
                return True
            else:
                logger.warning(f"Failed to register model from run {run_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error getting run info: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logger.error(f"Error saving model from run: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register_model.py <run_id> [model_name]")
        sys.exit(1)
    
    run_id = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    logger.info(f"Starting model registration for run {run_id}, NO_NEW_RUNS mode: {NO_NEW_RUNS}")
    
    # If model_name is provided, register model only
    if model_name:
        success = register_model_from_run(
            run_id=run_id,
            model_name=model_name,
            description=f"Model from run {run_id}",
            tags={"registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        )
    # Otherwise, save model info and register
    else:
        success = save_model_from_run(run_id)
    
    sys.exit(0 if success else 1) 