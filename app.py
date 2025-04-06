"""MLflow Lab API với Front-end"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
import numpy as np
import pandas as pd
from mlib import MLModel
import mlflow
import logging
from typing import Dict, Any
import traceback
import subprocess
import json
import os
import re
from datetime import datetime
import requests
import sys
import joblib
from mlflow.tracking import MlflowClient
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tempfile

# Thêm thư mục gốc vào sys.path để import các module từ thư mục con
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Thêm các đường dẫn tương đối
from mlflow_scripts import mlflow_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "mlops-flask-app-secret-key-2024"  # Cần thiết cho flash messages
model = MLModel()

# Template filter cho timestamp
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert timestamp (milliseconds) to readable date"""
    try:
        # Kiểm tra xem timestamp có phải đã là milliseconds (13 chữ số) không
        if len(str(timestamp)) >= 13:
            # Đã là milliseconds
            dt = datetime.fromtimestamp(timestamp / 1000)
        else:
            # Là seconds
            dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%d-%m-%Y %H:%M:%S')
    except:
        return "Invalid timestamp"

# Tạo thư mục templates nếu chưa tồn tại
os.makedirs('templates', exist_ok=True)

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

def run_tuning_command(command):
    """
    Chạy lệnh tuning hyperparameter và trả về kết quả
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Lỗi khi chạy lệnh: {stderr}")
            return False, stderr
        
        return True, stdout
    except Exception as e:
        logger.error(f"Lỗi khi chạy lệnh tuning: {str(e)}")
        return False, str(e)

def get_tuning_results():
    """
    Lấy kết quả tuning từ thư mục tuning_results
    """
    results = []
    results_dir = "tuning_results"
    
    if not os.path.exists(results_dir):
        return results
    
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(results_dir, file), 'r') as f:
                    result = json.load(f)
                    # Thêm tên file vào kết quả
                    result["file_name"] = file
                    # Trích xuất thời gian từ tên file
                    match = re.search(r'(\d{8}_\d{6})', file)
                    if match:
                        date_str = match.group(1)
                        try:
                            date_obj = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                            result["timestamp"] = date_obj.strftime('%d-%m-%Y %H:%M:%S')
                        except:
                            result["timestamp"] = "Unknown"
                    else:
                        result["timestamp"] = "Unknown"
                    
                    # Bổ sung thông tin từ MLflow nếu có MLflow run ID
                    if "mlflow_run_id" in result:
                        mlflow_data = get_mlflow_run_data(result["mlflow_run_id"])
                        if mlflow_data:
                            result.update(mlflow_data)
                    
                    results.append(result)
            except Exception as e:
                logger.error(f"Lỗi khi đọc file {file}: {str(e)}")
    
    # Sắp xếp kết quả theo thời gian (mới nhất lên đầu)
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results

def get_mlflow_run_data(run_id):
    """Lấy dữ liệu của một run từ MLflow API"""
    try:
        mlflow_api_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5002')
        if not mlflow_api_url.startswith("http"):
            mlflow_api_url = "http://localhost:5002"
            
        mlflow_api_url = f"{mlflow_api_url}/api/2.0/mlflow"
        
        # Gọi API MLflow để lấy thông tin về run
        response = requests.post(f"{mlflow_api_url}/runs/get", json={"run_id": run_id})
        
        if response.status_code != 200:
            logger.warning(f"Không thể lấy thông tin run từ MLflow API: {response.text}")
            return None
        
        run_data = response.json()
        
        # Trích xuất thông tin quan trọng từ response
        mlflow_result = {
            "mlflow_status": run_data.get("run", {}).get("info", {}).get("status"),
            "mlflow_metrics": {},
            "mlflow_params": {},
            "mlflow_tags": {},
            "mlflow_artifact_uri": run_data.get("run", {}).get("info", {}).get("artifact_uri")
        }
        
        # Xử lý metrics
        metrics = run_data.get("run", {}).get("data", {}).get("metrics", [])
        for metric in metrics:
            mlflow_result["mlflow_metrics"][metric["key"]] = metric["value"]
        
        # Xử lý params
        params = run_data.get("run", {}).get("data", {}).get("params", [])
        for param in params:
            mlflow_result["mlflow_params"][param["key"]] = param["value"]
        
        # Xử lý tags
        tags = run_data.get("run", {}).get("data", {}).get("tags", [])
        for tag in tags:
            mlflow_result["mlflow_tags"][tag["key"]] = tag["value"]
        
        return mlflow_result
    except Exception as e:
        logger.warning(f"Lỗi khi lấy dữ liệu từ MLflow API: {str(e)}")
        return None

def get_mlflow_experiments():
    """
    Lấy danh sách tất cả các experiment từ MLflow API
    """
    try:
        # Try a different approach using MLflow client
        # First approach: Using docker networking names
        mlflow_api_url = "http://mlflow:5002"
        logger.info(f"Connecting to MLflow API at {mlflow_api_url}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_api_url)
        
        # Use MLflow Python API instead of REST API
        try:
            experiments = mlflow.search_experiments()
            logger.info(f"Found {len(experiments)} experiments using MLflow Python API")
            return experiments
        except Exception as e:
            logger.warning(f"Error using MLflow Python API: {str(e)}")
        
        # Second approach: Try localhost
        try:
            mlflow_api_url = "http://localhost:5002"
            logger.info(f"Trying alternative connection to MLflow at {mlflow_api_url}")
            mlflow.set_tracking_uri(mlflow_api_url)
            experiments = mlflow.search_experiments()
            logger.info(f"Found {len(experiments)} experiments using localhost")
            return experiments
        except Exception as e:
            logger.warning(f"Error using localhost approach: {str(e)}")
        
        # Final fallback to direct REST API
        try:
            # Try with different endpoint formats
            endpoints = [
                "http://mlflow:5002/api/2.0/mlflow/experiments/search",
                "http://mlflow:5002/api/2.0/preview/mlflow/experiments/search",
                "http://localhost:5002/api/2.0/mlflow/experiments/search",
                "http://localhost:5002/api/2.0/preview/mlflow/experiments/search"
            ]
            
            for endpoint in endpoints:
                logger.info(f"Trying endpoint: {endpoint}")
                response = requests.get(endpoint)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Successfully connected to {endpoint}")
                    return data.get("experiments", [])
                else:
                    logger.warning(f"Failed to connect to {endpoint}: {response.status_code} - {response.text}")
            
            # If we get here, none of the endpoints worked
            logger.error("All MLflow API endpoints failed")
            return []
        except Exception as e:
            logger.error(f"Error in fallback approach: {str(e)}")
            return []
            
    except Exception as e:
        logger.warning(f"Lỗi khi lấy danh sách experiment từ MLflow API: {str(e)}")
        return []

def get_mlflow_experiment_runs(experiment_id):
    """
    Lấy danh sách các run của một experiment
    """
    try:
        logger.info(f"Fetching runs for experiment ID: {experiment_id}")
        
        # First approach: Try MLflow Python API with Docker network name
        try:
            # Set MLflow tracking URI with Docker network name
            mlflow_api_url = "http://mlflow:5002"
            logger.info(f"Using MLflow Python API with tracking URI: {mlflow_api_url}")
            mlflow.set_tracking_uri(mlflow_api_url)
            
            # Use MLflow Python client to search runs
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            if len(runs) > 0:
                logger.info(f"Found {len(runs)} runs using MLflow Python API")
                # Convert DataFrame to list of dictionaries format expected by template
                runs_list = []
                for _, row in runs.iterrows():
                    # Get full run data with additional info
                    run_data = mlflow.get_run(row.run_id)
                    
                    # Format tags to match template expectations
                    formatted_tags = {}
                    for k, v in run_data.data.tags.items():
                        # Handle nested tags (like mlflow.runName)
                        if '.' in k:
                            parts = k.split('.')
                            if parts[0] not in formatted_tags:
                                formatted_tags[parts[0]] = {}
                            if len(parts) == 2:
                                formatted_tags[parts[0]][parts[1]] = v
                            elif len(parts) == 3:
                                if parts[1] not in formatted_tags[parts[0]]:
                                    formatted_tags[parts[0]][parts[1]] = {}
                                formatted_tags[parts[0]][parts[1]][parts[2]] = v
                        else:
                            formatted_tags[k] = v
                    
                    # Add the original tags too for accessing with the ['key'] syntax
                    for k, v in run_data.data.tags.items():
                        formatted_tags[k] = v
                    
                    runs_list.append({
                        "info": {
                            "run_id": run_data.info.run_id,
                            "experiment_id": run_data.info.experiment_id,
                            "status": run_data.info.status,
                            "start_time": run_data.info.start_time,
                            "end_time": run_data.info.end_time,
                            "artifact_uri": run_data.info.artifact_uri
                        },
                        "data": {
                            "metrics": run_data.data.metrics,
                            "params": run_data.data.params,
                            "tags": formatted_tags
                        }
                    })
                return runs_list
            else:
                logger.warning("No runs found using MLflow Python API")
        except Exception as e:
            logger.warning(f"Error using MLflow Python API for runs: {str(e)}")

        # Second approach: Try REST API endpoints with different formats
        endpoints = [
            ("http://mlflow:5002/api/2.0/mlflow/runs/search", "POST"),
            ("http://localhost:5002/api/2.0/mlflow/runs/search", "POST")
        ]
        
        for endpoint, method in endpoints:
            try:
                logger.info(f"Trying REST API endpoint: {endpoint}")
                if method == "POST":
                    response = requests.post(endpoint, json={"experiment_ids": [experiment_id]})
                else:
                    response = requests.get(endpoint, params={"experiment_ids": [experiment_id]})
                
                if response.status_code == 200:
                    data = response.json()
                    runs = data.get("runs", [])
                    logger.info(f"Found {len(runs)} runs using REST API endpoint: {endpoint}")
                    
                    # Process the runs to match the template expectations
                    for run in runs:
                        if "data" in run and "tags" in run["data"]:
                            # Format tags to match template expectations
                            formatted_tags = {}
                            for tag in run["data"]["tags"]:
                                key = tag.get("key", "")
                                value = tag.get("value", "")
                                
                                # Handle nested tags (like mlflow.runName)
                                if '.' in key:
                                    parts = key.split('.')
                                    if parts[0] not in formatted_tags:
                                        formatted_tags[parts[0]] = {}
                                    if len(parts) == 2:
                                        formatted_tags[parts[0]][parts[1]] = value
                                    elif len(parts) == 3:
                                        if parts[1] not in formatted_tags[parts[0]]:
                                            formatted_tags[parts[0]][parts[1]] = {}
                                        formatted_tags[parts[0]][parts[1]][parts[2]] = value
                                
                                # Add the original key too
                                formatted_tags[key] = value
                            
                            # Replace the tags in the run with our formatted version
                            run["data"]["tags"] = formatted_tags
                    
                    return runs
                else:
                    logger.warning(f"Failed to get runs from {endpoint}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.warning(f"Error accessing REST API endpoint {endpoint}: {str(e)}")
        
        # If all attempts fail, return empty list
        logger.error("All methods to get experiment runs failed")
        return []
    except Exception as e:
        logger.warning(f"Lỗi khi lấy danh sách run từ MLflow API: {str(e)}")
        return []

def get_mlflow_registered_models():
    """
    Lấy danh sách các model đã đăng ký trong Model Registry
    """
    try:
        # MLflow API URL
        mlflow_api_url = "http://localhost:5002/api/2.0/mlflow"
        
        # Gửi request để lấy danh sách registered models
        response = requests.get(f"{mlflow_api_url}/registered-models/list")
        
        if response.status_code == 200:
            data = response.json()
            return data.get("registered_models", [])
        else:
            logger.warning(f"Không thể lấy danh sách registered models từ MLflow API: {response.text}")
            return []
    except Exception as e:
        logger.warning(f"Lỗi khi lấy danh sách registered models từ MLflow API: {str(e)}")
        return []

@app.route("/", methods=["GET"])
def home():
    """Trang chủ của ứng dụng"""
    # Lấy kết quả tuning gần nhất
    tuning_results = get_tuning_results()
    
    # Kiểm tra kết nối MLflow
    mlflow_status = "disconnected"
    try:
        response = requests.get("http://localhost:5002/api/2.0/mlflow/experiments/list")
        if response.status_code == 200:
            mlflow_status = "connected"
    except:
        pass
    
    # Kiểm tra xem có model đã train chưa
    model_trained = os.path.exists(os.path.join("models", "best_model.joblib"))
    model_info = None
    
    # Nếu model đã được train, đọc thông tin từ model_info.json
    if model_trained:
        try:
            with open(os.path.join("models", "model_info.json"), "r") as f:
                model_info = json.load(f)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file model_info.json: {str(e)}")
    
    # Lấy danh sách các experiment từ MLflow
    mlflow_experiments = []
    if mlflow_status == "connected":
        mlflow_experiments = get_mlflow_experiments()
    
    # Lấy danh sách các model đã đăng ký
    registered_models = []
    if mlflow_status == "connected":
        registered_models = get_mlflow_registered_models()
    
    return render_template("index.html", 
                          model_trained=model_trained,
                          model_info=model_info,
                          tuning_results=tuning_results,
                          mlflow_status=mlflow_status,
                          mlflow_experiments=mlflow_experiments,
                          registered_models=registered_models)

@app.route("/run_simple_tuning", methods=["POST"])
def run_simple_tuning():
    """Chạy simple hyperparameter tuning với tùy chọn từ người dùng"""
    try:
        # Log raw request data for debugging
        logger.info(f"Request form data: {request.form}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request method: {request.method}")
        
        model_type = request.form.get("model_type", "random_forest")
        param_space = request.form.get("param_space", "small")
        samples = request.form.get("samples", "1000")
        features = request.form.get("features", "20")
        cv = request.form.get("cv", "5")
        auto_register = request.form.get("auto_register", "0") == "1"
        
        # Debug log
        logger.info(f"Starting tuning with params: model={model_type}, space={param_space}, samples={samples}, features={features}, cv={cv}, auto_register={auto_register}")
        
        # Đường dẫn đầy đủ đến script
        script_path = os.path.join(app.root_path, "tuning_scripts", "simple_hyperparam_tuning.py")
        logger.info(f"Script path: {script_path}")
        
        # Kiểm tra tệp tồn tại
        if not os.path.exists(script_path):
            logger.error(f"Script not found at {script_path}")
            flash(f'Tuning thất bại: Script không tồn tại tại {script_path}', 'danger')
            return redirect(url_for("home"))
        
        command = f"python {script_path} --model {model_type} --space {param_space} --samples {samples} --features {features} --cv {cv}"
        logger.info(f"Running command: {command}")
        
        # Set environment variables for auto-registration
        env = os.environ.copy()
        if auto_register:
            env["AUTO_REGISTER_MODEL"] = "1"
            logger.info("Model will be automatically registered to Model Registry")
        else:
            env["AUTO_REGISTER_MODEL"] = "0"
        
        # Run tuning command with environment variables
        env_str = " ".join([f"{k}={v}" for k, v in env.items() if k.startswith("AUTO_") or k.startswith("MLFLOW_")])
        logger.info(f"Environment variables: {env_str}")
        
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("Tuning completed successfully")
            flash('Tuning thành công', 'success')
            
            # If auto_register is enabled, try direct registration
            if auto_register:
                logger.info("AUTO_REGISTER_MODEL is enabled, will try to register the best model")
                try:
                    # Direct approach to find and register the best model
                    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5002')
                    logger.info(f"Using MLflow tracking URI: {tracking_uri}")
                    
                    # Use MLflow REST API to get the latest run
                    mlflow_api_url = tracking_uri
                    if not mlflow_api_url.startswith("http"):
                        mlflow_api_url = "http://localhost:5002"
                    
                    logger.info(f"Using MLflow API URL: {mlflow_api_url}")
                    
                    # Step 1: Find the experiment ID
                    try:
                        response = requests.get(f"{mlflow_api_url}/api/2.0/mlflow/experiments/search", 
                                            json={"max_results": 10})
                        if response.status_code != 200:
                            logger.error(f"Failed to get experiments: {response.text}")
                            raise Exception(f"Failed to get experiments: {response.text}")
                        
                        experiments = response.json().get("experiments", [])
                        experiment_id = None
                        
                        for exp in experiments:
                            if exp.get("name") == "tuning_experiment":
                                experiment_id = exp.get("experiment_id")
                                break
                        
                        if experiment_id is None:
                            logger.error("Could not find experiment 'tuning_experiment'")
                            flash('Auto-register failed: Could not find experiment', 'warning')
                            return redirect(url_for("home"))
                        
                        logger.info(f"Found experiment with ID: {experiment_id}")
                        
                        # Step 2: Get the latest run with the highest F1-score
                        response = requests.post(f"{mlflow_api_url}/api/2.0/mlflow/runs/search", 
                                            json={"experiment_ids": [experiment_id], "max_results": 10})
                        
                        if response.status_code != 200:
                            logger.error(f"Failed to get runs: {response.text}")
                            raise Exception(f"Failed to get runs: {response.text}")
                        
                        runs = response.json().get("runs", [])
                        
                        if not runs:
                            logger.error("No runs found in experiment")
                            flash('Auto-register failed: No runs found', 'warning')
                            return redirect(url_for("home"))
                        
                        # Find run with highest F1-score
                        best_run = None
                        best_f1 = -1
                        
                        for run in runs:
                            metrics = {}
                            for metric in run.get("data", {}).get("metrics", []):
                                metrics[metric["key"]] = metric["value"]
                            
                            f1 = metrics.get("f1_score", -1)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_run = run
                        
                        if best_run is None:
                            logger.error("Could not find a run with f1_score")
                            flash('Auto-register failed: No run with f1_score found', 'warning')
                            return redirect(url_for("home"))
                        
                        run_id = best_run["info"]["run_id"]
                        logger.info(f"Found best run with ID: {run_id}, f1_score: {best_f1}")
                        
                        # Step 3: Try to register the model using MLflow Python API
                        try:
                            # Set tracking URI
                            mlflow.set_tracking_uri(tracking_uri)
                            
                            # Create model name
                            model_name = f"{model_type}_model"
                            model_uri = f"runs:/{run_id}/model"
                            
                            # Create model if it doesn't exist
                            client = MlflowClient()
                            try:
                                client.get_registered_model(model_name)
                                logger.info(f"Model {model_name} already exists")
                            except:
                                client.create_registered_model(model_name)
                                logger.info(f"Created model {model_name}")
                            
                            # Register model
                            version = mlflow.register_model(model_uri, model_name)
                            logger.info(f"Registered model {model_name} version {version.version}")
                            
                            # Add tags
                            client.set_model_version_tag(
                                name=model_name,
                                version=version.version,
                                key="f1_score",
                                value=str(best_f1)
                            )
                            client.set_model_version_tag(
                                name=model_name,
                                version=version.version,
                                key="registered_at",
                                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                            
                            flash(f'Model {model_name} successfully registered with version {version.version}', 'success')
                            return redirect(url_for("home"))
                            
                        except Exception as e:
                            logger.error(f"Failed to register model using Python API: {str(e)}")
                            logger.error(traceback.format_exc())
                            
                            # Fall back to REST API
                            logger.info("Falling back to REST API for model registration")
                            try:
                                # Step 1: Create model if it doesn't exist
                                model_name = f"{model_type}_model"
                                
                                response = requests.get(f"{mlflow_api_url}/api/2.0/mlflow/registered-models/get", 
                                                    params={"name": model_name})
                                
                                if response.status_code != 200:
                                    # Model doesn't exist, create it
                                    response = requests.post(f"{mlflow_api_url}/api/2.0/mlflow/registered-models/create", 
                                                        json={"name": model_name})
                                    
                                    if response.status_code != 200:
                                        logger.error(f"Failed to create model: {response.text}")
                                        raise Exception(f"Failed to create model: {response.text}")
                                    
                                    logger.info(f"Created model {model_name}")
                                
                                # Step 2: Register a new version
                                model_uri = f"runs:/{run_id}/model"
                                response = requests.post(f"{mlflow_api_url}/api/2.0/mlflow/model-versions/create", 
                                                    json={"name": model_name, "source": model_uri})
                                
                                if response.status_code != 200:
                                    logger.error(f"Failed to create model version: {response.text}")
                                    raise Exception(f"Failed to create model version: {response.text}")
                                
                                version = response.json()["model_version"]["version"]
                                logger.info(f"Registered model {model_name} version {version}")
                                
                                # Step 3: Add tags
                                tag_payload = {"name": model_name, "version": version, "key": "f1_score", "value": str(best_f1)}
                                response = requests.post(f"{mlflow_api_url}/api/2.0/mlflow/model-versions/set-tag", 
                                                    json=tag_payload)
                                
                                if response.status_code != 200:
                                    logger.warning(f"Failed to add f1_score tag: {response.text}")
                                
                                tag_payload = {"name": model_name, "version": version, "key": "registered_at", 
                                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                                response = requests.post(f"{mlflow_api_url}/api/2.0/mlflow/model-versions/set-tag", 
                                                    json=tag_payload)
                                
                                if response.status_code != 200:
                                    logger.warning(f"Failed to add registered_at tag: {response.text}")
                                
                                flash(f'Model {model_name} successfully registered with version {version}', 'success')
                                
                            except Exception as e2:
                                logger.error(f"Failed to register model using REST API: {str(e2)}")
                                logger.error(traceback.format_exc())
                                flash(f'Failed to register model: {str(e2)}', 'warning')
                        
                    except Exception as e:
                        logger.error(f"Error in auto-register process: {str(e)}")
                        logger.error(traceback.format_exc())
                        flash(f'Error in auto-register process: {str(e)}', 'warning')
                    
                except Exception as e:
                    logger.error(f"Error in auto-register: {str(e)}")
                    logger.error(traceback.format_exc())
                    flash(f'Lỗi khi tự động đăng ký model: {str(e)}', 'warning')
        else:
            logger.error(f"Tuning failed. Output: {stderr}")
            flash(f'Tuning thất bại: {stderr}', 'danger')
        
        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Lỗi khi chạy tuning: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Tuning thất bại: {str(e)}', 'danger')
        return redirect(url_for("home"))

@app.route("/tuning/simple", methods=["POST"])
def run_simple_tuning_alias():
    """Alias cho route run_simple_tuning"""
    return run_simple_tuning()

@app.route("/run_custom_tuning", methods=["POST"])
def run_custom_tuning():
    """Chạy hyperparameter tuning với tham số tùy chỉnh"""
    try:
        model_type = request.form.get("model_type", "random_forest")
        
        # Xử lý các tham số tùy chỉnh từ form
        custom_params = {}
        if model_type == "random_forest":
            # Các tham số cho Random Forest
            n_estimators = request.form.get("rf_n_estimators", "")
            max_depth = request.form.get("rf_max_depth", "")
            min_samples_split = request.form.get("rf_min_samples_split", "")
            min_samples_leaf = request.form.get("rf_min_samples_leaf", "")
            
            if n_estimators:
                custom_params["n_estimators"] = [int(x.strip()) for x in n_estimators.split(",")]
            if max_depth:
                custom_params["max_depth"] = [int(x.strip()) for x in max_depth.split(",")]
            if min_samples_split:
                custom_params["min_samples_split"] = [int(x.strip()) for x in min_samples_split.split(",")]
            if min_samples_leaf:
                custom_params["min_samples_leaf"] = [int(x.strip()) for x in min_samples_leaf.split(",")]
        
        elif model_type == "gradient_boosting":
            # Các tham số cho Gradient Boosting
            n_estimators = request.form.get("gb_n_estimators", "")
            learning_rate = request.form.get("gb_learning_rate", "")
            max_depth = request.form.get("gb_max_depth", "")
            min_samples_split = request.form.get("gb_min_samples_split", "")
            
            if n_estimators:
                custom_params["n_estimators"] = [int(x.strip()) for x in n_estimators.split(",")]
            if learning_rate:
                custom_params["learning_rate"] = [float(x.strip()) for x in learning_rate.split(",")]
            if max_depth:
                custom_params["max_depth"] = [int(x.strip()) for x in max_depth.split(",")]
            if min_samples_split:
                custom_params["min_samples_split"] = [int(x.strip()) for x in min_samples_split.split(",")]
        
        # Các tham số chung
        samples = request.form.get("samples", "1000")
        features = request.form.get("features", "20")
        cv = request.form.get("cv", "5")
        
        # Lưu tham số vào file tạm
        custom_params_file = f"custom_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(custom_params_file, "w") as f:
            json.dump(custom_params, f)
        
        # Tạo lệnh chạy với file tham số
        command = f"python tuning_scripts/custom_hyperparam_tuning.py --model {model_type} --params-file {custom_params_file} --samples {samples} --features {features} --cv {cv}"
        
        success, output = run_tuning_command(command)
        
        try:
            # Xóa file tham số tạm sau khi chạy xong
            os.remove(custom_params_file)
        except:
            pass
        
        if success:
            flash('Tuning tùy chỉnh thành công', 'success')
        else:
            flash(f'Tuning tùy chỉnh thất bại: {output}', 'danger')
        
        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Lỗi khi chạy tuning tùy chỉnh: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Tuning tùy chỉnh thất bại: {str(e)}', 'danger')
        return redirect(url_for("home"))

@app.route("/view_result/<filename>", methods=["GET"])
def view_result(filename):
    """Xem chi tiết kết quả tuning"""
    try:
        results_dir = "tuning_results"
        file_path = os.path.join(results_dir, filename)
        
        if not os.path.exists(file_path):
            flash('File kết quả không tồn tại', 'danger')
            return redirect(url_for("home"))
        
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        # Lấy thông tin từ MLflow nếu có experiment_id
        if "experiment_id" in result:
            try:
                # MLflow API URL
                mlflow_api_url = "http://localhost:5002/api/2.0/mlflow"
                
                # Lấy thông tin experiment
                exp_response = requests.get(
                    f"{mlflow_api_url}/experiments/get",
                    params={"experiment_id": result["experiment_id"]}
                )
                
                if exp_response.status_code == 200:
                    experiment_data = exp_response.json()
                    
                    # Lấy danh sách runs của experiment
                    run_response = requests.get(
                        f"{mlflow_api_url}/runs/search",
                        json={"experiment_ids": [result["experiment_id"]]}
                    )
                    
                    if run_response.status_code == 200:
                        runs = run_response.json().get("runs", [])
                        if runs:
                            # Lấy thông tin run mới nhất
                            latest_run = runs[0]
                            run_info = latest_run.get("info", {})
                            run_data = latest_run.get("data", {})
                            
                            # Xử lý metrics
                            metrics = {}
                            for metric in run_data.get("metrics", []):
                                metrics[metric["key"]] = metric["value"]
                            
                            # Xử lý params
                            params = {}
                            for param in run_data.get("params", []):
                                params[param["key"]] = param["value"]
                            
                            # Cập nhật thông tin vào result
                            result["metrics"] = metrics
                            result["best_params"] = params
                            result["mlflow_status"] = run_info.get("status")
                            
                            # Cập nhật thông tin cơ bản từ params
                            if "n_samples" in params:
                                result["n_samples_train"] = int(float(params["n_samples"]) * 0.8)
                                result["n_samples_test"] = int(float(params["n_samples"]) * 0.2)
                            if "n_features" in params:
                                result["n_features"] = int(params["n_features"])
                            if "cv" in params:
                                result["cv"] = int(params["cv"])
            except Exception as e:
                logger.warning(f"Không thể lấy dữ liệu từ MLflow API: {str(e)}")
                # Không dừng luồng xử lý nếu lỗi MLflow API
                pass
        
        return render_template("result_detail.html", result=result, filename=filename)
    except FileNotFoundError:
        flash('File kết quả không tồn tại', 'danger')
        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Lỗi khi xem kết quả: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Lỗi khi đọc file kết quả: {str(e)}', 'danger')
        return redirect(url_for("home"))

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": model.is_trained
    }), 200

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

@app.route("/predict", methods=["POST"])
def predict():
    """Dự đoán sử dụng mô hình đã huấn luyện"""
    try:
        feature_data = request.form.get("feature_data", "")
        
        # Check if we should use the best model
        use_best_model = request.form.get("use_best_model", "0") == "1"
        best_model_run_id = request.form.get("best_model_run_id", "")
        
        if not feature_data:
            flash("Vui lòng nhập dữ liệu features để phân loại", "danger")
            return redirect(url_for("home", active_tab="predict"))
        
        try:
            features = json.loads(feature_data)
        except json.JSONDecodeError:
            flash("Dữ liệu đầu vào không đúng định dạng JSON", "danger")
            return redirect(url_for("home", active_tab="predict"))
        
        if not isinstance(features, list):
            features = [features]
        
        # Determine whether to use the default model or the best model
        if use_best_model and best_model_run_id:
            # Load the model from the specified run
            logger.info(f"Using best model from run: {best_model_run_id}")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5002'))
            
            try:
                # Load model from the run
                model_path = f"runs:/{best_model_run_id}/model"
                model = mlflow.pyfunc.load_model(model_path)
                logger.info(f"Successfully loaded model from run: {best_model_run_id}")
                
                # Get run info for display
                client = MlflowClient()
                run = client.get_run(best_model_run_id)
                run_info = {
                    "run_id": best_model_run_id,
                    "model_type": run.data.params.get("model_type", "unknown"),
                    "accuracy": float(run.data.metrics.get("accuracy", 0)),
                    "f1_score": float(run.data.metrics.get("f1_score", 0)),
                    "precision": float(run.data.metrics.get("precision", 0)),
                    "recall": float(run.data.metrics.get("recall", 0)),
                    "features_count": int(run.data.params.get("n_features", 20)),
                    "samples_count": int(run.data.params.get("n_samples", 1000)),
                    "saved_date": "Temporary (not saved)"
                }
            except Exception as e:
                logger.error(f"Error loading model from run {best_model_run_id}: {str(e)}")
                logger.error(traceback.format_exc())
                flash(f"Lỗi khi tải mô hình từ run {best_model_run_id}: {str(e)}", "danger")
                return redirect(url_for("home", active_tab="predict"))
        else:
            # Load the default saved model
            logger.info("Using default saved model")
            model_info_path = os.path.join(app.root_path, "models", "model_info.json")
            model_path = os.path.join(app.root_path, "models", "best_model.joblib")
            
            if not os.path.exists(model_path) or not os.path.exists(model_info_path):
                flash("Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi phân loại.", "warning")
                return redirect(url_for("home", active_tab="predict"))
            
            # Load model info
            with open(model_info_path, 'r') as f:
                run_info = json.load(f)
            
            # Load the model
            model = joblib.load(model_path)
        
        # Process each sample for prediction
        results = []
        for i, sample_features in enumerate(features):
            # Convert to a format suitable for the model
            try:
                # Convert to numpy array
                X = format_features_for_prediction(sample_features, run_info.get("features_count", 20))
                
                # Get predictions
                y_pred = model.predict(X)[0]
                y_proba = model.predict_proba(X)[0]
                
                # Create prediction result
                prediction_result = {
                    "id": i + 1,
                    "features": sample_features,
                    "predicted_class": int(y_pred),
                    "probability": float(y_proba[int(y_pred)]),
                    "class_probabilities": {str(i): float(p) for i, p in enumerate(y_proba)}
                }
                
                results.append(prediction_result)
            except Exception as e:
                logger.error(f"Lỗi khi dự đoán mẫu {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
                flash(f"Lỗi khi dự đoán mẫu {i+1}: {str(e)}", "danger")
                return redirect(url_for("home", active_tab="predict"))
        
        return render_template("index.html", 
                               active_tab="predict",
                               model_trained=True,
                               model_info=run_info,
                               prediction_results=results)
    
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Lỗi khi dự đoán: {str(e)}", "danger")
        return redirect(url_for("home", active_tab="predict"))

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

@app.route("/mlflow_runs", methods=["GET"])
def get_mlflow_runs():
    """API để lấy danh sách runs từ MLflow"""
    try:
        # MLflow API URL
        mlflow_api_url = "http://localhost:5002/api/2.0/mlflow"
        
        # Lấy danh sách experiments
        response = requests.get(f"{mlflow_api_url}/experiments/list")
        if response.status_code != 200:
            return jsonify({"error": "Không thể kết nối tới MLflow API"}), 500
        
        experiments = response.json().get("experiments", [])
        
        all_runs = []
        for exp in experiments:
            exp_id = exp["experiment_id"]
            
            # Lấy danh sách runs cho mỗi experiment
            run_response = requests.get(
                f"{mlflow_api_url}/runs/search",
                json={"experiment_ids": [exp_id]}
            )
            
            if run_response.status_code == 200:
                runs = run_response.json().get("runs", [])
                all_runs.extend(runs)
        
        # Xử lý và format dữ liệu
        formatted_runs = []
        for run in all_runs:
            run_info = run.get("info", {})
            run_data = run.get("data", {})
            
            metrics = {}
            for metric in run_data.get("metrics", []):
                metrics[metric["key"]] = metric["value"]
            
            params = {}
            for param in run_data.get("params", []):
                params[param["key"]] = param["value"]
            
            formatted_run = {
                "run_id": run_info.get("run_id"),
                "experiment_id": run_info.get("experiment_id"),
                "status": run_info.get("status"),
                "start_time": run_info.get("start_time"),
                "end_time": run_info.get("end_time"),
                "metrics": metrics,
                "params": params
            }
            
            formatted_runs.append(formatted_run)
        
        return jsonify({"runs": formatted_runs})
    
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách MLflow runs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/run_save_best_model", methods=["GET"])
def run_save_best_model():
    """
    Chạy script lưu model tốt nhất và chuyển hướng về trang chủ
    """
    try:
        # Đường dẫn đến script save_best_model.py
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_scripts", "save_best_model.py")
        
        # Thiết lập biến môi trường MLFLOW_TRACKING_URI nếu đang chạy trong Docker
        environment = os.environ.copy()
        if "MLFLOW_TRACKING_URI" in os.environ:
            logger.info(f"Sử dụng MLFLOW_TRACKING_URI từ môi trường: {os.environ['MLFLOW_TRACKING_URI']}")
        else:
            # Nếu không có biến môi trường, sử dụng giá trị mặc định
            environment["MLFLOW_TRACKING_URI"] = "http://localhost:5002"
            logger.info(f"Sử dụng MLFLOW_TRACKING_URI mặc định: http://localhost:5002")
        
        # Chạy script với biến môi trường đã cấu hình
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Script save_best_model.py đã chạy thành công: {stdout}")
            flash('Đã lưu mô hình tốt nhất thành công', 'success')
            
            # Kiểm tra kết nối MLflow để xác nhận model đã được đăng ký
            try:
                mlflow_api_url = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002")
                response = requests.get(f"{mlflow_api_url}/api/2.0/mlflow/registered-models/list")
                
                if response.status_code == 200:
                    models = response.json().get("registered_models", [])
                    if models:
                        model_names = [model.get("name") for model in models]
                        flash(f'Các mô hình đã đăng ký: {", ".join(model_names)}', 'info')
                else:
                    logger.warning(f"Không thể lấy danh sách model từ MLflow: {response.text}")
            except Exception as e:
                logger.warning(f"Lỗi khi kiểm tra model trong MLflow: {str(e)}")
        else:
            logger.error(f"Lỗi khi chạy script save_best_model.py: {stderr}")
            flash(f'Không thể lưu mô hình tốt nhất: {stderr}', 'danger')
        
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"Lỗi khi chạy script save_best_model.py: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Lỗi khi lưu mô hình tốt nhất: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route("/find_best_model/<experiment_id>", methods=["GET"])
def find_best_model(experiment_id):
    """Tìm model tốt nhất dựa trên metric được chọn (mặc định là accuracy)"""
    try:
        metric = request.args.get("metric", "accuracy")  # Mặc định sử dụng accuracy
        run_id = request.args.get("run_id")
        logger.info(f"Finding best model for experiment {experiment_id}" + 
                   (f" based on {metric}" if run_id is None else f" with run_id {run_id}"))
        
        # Lấy danh sách runs từ experiment
        runs = get_mlflow_experiment_runs(experiment_id)
        if not runs:
            flash(f"Không tìm thấy runs nào trong experiment {experiment_id}", "warning")
            return redirect(url_for("home"))
        
        # Nếu có run_id, tìm run cụ thể
        if run_id:
            best_run = next((run for run in runs if run["info"]["run_id"] == run_id), None)
            if best_run is None:
                flash(f"Không tìm thấy run với ID {run_id}", "warning")
                return redirect(url_for("home"))
        else:
            # Tìm run có metric tốt nhất
            best_run = None
            best_metric_value = -1
            
            for run in runs:
                if "data" in run and "metrics" in run["data"]:
                    metrics = run["data"]["metrics"]
                    if metric in metrics and metrics[metric] > best_metric_value:
                        best_metric_value = metrics[metric]
                        best_run = run
            
            if best_run is None:
                flash(f"Không tìm thấy run nào có metric '{metric}'", "warning")
                return redirect(url_for("home"))
        
        # Lấy thông tin best run
        run_id = best_run["info"]["run_id"]
        run_name = best_run["data"]["tags"].get("mlflow.runName", run_id[:8])
        model_type = best_run["data"]["params"].get("model_type", "unknown")
        metrics_info = best_run["data"]["metrics"]
        accuracy = metrics_info.get("accuracy", 0)
        f1_score = metrics_info.get("f1_score", 0)
        
        # Hiển thị thông báo và chuyển hướng về trang chủ
        flash(f"Đã tìm thấy model tốt nhất: {model_type} (Run ID: {run_id[:8]}) với {metric}={best_metric_value:.4f}", "success")
        return redirect(url_for("home"))
        
    except Exception as e:
        logger.error(f"Lỗi khi tìm model tốt nhất: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Lỗi khi tìm model tốt nhất: {str(e)}", "danger")
        return redirect(url_for("home"))

@app.route("/register-model", methods=["POST"])
def register_model():
    """Đăng ký model vào MLflow Model Registry"""
    try:
        run_id = request.form.get("run_id")
        model_name = request.form.get("model_name")
        
        if not run_id or not model_name:
            flash("Vui lòng cung cấp Run ID và tên model", "danger")
            return redirect(url_for("home"))
        
        # Đường dẫn đến script register_model.py
        register_model_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "tuning_scripts",
            "register_model.py"
        )
        
        # Mô tả và tags
        description = request.form.get("description", f"Model registered from run {run_id}")
        f1_score = request.form.get("f1_score", "0.0")
        accuracy = request.form.get("accuracy", "0.0")
        
        # Thiết lập biến môi trường cho MLflow
        environment = os.environ.copy()
        if "MLFLOW_TRACKING_URI" not in environment:
            environment["MLFLOW_TRACKING_URI"] = "http://mlflow:5002"
        
        # Chạy script register_model.py
        process = subprocess.Popen(
            [
                "python", 
                register_model_script, 
                run_id, 
                model_name
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Model đã được đăng ký thành công: {stdout}")
            flash(f"Đã đăng ký model {model_name} thành công", "success")
        else:
            logger.error(f"Không thể đăng ký model: {stderr}")
            flash(f"Không thể đăng ký model: {stderr}", "danger")
        
        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Lỗi khi đăng ký model: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Lỗi khi đăng ký model: {str(e)}", "danger")
        return redirect(url_for("home"))

@app.route("/save-model/<run_id>", methods=["POST"])
def save_model_from_run(run_id):
    """Lưu model từ một run MLflow vào thư mục models"""
    try:
        # Get model name from form
        model_name = request.form.get("model_name", "")
        
        # Đường dẫn đến script register_model.py
        register_model_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "tuning_scripts",
            "register_model.py"
        )
        
        # Thiết lập biến môi trường cho MLflow
        environment = os.environ.copy()
        if "MLFLOW_TRACKING_URI" not in environment:
            environment["MLFLOW_TRACKING_URI"] = "http://mlflow:5002"
        
        # Set NO_NEW_RUNS environment variable to prevent creating new runs
        environment["NO_NEW_RUNS"] = "1"
        
        # Command arguments
        command = ["python", register_model_script, run_id]
        
        # Add model name as an argument if provided
        if model_name:
            command.append(model_name)
            
        # Chạy script register_model.py
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Model '{model_name}' đã được lưu thành công: {stdout}")
            flash(f"Đã lưu và đăng ký model '{model_name}' thành công", "success")
        else:
            logger.error(f"Không thể lưu model: {stderr}")
            flash(f"Không thể lưu model: {stderr}", "danger")
        
        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Lỗi khi lưu model: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Lỗi khi lưu model: {str(e)}", "danger")
        return redirect(url_for("home"))

@app.route("/test_tuning_script", methods=["GET"])
def test_tuning_script():
    """Test route để kiểm tra script tuning"""
    try:
        # Đường dẫn đầy đủ đến script
        script_path = os.path.join(app.root_path, "tuning_scripts", "simple_hyperparam_tuning.py")
        
        # Kiểm tra tệp tồn tại
        if not os.path.exists(script_path):
            logger.error(f"Script not found at {script_path}")
            return jsonify({
                "status": "error",
                "message": f"Script không tồn tại tại {script_path}",
                "root_path": app.root_path,
                "cwd": os.getcwd()
            }), 500
        
        # Thử chạy script với các tham số tối thiểu
        command = f"python {script_path} --model random_forest --space tiny --samples 100 --features 5 --cv 2 --no-mlflow"
        
        logger.info(f"Running test command: {command}")
        
        success, output = run_tuning_command(command)
        
        return jsonify({
            "status": "success" if success else "error",
            "output": output,
            "script_path": script_path,
            "root_path": app.root_path,
            "cwd": os.getcwd()
        })
    except Exception as e:
        logger.error(f"Lỗi khi test script tuning: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/model-details/<model_name>/<version>", methods=["GET"])
def view_model_details(model_name, version):
    """Hiển thị chi tiết về một phiên bản model cụ thể từ registry"""
    flash("Chức năng Model Registry đã bị tắt", "warning")
    return redirect(url_for("home"))

@app.route("/model/deploy/<model_name>/<version>", methods=["POST"])
def deploy_model(model_name, version):
    """Chuyển một model version sang trạng thái Production"""
    flash("Chức năng Model Registry đã bị tắt", "warning")
    return redirect(url_for("home"))

@app.route("/model/download/<model_name>/<version>", methods=["GET"])
def download_model(model_name, version):
    """Tải xuống một model cụ thể từ MLflow Model Registry"""
    flash("Chức năng Model Registry đã bị tắt", "warning")
    return redirect(url_for("home"))

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

def format_features_for_prediction(features, num_features):
    """Format features into a numpy array suitable for prediction"""
    import numpy as np
    
    # Create a numpy array with zeros
    X = np.zeros((1, num_features))
    
    # Try to fill the array from the features dictionary
    try:
        # If features are named (e.g., {"feature_1": 0.5, "feature_2": 1.2, ...})
        if all(key.startswith("feature_") for key in features.keys()):
            for key, value in features.items():
                if isinstance(key, str) and key.startswith("feature_"):
                    try:
                        # Extract the feature index (1-based)
                        idx = int(key.split("_")[1]) - 1
                        if 0 <= idx < num_features:
                            X[0, idx] = float(value)
                    except (ValueError, IndexError):
                        continue
        # If features are a list or array-like
        elif isinstance(features, (list, tuple)) and len(features) <= num_features:
            for i, value in enumerate(features):
                if i < num_features:
                    X[0, i] = float(value)
        # If features are arbitrary names
        else:
            for i, (key, value) in enumerate(features.items()):
                if i < num_features:
                    X[0, i] = float(value)
    except Exception as e:
        logger.warning(f"Error formatting features: {str(e)}")
        # Continue with zeros for missing features
    
    return X

if __name__ == "__main__":
    # Configure MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002"))
    mlflow.set_experiment("tuning_experiment")
    
    # Start Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)