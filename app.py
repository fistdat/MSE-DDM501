"""MLflow Lab API với Front-end"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
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
    """
    Lấy thông tin chi tiết về một run từ MLflow API
    """
    try:
        # MLflow API URL
        mlflow_api_url = "http://localhost:5002/api/2.0/mlflow"
        
        # Gửi request để lấy thông tin run
        response = requests.get(f"{mlflow_api_url}/runs/get", params={"run_id": run_id})
        
        if response.status_code == 200:
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
        else:
            logger.warning(f"Không thể lấy dữ liệu từ MLflow API cho run {run_id}: {response.text}")
            return None
    except Exception as e:
        logger.warning(f"Lỗi khi lấy dữ liệu từ MLflow API: {str(e)}")
        return None

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
    
    return render_template("index.html", 
                          model_trained=model_trained,
                          model_info=model_info,
                          tuning_results=tuning_results,
                          mlflow_status=mlflow_status)

@app.route("/run_simple_tuning", methods=["POST"])
def run_simple_tuning():
    """Chạy simple hyperparameter tuning với tùy chọn từ người dùng"""
    try:
        model_type = request.form.get("model_type", "random_forest")
        param_space = request.form.get("param_space", "small")
        samples = request.form.get("samples", "1000")
        features = request.form.get("features", "20")
        cv = request.form.get("cv", "5")
        
        command = f"python tuning_scripts/simple_hyperparam_tuning.py --model {model_type} --space {param_space} --samples {samples} --features {features} --cv {cv}"
        
        success, output = run_tuning_command(command)
        
        if success:
            flash('Tuning thành công', 'success')
        else:
            flash(f'Tuning thất bại: {output}', 'danger')
        
        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Lỗi khi chạy tuning: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Tuning thất bại: {str(e)}', 'danger')
        return redirect(url_for("home"))

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
    """
    Endpoint để thực hiện dự đoán sử dụng mô hình tốt nhất đã được huấn luyện.
    """
    # Kiểm tra xem đã có mô hình được huấn luyện chưa
    model_path = 'models/best_model.joblib'
    model_info_path = 'models/model_info.json'
    
    if not os.path.exists(model_path) or not os.path.exists(model_info_path):
        flash('Chưa có mô hình được huấn luyện. Vui lòng thực hiện tuning và lưu mô hình tốt nhất trước.', 'danger')
        return redirect(url_for('home'))
    
    try:
        # Lấy dữ liệu đầu vào từ form
        feature_data = request.form.get('feature_data')
        if not feature_data:
            flash('Dữ liệu đầu vào không hợp lệ', 'danger')
            return redirect(url_for('home'))
        
        # Parse JSON data
        input_data = json.loads(feature_data)
        
        # Tải mô hình
        model = joblib.load(model_path)
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Chuẩn bị dữ liệu cho dự đoán
        # Chuyển đổi dữ liệu đầu vào thành mảng numpy
        feature_arrays = []
        for sample in input_data:
            features = list(sample.values())
            feature_arrays.append(features)
        
        X = np.array(feature_arrays)
        
        # Thực hiện dự đoán
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Chuẩn bị kết quả
        prediction_results = []
        for i, pred in enumerate(predictions):
            # Lấy xác suất của lớp được dự đoán
            prob = probabilities[i][int(pred)]
            
            result = {
                'predicted_class': int(pred),
                'probability': float(prob),
                'features': input_data[i],
                'class_probabilities': {
                    '0': float(probabilities[i][0]),
                    '1': float(probabilities[i][1])
                }
            }
            prediction_results.append(result)
        
        # Thêm thông báo flash về kết quả dự đoán
        flash(f'Đã thực hiện dự đoán cho {len(prediction_results)} mẫu dữ liệu', 'success')
        
        # Render trang index với kết quả dự đoán và thông tin về model
        tuning_results = get_tuning_results()
        return render_template('index.html', 
                              model_trained=True, 
                              model_info=model_info,
                              tuning_results=tuning_results,
                              prediction_results=prediction_results,
                              active_tab='predict',
                              features=len(input_data[0]))
                                
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện dự đoán: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Có lỗi xảy ra khi thực hiện dự đoán: {str(e)}', 'danger')
        return redirect(url_for('home'))

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
        
        # Chạy script
        command = f"python {script_path}"
        success, output = run_tuning_command(command)
        
        if success:
            flash('Đã lưu mô hình tốt nhất thành công', 'success')
        else:
            flash(f'Không thể lưu mô hình tốt nhất: {output}', 'danger')
        
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"Lỗi khi chạy script save_best_model.py: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Lỗi khi lưu mô hình tốt nhất: {str(e)}', 'danger')
        return redirect(url_for('home'))

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

if __name__ == "__main__":
    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("tuning_experiment")
    
    # Start Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)