"""
Tiện ích MLflow đơn giản hóa
"""

import os
import socket
import subprocess
import time
import requests
import sys
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình MLflow
MLFLOW_PORT = 5002
MLFLOW_HOST = "0.0.0.0"
MLFLOW_URI = f"http://localhost:{MLFLOW_PORT}"
ARTIFACT_LOCATION = "./mlruns"

def is_port_available(port, host="0.0.0.0"):
    """Kiểm tra xem cổng có khả dụng không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def is_mlflow_running(uri=MLFLOW_URI, max_retries=1):
    """Kiểm tra xem MLflow server đã chạy chưa"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{uri}/api/2.0/mlflow/experiments/list")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(1)
    return False

def ensure_artifact_dir():
    """Đảm bảo thư mục artifact tồn tại"""
    if not os.path.exists(ARTIFACT_LOCATION):
        os.makedirs(ARTIFACT_LOCATION)
        logger.info(f"Đã tạo thư mục artifact: {ARTIFACT_LOCATION}")

def start_mlflow_server(wait=True):
    """Khởi động MLflow server nếu chưa chạy"""
    # Kiểm tra xem MLflow đã chạy chưa
    if is_mlflow_running():
        logger.info("MLflow server đã chạy")
        return True
    
    # Kiểm tra cổng
    if not is_port_available(MLFLOW_PORT):
        logger.warning(f"Cổng {MLFLOW_PORT} đang bị sử dụng, không thể khởi động MLflow server")
        logger.info(f"Để giải phóng cổng, chạy: lsof -ti:{MLFLOW_PORT} | xargs kill -9")
        return False
    
    # Đảm bảo thư mục artifact tồn tại
    ensure_artifact_dir()
    
    # Khởi động MLflow server
    try:
        logger.info(f"Khởi động MLflow server tại {MLFLOW_URI}...")
        
        # Sử dụng cấu hình đơn giản
        cmd = [
            "mlflow", "server",
            "--host", MLFLOW_HOST,
            "--port", str(MLFLOW_PORT),
            "--default-artifact-root", ARTIFACT_LOCATION
        ]
        
        # Khởi động MLflow trong process riêng biệt
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Chờ server khởi động nếu cần
        if wait:
            for _ in range(5):
                time.sleep(1)
                if is_mlflow_running():
                    logger.info("MLflow server đã khởi động thành công")
                    return True
            
            # Kiểm tra xem process có lỗi không
            if process.poll() is not None:
                _, stderr = process.communicate()
                logger.error(f"MLflow server bị lỗi: {stderr.decode()}")
                return False
            
            logger.warning("Không thể xác nhận MLflow server đã khởi động sau 5 giây")
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi khởi động MLflow server: {e}")
        return False

def stop_mlflow_server():
    """Dừng MLflow server nếu đang chạy"""
    try:
        import platform
        if platform.system() == "Windows":
            # Phương pháp cho Windows
            subprocess.run(
                f"FOR /F \"tokens=5\" %P IN ('netstat -ano ^| findstr {MLFLOW_PORT}') DO taskkill /F /PID %P",
                shell=True
            )
        else:
            # Phương pháp cho Unix/Linux/MacOS
            subprocess.run(f"lsof -ti:{MLFLOW_PORT} | xargs kill -9", shell=True)
        logger.info("Đã dừng MLflow server")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi dừng MLflow server: {e}")
        return False

def setup_mlflow_tracking(experiment_name="hyperparameter_tuning", auto_start=True):
    """Thiết lập MLflow tracking"""
    try:
        # Kiểm tra và khởi động server nếu cần
        server_running = is_mlflow_running()
        
        if not server_running and auto_start:
            if not start_mlflow_server():
                logger.warning("Không thể khởi động MLflow server, sẽ sử dụng tracking cục bộ")
                return False
            server_running = True
        
        # Thiết lập tracking URI nếu server đang chạy
        if server_running:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            
            # Tạo experiment nếu chưa tồn tại
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"Đã thiết lập MLflow tracking với experiment: {experiment_name}")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập MLflow tracking: {e}")
        return False

# Hàm chính khi chạy file này trực tiếp
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_mlflow_server()
    else:
        # Khởi động server và giữ cho nó chạy
        if start_mlflow_server(wait=False):
            print(f"\nMLflow server đang chạy tại {MLFLOW_URI}")
            print("Nhấn Ctrl+C để dừng")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nDừng MLflow server...")
                stop_mlflow_server()
                print("Đã dừng MLflow server") 