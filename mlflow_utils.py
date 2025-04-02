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
from mlflow_config import (
    TRACKING_URI, 
    ARTIFACTS_PATH, 
    MLFLOW_PORT, 
    setup_mlflow, 
    DEFAULT_EXPERIMENT_NAME,
    create_experiment_if_not_exists
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình MLflow
MLFLOW_HOST = "0.0.0.0"
MLFLOW_UI_URI = f"http://localhost:{MLFLOW_PORT}"

def is_port_available(port, host="0.0.0.0"):
    """Kiểm tra xem cổng có khả dụng không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def is_mlflow_running(uri=MLFLOW_UI_URI, max_retries=3):
    """Kiểm tra xem MLflow server đã chạy chưa"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{uri}/api/2.0/mlflow/experiments/list")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(2)  # Tăng thời gian chờ giữa các lần thử
    return False

def ensure_directories():
    """Đảm bảo các thư mục cần thiết tồn tại"""
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

def start_mlflow_server(wait=True, reset=False, timeout=30):
    """
    Khởi động MLflow server
    
    Args:
        wait (bool): Chờ server khởi động nếu True
        reset (bool): Xóa dữ liệu cũ nếu True
        timeout (int): Thời gian tối đa chờ server khởi động (giây)
    """
    # Kiểm tra xem MLflow đã chạy chưa
    if is_mlflow_running():
        logger.info("MLflow server đã chạy")
        return True
    
    # Kiểm tra cổng
    if not is_port_available(MLFLOW_PORT):
        logger.warning(f"Cổng {MLFLOW_PORT} đang bị sử dụng, không thể khởi động MLflow server")
        logger.info(f"Để giải phóng cổng, chạy: lsof -ti:{MLFLOW_PORT} | xargs kill -9")
        return False
    
    # Thiết lập MLflow với SQLite backend
    if not setup_mlflow(reset=reset):
        logger.error("Không thể thiết lập MLflow, không khởi động server")
        return False
    
    # Đảm bảo thư mục cần thiết tồn tại
    ensure_directories()
    
    # Tạo experiment mặc định nếu chưa tồn tại
    logger.info("Đợi 2 giây để đảm bảo database đã khởi tạo...")
    time.sleep(2)
    
    # Thử tạo experiment trước khi khởi động server
    for attempt in range(2):  # Thử 2 lần
        try:
            experiment_id = create_experiment_if_not_exists()
            if experiment_id:
                logger.info(f"Đã tạo/kiểm tra experiment với ID: {experiment_id}")
                break
        except Exception as e:
            logger.warning(f"Lần thử {attempt+1}: Không thể tạo experiment: {str(e)}")
            time.sleep(1)
    
    # Khởi động MLflow server
    try:
        logger.info(f"Khởi động MLflow server tại {MLFLOW_UI_URI} với backend {TRACKING_URI}...")
        
        # Sử dụng cấu hình với SQLite backend
        cmd = [
            "mlflow", "server",
            "--host", MLFLOW_HOST,
            "--port", str(MLFLOW_PORT),
            "--backend-store-uri", TRACKING_URI,
            "--default-artifact-root", ARTIFACTS_PATH
        ]
        
        # Khởi động MLflow server trực tiếp thông qua subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Nếu không cần đợi, trả về ngay
        if not wait:
            logger.info(f"MLflow server đã được khởi động trong background với PID: {process.pid}")
            return True
        
        # Khởi động trong process này nếu wait=True
        logger.info("MLflow server đang khởi động...")
        start_time = time.time()
        waiting_dots = 1
        
        # Đợi server khởi động
        while time.time() - start_time < timeout:
            if is_mlflow_running():
                logger.info("MLflow server đã khởi động thành công")
                return True
            
            # Hiển thị trạng thái đợi
            if waiting_dots <= 5:
                sys.stdout.write(f"\rĐang đợi server khởi động{'.' * waiting_dots}{' ' * (5-waiting_dots)}")
                sys.stdout.flush()
                waiting_dots = (waiting_dots % 5) + 1
            
            # Kiểm tra xem process có lỗi không
            if process.poll() is not None:
                _, stderr = process.communicate()
                logger.error(f"MLflow server bị lỗi: {stderr.decode()}")
                return False
                
            time.sleep(2)
        
        logger.warning(f"Không thể xác nhận MLflow server đã khởi động sau {timeout} giây")
        return False
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

def setup_mlflow_tracking(experiment_name=DEFAULT_EXPERIMENT_NAME, auto_start=True, reset=False):
    """
    Thiết lập MLflow tracking
    
    Args:
        experiment_name (str): Tên của experiment
        auto_start (bool): Tự động khởi động server nếu True
        reset (bool): Xóa dữ liệu cũ nếu True
    """
    try:
        # Thiết lập cấu hình SQLite
        if not setup_mlflow(reset=reset):
            logger.error("Không thể thiết lập MLflow")
            return False
        
        # Kiểm tra và khởi động server nếu cần
        server_running = is_mlflow_running()
        
        if not server_running and auto_start:
            if not start_mlflow_server(reset=reset):
                logger.warning("Không thể khởi động MLflow server, chỉ sử dụng tracking cục bộ")
                server_running = False
            else:
                server_running = True
        
        # Tạo experiment nếu chưa tồn tại và thiết lập nó
        if server_running:
            # Chờ một chút để đảm bảo server đã sẵn sàng
            time.sleep(1)
            
            experiment_id = create_experiment_if_not_exists(experiment_name)
            if not experiment_id:
                logger.error(f"Không thể tạo/thiết lập experiment {experiment_name}")
                return False
        
        logger.info(f"Đã thiết lập MLflow tracking với experiment: {experiment_name}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập MLflow tracking: {str(e)}")
        return False

# Hàm chính khi chạy file này trực tiếp
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop":
            stop_mlflow_server()
        elif sys.argv[1] == "reset":
            # Dừng server nếu đang chạy
            stop_mlflow_server()
            time.sleep(2)  # Đợi một chút để đảm bảo server đã dừng
            # Khởi động lại với reset data
            if start_mlflow_server(wait=True, reset=True):
                print(f"\nMLflow server đã khởi động lại với dữ liệu mới tại {MLFLOW_UI_URI}")
                print("Nhấn Ctrl+C để dừng")
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nDừng MLflow server...")
                    stop_mlflow_server()
                    print("Đã dừng MLflow server")
    else:
        # Khởi động server và giữ cho nó chạy
        if start_mlflow_server(wait=True):
            print(f"\nMLflow server đang chạy tại {MLFLOW_UI_URI}")
            print("Nhấn Ctrl+C để dừng")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nDừng MLflow server...")
                stop_mlflow_server()
                print("Đã dừng MLflow server") 