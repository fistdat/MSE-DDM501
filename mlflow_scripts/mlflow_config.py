"""
Cấu hình MLflow với backend database để hỗ trợ Model Registry
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
import logging
import shutil
import sqlalchemy
import time

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thư mục chứa dữ liệu MLflow
MLFLOW_DIR = os.path.join(os.getcwd(), "mlflow_data")
ARTIFACTS_PATH = os.path.join(MLFLOW_DIR, "artifacts")
DB_PATH = os.path.join(MLFLOW_DIR, "mlflow.db")

# URI cho backend storage (SQLite)
# Kiểm tra xem có đang chạy trong Docker không
if os.environ.get('MLFLOW_TRACKING_URI'):
    TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
else:
    TRACKING_URI = f"sqlite:///{DB_PATH}"

# Port cho MLflow UI
MLFLOW_PORT = 5002

# Tên experiment mặc định
DEFAULT_EXPERIMENT_NAME = "tuning_experiment"

def clear_existing_data():
    """
    Xóa tất cả dữ liệu MLflow hiện có để tránh xung đột
    """
    try:
        # Xóa directory cũ mlruns nếu còn tồn tại (từ cấu hình cũ)
        if os.path.exists("mlruns"):
            shutil.rmtree("mlruns")
            logger.info("Đã xóa thư mục mlruns cũ")
            
        # Xóa thư mục MLflow data hiện tại
        if os.path.exists(MLFLOW_DIR):
            shutil.rmtree(MLFLOW_DIR)
            logger.info(f"Đã xóa thư mục dữ liệu MLflow: {MLFLOW_DIR}")
            
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xóa dữ liệu MLflow: {str(e)}")
        return False

def check_sqlite_connection():
    """
    Kiểm tra kết nối tới SQLite database
    """
    try:
        # Kiểm tra thư mục tồn tại
        db_dir = os.path.dirname(DB_PATH)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        # Tạo connection string
        conn_str = TRACKING_URI
        
        # Chỉ tạo engine nhưng không kết nối
        engine = sqlalchemy.create_engine(conn_str, echo=False)
        
        # Thử execute một câu query đơn giản
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text("SELECT 1"))
            
        return True
    except Exception as e:
        logger.error(f"Lỗi khi kết nối tới SQLite database: {str(e)}")
        return False

def setup_mlflow(reset=False):
    """
    Thiết lập MLflow với backend database SQLite
    để hỗ trợ Model Registry
    
    Args:
        reset (bool): Xóa dữ liệu cũ nếu True
    """
    try:
        # Xóa dữ liệu cũ nếu yêu cầu
        if reset:
            clear_existing_data()
            
        # Tạo thư mục cho dữ liệu MLflow nếu chưa tồn tại
        os.makedirs(MLFLOW_DIR, exist_ok=True)
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)
        
        # Thiết lập tracking URI
        mlflow.set_tracking_uri(TRACKING_URI)
        logger.info(f"Đã thiết lập MLflow tracking URI: {TRACKING_URI}")
        
        # Thử kết nối với MLflow server
        try:
            client = MlflowClient()
            client.search_experiments()  # Kiểm tra xem có kết nối được không
            logger.info("Kết nối thành công tới MLflow server")
        except Exception as e:
            logger.warning(f"Cảnh báo: Không thể kết nối với MLflow server: {str(e)}")
            logger.warning("Sẽ tiếp tục thực hiện nhưng có thể gặp lỗi sau này")
        
        # Tạo experiment mặc định nếu chưa tồn tại
        create_experiment_if_not_exists()
        
        # Chờ một chút để MLflow khởi tạo database (nếu cần)
        time.sleep(1)
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập MLflow: {str(e)}")
        return False

def get_tracking_uri():
    """Trả về tracking URI đã cấu hình"""
    return TRACKING_URI

def get_artifacts_path():
    """Trả về đường dẫn lưu artifacts"""
    return ARTIFACTS_PATH

def create_experiment_if_not_exists(experiment_name=DEFAULT_EXPERIMENT_NAME):
    """
    Tạo experiment nếu chưa tồn tại
    
    Args:
        experiment_name (str): Tên experiment cần tạo
    
    Returns:
        str: ID của experiment
    """
    try:
        # Thiết lập tracking URI trước khi gọi API
        mlflow.set_tracking_uri(TRACKING_URI)
        
        # Thử lấy danh sách experiment để kiểm tra kết nối
        try:
            # Thay vì dùng MlflowClient.list_experiments, sử dụng mlflow.search_experiments
            mlflow.search_experiments()
            logger.info("Đã kết nối thành công tới MLflow API")
        except Exception as e:
            logger.error(f"Lỗi khi truy cập MLflow API: {str(e)}")
            return None
            
        # Kiểm tra xem experiment đã tồn tại chưa
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Tạo experiment mới
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Đã tạo experiment mới: {experiment_name} với ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Experiment {experiment_name} đã tồn tại với ID: {experiment_id}")
            
        # Thiết lập experiment hiện tại
        mlflow.set_experiment(experiment_name)
        
        return experiment_id
    except Exception as e:
        logger.error(f"Lỗi khi tạo/kiểm tra experiment: {str(e)}")
        return None

if __name__ == "__main__":
    # Reset và thiết lập lại MLflow
    if setup_mlflow(reset=True):
        logger.info(f"MLflow đã được cấu hình với tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Artifacts sẽ được lưu tại: {ARTIFACTS_PATH}")
        
        # Chờ một chút để MLflow tạo database
        time.sleep(2)
        
        # Tạo experiment mặc định
        experiment_id = create_experiment_if_not_exists()
        if experiment_id:
            logger.info(f"Experiment ID: {experiment_id}")
        else:
            logger.error("Không thể tạo experiment mặc định")
    else:
        logger.error("Không thể thiết lập MLflow") 