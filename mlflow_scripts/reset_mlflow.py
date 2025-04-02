"""
Script để reset và khởi động lại MLflow từ đầu
"""

import os
import time
import logging
import subprocess
from mlflow_config import (
    clear_existing_data, 
    setup_mlflow, 
    create_experiment_if_not_exists,
    DEFAULT_EXPERIMENT_NAME,
    TRACKING_URI
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reset_and_init_mlflow():
    """
    Reset và khởi tạo lại MLflow từ đầu
    """
    logger.info("Bắt đầu reset MLflow...")
    
    # Đóng tất cả kết nối tới MLflow
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    # Dừng MLflow server nếu đang chạy
    try:
        subprocess.run("lsof -ti:5002 | xargs kill -9", shell=True)
        logger.info("Đã dừng MLflow server nếu đang chạy")
        time.sleep(1)  # Đợi server dừng hoàn toàn
    except:
        pass
    
    # Xóa dữ liệu cũ
    if clear_existing_data():
        logger.info("Đã xóa dữ liệu MLflow cũ")
    else:
        logger.error("Không thể xóa dữ liệu MLflow cũ")
        return False
    
    # Thiết lập MLflow mới
    if setup_mlflow(reset=True):
        logger.info("Đã thiết lập MLflow mới")
    else:
        logger.error("Không thể thiết lập MLflow mới")
        return False
    
    # Chờ một chút để MLflow tạo database
    logger.info("Đợi MLflow khởi tạo database...")
    time.sleep(3)
    
    # Tạo experiment mặc định
    import mlflow
    from mlflow.tracking import MlflowClient
    
    for attempt in range(3):  # Thử tối đa 3 lần
        try:
            # Đảm bảo tracking URI được thiết lập
            mlflow.set_tracking_uri(TRACKING_URI)
            
            # Thử truy cập client trước - thay đổi cách kiểm tra kết nối
            client = MlflowClient()
            # Thay vì gọi list_experiments, kiểm tra kết nối bằng cách lấy experiments
            # thông qua search_experiments của mlflow API
            mlflow.search_experiments()
            logger.info("Kết nối MLflow client thành công")
            
            # Tạo experiment
            experiment_id = create_experiment_if_not_exists(DEFAULT_EXPERIMENT_NAME)
            if experiment_id:
                logger.info(f"Đã tạo experiment '{DEFAULT_EXPERIMENT_NAME}' với ID: {experiment_id}")
                break
            else:
                logger.warning(f"Lần thử {attempt+1}: Không thể tạo experiment, thử lại sau 2 giây...")
                time.sleep(2)
        except Exception as e:
            logger.warning(f"Lần thử {attempt+1}: Lỗi khi tạo experiment: {str(e)}")
            time.sleep(2)
    else:  # Nếu vòng lặp kết thúc mà không break
        logger.error(f"Không thể tạo experiment '{DEFAULT_EXPERIMENT_NAME}' sau 3 lần thử")
        return False
    
    # Kiểm tra lại để đảm bảo experiment đã được tạo thành công
    try:
        experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
        if experiment:
            logger.info(f"Xác nhận: Experiment '{DEFAULT_EXPERIMENT_NAME}' đã tồn tại với ID: {experiment.experiment_id}")
            logger.info("Reset MLflow thành công!")
            return True
        else:
            logger.error(f"Xác nhận: Experiment '{DEFAULT_EXPERIMENT_NAME}' vẫn chưa được tạo")
            return False
    except Exception as e:
        logger.error(f"Lỗi khi xác nhận experiment: {str(e)}")
        return False

if __name__ == "__main__":
    if reset_and_init_mlflow():
        logger.info("Bây giờ bạn có thể khởi động MLflow server: 'make mlflow'")
        logger.info("Sau đó chạy thí nghiệm: 'make simple-tuning'")
        logger.info("Và cuối cùng lưu mô hình tốt nhất: 'make save-best-model'")
    else:
        logger.error("Reset MLflow thất bại!") 