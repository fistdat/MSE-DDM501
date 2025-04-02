"""
Script để quản lý MLflow experiments (liệt kê, khôi phục, xóa vĩnh viễn)
"""

import mlflow
import argparse
import sys
import logging
from mlflow.tracking import MlflowClient

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình MLflow
MLFLOW_URI = "http://localhost:5002"

def connect_mlflow():
    """Kết nối tới MLflow server"""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        return True
    except Exception as e:
        logger.error(f"Không thể kết nối tới MLflow server: {e}")
        return False

def list_experiments(show_deleted=False):
    """Liệt kê các experiments (bao gồm cả đã xóa nếu chỉ định)"""
    if not connect_mlflow():
        return
        
    try:
        # Liệt kê tất cả experiments, bao gồm cả đã xóa nếu chỉ định
        client = MlflowClient()
        
        if show_deleted:
            # Lấy tất cả experiments bao gồm cả đã xóa
            experiments = client.list_experiments()
        else:
            # Chỉ lấy experiments đang hoạt động
            experiments = [exp for exp in client.list_experiments() 
                          if exp.lifecycle_stage == "active"]
        
        if not experiments:
            print("Không tìm thấy experiments nào.")
            return
            
        print("\n" + "=" * 80)
        print(f"DANH SÁCH EXPERIMENTS {'(BAO GỒM ĐÃ XÓA)' if show_deleted else ''}:")
        print("=" * 80)
        
        for exp in experiments:
            status = "DELETED" if exp.lifecycle_stage == "deleted" else "ACTIVE"
            print(f"ID: {exp.experiment_id}, Tên: {exp.name}, Trạng thái: {status}")

        print("=" * 80)
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê experiments: {e}")

def restore_experiment(experiment_name=None, experiment_id=None):
    """Khôi phục experiment đã xóa"""
    if not connect_mlflow():
        return
        
    try:
        # Tìm experiment cần khôi phục
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                print(f"Không tìm thấy experiment có tên '{experiment_name}'.")
                return
            experiment_id = experiment.experiment_id
        
        if not experiment_id:
            print("Cần cung cấp tên hoặc ID của experiment cần khôi phục.")
            return
            
        # Khôi phục experiment
        client = MlflowClient()
        client.restore_experiment(experiment_id)
        print(f"Đã khôi phục experiment có ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Lỗi khi khôi phục experiment: {e}")

def delete_experiment(experiment_name=None, experiment_id=None, permanently=False):
    """Xóa experiment (tạm thời hoặc vĩnh viễn)"""
    if not connect_mlflow():
        return
        
    try:
        # Tìm experiment cần xóa
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                print(f"Không tìm thấy experiment có tên '{experiment_name}'.")
                return
            experiment_id = experiment.experiment_id
        
        if not experiment_id:
            print("Cần cung cấp tên hoặc ID của experiment cần xóa.")
            return
            
        client = MlflowClient()
        
        if permanently:
            # Yêu cầu xác nhận trước khi xóa vĩnh viễn
            confirm = input(f"XÁC NHẬN xóa VĨNH VIỄN experiment ID {experiment_id}? (yes/no): ")
            if confirm.lower() != "yes":
                print("Đã hủy xóa vĩnh viễn.")
                return
                
            # Xóa vĩnh viễn experiment - MLflow API không hỗ trợ xóa vĩnh viễn
            # Đây là thao tác gọi internal API
            try:
                client._tracking_client._delete_experiment(experiment_id)
                print(f"Đã xóa VĨNH VIỄN experiment có ID: {experiment_id}")
            except Exception as e:
                logger.error(f"Lỗi khi xóa vĩnh viễn experiment: {e}")
                print("Không thể xóa vĩnh viễn experiment. Thử dùng xóa tạm thời thay thế.")
        else:
            # Xóa tạm thời experiment 
            client.delete_experiment(experiment_id)
            print(f"Đã xóa (đánh dấu đã xóa) experiment có ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Lỗi khi xóa experiment: {e}")

def parse_args():
    """Xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Quản lý MLflow experiments")
    parser.add_argument("--list", action="store_true", 
                        help="Liệt kê tất cả experiments")
    parser.add_argument("--list-all", action="store_true", 
                        help="Liệt kê tất cả experiments, bao gồm cả đã xóa")
    parser.add_argument("--restore-id", type=str, 
                        help="Khôi phục experiment theo ID")
    parser.add_argument("--restore-name", type=str, 
                        help="Khôi phục experiment theo tên")
    parser.add_argument("--delete-id", type=str, 
                        help="Xóa experiment theo ID")
    parser.add_argument("--delete-name", type=str, 
                        help="Xóa experiment theo tên")
    parser.add_argument("--permanent", action="store_true", 
                        help="Xóa vĩnh viễn thay vì đánh dấu đã xóa")
    
    return parser.parse_args()

def main():
    """Hàm chính"""
    args = parse_args()
    
    if args.list:
        list_experiments(show_deleted=False)
    elif args.list_all:
        list_experiments(show_deleted=True)
    elif args.restore_id:
        restore_experiment(experiment_id=args.restore_id)
    elif args.restore_name:
        restore_experiment(experiment_name=args.restore_name)
    elif args.delete_id:
        delete_experiment(experiment_id=args.delete_id, permanently=args.permanent)
    elif args.delete_name:
        delete_experiment(experiment_name=args.delete_name, permanently=args.permanent)
    else:
        print("Hãy chỉ định một hành động. Xem --help để biết thêm chi tiết.")
        list_experiments(show_deleted=True)

if __name__ == "__main__":
    main() 