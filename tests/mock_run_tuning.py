"""
Mock cho hàm run_tuning_command trong app.py để sử dụng trong testing
"""

import os
import json
import time
from datetime import datetime

def mock_run_tuning_command(command, params=None):
    """
    Mock cho hàm run_tuning_command, trả về dữ liệu giả lập
    
    Args:
        command: Lệnh cần thực hiện
        params: Các tham số
        
    Returns:
        Tuple (success, output, result_file)
    """
    # Tạo thư mục tuning_results nếu chưa tồn tại
    os.makedirs('tuning_results', exist_ok=True)
    
    # Thời gian hiện tại theo định dạng YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Thời gian unix để tạo tên file duy nhất
    unix_time = int(time.time())
    
    if params is None:
        params = {}
    
    # Xác định loại mô hình và không gian tham số
    model_type = params.get('model_type', 'random_forest')
    param_space = params.get('param_space', 'small')
    
    # Tạo tên file kết quả
    result_file = f"{model_type}_{param_space}_{timestamp}.json"
    result_path = os.path.join('tuning_results', result_file)
    
    # Tạo dữ liệu kết quả mẫu
    result_data = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "param_space": param_space,
            "model_type": model_type,
            "n_features": params.get('features', 20),
            "n_samples": params.get('samples', 1000),
            "cv": params.get('cv', 5)
        },
        "cv_results": {
            "mean_fit_time": [0.25],
            "mean_score_time": [0.1],
            "mean_test_score": [0.85],
            "std_test_score": [0.05],
            "split0_test_score": [0.83],
            "split1_test_score": [0.86],
            "split2_test_score": [0.87]
        },
        "best_score": 0.85,
        "metrics": {
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.86,
            "f1_score": 0.85
        }
    }
    
    # Lưu kết quả vào file
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    
    # Mô phỏng lỗi trong một số trường hợp
    if 'fail' in command:
        return False, "Lỗi khi chạy tuning. Xem chi tiết trong log.", None
    
    # Trả về kết quả mock
    return True, "Tuning thành công!", result_file 