"""
Test các chức năng Frontend của ứng dụng Flask
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
from datetime import datetime

from flask import template_rendered, session, get_flashed_messages
from contextlib import contextmanager

# Đảm bảo có thể import các module từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mock_app import app
from tests.test_helpers import captured_templates, check_flash_in_response


class TestAppFrontend(unittest.TestCase):
    """
    Test các chức năng Frontend của ứng dụng
    """
    
    def setUp(self):
        """
        Thiết lập cho mỗi test case
        """
        self.app = app.test_client()
        self.app.testing = True
        
        # Đảm bảo thư mục cần thiết tồn tại
        os.makedirs('tuning_results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Tạo file test_result.json nếu chưa tồn tại
        if not os.path.exists('tuning_results/test_result.json'):
            with open('tuning_results/test_result.json', 'w') as f:
                json.dump({
                    "model_type": "random_forest",
                    "timestamp": "2023-11-15T10:30:45",
                    "parameters": {},
                    "cv_results": {},
                    "best_score": 0.85,
                    "metrics": {
                        "accuracy": 0.88,
                        "f1_score": 0.85
                    }
                }, f)
        
        # Tạo dữ liệu mẫu
        self.create_sample_results()
    
    def tearDown(self):
        """
        Dọn dẹp sau mỗi test case
        """
        pass
    
    def create_sample_results(self):
        """
        Tạo dữ liệu mẫu cho test
        """
        self.sample_results = []
        
        # Mẫu kết quả Random Forest
        rf_result = {
            "model_type": "random_forest",
            "param_space": "small",
            "n_samples_train": 800,
            "n_samples_test": 200,
            "n_features": 20,
            "cv": 5,
            "best_params": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "best_cv_f1": 0.85,
            "metrics": {
                "accuracy": 0.88,
                "precision": 0.87,
                "recall": 0.84,
                "f1": 0.855
            },
            "tuning_time_seconds": 15.5,
            "total_combinations": 6,
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10]
            },
            "file_name": "random_forest_small_20250402_123456.json",
            "timestamp": "02-04-2025 12:34:56",
            "mlflow_run_id": "abc123def456"
        }
        
        # Mẫu kết quả tùy chỉnh
        custom_result = {
            "model_type": "random_forest",
            "param_space": "custom",
            "n_samples_train": 800,
            "n_samples_test": 200,
            "n_features": 20,
            "cv": 5,
            "best_params": {
                "n_estimators": 150,
                "max_depth": 15,
                "min_samples_split": 5
            },
            "best_cv_f1": 0.87,
            "metrics": {
                "accuracy": 0.90,
                "precision": 0.89,
                "recall": 0.86,
                "f1": 0.875
            },
            "tuning_time_seconds": 18.7,
            "total_combinations": 6,
            "param_grid": {
                "n_estimators": [100, 150, 200],
                "max_depth": [10, 15],
                "min_samples_split": [5]
            },
            "file_name": "random_forest_custom_20250402_123600.json",
            "timestamp": "02-04-2025 12:36:00"
        }
        
        self.sample_results = [rf_result, custom_result]  # Sắp xếp theo thời gian gần nhất
        
        # Lưu mẫu kết quả vào thư mục test
        for result in self.sample_results:
            file_path = os.path.join('tuning_results', result["file_name"])
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2)
    
    def test_home_page_render(self):
        """
        Test trang chủ hiển thị đúng
        """
        with captured_templates(app) as templates:
            response = self.app.get('/')
            
            # Kiểm tra status code
            self.assertEqual(response.status_code, 200)
            
            # Kiểm tra template đã render
            self.assertEqual(len(templates), 1)
            template, context = templates[0]
            self.assertEqual(template.name, 'index.html')
    
    def test_simple_tuning_form_submit(self):
        """
        Test gửi form tuning đơn giản
        """
        response = self.app.post('/run_simple_tuning', data={
            'model_type': 'random_forest',
            'param_space': 'small',
            'samples': '1000',
            'features': '20',
            'cv': '5'
        }, follow_redirects=True)
        
        # Kiểm tra status code và redirect
        self.assertEqual(response.status_code, 200)
        
        # Kiểm tra flash message trong HTML response
        self.assertTrue(check_flash_in_response(response.data, 'success'),
                       "Không tìm thấy flash message 'success' trong response")
    
    def test_custom_tuning_form_submit(self):
        """
        Test gửi form tuning tùy chỉnh
        """
        form_data = {
            'model_type': 'gradient_boosting',
            'n_estimators': '50,100,200',
            'learning_rate': '0.01,0.1',
            'max_depth': '3,5',
            'samples': '1500',
            'features': '25',
            'cv': '3'
        }
        
        response = self.app.post('/run_custom_tuning', data=form_data, follow_redirects=True)
        
        # Kiểm tra status code và redirect
        self.assertEqual(response.status_code, 200)
        
        # Kiểm tra flash message trong HTML response
        self.assertTrue(check_flash_in_response(response.data, 'success'),
                       "Không tìm thấy flash message 'success' trong response")
    
    def test_view_result_detail(self):
        """
        Test xem chi tiết kết quả tuning
        """
        with captured_templates(app) as templates:
            response = self.app.get('/view_result/test_result.json')
            
            # Chấp nhận cả status code 200 và 302 (redirect)
            self.assertIn(response.status_code, [200, 302])
            
            if response.status_code == 200 and templates:
                # Nếu status code là 200, kiểm tra template và context
                self.assertEqual(len(templates), 1)
                template, context = templates[0]
                self.assertEqual(template.name, 'result_detail.html')
                self.assertIn('result', context)
    
    def test_view_nonexistent_result(self):
        """
        Test xem kết quả không tồn tại
        """
        response = self.app.get('/view_result/nonexistent.json', follow_redirects=True)
        
        # Kiểm tra status code và redirect
        self.assertEqual(response.status_code, 200)
        
        # Kiểm tra flash message trong HTML response
        self.assertTrue(check_flash_in_response(response.data, 'danger'),
                       "Không tìm thấy flash message 'danger' trong response")
    
    def test_failed_simple_tuning(self):
        """
        Test kịch bản tuning thất bại
        """
        form_data = {
            'model_type': 'fail_model',  # Trigger lỗi trong mock
            'param_space': 'fail',
            'samples': '1000',
            'features': '20',
            'cv': '5'
        }
        
        response = self.app.post('/run_simple_tuning', data=form_data, follow_redirects=True)
        
        # Kiểm tra status code và redirect
        self.assertEqual(response.status_code, 200)
        
        # Kiểm tra flash message trong HTML response
        self.assertTrue(check_flash_in_response(response.data, 'danger'),
                       "Không tìm thấy flash message 'danger' trong response")
    
    def test_api_health_check(self):
        """
        Test API health check
        """
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')


if __name__ == '__main__':
    unittest.main() 