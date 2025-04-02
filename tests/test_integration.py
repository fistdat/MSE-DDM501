"""
Test tích hợp cho toàn bộ luồng hoạt động của ứng dụng
"""

import os
import json
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from app import app


class TestIntegrationFlow(unittest.TestCase):
    """
    Test tích hợp các luồng chính của ứng dụng
    """
    
    def setUp(self):
        """
        Thiết lập cho mỗi test case
        """
        # Cấu hình testing
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SERVER_NAME'] = 'localhost'
        self.client = self.app.test_client()
        
        # Tạo thư mục tạm để lưu kết quả
        self.temp_dir = tempfile.mkdtemp()
        self.original_tuning_dir = 'tuning_results'
        
        # Tạo hoặc bảo toàn thư mục tuning_results
        if not os.path.exists(self.original_tuning_dir):
            os.makedirs(self.original_tuning_dir)
    
    def tearDown(self):
        """
        Dọn dẹp sau mỗi test case
        """
        # Xóa thư mục tạm
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('app.run_tuning_command')
    @patch('app.mlflow')
    def test_full_tuning_workflow(self, mock_mlflow, mock_run_tuning):
        """
        Test toàn bộ quy trình: từ trang chủ -> chạy tuning -> xem kết quả
        """
        # 1. Chuẩn bị mock data
        mock_run_tuning.return_value = (True, "Tuning đã chạy thành công")
        
        result_file_name = "random_forest_small_20250402_123456.json"
        result_data = {
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
            "file_name": result_file_name,
            "timestamp": "02-04-2025 12:34:56",
            "mlflow_run_id": "abc123def456"
        }
        
        # 2. Mock lưu kết quả
        with self.app.test_request_context():
            with patch('os.makedirs'), \
                 patch('json.dump') as mock_json_dump, \
                 patch('builtins.open', unittest.mock.mock_open()) as mock_open, \
                 patch('app.get_tuning_results', return_value=[result_data]):
                
                # 3. Kiểm tra trang chủ ban đầu
                home_response = self.client.get('/')
                self.assertEqual(home_response.status_code, 200)
                
                # 4. Gửi form chạy tuning đơn giản
                form_data = {
                    'model_type': 'random_forest',
                    'param_space': 'small',
                    'samples': '1000',
                    'features': '20',
                    'cv': '5'
                }
                
                tuning_response = self.client.post('/run_simple_tuning', 
                                                 data=form_data, 
                                                 follow_redirects=True)
                
                # Kiểm tra chuyển hướng về trang chủ sau khi tuning
                self.assertEqual(tuning_response.status_code, 200)
                
                # Kiểm tra thông báo thành công
                self.assertIn(b'Tuning th\xc3\xa0nh c\xc3\xb4ng', tuning_response.data)
                
                # 5. Mô phỏng việc mở file kết quả
                with patch('os.path.exists', return_value=True), \
                     patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(result_data))):
                     
                    # Xem chi tiết kết quả
                    detail_response = self.client.get(f'/view_result/{result_file_name}')
                    
                    # Kiểm tra hiển thị chi tiết kết quả
                    self.assertEqual(detail_response.status_code, 200)
                    
                    # Kiểm tra nội dung chi tiết
                    self.assertIn(b'Random Forest', detail_response.data)
                    self.assertIn(b'Tham s\xe1\xbb\x91 t\xe1\xbb\x91t nh\xe1\xba\xa5t', detail_response.data)

    @patch('app.datetime')
    @patch('app.run_tuning_command')
    @patch('app.mlflow')
    def test_custom_tuning_flow(self, mock_mlflow, mock_run_tuning, mock_datetime):
        """
        Test quy trình tuning tùy chỉnh
        """
        # 1. Setup mocks
        mock_run_tuning.return_value = (True, "Tuning tùy chỉnh thành công")
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20250402_123600"
        mock_datetime.now.return_value = mock_now
        
        result_file_name = "random_forest_custom_20250402_123600.json"
        result_data = {
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
            "file_name": result_file_name,
            "timestamp": "02-04-2025 12:36:00"
        }
        
        # 2. Test quy trình
        with self.app.test_request_context():
            with patch('os.makedirs'), \
                 patch('json.dump') as mock_json_dump, \
                 patch('builtins.open', unittest.mock.mock_open()) as mock_open, \
                 patch('app.get_tuning_results', return_value=[result_data]):
                
                # 3. Gửi form chạy tuning tùy chỉnh
                form_data = {
                    'model_type': 'random_forest',
                    'rf_n_estimators': '100, 150, 200',
                    'rf_max_depth': '10, 15',
                    'rf_min_samples_split': '5',
                    'samples': '1000',
                    'features': '20',
                    'cv': '5'
                }
                
                tuning_response = self.client.post('/run_custom_tuning', 
                                                  data=form_data, 
                                                  follow_redirects=True)
                
                # Kiểm tra chuyển hướng và thông báo thành công
                self.assertEqual(tuning_response.status_code, 200)
                self.assertIn(b'Tuning t\xc3\xb9y ch\xe1\xbb\x89nh th\xc3\xa0nh c\xc3\xb4ng', tuning_response.data)
                
                # 4. Kiểm tra gọi hàm tạo file tham số
                mock_open.assert_called()
                
                # 5. Kiểm tra chạy lệnh tuning
                mock_run_tuning.assert_called_once()
                
                # 6. Mô phỏng việc mở file kết quả
                with patch('os.path.exists', return_value=True), \
                     patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(result_data))):
                    
                    # Xem chi tiết kết quả
                    detail_response = self.client.get(f'/view_result/{result_file_name}')
                    
                    # Kiểm tra hiển thị chi tiết kết quả
                    self.assertEqual(detail_response.status_code, 200)
                    self.assertIn(b'Random Forest', detail_response.data)
                    self.assertIn(b'custom', detail_response.data)
                    self.assertIn(b'0.875', detail_response.data)  # F1 score

    @patch('app.run_tuning_command')
    def test_error_handling(self, mock_run_tuning):
        """
        Test xử lý lỗi trong quá trình tuning
        """
        # Mock lỗi khi chạy tuning
        mock_run_tuning.return_value = (False, "Không thể tạo mô hình: dữ liệu không hợp lệ")
        
        # Gửi form tuning đơn giản
        form_data = {
            'model_type': 'random_forest',
            'param_space': 'small',
            'samples': '1000',
            'features': '20',
            'cv': '5'
        }
        
        # Gửi request và kiểm tra phản hồi
        with self.app.test_request_context():
            response = self.client.post('/run_simple_tuning', 
                                       data=form_data, 
                                       follow_redirects=True)
            
            # Kiểm tra status code và thông báo lỗi
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Tuning th\xe1\xba\xa5t b\xe1\xba\xa1i', response.data)
    
    def test_nonexistent_result_handling(self):
        """
        Test xử lý khi yêu cầu kết quả không tồn tại
        """
        # Mô phỏng file không tồn tại
        with patch('os.path.exists', return_value=False):
            response = self.client.get('/view_result/nonexistent.json', follow_redirects=True)
            
            # Kiểm tra chuyển hướng về trang chủ với thông báo lỗi
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'File k\xe1\xba\xbft qu\xe1\xba\xa3 kh\xc3\xb4ng t\xe1\xbb\x93n t\xe1\xba\xa1i', response.data)
    
    def test_health_endpoint(self):
        """
        Test endpoint health API
        """
        response = self.client.get('/health')
        
        # Kiểm tra response JSON
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')


if __name__ == '__main__':
    unittest.main() 