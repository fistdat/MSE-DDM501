"""
Test sự khả năng chịu lỗi của templates đối với dữ liệu không đầy đủ
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from flask import template_rendered, render_template
from contextlib import contextmanager

from app import app


@contextmanager
def captured_templates(app):
    """Context manager để bắt các template đã render"""
    recorded = []
    def record(sender, template, context, **extra):
        recorded.append((template, context))
    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)


class TestTemplateValidation(unittest.TestCase):
    """Test templates với dữ liệu không đầy đủ"""
    
    def setUp(self):
        """Thiết lập cho mỗi test case"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Tạo các kiểu dữ liệu thiếu khác nhau để test
        self.create_test_data()
    
    def create_test_data(self):
        """Tạo các dạng dữ liệu thiếu khác nhau để test"""
        # Kết quả không có n_samples_train
        self.missing_train_samples = {
            "model_type": "random_forest",
            "param_space": "small",
            # Thiếu n_samples_train
            "n_samples_test": 200,
            "n_features": 20,
            "cv": 5,
            "metrics": {
                "accuracy": 0.88,
                "precision": 0.87,
                "recall": 0.84,
                "f1": 0.855
            },
            "file_name": "missing_train.json",
            "timestamp": "02-04-2025 12:34:56"
        }
        
        # Kết quả không có metrics
        self.missing_metrics = {
            "model_type": "random_forest",
            "param_space": "small",
            "n_samples_train": 800,
            "n_samples_test": 200,
            "n_features": 20,
            "cv": 5,
            # Thiếu metrics
            "file_name": "missing_metrics.json",
            "timestamp": "02-04-2025 12:35:00"
        }
        
        # Kết quả chỉ có thông tin tối thiểu
        self.minimal_result = {
            "model_type": "random_forest",
            "param_space": "tiny",
            "file_name": "minimal.json",
            "timestamp": "02-04-2025 12:36:00"
        }
        
        # Kết quả với metrics thiếu một số trường
        self.partial_metrics = {
            "model_type": "gradient_boosting",
            "param_space": "medium", 
            "n_samples_train": 1000,
            "n_samples_test": 250,
            "metrics": {
                "accuracy": 0.90,
                # Thiếu precision
                "recall": 0.89,
                "f1": 0.895
            },
            "file_name": "partial_metrics.json",
            "timestamp": "02-04-2025 12:37:00"
        }
        
        # Tạo danh sách tất cả các kết quả test
        self.all_test_results = [
            self.missing_train_samples,
            self.missing_metrics,
            self.minimal_result,
            self.partial_metrics
        ]
    
    def test_index_template_with_missing_data(self):
        """Test template index.html với dữ liệu không đầy đủ"""
        with patch('app.get_tuning_results', return_value=self.all_test_results):
            # Kết quả không nên gây ra lỗi
            response = self.client.get('/')
            self.assertEqual(response.status_code, 200)
    
    def test_result_detail_with_missing_train_samples(self):
        """Test template result_detail.html với kết quả thiếu dữ liệu huấn luyện"""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.missing_train_samples))):
            response = self.client.get('/view_result/missing_train.json')
            self.assertEqual(response.status_code, 200)
    
    def test_result_detail_with_missing_metrics(self):
        """Test template result_detail.html với kết quả thiếu metrics"""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.missing_metrics))):
            response = self.client.get('/view_result/missing_metrics.json')
            self.assertEqual(response.status_code, 200)
    
    def test_result_detail_with_minimal_result(self):
        """Test template result_detail.html với kết quả có thông tin tối thiểu"""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.minimal_result))):
            response = self.client.get('/view_result/minimal.json')
            self.assertEqual(response.status_code, 200)
    
    def test_result_detail_with_partial_metrics(self):
        """Test template result_detail.html với kết quả có metrics thiếu field"""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.partial_metrics))):
            response = self.client.get('/view_result/partial_metrics.json')
            self.assertEqual(response.status_code, 200)
    
    def test_result_detail_content(self):
        """Kiểm tra nội dung hiển thị khi thiếu dữ liệu"""
        with self.app.test_request_context():
            # Render template với dữ liệu thiếu và kiểm tra kết quả
            result = render_template('result_detail.html', 
                                    result=self.missing_train_samples, 
                                    filename="missing_train.json")
            
            # Kiểm tra các giá trị N/A xuất hiện đúng chỗ
            self.assertIn('N/A', result)
            # Kiểm tra các giá trị có thực tế xuất hiện
            self.assertIn('200', result)  # n_samples_test
            self.assertIn('0.88', result)  # accuracy
    
    def test_all_templates_with_all_test_data(self):
        """Test tất cả templates với tất cả dữ liệu test"""
        with self.app.test_request_context():
            # Kiểm tra từng kết quả test với template result_detail.html
            for result_data in self.all_test_results:
                rendered = render_template('result_detail.html', 
                                         result=result_data, 
                                         filename=result_data["file_name"])
                # Không có lỗi nào xảy ra là đủ để pass test
                self.assertIsNotNone(rendered)
                # Template phải chứa model_type (có thể đã được chuyển đổi sang title case)
                model_type = result_data["model_type"]
                # Kiểm tra cả snake_case và title case
                self.assertTrue(
                    model_type in rendered or model_type.capitalize() in rendered,
                    f"Model type '{model_type}' hoặc '{model_type.capitalize()}' không có trong kết quả"
                )


if __name__ == '__main__':
    unittest.main() 