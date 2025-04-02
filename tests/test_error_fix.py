"""
Test và sửa lỗi hiển thị template
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from flask import template_rendered
from contextlib import contextmanager

from app import app


@contextmanager
def captured_templates(app):
    """
    Context manager để bắt các template đã render
    """
    recorded = []
    def record(sender, template, context, **extra):
        recorded.append((template, context))
    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)


def fix_template_index_html():
    """
    Sửa lỗi 'dict object' has no attribute 'n_samples_train' trong template index.html
    
    Lỗi này xảy ra trong phần hiển thị số mẫu ở trang kết quả, cần đảm bảo rằng thuộc tính này tồn tại
    """
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tìm dòng có vấn đề
    original_line = '<p><strong>Số mẫu:</strong> {{ result.n_samples_train + result.n_samples_test }}</p>'
    
    # Thay bằng đoạn code kiểm tra existence trước khi cộng
    fixed_line = """<p><strong>Số mẫu:</strong> 
                        {% if result.n_samples_train is defined and result.n_samples_test is defined %}
                            {{ result.n_samples_train + result.n_samples_test }}
                        {% elif result.n_samples_train is defined %}
                            {{ result.n_samples_train }}
                        {% elif result.n_samples_test is defined %}
                            {{ result.n_samples_test }}
                        {% else %}
                            N/A
                        {% endif %}
                    </p>"""
    
    # Thay thế tất cả các chỗ xuất hiện
    updated_content = content.replace(original_line, fixed_line)
    
    # Lưu lại template đã cập nhật
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Đã sửa lỗi trong file templates/index.html")


def create_sample_result_with_total_samples():
    """
    Tạo một phiên bản chỉnh sửa của template xử lý tình huống không có n_samples_train
    """
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tìm dòng có vấn đề
    original_line = '<p><strong>Số mẫu:</strong> {{ result.n_samples_train + result.n_samples_test }}</p>'
    
    # Thay bằng một đoạn code kiểm tra total_samples trước, rồi mới tính tổng
    fixed_line = """<p><strong>Số mẫu:</strong> 
                        {% if result.total_samples is defined %}
                            {{ result.total_samples }}
                        {% elif result.n_samples_train is defined and result.n_samples_test is defined %}
                            {{ result.n_samples_train + result.n_samples_test }}
                        {% else %}
                            N/A
                        {% endif %}
                    </p>"""
    
    # Thay thế
    updated_content = content.replace(original_line, fixed_line)
    
    # Lưu thành một file tạm để test
    test_template_path = 'templates/test_index.html'
    with open(test_template_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Đã tạo file test template: {test_template_path}")
    return test_template_path


class TestTemplateError(unittest.TestCase):
    """
    Test các lỗi trong template và giải pháp sửa chữa
    """
    
    def setUp(self):
        """
        Thiết lập cho mỗi test case
        """
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Tạo thư mục tuning_results cho test
        self.test_tuning_dir = 'test_tuning_results'
        if not os.path.exists(self.test_tuning_dir):
            os.makedirs(self.test_tuning_dir)
    
    def tearDown(self):
        """
        Dọn dẹp sau mỗi test
        """
        # Xóa thư mục test nếu tồn tại
        if os.path.exists(self.test_tuning_dir):
            import shutil
            shutil.rmtree(self.test_tuning_dir)
        
        # Xóa file template test nếu tồn tại
        test_template = 'templates/test_index.html'
        if os.path.exists(test_template):
            os.remove(test_template)
    
    def test_template_error_diagnosis(self):
        """
        Kiểm tra lỗi khi hiển thị kết quả không có thuộc tính n_samples_train
        """
        # Tạo dữ liệu mẫu lỗi - thiếu n_samples_train
        error_result = {
            "model_type": "random_forest",
            "param_space": "small",
            # Thiếu n_samples_train
            "n_samples_test": 200,
            "n_features": 20,
            "metrics": {
                "accuracy": 0.88,
                "precision": 0.87,
                "recall": 0.84,
                "f1": 0.855
            },
            "file_name": "error_case.json",
            "timestamp": "02-04-2025 12:34:56"
        }
        
        # Lưu mẫu lỗi
        with open(os.path.join(self.test_tuning_dir, "error_case.json"), 'w') as f:
            json.dump(error_result, f)
        
        # Mock hàm get_tuning_results để trả về mẫu lỗi
        with patch('app.get_tuning_results', return_value=[error_result]):
            # Kiểm tra có lỗi khi render template
            with self.assertRaises(Exception) as context:
                self.client.get('/')
            
            # Kiểm tra xem có phải lỗi chúng ta đang tìm không
            self.assertIn("'dict' object has no attribute 'n_samples_train'", str(context.exception))
    
    def test_fixed_template(self):
        """
        Kiểm tra template đã sửa có hoạt động không
        """
        # Tạo file template sửa lỗi
        test_template_path = create_sample_result_with_total_samples()
        
        # Tạo dữ liệu mẫu lỗi - thiếu n_samples_train
        error_result = {
            "model_type": "random_forest",
            "param_space": "small",
            # Thiếu n_samples_train
            "n_samples_test": 200,
            "n_features": 20,
            "metrics": {
                "accuracy": 0.88,
                "precision": 0.87,
                "recall": 0.84,
                "f1": 0.855
            },
            "file_name": "error_case.json",
            "timestamp": "02-04-2025 12:34:56"
        }
        
        # Lưu mẫu lỗi
        with open(os.path.join(self.test_tuning_dir, "error_case.json"), 'w') as f:
            json.dump(error_result, f)
        
        # Patching hàm render_template để sử dụng template đã sửa
        with patch('flask.render_template', side_effect=lambda template_name, **kwargs: 
                  self.app.jinja_env.get_template('test_index.html').render(**kwargs) 
                  if template_name == 'index.html' else 
                  self.app.jinja_env.get_template(template_name).render(**kwargs)):
            
            # Mock hàm get_tuning_results để trả về mẫu lỗi
            with patch('app.get_tuning_results', return_value=[error_result]):
                # Kiểm tra không còn lỗi
                response = self.client.get('/')
                self.assertEqual(response.status_code, 200)
    
    def test_apply_fix(self):
        """
        Áp dụng giải pháp sửa lỗi và kiểm tra
        """
        # Lưu bản sao của template gốc
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        try:
            # Sửa lỗi template
            fix_template_index_html()
            
            # Tạo dữ liệu mẫu lỗi - thiếu n_samples_train
            error_result = {
                "model_type": "random_forest",
                "param_space": "small",
                # Thiếu n_samples_train
                "n_samples_test": 200,
                "n_features": 20,
                "metrics": {
                    "accuracy": 0.88,
                    "precision": 0.87,
                    "recall": 0.84,
                    "f1": 0.855
                },
                "file_name": "error_case.json",
                "timestamp": "02-04-2025 12:34:56"
            }
            
            # Lưu mẫu lỗi
            with open(os.path.join(self.test_tuning_dir, "error_case.json"), 'w') as f:
                json.dump(error_result, f)
            
            # Mock hàm get_tuning_results để trả về mẫu lỗi
            with patch('app.get_tuning_results', return_value=[error_result]):
                # Kiểm tra không còn lỗi
                response = self.client.get('/')
                self.assertEqual(response.status_code, 200)
        
        finally:
            # Khôi phục template gốc
            with open('templates/index.html', 'w', encoding='utf-8') as f:
                f.write(original_content)


if __name__ == "__main__":
    # Chạy test để phát hiện lỗi
    test_error = TestTemplateError()
    test_error.setUp()
    
    try:
        print("Kiểm tra lỗi template...")
        test_error.test_template_error_diagnosis()
    except Exception as e:
        print(f"Phát hiện lỗi: {e}")
        
        # Sửa lỗi
        print("Đang sửa lỗi template...")
        fix_template_index_html()
        
        # Kiểm tra lại
        print("Kiểm tra lại sau khi sửa...")
        test_error.test_fixed_template()
        print("Đã sửa lỗi thành công!")
    
    test_error.tearDown() 