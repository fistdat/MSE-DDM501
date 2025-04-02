"""
Test cho chức năng tìm và lưu mô hình tốt nhất
"""

import unittest
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from save_best_model import find_best_model

class TestSaveBestModel(unittest.TestCase):
    """Test các chức năng của module save_best_model"""
    
    @patch('save_best_model.setup_mlflow')
    @patch('save_best_model.mlflow')
    @patch('save_best_model.joblib')
    def test_find_best_model_success(self, mock_joblib, mock_mlflow, mock_setup_mlflow):
        """Test trường hợp tìm và lưu mô hình thành công"""
        # Tạo mock data cho experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Tạo mock data cho runs
        mock_runs = pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3'],
            'metrics.f1': [0.8, 0.85, 0.75],
            'metrics.accuracy': [0.82, 0.87, 0.76],
            'metrics.precision': [0.79, 0.84, 0.74],
            'metrics.recall': [0.81, 0.86, 0.77],
            'params.model_type': ['random_forest', 'gradient_boosting', 'random_forest'],
            'params.model__n_estimators': ['100', '200', '50'],
            'params.model__max_depth': ['10', '5', '15']
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Mock load_model
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        # Mock register_model
        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_mlflow.register_model.return_value = mock_model_version
        
        # Mock MlflowClient
        mock_client = MagicMock()
        mock_mlflow.tracking.client.MlflowClient.return_value = mock_client
        
        # Đảm bảo thư mục models tồn tại
        os.makedirs('models', exist_ok=True)
        
        # Chạy hàm cần test
        result = find_best_model()
        
        # Kiểm tra các mock đã được gọi đúng
        mock_setup_mlflow.assert_called_once()
        mock_mlflow.get_experiment_by_name.assert_called_once_with("tuning_experiment")
        mock_mlflow.search_runs.assert_called_once()
        mock_mlflow.sklearn.load_model.assert_called_once()
        mock_mlflow.register_model.assert_called_once()
        mock_joblib.dump.assert_called_once()
        
        # Kiểm tra kết quả
        self.assertIsNotNone(result)
        self.assertEqual(result['run_id'], 'run2')  # Run với F1 cao nhất
        self.assertEqual(result['model_type'], 'gradient_boosting')
        self.assertEqual(result['f1_score'], 0.85)
        self.assertIn('model_registry', result)
        self.assertEqual(result['model_registry']['name'], 'best_gradient_boosting_model')
        
        # Kiểm tra file kết quả
        self.assertTrue(os.path.exists('models/best_model_info.json'))
        
        # Xóa file sau khi test
        if os.path.exists('models/best_model_info.json'):
            os.remove('models/best_model_info.json')
    
    @patch('save_best_model.mlflow')
    @patch('save_best_model.setup_mlflow')
    def test_find_best_model_no_experiment(self, mock_setup_mlflow, mock_mlflow):
        """Test trường hợp không tìm thấy experiment"""
        # Trả về None cho experiment
        mock_mlflow.get_experiment_by_name.return_value = None
        
        # Chạy hàm cần test
        result = find_best_model()
        
        # Kiểm tra kết quả
        self.assertIsNone(result)
    
    @patch('save_best_model.mlflow')
    @patch('save_best_model.setup_mlflow')
    def test_find_best_model_empty_runs(self, mock_setup_mlflow, mock_mlflow):
        """Test trường hợp không có runs nào"""
        # Tạo mock data cho experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Trả về DataFrame rỗng
        mock_mlflow.search_runs.return_value = pd.DataFrame()
        
        # Chạy hàm cần test
        result = find_best_model()
        
        # Kiểm tra kết quả
        self.assertIsNone(result)
    
    @patch('save_best_model.mlflow')
    @patch('save_best_model.setup_mlflow')
    def test_find_best_model_missing_metrics(self, mock_setup_mlflow, mock_mlflow):
        """Test trường hợp không có metrics.f1"""
        # Tạo mock data cho experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Tạo DataFrame không có cột metrics.f1
        mock_runs = pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3'],
            'metrics.accuracy': [0.82, 0.87, 0.76]
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Chạy hàm cần test
        result = find_best_model()
        
        # Kiểm tra kết quả
        self.assertIsNone(result)
    
    @patch('save_best_model.MlflowClient')
    @patch('save_best_model.mlflow')
    @patch('save_best_model.check_sqlite_connection')
    def test_register_model_functionality(self, mock_check_connection, mock_mlflow, mock_client_class):
        """Test trực tiếp chức năng register_model"""
        # Import register_model
        from save_best_model import register_model
        
        # Đảm bảo check_sqlite_connection trả về True
        mock_check_connection.return_value = True
        
        # Mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock register_model
        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_mlflow.register_model.return_value = mock_model_version
        
        # Trường hợp: Mô hình chưa tồn tại
        mock_client.get_registered_model.side_effect = Exception("Model not found")
        
        # Test register_model
        result = register_model(
            model_uri="runs:/test123/model",
            name="test_model",
            description="Test description",
            tags={"tag1": "value1"}
        )
        
        # Kiểm tra client.create_registered_model được gọi
        self.assertTrue(mock_client.create_registered_model.called)
        self.assertTrue(mock_mlflow.register_model.called)
        
        # Kiểm tra kết quả
        self.assertEqual(result.version, "1")

def run_tests():
    """Chạy các test"""
    unittest.main()

if __name__ == "__main__":
    run_tests() 