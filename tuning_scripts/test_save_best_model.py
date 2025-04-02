"""
Unit Tests for save_best_model module
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys
import json
import tempfile
import shutil

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning_scripts.save_best_model import (
    find_best_run_by_metric,
    register_model,
    save_best_model,
    CustomJSONEncoder
)

class TestSaveBestModel(unittest.TestCase):
    """Test case cho module save_best_model"""
    
    def setUp(self):
        """Set up các test"""
        # Tạo sample data cho runs
        self.sample_runs_data = {
            'run_id': ['run1', 'run2', 'run3'],
            'metrics.f1_score': [0.75, 0.85, 0.65],
            'metrics.accuracy': [0.78, 0.88, 0.70],
            'metrics.precision': [0.72, 0.82, 0.68],
            'metrics.recall': [0.74, 0.84, 0.64],
            'params.model_type': ['random_forest', 'gradient_boosting', 'random_forest'],
            'params.n_estimators': ['100', '200', '50'],
            'params.max_depth': ['10', '5', '15'],
            'params.n_features': ['20', '20', '20'],
            'params.n_samples': ['1000', '1000', '1000']
        }
        self.sample_runs = pd.DataFrame(self.sample_runs_data)
        
        # Tạo thư mục tạm cho models
        self.temp_models_dir = tempfile.mkdtemp()
        self.original_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    def tearDown(self):
        """Dọn dẹp sau các test"""
        # Xóa thư mục tạm
        shutil.rmtree(self.temp_models_dir)
    
    @patch('tuning_scripts.save_best_model.setup_mlflow')
    @patch('tuning_scripts.save_best_model.mlflow')
    def test_find_best_run_by_metric(self, mock_mlflow, mock_setup_mlflow):
        """Test chức năng tìm run tốt nhất theo metric"""
        # Cài đặt mock
        mock_setup_mlflow.return_value = True
        
        # Tạo mock cho experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp1"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Tạo mock cho search_runs
        mock_mlflow.search_runs.return_value = self.sample_runs
        
        # Tạo mock cho run
        mock_run = MagicMock()
        mock_run.info = MagicMock(run_id="run2")
        mock_run.data.metrics = {"f1_score": 0.85, "accuracy": 0.88}
        mock_run.data.params = {"model_type": "gradient_boosting", "n_estimators": "200"}
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        # Gọi hàm cần test
        run_info, metrics, params = find_best_run_by_metric(metric="f1_score", ascending=False)
        
        # Kiểm tra kết quả
        self.assertEqual(run_info.run_id, "run2")
        self.assertEqual(metrics["f1_score"], 0.85)
        self.assertEqual(params["model_type"], "gradient_boosting")
        
        # Kiểm tra các mock đã được gọi đúng
        mock_setup_mlflow.assert_called_once()
        mock_mlflow.get_experiment_by_name.assert_called_once()
        mock_mlflow.search_runs.assert_called_once_with(experiment_ids=["exp1"])
        mock_client.get_run.assert_called_once_with("run2")
    
    @patch('tuning_scripts.save_best_model.check_sqlite_connection')
    @patch('tuning_scripts.save_best_model.mlflow')
    @patch('tuning_scripts.save_best_model.MlflowClient')
    def test_register_model_functionality(self, mock_client_class, mock_mlflow, mock_check_connection):
        """Test trực tiếp chức năng register_model"""
        # Đảm bảo check_sqlite_connection trả về True
        mock_check_connection.return_value = True
        
        # Tạo mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock cho get_registered_model
        mock_client.get_registered_model.side_effect = Exception("Model not found")
        
        # Mock cho mlflow.register_model
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow.register_model.return_value = mock_version
        
        # Gọi hàm cần test
        result = register_model(
            model_uri="runs:/abc123/model",
            name="test_model",
            description="Test description",
            tags={"tag1": "value1"}
        )
        
        # Kiểm tra kết quả
        self.assertIsNotNone(result, "register_model should return a result")
        mock_check_connection.assert_called_once()
        mock_client.create_registered_model.assert_called_once()
        mock_mlflow.register_model.assert_called_once()
        mock_client.update_model_version.assert_called_once()
    
    @patch('tuning_scripts.save_best_model.find_best_run_by_metric')
    @patch('tuning_scripts.save_best_model.mlflow')
    @patch('tuning_scripts.save_best_model.joblib')
    @patch('tuning_scripts.save_best_model.os.path.dirname')
    @patch('tuning_scripts.save_best_model.os.makedirs')
    @patch('tuning_scripts.save_best_model.register_model')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dump')
    def test_save_best_model(self, mock_json_dump, mock_open, mock_register_model, mock_makedirs, 
                           mock_dirname, mock_joblib, mock_mlflow, mock_find_best_run):
        """Test chức năng lưu mô hình tốt nhất"""
        # Cài đặt mock cho find_best_run_by_metric
        mock_run_info = MagicMock(
            run_id="run2",
            start_time=1000000000,
            artifact_uri="artifacts/run2",
            experiment_id="exp1",
            run_name="Test Run"
        )
        mock_metrics = {
            "f1_score": 0.85,
            "accuracy": 0.88,
            "precision": 0.82,
            "recall": 0.84
        }
        mock_params = {
            "model_type": "gradient_boosting",
            "n_estimators": "200",
            "max_depth": "5",
            "n_features": "20",
            "n_samples": "1000"
        }
        mock_find_best_run.return_value = (mock_run_info, mock_metrics, mock_params)
        
        # Cài đặt mock cho dirname
        mock_dirname.return_value = self.temp_models_dir
        
        # Cài đặt mock cho mlflow
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        # Cài đặt mock cho register_model
        mock_version = MagicMock(version="1")
        mock_register_model.return_value = mock_version
        
        # Gọi hàm cần test
        result = save_best_model()
        
        # Kiểm tra kết quả
        self.assertTrue(result, "save_best_model should return True when successful")
        
        # Kiểm tra các mock đã được gọi đúng
        mock_find_best_run.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_mlflow.sklearn.load_model.assert_called_once()
        mock_joblib.dump.assert_called_once()
        mock_register_model.assert_called_once()
        mock_open.assert_called()
        mock_json_dump.assert_called()

if __name__ == '__main__':
    unittest.main() 