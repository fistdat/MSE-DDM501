"""
Fixtures và các hàm phụ trợ cho việc test
"""
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Mock MLflow để test không cần kết nối đến MLflow server
    """
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model'), \
         patch('mlflow.start_run', return_value=MagicMock()), \
         patch('mlflow.end_run'):
        yield
        
@pytest.fixture
def reset_mlflow_experiment():
    """
    Reset MLflow experiment active
    """
    with patch('mlflow.get_experiment_by_name', return_value=MagicMock(
        experiment_id='1',
        lifecycle_stage='active'
    )):
        yield 