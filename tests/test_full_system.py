"""
Test suite for testing the full system functionality
"""
import pytest
import requests
import json
import time
import os
from unittest.mock import patch, MagicMock
import mlflow
from mlflow.tracking import MlflowClient

# Test configuration
FLASK_URL = "http://localhost:5001"
MLFLOW_URL = "http://localhost:5002"

@pytest.fixture(scope="session")
def wait_for_services():
    """Wait for services to be ready"""
    max_retries = 30
    retry_interval = 2
    
    for _ in range(max_retries):
        try:
            # Check Flask app
            response = requests.get(f"{FLASK_URL}/")
            if response.status_code == 200:
                # Check MLflow
                response = requests.get(f"{MLFLOW_URL}/")
                if response.status_code == 200:
                    return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(retry_interval)
    
    raise Exception("Services did not become ready in time")

def test_flask_app_availability(wait_for_services):
    """Test if Flask app is accessible"""
    response = requests.get(f"{FLASK_URL}/")
    assert response.status_code == 200
    assert "MLOps Lab 02" in response.text

def test_mlflow_availability(wait_for_services):
    """Test if MLflow UI is accessible"""
    response = requests.get(f"{MLFLOW_URL}/")
    assert response.status_code == 200
    assert "MLflow" in response.text

def test_mlflow_experiment_creation():
    """Test MLflow experiment creation and tracking"""
    # Set up MLflow client
    client = MlflowClient(tracking_uri=MLFLOW_URL)
    
    # Create a test experiment
    experiment_name = "test_experiment"
    experiment_id = client.create_experiment(experiment_name)
    assert experiment_id is not None
    
    # Start a run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log some test metrics
        mlflow.log_metric("test_metric", 0.95)
        mlflow.log_param("test_param", "value")
        
        # Get run info
        run_info = client.get_run(run.info.run_id)
        assert run_info.data.metrics["test_metric"] == 0.95
        assert run_info.data.params["test_param"] == "value"

def test_model_tuning():
    """Test model tuning functionality"""
    # Prepare test data
    test_data = {
        "model_type": "random_forest",
        "param_grid": json.dumps({
            "n_estimators": [50, 100],
            "max_depth": [3, 5]
        }),
        "cv": "2"
    }
    
    # Send tuning request
    response = requests.post(
        f"{FLASK_URL}/run_simple_tuning",
        data=test_data
    )
    
    assert response.status_code == 200
    
    # Wait for tuning to complete
    max_retries = 30
    retry_interval = 2
    
    for _ in range(max_retries):
        # Check tuning results
        response = requests.get(f"{FLASK_URL}/")
        if "Tuning thành công" in response.text:
            break
        time.sleep(retry_interval)
    
    # Verify results are saved
    assert os.path.exists("tuning_results")
    assert len(os.listdir("tuning_results")) > 0

def test_model_prediction():
    """Test model prediction functionality"""
    # First ensure we have a trained model
    test_data = {
        "model_type": "random_forest",
        "param_grid": json.dumps({
            "n_estimators": [50],
            "max_depth": [3]
        }),
        "cv": "2"
    }
    
    # Train a model
    response = requests.post(
        f"{FLASK_URL}/run_simple_tuning",
        data=test_data
    )
    
    # Wait for training to complete
    for _ in range(30):
        response = requests.get(f"{FLASK_URL}/")
        if "Tuning thành công" in response.text:
            break
        time.sleep(2)
    
    # Test prediction
    test_features = {
        "feature_data": json.dumps([[1.0, 2.0, 3.0, 4.0, 5.0]])
    }
    
    response = requests.post(
        f"{FLASK_URL}/predict",
        data=test_features
    )
    
    assert response.status_code == 200
    assert "prediction_results" in response.text

def test_error_handling():
    """Test error handling in various scenarios"""
    # Test invalid model type
    test_data = {
        "model_type": "invalid_model",
        "param_grid": "{}",
        "cv": "2"
    }
    
    response = requests.post(
        f"{FLASK_URL}/run_simple_tuning",
        data=test_data
    )
    assert response.status_code == 200  # Flask returns 200 but shows error in template
    assert "error" in response.text.lower()
    
    # Test invalid prediction request
    test_features = {
        "feature_data": "invalid"  # Invalid feature format
    }
    
    response = requests.post(
        f"{FLASK_URL}/predict",
        data=test_features
    )
    assert response.status_code == 200  # Flask returns 200 but shows error in template
    assert "error" in response.text.lower()

def test_mlflow_integration():
    """Test MLflow integration with the system"""
    # Verify experiment exists
    client = MlflowClient(tracking_uri=MLFLOW_URL)
    experiment = client.get_experiment_by_name("tuning_experiment")
    assert experiment is not None
    
    # Verify runs are being logged
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) > 0
    
    # Verify artifacts are being saved
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        assert len(artifacts) > 0 