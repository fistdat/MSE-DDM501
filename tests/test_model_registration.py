"""
Test suite for testing model saving and registration functionality
"""
import pytest
import requests
import json
import time
import os
import re
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

def test_runs_page_access(wait_for_services):
    """Test if the runs page is accessible and correctly displays run information"""
    # Access experiment runs page
    response = requests.get(f"{FLASK_URL}/experiments/1/runs")
    assert response.status_code == 200
    
    # Check for expected content in the response
    assert "Danh sách Runs" in response.text
    assert "form action=\"/save-model/" in response.text  # Check for save model forms

def test_get_valid_run_id(wait_for_services):
    """Get a valid run ID from the experiment runs page"""
    response = requests.get(f"{FLASK_URL}/experiments/1/runs")
    
    # Use regex to extract run_id from the page
    run_id_match = re.search(r'form action="/save-model/([^"]+)"', response.text)
    assert run_id_match is not None, "Could not find a save-model form with run_id on the page"
    
    run_id = run_id_match.group(1)
    assert len(run_id) > 0, "Run ID should not be empty"
    
    # Return the run_id for use in other tests
    return run_id

def test_model_saving(wait_for_services):
    """Test model saving functionality"""
    # Get a valid run_id
    run_id = test_get_valid_run_id()
    
    # Prepare test data for model saving
    test_data = {
        "model_name": f"test_model_{int(time.time())}"  # Use timestamp to create unique name
    }
    
    # Send save model request
    response = requests.post(
        f"{FLASK_URL}/save-model/{run_id}",
        data=test_data,
        allow_redirects=False  # Don't follow redirect to check status code
    )
    
    # Check for redirect (302 status code)
    assert response.status_code in [302, 303], "Should redirect after successful model saving"
    
    # Follow redirect and check for success message
    redirect_response = requests.get(response.headers['Location'])
    assert "Đã lưu và đăng ký model" in redirect_response.text, "Success message not found"
    
def test_register_model_script():
    """Test the register_model.py script directly"""
    # First, get a valid run_id
    run_id = test_get_valid_run_id()
    
    # Model name with timestamp to ensure uniqueness
    model_name = f"test_model_direct_{int(time.time())}"
    
    # Define the command to run the script
    import subprocess
    
    try:
        # Run the register_model.py script
        result = subprocess.run(
            ["docker", "exec", "mlops-flask", "python", "/app/tuning_scripts/register_model.py", 
             run_id, model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check the output for success indicators
        assert "Successfully created model version" in result.stdout, "Model version creation failed"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"register_model.py script failed: {e.stderr}")

def test_mlflow_model_registry():
    """Test that models appear in the MLflow Model Registry"""
    # Set up MLflow client
    client = MlflowClient(tracking_uri=MLFLOW_URL)
    
    # List registered models
    registered_models = client.search_registered_models()
    
    # Check if we have any registered models
    assert len(registered_models) > 0, "No registered models found in MLflow Model Registry"
    
    # Print info about registered models (for debugging)
    for model in registered_models:
        print(f"Found registered model: {model.name}")
        versions = client.get_latest_versions(model.name)
        print(f"  Latest version: {versions[0].version}")

def test_register_model_endpoint():
    """Test that the register model endpoint works"""
    # Since we removed the Model Registry functionality, this test is now skipped
    pytest.skip("Model Registry functionality has been removed")
    
    # Setup
    # Find a valid run ID
    response = requests.get(f"{FLASK_URL}/experiments/1/runs")
    assert response.status_code == 200
    
    # Extract the first run ID
    run_id_match = re.search(r'data-run-id="([a-f0-9]+)"', response.text)
    if not run_id_match:
        pytest.skip("No runs found to test with")
    
    run_id = run_id_match.group(1)
    model_name = f"test_model_{int(time.time())}"
    
    # Call the API
    data = {
        "run_id": run_id,
        "model_name": model_name,
        "description": "Test model from automated test"
    }
    response = requests.post(f"{FLASK_URL}/register-model", data=data)
    
    # Verify
    assert response.status_code == 200 or response.status_code == 302
    
    # Check if the model was registered
    client = MlflowClient()
    try:
        models = client.search_registered_models()
        model_names = [model.name for model in models]
        assert model_name in model_names
    except Exception as e:
        pytest.fail(f"Failed to verify model registration: {str(e)}")
    finally:
        # Clean up
        try:
            client.delete_registered_model(model_name)
        except:
            pass 