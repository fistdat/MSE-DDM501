#!/usr/bin/env python
"""
Script test tích hợp cho ứng dụng Docker.
Kiểm tra service Flask và MLflow đang chạy đúng cách.
"""

import os
import unittest
import requests
import time
import json
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants - URL endpoints for testing
FLASK_API_URL = "http://localhost:5001"
MLFLOW_API_URL = "http://localhost:5002"

class DockerIntegrationTest(unittest.TestCase):
    """Test integration for Docker containers"""
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment: Start Docker containers if needed.
        This runs once before all tests.
        """
        logger.info("Setting up Docker integration test")
        
        # Check if Docker is running
        try:
            subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("Docker is running")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker is not running or Docker CLI not found")
            raise unittest.SkipTest("Docker is not running. Skipping integration tests.")
            
        # Check if containers are already running
        try:
            status = subprocess.run(["docker", "ps", "-q", "-f", "name=mlops-lab02"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
            if not status.stdout.strip():
                logger.info("Starting Docker Compose services")
                subprocess.run(["docker-compose", "up", "-d"], check=True)
                # Wait for services to start
                logger.info("Waiting for services to start...")
                time.sleep(10)  # Allow services to start
        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting Docker Compose: {e}")
            raise
            
        # Verify services are up
        cls._verify_services_up()
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up after tests: Keep Docker containers running.
        This runs once after all tests.
        """
        logger.info("Docker containers remain running after tests.")
    
    @classmethod
    def _verify_services_up(cls, max_attempts=5, wait_time=5):
        """
        Verify that both services (Flask API and MLflow) are running.
        
        Args:
            max_attempts: Maximum number of attempts to check services
            wait_time: Time to wait between attempts (seconds)
        
        Raises:
            unittest.SkipTest: If services don't come up
        """
        flask_up = False
        mlflow_up = False
        
        for attempt in range(max_attempts):
            try:
                # Check Flask API
                if not flask_up:
                    flask_response = requests.get(f"{FLASK_API_URL}/health", timeout=2)
                    if flask_response.status_code == 200:
                        logger.info("Flask API is up")
                        flask_up = True
                
                # Check MLflow
                if not mlflow_up:
                    mlflow_response = requests.get(f"{MLFLOW_API_URL}/api/2.0/mlflow/experiments/search?max_results=1000", timeout=2)
                    if mlflow_response.status_code == 200:
                        logger.info("MLflow server is up")
                        mlflow_up = True
                
                if flask_up and mlflow_up:
                    return True
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(f"Attempt {attempt+1}/{max_attempts}: Services not ready yet. Error: {e}")
            
            time.sleep(wait_time)
        
        # If we got here, not all services are up
        error_message = f"Services are not available after {max_attempts} attempts: "
        error_message += "Flask API: " + ("UP" if flask_up else "DOWN") + ", "
        error_message += "MLflow: " + ("UP" if mlflow_up else "DOWN")
        logger.error(error_message)
        raise unittest.SkipTest(error_message)
    
    def test_flask_api_health(self):
        """Test Flask API health endpoint"""
        response = requests.get(f"{FLASK_API_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue("message" in data)
    
    def test_flask_api_root(self):
        """Test Flask API root endpoint returns HTML"""
        response = requests.get(FLASK_API_URL)
        self.assertEqual(response.status_code, 200)
        self.assertTrue("text/html" in response.headers.get("Content-Type", ""))
        # Verify it's the right HTML by checking for some key content
        self.assertIn("MLOps Final Project", response.text)
    
    def test_mlflow_experiments_list(self):
        """Test MLflow can list experiments"""
        response = requests.get(f"{MLFLOW_API_URL}/api/2.0/mlflow/experiments/search?max_results=1000")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("experiments" in data)
        # MLflow should always have at least the Default experiment
        self.assertTrue(len(data["experiments"]) >= 1)
    
    def test_flask_api_predict(self):
        """Test Flask API prediction endpoint"""
        test_data = {"feature_data": [{"feature_1": 0.5, "feature_2": 0.7}]}
        response = requests.post(
            f"{FLASK_API_URL}/api/predict",
            json=test_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("prediction" in data)
        # Verify prediction is a list of numbers
        self.assertTrue(isinstance(data["prediction"], list))
        self.assertTrue(all(isinstance(p, (int, float)) for p in data["prediction"]))
    
    def test_flask_api_models_list(self):
        """Test Flask API model listing endpoint"""
        response = requests.get(f"{FLASK_API_URL}/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertTrue(isinstance(models, list))
        # If models exist, check they have the right structure
        if models:
            self.assertTrue("name" in models[0])
            self.assertTrue("version" in models[0])
            self.assertTrue("accuracy" in models[0])

if __name__ == "__main__":
    unittest.main() 