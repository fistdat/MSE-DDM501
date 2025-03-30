"""
Kiểm thử cho module mlib.py
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from mlib import MLModel

def test_model_initialization():
    """Kiểm tra khởi tạo model"""
    model = MLModel()
    assert model.is_trained == False
    assert model.model is not None
    assert model.metrics == {}

def test_model_train():
    """Kiểm tra huấn luyện model"""
    model = MLModel()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    with patch('mlflow.log_metrics'), \
         patch('mlflow.log_params'), \
         patch('mlflow.sklearn.log_model'):
        metrics = model.train(X, y)
        
    assert model.is_trained == True
    assert model.model is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    
def test_model_predict():
    """Kiểm tra dự đoán model"""
    model = MLModel()
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    
    with patch('mlflow.log_metrics'), \
         patch('mlflow.log_params'), \
         patch('mlflow.sklearn.log_model'):
        model.train(X_train, y_train)
    
    X_test = np.array([[2, 3], [6, 7]])
    predictions = model.predict(X_test)
    assert len(predictions) == 2
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    
def test_model_predict_proba():
    """Kiểm tra dự đoán xác suất model"""
    model = MLModel()
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    
    with patch('mlflow.log_metrics'), \
         patch('mlflow.log_params'), \
         patch('mlflow.sklearn.log_model'):
        model.train(X_train, y_train)
    
    X_test = np.array([[2, 3], [6, 7]])
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (2, 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(np.sum(probabilities, axis=1), np.ones(2))
    
def test_get_metrics():
    """Kiểm tra lấy metrics"""
    model = MLModel()
    assert model.get_metrics() == {}
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    with patch('mlflow.log_metrics'), \
         patch('mlflow.log_params'), \
         patch('mlflow.sklearn.log_model'):
        metrics = model.train(X, y)
    
    assert model.get_metrics() == metrics
    assert "accuracy" in model.get_metrics()
    
def test_predict_without_training():
    """Kiểm tra dự đoán khi chưa huấn luyện model"""
    model = MLModel()
    model.is_trained = False
    X = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError):
        model.predict(X)
        
    with pytest.raises(ValueError):
        model.predict_proba(X) 