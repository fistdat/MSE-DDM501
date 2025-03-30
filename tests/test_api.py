"""
Kiểm thử cho API trong app.py
"""
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app import app
from mlib import MLModel

@pytest.fixture
def client():
    """Tạo client test Flask"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Kiểm tra endpoint health"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'model_trained' in data
    assert isinstance(data['model_trained'], bool)

def test_train_endpoint_valid_data(client):
    """Kiểm tra endpoint train với dữ liệu hợp lệ"""
    with patch.object(MLModel, 'train', return_value={
        'accuracy': 0.95,
        'precision': 0.9,
        'recall': 0.89,
        'f1': 0.88
    }):
        payload = {
            'data': [[1, 2], [3, 4], [5, 6], [7, 8]],
            'target': [0, 1, 0, 1]
        }
        response = client.post('/train', json=payload)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'message' in data
        assert data['message'] == 'Model trained successfully'
        assert 'metrics' in data
        assert 'accuracy' in data['metrics']

def test_train_endpoint_invalid_data(client):
    """Kiểm tra endpoint train với dữ liệu không hợp lệ"""
    # Thiếu trường 'data'
    payload = {
        'target': [0, 1, 0, 1]
    }
    response = client.post('/train', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    
    # Thiếu trường 'target'
    payload = {
        'data': [[1, 2], [3, 4], [5, 6], [7, 8]]
    }
    response = client.post('/train', json=payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_endpoint_not_trained(client):
    """Kiểm tra endpoint predict khi chưa huấn luyện model"""
    # Tạo một model mới chưa được huấn luyện
    model = MLModel()
    model.is_trained = False
    with patch.object(app, 'model', model):
        payload = {
            'data': [[1, 2], [3, 4]]
        }
        
        response = client.post('/predict', json=payload)
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data

def test_predict_endpoint_trained(client):
    """Kiểm tra endpoint predict sau khi đã huấn luyện model"""
    # Tạo một model đã được huấn luyện
    model = MLModel()
    model.is_trained = True
    with patch.object(app, 'model', model), \
         patch.object(model, 'predict', return_value=np.array([0, 1])), \
         patch.object(model, 'predict_proba', return_value=np.array([[0.8, 0.2], [0.3, 0.7]])):
        
        # Test predict
        predict_payload = {
            'data': [[2, 3], [6, 7]]
        }
        response = client.post('/predict', json=predict_payload)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data
        assert 'probabilities' in data
        assert len(data['predictions']) == 2
        assert len(data['probabilities']) == 2
        assert len(data['probabilities'][0]) == 2  # Hai lớp

def test_metrics_endpoint_not_trained(client):
    """Kiểm tra endpoint metrics khi chưa huấn luyện model"""
    # Tạo một model mới chưa được huấn luyện
    model = MLModel()
    model.is_trained = False
    with patch.object(app, 'model', model), \
         patch.object(model, 'get_metrics', return_value={}):
        
        response = client.get('/metrics')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'No metrics available'

def test_metrics_endpoint_trained(client):
    """Kiểm tra endpoint metrics sau khi đã huấn luyện model"""
    # Tạo một model đã được huấn luyện với metrics
    model = MLModel()
    model.is_trained = True
    metrics = {
        'accuracy': 0.95,
        'precision': 0.9,
        'recall': 0.89,
        'f1': 0.88
    }
    with patch.object(app, 'model', model), \
         patch.object(model, 'get_metrics', return_value=metrics):
        
        response = client.get('/metrics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'accuracy' in data
        assert 'precision' in data
        assert 'recall' in data
        assert 'f1' in data 