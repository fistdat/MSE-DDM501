"""MLflow Lab API"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from mlib import MLModel
import mlflow
import logging
from typing import Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = MLModel()

def validate_data(data: Dict[str, Any]) -> bool:
    """
    Validate input data
    
    Args:
        data: Input data dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not data or "data" not in data:
        return False
    return True

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": model.is_trained
    }), 200

@app.route("/train", methods=["POST"])
def train():
    """Train the model with provided data"""
    try:
        data = request.get_json()
        if not validate_data(data):
            return jsonify({
                "error": "Invalid input data",
                "message": "Data must contain 'data' field with training data"
            }), 400
            
        # Check if target is provided
        if "target" not in data:
            return jsonify({
                "error": "Invalid input data",
                "message": "Data must contain 'target' field with labels"
            }), 400
            
        # Convert data to DataFrame and numpy array
        X = np.array(data["data"])
        y = np.array(data["target"], dtype=int)
        
        # Train model
        metrics = model.train(X, y)
        
        return jsonify({
            "message": "Model trained successfully",
            "metrics": metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Training failed",
            "message": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the trained model"""
    try:
        data = request.get_json()
        if not validate_data(data):
            return jsonify({
                "error": "Invalid input data",
                "message": "Data must contain 'data' field with features"
            }), 400
            
        # Convert data to numpy array
        X = np.array(data["data"])
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get current model metrics"""
    try:
        metrics = model.get_metrics()
        if not metrics:
            return jsonify({
                "error": "No metrics available",
                "message": "Model has not been trained yet"
            }), 404
            
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to get metrics",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested URL was not found on the server"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error has occurred"
    }), 500

if __name__ == "__main__":
    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("tuning_experiment")
    
    # Start Flask app
    app.run(host="0.0.0.0", port=5001)