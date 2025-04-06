# MLOps Project - Hyperparameter Tuning

MLOps project with Flask API, MLflow and Hyperparameter Tuning for training and deploying machine learning models.

## Project Structure

```
MLOps-Lab02/
├── app.py                         # Main Flask API
├── mlib.py                        # Core ML library
├── tuning_scripts/                # Directory containing tuning scripts
│   ├── simple_hyperparam_tuning.py    # Simple hyperparameter tuning script
│   ├── custom_hyperparam_tuning.py    # Custom hyperparameter tuning script
│   ├── save_best_model.py             # Script to save best model from tuning results
│   └── register_model.py              # Model registration script
├── mlflow_scripts/                # Directory containing MLflow scripts
│   ├── mlflow_utils.py                # MLflow utilities
│   ├── mlflow_config.py               # MLflow configuration
│   ├── run_mlflow_server.py           # Script to start MLflow server
│   └── restart_mlflow.sh              # Script to restart MLflow server
├── models/                        # Directory containing saved models
│   └── best_model.joblib          # Best saved model
├── templates/                     # HTML templates for Flask UI
│   ├── index.html                 # Main application page (3 tabs)
│   └── result_detail.html         # Tuning result details page
├── tuning_results/                # Directory containing tuning results
├── tests/                         # Unit tests and Integration tests
├── mlflow_data/                   # MLflow data 
├── requirements.txt               # Required libraries
└── Makefile                       # Build and utility commands
```

## System Requirements

- Python 3.9+
- Libraries: flask, scikit-learn, pandas, numpy, joblib, mlflow, matplotlib

## Installation

1. Create a virtual environment:
```bash
make venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
make install
```

## Running the Application

1. Start the MLflow server:
```bash
make start-mlflow
```

2. Run the Flask application:
```bash
python app.py
```

Open your browser and navigate to: http://localhost:5001

### Using Docker

#### Single Container

```bash
make docker-build
make docker-run
```

If you need to update files in the Docker container:
```bash
make docker-cp-files
```

#### Docker Compose (Recommended)

For a complete setup with both Flask and MLflow services:

```bash
# Build and run with Docker Compose
docker compose -f docker/docker-compose.yml up --build

# Or use the provided script
bash docker/build-and-run.sh
```

This will start:
- MLflow server on port 5002
- Flask application on port 5001

### Docker Hub Images

The application has been packaged and pushed to Docker Hub. You can use the images directly:

```bash
# Pull images from Docker Hub
docker pull fistdat/mlops-flask:latest
docker pull fistdat/mlops-mlflow:latest

# Run with docker-compose
curl -O https://raw.githubusercontent.com/fistdat/MSE-DDM501/main/docker/docker-compose.yml
docker compose up
```

Available Docker images:
- `fistdat/mlops-flask:latest` - Flask application (latest version)
- `fistdat/mlops-mlflow:latest` - MLflow server (latest version)
- Tagged versions are also available (e.g., `v1.7`, `v1.8`, etc.)

### CI/CD Integration with CircleCI

This project is integrated with CircleCI to automate the build and deploy process to Docker Hub. Whenever changes are pushed to the `main` branch:

1. CircleCI automatically builds the Docker image
2. CircleCI pushes the image to Docker Hub with both `latest` and new version (vX.X) tags
3. The pipeline is configured to run tests before building and pushing

Details about the CI/CD setup and configuration can be found in the [CI_SETUP.md](CI_SETUP.md) file.

## User Interface

The application provides a web user interface with 3 main tabs:

1. **Simple Tuning**:
   - Select model type (Random Forest, Gradient Boosting)
   - Choose parameter space (Tiny, Small, Medium)
   - Specify number of samples, features and folds
   - Track progress and results from MLflow

2. **Custom Tuning**:
   - Detailed hyperparameter options for each model type
   - Adjust experiment scale and configuration
   - Control hyperparameter optimization

3. **Classification**:
   - Use the best trained model to classify new data
   - Generate random data for quick testing
   - View classification results and detailed probabilities

## API Endpoints

### Health Check
- URL: `/health`
- Method: GET
- Response: Application and model status

### Train Model
- URL: `/train`
- Method: POST
- Body: JSON with "data" field containing training data
- Response: Metrics after training

### Predict
- URL: `/predict`
- Method: POST
- Body: Form data with "feature_data" field containing features in JSON format
- Response: Classification results and probabilities

### Get Metrics
- URL: `/metrics`
- Method: GET
- Response: Current model metrics

## Hyperparameter Tuning

### Running Tuning from Command Line

```bash
# Simple tuning with default options (Random Forest, small space)
python tuning_scripts/simple_hyperparam_tuning.py
# or
make simple-tuning

# Tuning with smaller parameter space (faster)
python tuning_scripts/simple_hyperparam_tuning.py --space tiny
# or
make simple-tuning-tiny

# Tuning with Gradient Boosting
python tuning_scripts/simple_hyperparam_tuning.py --model gradient_boosting
# or
make simple-tuning-gb

# Tuning with more data samples
python tuning_scripts/simple_hyperparam_tuning.py --samples 2000 --features 30
# or
make simple-tuning-large
```

Available options for simple_hyperparam_tuning.py:
- `--model`: Model type (`random_forest`, `gradient_boosting`)
- `--space`: Parameter space size (`tiny`, `small`, `medium`)
- `--samples`: Number of data samples
- `--features`: Number of features
- `--cv`: Number of cross-validation folds
- `--no-mlflow`: Disable MLflow tracking

### Viewing Results

#### MLflow UI

Access: http://localhost:5002

#### Local Results

All tuning results are automatically saved in the `tuning_results/` directory.

## Saving the Best Model

After running tuning, you can save the best model:

```bash
# Save the best model from tuning results
make save-best-model
```

The `save_best_model.py` script will:
1. Search the MLflow experiment `tuning_experiment` to identify the run with the highest F1-score
2. Save the model to the `models/` directory as a joblib file
3. Save detailed information about the best model to `model_info.json`

## Classifying Data with the Best Model

After saving the best model, you can use it to classify new data:

1. **Through the web interface**:
   - Access the "Classification" tab in the user interface
   - Enter input data in JSON format or use the "Generate Random Data" button
   - Click "Classify" to see the results

2. **Through the API**:
   ```bash
   curl -X POST http://localhost:5001/predict \
     -d "feature_data=[{\"feature_1\": 0.5, \"feature_2\": 0.3, ...}]"
   ```

3. **Using the test command**:
   ```bash
   make test-predict
   ```

## Development

### Testing
```bash
make test
```

### Linting
```bash
make lint
```

### Format Code
```bash
make format
```

### Clean Up
```bash
make clean          # Clean cache, __pycache__, etc.
make clean-mlflow   # Remove MLflow data and tuning results
make clean-all      # Remove everything, including models
```