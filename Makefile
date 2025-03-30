.PHONY: install test lint format clean docker-build docker-run mlflow stop-mlflow venv dev check clean-mlflow simple-tuning simple-tuning-tiny simple-tuning-gb simple-tuning-large mlflow-experiments mlflow-restore

# Python version
PYTHON = python3.12

# Install dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

# Run tests with coverage
test:
	pytest --cov=. --cov-report=html

# Run linting
lint:
	pylint *.py
	black --check .

# Format code
format:
	black .

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# Docker commands
docker-build:
	docker build -t mlops-lab02 .

docker-run:
	docker run -p 5001:5001 mlops-lab02

# MLflow server
mlflow:
	python mlflow_utils.py

# Simple Hyperparameter Tuning
simple-tuning:
	python simple_hyperparam_tuning.py

simple-tuning-tiny:
	python simple_hyperparam_tuning.py --space tiny

simple-tuning-gb:
	python simple_hyperparam_tuning.py --model gradient_boosting

simple-tuning-large:
	python simple_hyperparam_tuning.py --samples 2000 --features 30

# Clean MLflow artifacts
clean-mlflow:
	rm -rf mlruns
	rm -rf tuning_results

# Stop MLflow server
stop-mlflow:
	python mlflow_utils.py stop

# Create virtual environment
venv:
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Development setup
dev: venv
	. venv/bin/activate && pip install -e .

# Run all checks
check: lint test

# MLflow experiments management
mlflow-experiments:
	python restore_experiments.py --list-all

mlflow-restore:
	python restore_experiments.py --restore-name tuning_experiment