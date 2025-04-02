.PHONY: install test lint format clean docker-build docker-run venv dev check clean-mlflow simple-tuning simple-tuning-tiny simple-tuning-gb simple-tuning-large save-best-model test-save-model reset-mlflow start-mlflow

# Python version
PYTHON = python3.12

# ===== CÀI ĐẶT & MÔI TRƯỜNG =====

# Install dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

# Create virtual environment
venv:
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Development setup
dev: venv
	. venv/bin/activate && pip install -e .

# ===== KIỂM TRA & ĐỊNH DẠNG CODE =====

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

# Run all checks
check: lint test

# ===== DOCKER =====

# Docker commands
docker-build:
	docker build -t mlops-lab02 .

docker-run:
	docker run -p 5001:5001 mlops-lab02

# ===== TUNING SIÊU THAM SỐ =====

# Simple Hyperparameter Tuning
simple-tuning:
	python simple_hyperparam_tuning.py

simple-tuning-tiny:
	python simple_hyperparam_tuning.py --space tiny

simple-tuning-gb:
	python simple_hyperparam_tuning.py --model gradient_boosting

simple-tuning-large:
	python simple_hyperparam_tuning.py --samples 2000 --features 30

# ===== LƯU & QUẢN LÝ MÔ HÌNH =====

# Save best model from tuning results
save-best-model:
	python save_best_model.py

# Test save best model functionality
test-save-model:
	python test_save_best_model.py

# ===== MLFLOW =====

# Khởi động lại MLflow hoàn toàn (xóa dữ liệu cũ và khởi động lại)
reset-mlflow:
	./restart_mlflow.sh

# Khởi động MLflow server
start-mlflow:
	python run_mlflow_server.py --host 127.0.0.1 --port 5002 \
		--backend-store-uri "./mlflow_data/mlflow.db" \
		--default-artifact-root "./mlflow_data/artifacts"

# Clean MLflow artifacts
clean-mlflow:
	rm -rf mlruns
	rm -rf tuning_results

# ===== DỌN DẸP =====

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