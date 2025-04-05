.PHONY: install test lint format clean docker-build docker-run venv dev simple-tuning simple-tuning-tiny simple-tuning-gb simple-tuning-large save-best-model test-predict reset-mlflow start-mlflow docker-cp-files

# Python version
PYTHON = python3

# ===== CÀI ĐẶT & MÔI TRƯỜNG =====

# Install dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

# Create virtual environment
venv:
	$(PYTHON) -m venv venv
	@echo "Môi trường ảo đã được tạo. Kích hoạt bằng 'source venv/bin/activate' (Linux/Mac) hoặc 'venv\\Scripts\\activate' (Windows)"

# Development setup
dev: venv
	. venv/bin/activate && pip install -e .

# ===== KIỂM TRA & ĐỊNH DẠNG CODE =====

# Run tests
test:
	PYTHONPATH=. python -m unittest discover -s tests

# Run linting
lint:
	pylint app.py
	pylint tuning_scripts/*.py mlflow_scripts/*.py tests/*.py
	black --check .

# Format code
format:
	black .

# ===== DOCKER =====

# Build Docker image
docker-build:
	docker build -t mlops-flask .

# Run Docker container
docker-run:
	docker run -p 5001:5001 -p 5002:5002 mlops-flask

# Copy updated files to Docker container
docker-cp-files:
	docker cp templates/index.html mlops-flask:/app/templates/
	docker cp app.py mlops-flask:/app/
	docker restart mlops-flask

# ===== TUNING SIÊU THAM SỐ =====

# Simple Hyperparameter Tuning
simple-tuning:
	python tuning_scripts/simple_hyperparam_tuning.py

# Tuning với không gian tham số nhỏ (nhanh)
simple-tuning-tiny:
	python tuning_scripts/simple_hyperparam_tuning.py --space tiny

# Tuning với Gradient Boosting
simple-tuning-gb:
	python tuning_scripts/simple_hyperparam_tuning.py --model gradient_boosting

# Tuning với nhiều dữ liệu hơn
simple-tuning-large:
	python tuning_scripts/simple_hyperparam_tuning.py --samples 2000 --features 30

# Custom Hyperparameter Tuning - Tuning tùy chỉnh
custom-tuning:
	python tuning_scripts/custom_hyperparam_tuning.py

# ===== LƯU MÔ HÌNH =====

# Save best model from tuning results
save-best-model:
	python tuning_scripts/save_best_model.py

# Test predict functionality with sample data
test-predict:
	python -c "import json, requests; print(requests.post('http://localhost:5001/predict', data={'feature_data': json.dumps([{f'feature_{i}': float(i) for i in range(20)}])}).json())"

# ===== MLFLOW =====

# Khởi động lại MLflow hoàn toàn (xóa dữ liệu cũ và khởi động lại)
reset-mlflow:
	cd mlflow_scripts && ./restart_mlflow.sh

# Khởi động MLflow server
start-mlflow:
	python mlflow_scripts/run_mlflow_server.py --host 127.0.0.1 --port 5002 \
		--backend-store-uri "./mlflow_data/mlflow.db" \
		--default-artifact-root "./mlflow_data/artifacts"

# ===== RUN APPLICATION =====

# Chạy ứng dụng Flask
run-app:
	python app.py

# ===== DỌN DẸP =====

# Clean MLflow artifacts and tuning results
clean-mlflow:
	rm -rf mlruns mlflow_data/mlflow.db
	rm -rf mlartifacts
	rm -rf tuning_results/*.json

# Clean up Python artifacts
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

# Clean everything (except venv)
clean-all: clean clean-mlflow
	rm -f .DS_Store
	rm -f models/*.joblib
	rm -f models/*.json