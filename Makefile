.PHONY: install test lint format clean docker-build docker-run venv dev check clean-mlflow clean-all simple-tuning simple-tuning-tiny simple-tuning-gb simple-tuning-large save-best-model test-save-model reset-mlflow start-mlflow test-predict lint-html fix-html docker-compose-build docker-compose-up docker-compose-up-d docker-compose-down docker-compose-logs docker-compose-clean docker-compose-test

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

# Run tests with coverage
test:
	PYTHONPATH=. python -m unittest discover -s tests

# Run specific tests
test-app:
	PYTHONPATH=. python -m unittest tests.test_app_frontend

# Run integration tests
test-integration:
	cd tests && python test_integration.py

# Run Docker integration tests
docker-compose-test:
	cd tests && bash run_docker_tests.sh

# Run linting
lint:
	pylint app.py
	pylint tuning_scripts/*.py mlflow_scripts/*.py tests/*.py
	black --check .

# Lint HTML templates
lint-html:
	cd templates && html-minifier --lint index.html

# Fix HTML issues
fix-html:
	cd templates && html-lint --fix index.html || echo "Chạy lệnh này yêu cầu cài đặt html-lint"
	@echo "Để cài đặt html-lint, chạy: npm install -g html-lint"

# Format code
format:
	black .

# Run all checks
check: lint test

# ===== DOCKER =====

# Build Docker image
docker-build:
	docker build -t mlops-final-project .

# Run Docker container
docker-run:
	docker run -p 5001:5001 -p 5002:5002 mlops-final-project

# ===== DOCKER COMPOSE =====

docker-compose-build:
	docker compose build

docker-compose-up:
	docker compose up

docker-compose-up-d:
	docker compose up -d

docker-compose-down:
	docker compose down

docker-compose-logs:
	docker compose logs -f

docker-compose-clean:
	docker compose down -v

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

# ===== LƯU & QUẢN LÝ MÔ HÌNH =====

# Save best model from tuning results
save-best-model:
	python tuning_scripts/save_best_model.py

# Test save best model functionality
test-save-model:
	python -m unittest tuning_scripts.test_save_best_model

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

# Khôi phục experiments từ backup
restore-experiments:
	python mlflow_scripts/restore_experiments.py

# ===== RUN APPLICATION =====

# Chạy ứng dụng Flask
run-app:
	python app.py

# Chạy ứng dụng với mô hình tốt nhất
run-app-best-model:
	python app.py --use-best-model

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

# ===== QUẢN LÝ PHIÊN BẢN =====

# Gắn tag và đẩy lên GitHub
tag-version:
	@read -p "Nhập số phiên bản (v1.1, v1.2, etc): " VERSION; \
	read -p "Nhập mô tả phiên bản: " MESSAGE; \
	git tag -a $$VERSION -m "$$MESSAGE"; \
	echo "Đã tạo tag $$VERSION"; \
	echo "Để đẩy tag lên remote, chạy: git push origin $$VERSION"

# Liệt kê các tag
list-tags:
	git tag -n