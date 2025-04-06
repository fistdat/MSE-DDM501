# Docker Usage Guide for MLOps Project

This project provides Docker images for both Flask application and MLflow server, making it easy to run the entire MLOps setup with minimal configuration.

## Available Images

All images are hosted on Docker Hub in the `fistdat` repository:

- **Flask Application**: `fistdat/mlops-flask`
  - Latest version: `fistdat/mlops-flask:latest`
  - Specific versions: `fistdat/mlops-flask:v1.X` (where X is the build number)

- **MLflow Server**: `fistdat/mlops-mlflow`
  - Latest version: `fistdat/mlops-mlflow:latest`
  - Specific versions: `fistdat/mlops-mlflow:v1.X` (where X is the build number)

## Running with Docker Compose (Recommended)

The simplest way to use these images is with Docker Compose:

1. Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  mlflow:
    image: fistdat/mlops-mlflow:latest
    container_name: mlops-mlflow
    ports:
      - "5002:5002"
    volumes:
      - mlflow_data:/app/mlflow_data
      - mlruns:/app/mlruns
    networks:
      - mlops-network
    restart: unless-stopped
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5002

  flask-app:
    image: fistdat/mlops-flask:latest
    container_name: mlops-flask
    ports:
      - "5001:5001"
    volumes:
      - models:/app/models
      - tuning_results:/app/tuning_results
    networks:
      - mlops-network
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5002
    restart: unless-stopped

networks:
  mlops-network:
    driver: bridge

volumes:
  mlflow_data:
  models:
  tuning_results:
  mlruns:
```

2. Start the services:

```bash
docker compose up -d
```

3. Access the applications:
   - Flask Application: http://localhost:5001
   - MLflow UI: http://localhost:5002

## Running Individual Containers

If you prefer to run containers individually:

### MLflow Server

```bash
docker run -d \
  --name mlops-mlflow \
  -p 5002:5002 \
  -e MLFLOW_TRACKING_URI=http://localhost:5002 \
  fistdat/mlops-mlflow:latest
```

### Flask Application

```bash
docker run -d \
  --name mlops-flask \
  -p 5001:5001 \
  -e MLFLOW_TRACKING_URI=http://mlops-mlflow:5002 \
  --link mlops-mlflow \
  fistdat/mlops-flask:latest
```

## Volumes and Persistence

When using Docker Compose, the following volumes are created for data persistence:
- `mlflow_data`: MLflow database and artifacts
- `mlruns`: MLflow runs
- `models`: Saved models
- `tuning_results`: Results from hyperparameter tuning

## Environment Variables

Key environment variables you can customize:

### MLflow Server
- `MLFLOW_TRACKING_URI`: URI for MLflow tracking server
- `MLFLOW_EXPERIMENT_NAME`: Default experiment name

### Flask Application
- `MLFLOW_TRACKING_URI`: URI for MLflow tracking server
- `FLASK_ENV`: Production or development mode

## Troubleshooting

1. **Connection Issues Between Containers**: 
   - Ensure they're on the same network
   - Verify that the MLflow container starts before the Flask container

2. **MLflow Connection Errors**:
   - Check that the `MLFLOW_TRACKING_URI` is correctly set to `http://mlflow:5002` in the Flask container

3. **Missing Data**:
   - Verify that volumes are properly mounted
   - Check container logs with `docker logs mlops-flask` or `docker logs mlops-mlflow`

## Updating Images

To update to the latest images:

```bash
docker pull fistdat/mlops-flask:latest
docker pull fistdat/mlops-mlflow:latest
docker compose down
docker compose up -d
```

## CI/CD Integration

These Docker images are automatically built and pushed to Docker Hub via CircleCI whenever changes are pushed to the main branch of the repository. 