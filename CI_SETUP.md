# Continuous Integration (CI/CD) Setup with CircleCI

This project utilizes CircleCI to automate the process of building and deploying Docker images to Docker Hub whenever changes are pushed to the `main` branch.

## CI/CD Structure

- **GitHub Repository**: [fistdat/MSE-DDM501](https://github.com/fistdat/MSE-DDM501)
- **Docker Hub Repository**: [fistdat/mlops-flask](https://hub.docker.com/repository/docker/fistdat/mlops-flask/general)
- **CI/CD Service**: CircleCI

## Setup Instructions

### 1. Create a CircleCI Account

1. Register for an account at [CircleCI](https://circleci.com/) using your GitHub account
2. Connect CircleCI to your GitHub repository (fistdat/MSE-DDM501)

### 2. Configure Docker Hub Credentials in CircleCI

1. Log in to CircleCI
2. Select Organization Settings (or Project Settings)
3. Select "Contexts"
4. Create a new context named "docker-hub-credentials"
5. Add two environment variables:
   - DOCKER_USERNAME: your Docker Hub username
   - DOCKER_PASSWORD: your password or access token (using a Personal Access Token is recommended)

### 3. CircleCI Configuration File

The project has been configured with a `.circleci/config.yml` file that performs the following steps:

- Checks out code from GitHub
- Builds a Docker image using Dockerfile.flask
- Tags the image with version information
- Pushes the image to Docker Hub with appropriate tags

## CI/CD Workflow

1. Developer pushes code to the `main` branch of the GitHub repository
2. CircleCI automatically detects the change and triggers the workflow
3. CircleCI builds the Docker image based on Dockerfile.flask
4. CircleCI logs in to Docker Hub using the configured credentials
5. The image is pushed to Docker Hub under two tags:
   - `fistdat/mlops-flask:latest`
   - `fistdat/mlops-flask:v1.X` (X is the CircleCI build number)

## Monitoring and Verification

- Monitor the build process on the CircleCI Dashboard
- Verify that the image has been pushed to Docker Hub
- Use the command `docker pull fistdat/mlops-flask:latest` to pull the latest image

## Common Issues and Solutions

1. **Docker Hub Authentication Error**:
   - Verify the DOCKER_USERNAME and DOCKER_PASSWORD in the context
   - Ensure the access token is still valid

2. **Docker Build Error**:
   - Check the logs in CircleCI
   - Ensure the Dockerfile has no errors

3. **Workflow Not Triggered**:
   - Verify the branch filters configuration
   - Make sure you're pushing to the `main` branch

## Future Improvements

- Add automated testing before building
- Add security scanning for Docker images
- Integrate notifications via Slack/Email for build success/failure 