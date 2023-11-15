#!/bin/bash

# Set env vars.
echo "export ALPACA_API_KEY=${ALPACA_API_KEY}" >> /etc/environment
echo "export ALPACA_API_SECRET=${ALPACA_API_SECRET}" >> /etc/environment
echo "export QDRANT_API_KEY=${QDRANT_API_KEY}" >> /etc/environment
echo "export QDRANT_URL=${QDRANT_URL}" >> /etc/environment

# Install Docker.
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

sudo usermod -aG docker ubuntu
newgrp docker

# Restart Docker.
sudo systemctl start docker
sudo systemctl enable docker

# Install AWS CLI.
sudo apt update
sudo apt install awscli -y

# Sleep for 60 seconds to allow the instance to fully initialize.
echo "Sleeping for 60 seconds to allow the instance to fully initialize..."
sleep 60

# Authenticate Docker to the ECR registry.
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_IMAGE_ECR_REGISTRY_URI}

# Pull Docker image from ECR.
echo "Pulling Docker image from ECR: ${DOCKER_IMAGE_ECR_REGISTRY_URI}/${AWS_ECR_REPO_NAME}:latest"
docker pull ${DOCKER_IMAGE_ECR_REGISTRY_URI}:latest

# Run Docker image.
echo "Running Docker image: ${DOCKER_IMAGE_ECR_REGISTRY_URI}/${AWS_ECR_REPO_NAME}:latest"
source /etc/environment && docker run --rm \
    -e BYTEWAX_PYTHON_FILE_PATH=tools.run_real_time:build_flow \
    -e ALPACA_API_KEY=${ALPACA_API_KEY} \
    -e ALPACA_API_SECRET=${ALPACA_API_SECRET} \
    -e QDRANT_API_KEY=${QDRANT_API_KEY} \
    -e QDRANT_URL=${QDRANT_URL} \
    --name streaming_pipeline \
    ${DOCKER_IMAGE_ECR_REGISTRY_URI}:latest
