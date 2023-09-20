#!/bin/bash

# Set env vars.
echo "export ALPACA_API_KEY=${ALPACA_API_KEY}" >> /etc/environment
echo "export ALPACA_API_SECRET=${ALPACA_API_SECRET}" >> /etc/environment
echo "export QDRANT_API_KEY=${QDRANT_API_KEY}" >> /etc/environment
echo "export QDRANT_URL=${QDRANT_URL}" >> /etc/environment

# Clone the repo.
git clone https://github.com/iusztinpaul/hands-on-llms.git

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

# Clone the repo.
git clone -b sp-3-deploy-streaming-pipeline https://github.com/iusztinpaul/hands-on-llms.git ~/hands-on-llms
cd ~/hands-on-llms/modules/streaming_pipeline

# Build & run the Docker image.
sudo apt update
sudo apt install build-essential make -y

make build
source /etc/environment && make run_docker
