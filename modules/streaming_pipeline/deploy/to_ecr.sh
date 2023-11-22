#!/bin/bash

# Set variables
REPO_NAME=streaming_pipeline
LOCAL_IMAGE_NAME=streaming_pipeline
AWS_PROFILE=default
AWS_REGION=eu-central-1

# Create the ECR repository (if it doesn't exist)
aws ecr describe-repositories --repository-names ${REPO_NAME} || aws ecr create-repository \
  --repository-name $REPO_NAME \
  --profile $AWS_PROFILE \
  --region $AWS_REGION

# Query the ECR registry URI.
ECR_REGISTRY_URI=$(aws ecr describe-repositories --repository-names ${REPO_NAME} --query "repositories[?repositoryName==\`${REPO_NAME}\`].repositoryUri" --output text --profile $AWS_PROFILE --region $AWS_REGION)

# Authenticate Docker to the ECR registry
aws ecr get-login-password \
  --region $AWS_REGION \
  --profile $AWS_PROFILE | docker login --username AWS --password-stdin $ECR_REGISTRY_URI

# Build the Docker image.
docker build -t $LOCAL_IMAGE_NAME:latest -f deploy/Dockerfile .

# Tag the local Docker image
docker tag $LOCAL_IMAGE_NAME:latest $ECR_REGISTRY_URI/$REPO_NAME:latest

# Push the Docker image to ECR
docker push $ECR_REGISTRY_URI/$REPO_NAME:latest
