#!/bin/bash

# Set variables
ECR_REGISTRY_URI=994231256807.dkr.ecr.eu-central-1.amazonaws.com
REPO_NAME=streaming_pipeline
LOCAL_IMAGE_NAME=streaming_pipeline
AWS_PROFILE=default
AWS_REGION=eu-central-1

# Create the ECR repository (if it doesn't exist)
aws ecr describe-repositories --repository-names ${REPO_NAME} || aws ecr create-repository \
  --repository-name $REPO_NAME \
  --profile $AWS_PROFILE \
  --region $AWS_REGION

# Authenticate Docker to the ECR registry
aws ecr get-login-password \
  --region $AWS_REGION \
  --profile $AWS_PROFILE | docker login --username AWS --password-stdin $ECR_REGISTRY_URI

# Tag the local Docker image
docker tag $LOCAL_IMAGE_NAME:latest $ECR_REGISTRY_URI/$REPO_NAME:latest

# Push the Docker image to ECR
docker push $ECR_REGISTRY_URI/$REPO_NAME:latest
