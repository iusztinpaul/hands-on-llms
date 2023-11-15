#!/bin/bash

# Source the environment variables if .env file exists
if [ -f ".env" ]; then
    source .env
else
    echo ".env file does not exist."
fi

# Query the ECR registry URI.
export ECR_REGISTRY_URI=$(aws ecr describe-repositories --repository-names ${AWS_ECR_REPO_NAME} --query "repositories[?repositoryName==\`${AWS_ECR_REPO_NAME}\`].repositoryUri" --output text --region $AWS_REGION)
if [ -z "$ECR_REGISTRY_URI" ]; then
    echo "ECR_REGISTRY_URI is not set. Most probably because the AWS_ECR_REPO_NAME=${AWS_ECR_REPO_NAME} ECR repository does not exist. Exiting script."
    exit 1
fi

# Extract all variables from the template
variables=$(grep -oP '\$\{\K[^}]*' deploy/user_data_template.sh)

# Define your list of strings
allowed_variables=("ECR_REGISTRY_URI")

# Flag to indicate if all variables are set
all_set=true

# Iterate through all variables and check if they are set
for var in $variables; do
  if [[ -z "${!var}" ]]; then
    echo "Environment variable $var is not set."
    all_set=false
  fi
done

# Only run envsubst if all variables are set
if $all_set; then
  envsubst < deploy/user_data_template.sh > deploy/user_data.sh

  exit 0
else
  echo "Not all variables are set in your '.env' file. Aborting."

  exit 1
fi
