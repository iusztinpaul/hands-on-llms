#!/bin/bash

# Get the default VPC ID
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?IsDefault==`true`].VpcId | [0]' --output text)

# Get the first subnet ID of the default VPC
DEFAULT_SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC_ID" --query 'Subnets[0].SubnetId' --output text)

AMI_IMAGE="ami-04e601abe3e1a910f"

USER_DATA_FILE=`pwd`/deploy/user_data.sh
if [ ! -f "$USER_DATA_FILE" ]; then
  echo "User data file not found: $USER_DATA_FILE"
  exit 1
fi

# Output the values
echo "Default VPC ID: $DEFAULT_VPC_ID"
echo "First Subnet ID in Default VPC: $DEFAULT_SUBNET_ID"
echo "AMI Image ID: $AMI_IMAGE"
echo "User data file: $USER_DATA_FILE"

# Create Security Group
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name 'hands-on-llms-sg' \
    --description 'Security group for the Hands-on LLMS project.' \
    --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=hands-on-llms-sg}]' \
    --vpc-id $DEFAULT_VPC_ID | jq -r '.GroupId') 

# Print the extracted Security Group ID
echo "Created Security Group with ID: $SECURITY_GROUP_ID"

# The output will provide a Security Group ID, make note of it for the next steps
aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 443 --cidr 0.0.0.0/0

KEY_NAME='AWSHandsOnLLmsKey'
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text >  ~/.ssh/${KEY_NAME}.pem

# TODO: Surpress output
aws ec2 run-instances \
    --image-id ${AMI_IMAGE} \
    --count 1 \
    --instance-type t2.micro \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SECURITY_GROUP_ID} \
    --subnet-id ${DEFAULT_SUBNET_ID} \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=streaming-pipeline-server}]' 'ResourceType=volume,Tags=[{Key=Name,Value=demo-server-disk}]' \
    --user-data "file://${USER_DATA_FILE}"
