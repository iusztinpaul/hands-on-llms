#!/bin/bash

# Get the default VPC ID
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?IsDefault==`true`].VpcId | [0]' --output text)

# Get the first subnet ID of the default VPC
DEFAULT_SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC_ID" --query 'Subnets[0].SubnetId' --output text)

# Define EC2 AMI Image ID
AMI_IMAGE="ami-04e601abe3e1a910f"

echo "Default VPC ID: $DEFAULT_VPC_ID"
echo "First Subnet ID in Default VPC: $DEFAULT_SUBNET_ID"
echo "AMI Image ID: $AMI_IMAGE"


# Create user data file.
USER_DATA_FILE=`pwd`/deploy/user_data.sh

./deploy/create_user_data.sh
create_user_data_result_code=$?

echo "Create user data result: $create_user_data_result_code"
if [ "$create_user_data_result_code" -eq 0 ]; then
  echo "Created user data file at $USER_DATA_FILE."
else
  echo "Could not create user data file. Aborting."
  exit 1
fi


# Create Security Group.
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name 'hands-on-llms-sg' \
    --description 'Security group for the Hands-on LLMS project.' \
    --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=hands-on-llms-sg}]' \
    --vpc-id $DEFAULT_VPC_ID | jq -r '.GroupId') 

echo "Created Security Group with ID: $SECURITY_GROUP_ID"

# The output will provide a Security Group ID, make note of it for the next steps
aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 443 --cidr 0.0.0.0/0


KEY_NAME='AWSHandsOnLLmsKey'
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text >  ~/.ssh/${KEY_NAME}.pem


aws ec2 run-instances \
    --image-id ${AMI_IMAGE} \
    --count 1 \
    --instance-type t2.micro \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SECURITY_GROUP_ID} \
    --subnet-id ${DEFAULT_SUBNET_ID} \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=streaming-pipeline-server}]' 'ResourceType=volume,Tags=[{Key=Name,Value=demo-server-disk}]' \
    --user-data "file://${USER_DATA_FILE}" > /dev/null 2>&1

echo "Created EC2 instance."
