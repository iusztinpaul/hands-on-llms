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

# Create SSH key.
KEY_NAME='AWSHandsOnLLmsKey'
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text >  ~/.ssh/${KEY_NAME}.pem

# Create EC2 instance.
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ${AMI_IMAGE} \
    --count 1 \
    --instance-type t2.small \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SECURITY_GROUP_ID} \
    --subnet-id ${DEFAULT_SUBNET_ID} \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=streaming-pipeline-server}]' 'ResourceType=volume,Tags=[{Key=Name,Value=demo-server-disk}]' \
    --user-data "file://${USER_DATA_FILE}" \
    --output text --query 'Instances[0].InstanceId')

# Check if the instance ID was successfully captured
if [ -z "$INSTANCE_ID" ]; then
    echo "Failed to launch EC2 instance or capture its ID."
    exit 1
fi

echo "Launched EC2 instance with ID: $INSTANCE_ID"

# Wait for the instance to be in the 'running' state
echo "Waiting for the instance to be in the running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID

# Check if the wait command succeeded
if [ $? -eq 0 ]; then
    echo "Instance is running."
else
    echo "Failed to wait for the instance to enter the running state."
    exit 1
fi

# Create an IAM role with trust relationship for EC2 and ECR access
ROLE_NAME="EC2_ECR_Access_Role"
INSTANCE_PROFILE_NAME="EC2_ECR_Access_Instance_Profile"
TRUST_POLICY_FILE="trust_policy.json"
POLICY_ARN="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"

# Trust policy that allows EC2 to assume this role
cat > ${TRUST_POLICY_FILE} <<- EOM
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOM

# Create IAM role
aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document file://${TRUST_POLICY_FILE}

# Attach policy to role
aws iam attach-role-policy --role-name ${ROLE_NAME} --policy-arn ${POLICY_ARN}

# Create instance profile and add the role to it
aws iam create-instance-profile --instance-profile-name ${INSTANCE_PROFILE_NAME}
aws iam add-role-to-instance-profile --instance-profile-name ${INSTANCE_PROFILE_NAME} --role-name ${ROLE_NAME}

# Loop until the command is successful or the maximum number of attempts is reached
MAX_ATTEMPTS=20
ATTEMPT=1
while ! aws ec2 associate-iam-instance-profile --instance-id ${INSTANCE_ID} --iam-instance-profile Name=${INSTANCE_PROFILE_NAME}; do
    if [ ${ATTEMPT} -eq ${MAX_ATTEMPTS} ]; then
        echo "Reached maximum number of attempts. Exiting."
        exit 1
    fi

    echo "Attempt ${ATTEMPT} failed. Retrying in 10 seconds..."
    sleep 10
    ATTEMPT=$((ATTEMPT+1))
done

echo "Successfully created and attached the instance profile to your new EC2 VM."
