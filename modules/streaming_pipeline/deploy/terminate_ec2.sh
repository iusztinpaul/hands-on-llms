#!/bin/bash

# Terminate EC2 instances.
EC2_INSTANCE_IDS=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=streaming-pipeline-server" | jq -r '.Reservations[].Instances[] | .InstanceId')
aws ec2 terminate-instances --instance-ids ${EC2_INSTANCE_IDS}

# Delete key pair.
aws ec2 delete-key-pair --key-name AWSHandsOnLLmsKey

# TODO: Add a while loop to wait for the instances to terminate.

# Delete security groups.
SECURITY_GROUP_IDS=$(aws ec2 describe-security-groups --query 'SecurityGroups[?GroupName==`hands-on-llms-sg`].GroupId' --output text)
# Loop through each Security Group ID and attempt to delete it
for sg_id in $SECURITY_GROUP_IDS; do
  echo "Attempting to delete Security Group ID: $sg_id..."
  
  # Attempt to delete the Security Group
  delete_output=$(aws ec2 delete-security-group --group-id "$sg_id" 2>&1)
  
  # Check if the delete was successful
  if [[ $? -eq 0 ]]; then
    echo "Successfully deleted Security Group ID: $sg_id"
  else
    echo "Failed to delete Security Group ID: $sg_id"
    echo "Error: $delete_output"
  fi
  
  echo "-----"
done
