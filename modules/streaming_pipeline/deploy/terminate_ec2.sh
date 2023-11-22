#!/bin/bash

echo "Deleting roles & policies..."
aws iam detach-role-policy --role-name "EC2_ECR_Access_Role" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam remove-role-from-instance-profile --instance-profile-name "EC2_ECR_Access_Instance_Profile" --role-name "EC2_ECR_Access_Role"
aws iam delete-role --role-name "EC2_ECR_Access_Role"
aws iam delete-instance-profile --instance-profile-name "EC2_ECR_Access_Instance_Profile"
echo "Successfully deleted roles & policies..."

### Terminate EC2 instances. ###
EC2_INSTANCE_IDS=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=streaming-pipeline-server" | jq -r '.Reservations[].Instances[] | .InstanceId')
aws ec2 terminate-instances --instance-ids ${EC2_INSTANCE_IDS} > /dev/null 2>&1

# Loop until all instances are terminated
while true; do
  # Query the status of instances with the tag "Name=streaming-pipeline-server"
  STATUS=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=streaming-pipeline-server" --query "Reservations[].Instances[].State.Name" --output text)

  # Check if the result is empty, which means all instances are terminated
  if [ -z "$STATUS" ]; then
    echo "All instances are terminated."
    break
  fi

  # Check if all instances are terminated
  ALL_TERMINATED=true
  for state in $STATUS; do
    if [ "$state" != "terminated" ]; then
      ALL_TERMINATED=false
      break
    fi
  done

  if [ "$ALL_TERMINATED" = true ]; then
    echo "All instances are terminated."
    break
  else
    echo "Waiting for instances to terminate. Current statuses: $STATUS"
    sleep 10  # wait for 10 seconds before checking again
  fi
done

echo "Deleting key pair..."
aws ec2 delete-key-pair --key-name AWSHandsOnLLmsKey
echo "Successfully deleted key pair..."

### Delete security groups. ###
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
