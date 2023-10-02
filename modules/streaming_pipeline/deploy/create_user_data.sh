#!/bin/bash

# Source the environment variables
source .env

# Extract all variables from the template
variables=$(grep -oP '\$\{\K[^}]*' deploy/user_data_template.sh)

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
