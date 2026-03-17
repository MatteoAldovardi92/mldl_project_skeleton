#!/bin/bash

# Path to the secrets file
SECRET_FILE="secret.txt"

# Check if secret.txt exists
if [ ! -f "$SECRET_FILE" ]; then
    echo "$SECRET_FILE not found!"
    exit 1
fi

# Extract GITHUB_TOKEN from secret.txt
GITHUB_TOKEN=$(grep '^GITHUB_TOKEN=' "$SECRET_FILE" | cut -d '=' -f2)

if [ -z "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN not found in $SECRET_FILE!"
    exit 1
fi

# Assuming the repo belongs to MatteoAldovardi92 based on earlier cells
REPO_URL="github.com/MatteoAldovardi92/mldl_project_skeleton.git"

# Configure the git remote origin to include the token
git remote set-url origin "https://${GITHUB_TOKEN}@${REPO_URL}"

# Set dummy user details if needed for commits (change these as needed)
git config --global user.email "colab@example.com"
git config --global user.name "Colab User"

echo "Git remote configured successfully using the token from $SECRET_FILE!"
