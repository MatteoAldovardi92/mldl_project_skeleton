#!/bin/bash

echo "🚀 Starting the ML pipeline..."

# Load secrets into environment variables so train.py can access them
if [ -f "secret.txt" ]; then
    export $(grep -v '^#' secret.txt | xargs)
    echo "✅ Secrets loaded."
else
    echo "⚠️ secret.txt not found. Continuing without it..."
fi

echo "🧠 Running train.py..."
python train.py

echo "✅ Pipeline finished!"
