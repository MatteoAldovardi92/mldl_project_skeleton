#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting dataloader creationg..."
python utils/data.py

echo "Starting training..."
python train.py

echo "Training finished. Starting evaluation..."
python eval.py

echo "Pipeline complete!"
