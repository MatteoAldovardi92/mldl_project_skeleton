# Machine Learning Project Skeleton

A clean, modular deep learning project structure using PyTorch, designed with machine learning engineering best practices in mind, including out-of-the-box **Weights & Biases (wandb)** integration for experiment tracking.

## 📂 Project Structure

- `models/`: Neural network architecture definitions (e.g., `model.py` containing `CustomNet`).
- `dataset/`: Dataloader instantiation and tensor batching.
- `utils/`: Core utilities such as data transforms and dataset loading logic (`data.py`).
- `data/`: Standard directory to store the raw datasets (like Tiny ImageNet).
- `checkpoints/`: Directory where trained model states (`.pth`) are saved for evaluation or resuming training.
- `train.py`: The main script to train the model, iterate through epochs, and validate.
- `eval.py`: Script to load a trained model checkpoint and evaluate it against a validation set.
- `run_pipeline.sh`: A shell script to load environment variables (like API keys) and start the training process.
- `secret.txt` (Not tracked by Git): A private file to store sensitive environment variables, such as API keys.

## 🚀 Getting Started

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline:**
   Before running the pipeline, make sure you configure your credentials inside `secret.txt` (if any, like dataset or git credentials). Then execute:
   ```bash
   bash run_pipeline.sh
   ```
   Or to run training manually:
   ```bash
   python train.py
   ```

## 📊 Weights & Biases (wandb) Logging

This project natively integrates with [Weights & Biases](https://wandb.ai/) to seamlessly track your machine learning experiments. Below is an overview of how `wandb` captures metrics and models throughout the lifecycle:

1. **Initialization:** 
   When starting a training run in `train.py`, `wandb.init(project="mldl-project")` prepares the logger and groups the run on the web dashboard.
   
2. **Model Graph & Gradients:**
   `wandb.watch(model, criterion, log="all", log_freq=10)` hooks into the PyTorch model and logs network topology, gradients, and parameter distributions every 10 batches.

3. **Live Metrics Tracking:** 
   Throughout the training phase, both training and validation performance are tracked dynamically:
   ```python
   wandb.log({"train_loss": train_loss, "train_acc": train_accuracy, "epoch": epoch})
   ```
   This generates interactive charts on your dashboard to monitor over-fitting or under-fitting in real time.

4. **Model Artifacts & Summaries:**
   Once training finishes, the best validation accuracy is logged using `wandb.run.summary`. Most importantly, the final PyTorch model checkpoint (`my_model.pth`) is automatically synced to the wandb cloud via `wandb.save(save_path)`, allowing you to download the trained weights directly from the platform later!
