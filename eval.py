import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.metrics import f1_score
import numpy as np

# Make sure to import your model class from train.py!
# from train import CustomNet

# Placeholder for your CustomNet (replace with your actual import)
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(64 * 224 * 224, 200) # Simplified for example
    def forward(self, x):
        x = self.conv1(x).relu()
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# Define the transforms for the images
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Point to the directory where you extracted the dataset
# Note: test set often doesn't have subfolders per class by default in tiny-imagenet, 
# so you might need a custom dataset class for test, but assuming it's structured correctly:
test_dataset = ImageFolder(root='mldl_project_skeleton/data/tiny-imagenet/tiny-imagenet-200/val', transform=transform)

# Create the DataLoaders (shuffle=False for evaluation)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

model_folder_path = "/content/mldl_project_skeleton/models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F1_macro_list = []

# Make sure the directory exists before iterating
if os.path.exists(model_folder_path):
    for model_file in os.listdir(model_folder_path):
        if model_file.endswith('.pth') or model_file.endswith('.pt'):
            model_path = os.path.join(model_folder_path, model_file)
            print(f"Evaluating {model_file}...")
            
            # 1. Instantiate model and load weights
            model = CustomNet().to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval() # Set model to evaluation mode
            
            all_preds = []
            all_targets = []
            
            # 2. Evaluation loop
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # 3. Calculate metrics
            f1 = f1_score(all_targets, all_preds, average='macro')
            F1_macro_list.append((model_file, f1))
            print(f"F1 Macro for {model_file}: {f1:.4f}")
else:
    print(f"Directory {model_folder_path} does not exist yet.")

print("Final Results:", F1_macro_list)