import torch
import torch.nn as nn
from dataset.dataloader import get_dataloaders
from train import CustomNet

def evaluate_model(model_path):
    print(f"Loading model from {model_path}...")
    model = CustomNet().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, val_loader = get_dataloaders(batch_size=32, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_accuracy = 100. * correct / total
    val_loss = val_loss / len(val_loader)
    
    print(f"Final Evaluation - Loss: {val_loss:.6f}, Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model('checkpoints/my_model.pth')
