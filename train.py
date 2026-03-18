import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Import our custom dataloader logic
from dataset.dataloader import get_dataloaders

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 200) # 200 classes for Tiny ImageNet
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) # Corrected argument order
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    wandb.log({"train_loss": train_loss, "train_acc": train_accuracy, "epoch": epoch})

def validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    wandb.log({"val_loss": val_loss, "val_acc": val_accuracy, "epoch": epoch})
    return val_accuracy

if __name__ == "__main__":
    # Initialize wandb run
    wandb.init(project="mldl-project", name="custom-net-run")

    # Setup data, model, criterion, optimizer
    train_loader, val_loader = get_dataloaders(batch_size=32, num_workers=2)
    
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    wandb.watch(model, criterion, log="all", log_freq=10)

    best_acc = 0.0
    num_epochs = 10
    
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion, epoch)
        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.run.summary["best_val_accuracy"] = best_acc

    print('Saving model...')
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/my_model.pth'
    torch.save(model.state_dict(), save_path)
    
    wandb.save(save_path)
    wandb.finish()
