from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Pointing to the data/ folder
tiny_imagenet_dataset_train = ImageFolder(root='data/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='data/tiny-imagenet/tiny-imagenet-200/val', transform=transform)
