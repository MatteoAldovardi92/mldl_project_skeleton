import torch 

def get_model():

  class CustomNet(nn.Module):
      def __init__(self):
          super().__init__()

          self.features = nn.Sequential(
              nn.Conv2d(3, 32, kernel_size=3, padding=1),   # Input: (B, 3, 112, 112)
              nn.ReLU(),
              nn.MaxPool2d(2),                              # Output: (B, 32, 56, 56)

              nn.Conv2d(32, 64, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(2),                              # Output: (B, 64, 28, 28)

              nn.Conv2d(64, 128, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(2)                               # Output: (B, 128, 14, 14)
          )

          self.classifier = nn.Sequential(
              nn.Flatten(),
              # 128 channels * 28 * 28 spatial size
              nn.Linear(128 * 14 * 14, 256),
              nn.ReLU(),
              nn.Linear(256, 200)                           # 200 classes for Tiny ImageNet
          )

      def forward(self, x):
          x = self.features(x)
          x = self.classifier(x)
          return x

  m = CustomNet() # the function has to return an instance
  return m