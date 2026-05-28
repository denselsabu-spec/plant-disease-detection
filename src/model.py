import torch
import torch.nn as nn

class PlantDiseaseCNN(nn.Module):

    def __init__(self, num_classes):
        super(PlantDiseaseCNN,self).__init__()

        self.features = nn.Sequential(

            #Conv block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #convo Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #convo block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128*28*28,512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)

        x = self.classifier(x)

        return x