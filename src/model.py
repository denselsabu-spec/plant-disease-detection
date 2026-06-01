import torch
import torch.nn as nn
from torchvision import models

#load pretrained ResNet18
model = models. resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

#Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad=False

#number of plant disease classes
num_classes = 15

#replace final fully connect layer
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

#unfreeze final layer
for param in model.fc.parameters():
    param.requres_grad=True
