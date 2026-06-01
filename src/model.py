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

#unfreeze layer 4 of the imagenet bc this layyer
#has th highest_level image features
for param in model.layer4.parameters():
    param.requires_grad = True

#verify trainable parameters
print("Trainable Layer:\n")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

#Count trainable parameters
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"\nTrainable Parameters: {trainable_params:,}")


#replace final fully connect layer
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

#unfreeze final layer
for param in model.fc.parameters():
    param.requires_grad=True
