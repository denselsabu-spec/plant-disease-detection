
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import model
from data_loader import train_loader, val_loader

#device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#move model to device
model = model.to(device)

#loss function
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

#store metrics
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

#number of epochs
num_epochs = 15

for epoch in range(num_epochs):

    #training mode
    model.train()

    running_train_loss = 0.0

    correct_train = 0
    total_train = 0

    for images, labels in train_loader:

        #move tensors to device
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)

        #calculate loss
        loss = criterion(outputs, labels)

        #zero gradients
        optimizer.zero_grad()

        #backpropagation
        loss.backward()

        #update weights
        optimizer.step()

        running_train_loss += loss.item()

        #training accuracy
        _, predicted = torch.max(outputs, 1)

        total_train += labels.size(0)

        correct_train += (predicted == labels).sum().item()

    #average training loss
    epoch_train_loss = running_train_loss / len(train_loader)

    #training accuracy
    epoch_train_accuracy = 100 * correct_train / total_train

    train_losses.append(epoch_train_loss)

    train_accuracies.append(epoch_train_accuracy)

    #validation mode
    model.eval()

    running_val_loss = 0.0

    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for images, labels in val_loader:

            #move tensors to device
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)

            #calculate loss
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            #validation accuracy
            _, predicted = torch.max(outputs, 1)

            total_val += labels.size(0)

            correct_val += (predicted == labels).sum().item()

    #average validation loss
    epoch_val_loss = running_val_loss / len(val_loader)

    #validation accuracy
    epoch_val_accuracy = 100 * correct_val / total_val

    val_losses.append(epoch_val_loss)

    val_accuracies.append(epoch_val_accuracy)

    #print metrics
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Train Accuracy: {epoch_train_accuracy:.2f}% "
        f"Val Loss: {epoch_val_loss:.4f} "
        f"Val Accuracy: {epoch_val_accuracy:.2f}%"
    )

#plot loss curves
plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()

plt.savefig("training_loss_resnet.png")

print("Loss curve saved")

#plot accuracy curves
plt.figure(figsize=(8,5))

plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")

plt.legend()

plt.savefig("training_accuracy_resnet.png")

print("Accuracy curve saved")

#save model weights
torch.save(
    model.state_dict(),
    "models/plants_disease_resnet.pth"
)

print("Model weights saved")

