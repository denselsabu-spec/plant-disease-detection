import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import PlantDiseaseCNN
from data_loader import train_loader, val_loader

#device configuration for macbook
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


print(f"Using device: {device}")

#Number of classes
num_classes = len(train_loader.dataset.classes)

#initialize model
model = PlantDiseaseCNN(num_classes=num_classes).to(device)

#loss fucnction
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#store losses
train_losses=[]
val_losses=[]

#Number of epochs
num_epochs=5

for epoch in range(num_epochs):

    #training
    model.train()

    running_train_loss=0.0

    for images,labels in train_loader:

        #Move tensors to device
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs=model(images)

        #calculate loss
        loss = criterion(outputs,labels)
        
        #Zero gradients
        optimizer.zero_grad()

        #backpropogation
        loss.backward()

        #update weights
        optimizer.step()

        running_train_loss +=loss.item()

    epoch_train_loss = running_train_loss/len(train_loader)

    train_losses.append(epoch_train_loss)


    #validation
    model.eval()

    running_val_loss =0.0
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}]"
        f"Train Loss: {epoch_train_loss:.4f}"
        f"Val Loss: {epoch_val_loss:.4f}"
    )

    #plot loss curves
    plt.figure(figsize=(8,5))

    plt.plot(train_losses,label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")

    plt.legend()

    plt.savefig("training_loss.png")
    print("loss curve saved")
    #save model weights
    torch.save(
        model.state_dict(),
        "models/plants_disease_cnn.pth"
    )

print("Model weights saved")