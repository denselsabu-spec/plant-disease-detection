from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#image preprocessing transforms
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

#load training dataset
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)

#load validation dataset
val_dataset = datasets.ImageFolder(root='data/val',transform=transform)

#create dataloaders
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

val_loader = DataLoader(val_dataset,batch_size =32,shuffle=False)

#print dataset information
print("Classes:")
print(train_dataset.classes)

print("\nNumber of training images:")
print(len(train_dataset))

print("\nNumber of validation images:")
print(len(val_dataset))

#test one batch
images, labels = next(iter(train_loader))

print("\nImage batch shape:")
print(images.shape)

print("\nLabel batch shape:")
print(labels.shape)