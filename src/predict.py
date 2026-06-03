import torch
from PIL import Image
from torchvision import transforms

from src.model1 import PlantDiseaseCNN
from data_loader import train_dataset

#Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#Classes
classes = train_dataset.classes

#load model
model = PlantDiseaseCNN(num_classes=len(classes))

model.load_state_dict(
    torch.load(
        "models/plants_disease_cnn.pth",
        map_location=device
    )
)
model.to(device)

model.eval()

#Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
#load image
image = Image.open("test_leaf.jpeg").convert("RGB")

image = transform(image)

#add batch dimension
image = image.unsqueeze(0)
image = image.to(device)

#predicton
with torch.no_grad():
    outputs = model(image)

    _, predicted = torch.max(outputs,1)

    predicted_class = classes[predicted.item()]
print(f"\nPredicted Class: {predicted_class}")

probabilities = torch.softmax(outputs, dim=1)
confidence, predicted = torch.max(probabilities, 1)

print(f"Predicted Class: {classes[predicted.item()]}")
print(f"Confidence: {confidence.item() * 100:.2f}%")