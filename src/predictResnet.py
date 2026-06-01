import sys
import torch

from PIL import Image
from torchvision import transforms

from model import model
from data_loader import train_dataset

#device configuration 
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using dvice: {device}")

#class names
classes = train_dataset.classes

#load trained weights
model.load_state_dict(
    torch.load(
        "models/plants_disease_resnet.pth",
        map_location = device
    )
)

#move model to device
model = model.to(device)

#evaluton mode
model.eval()

#check command_line arguement
if len(sys.argv) < 2:
    print("Usage: python src/predict.py <image_path>")
    sys.exit(1)

#image path from terminal
image_path = sys.argv[1]

#image preprocessing 
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()
                                ])

#load mage
image = Image.open(image_path).convert("RGB")

#apply transforms
image = transform(image)

#move image to dimension
image = image.unsqueeze(0)

#move image to device
image = image.to(device)

#prediction
with torch.no_grad():

    outputs = model(image)

    probabilisties = torch.softmax(outputs,dim=1)

    confidence,predicted = torch.max(probabilisties,1)
    top_probs, top_indices = torch.topk(probabilisties,5)

print("\nTop 5 Predictions:")
for prob, idx in zip(top_probs[0], top_indices[0]):
    print(
        f"{classes[idx.item()]}: "
        f"{prob.item()*100:.2f}"
    )
predicted_class = classes[predicted.item()]

print(f"\nPredicted Class: {predicted_class}")
print(f"Confidence: {confidence.item()*100:.2f}%")