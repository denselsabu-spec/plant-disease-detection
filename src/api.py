from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO

from src.model import model

app = FastAPI()

#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#load trained weights
model.load_state_dict(
    torch.load(
        "models/plants_disease_resnet_best.pth",
        map_location = device
    )
)

model.to(device)
model.eval()

#class name
classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

#image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.get("/")
def home():
    return{"message": "Plant Disease Detection API"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")
    

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)
        probabilities = torch.softmax(outputs,dim = 1)

        confidence, predicted = torch.max(probabilities, 1)
        
    return{
        "predicted_class":
            classes[predicted.item()],
        "confidence":
            round(confidence.item()*100,2)
    }
