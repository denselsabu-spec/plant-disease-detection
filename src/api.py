from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO

from src.model import model

# Create FastAPI application
app = FastAPI()

# Serve frontend files (HTML, CSS, JS)
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

# -------------------------------
# Device Selection
# -------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# -------------------------------
# Load Trained Model
# -------------------------------
model.load_state_dict(
    torch.load(
        "models/plants_disease_resnet_best.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# -------------------------------
# Class Names
# -------------------------------
classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -------------------------------
# Image Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# Homepage Route
# -------------------------------
@app.get("/")
def home():
    return FileResponse("frontend/index.html")

# -------------------------------
# Prediction Route
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image bytes
    image_bytes = await file.read()

    # Convert bytes to image
    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Apply preprocessing
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Move image to device
    image = image.to(device)

    # Disable gradient calculations
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    # Return prediction as JSON
    return {
        "predicted_class":
            classes[predicted.item()],

        "confidence":
            round(confidence.item() * 100, 2)
    }