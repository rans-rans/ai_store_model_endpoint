from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision.models as models
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from PIL import Image
import io
import base64
import json

app = FastAPI()

# Define the model architecture (same as used during training)
model = models.resnet18()
model.fc = torch.nn.Linear(
    model.fc.in_features, 10
)  # Replace num_classes with your actual number of classes

# Load the saved model weights
model.load_state_dict(torch.load("torch_model.pth"))

# Set the model to evaluation mode
model.eval()

# Define the transform to apply to the image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageRequest(BaseModel):
    image: str


def base64_to_pil_image(base64_str: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

# Testing the model on a new image
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor

def get_label_from_index(class_index, mappings_file='class_to_idx.json'):
    idx_to_class = load_class_mappings(mappings_file)
    return idx_to_class.get(class_index, "Unknown")


def load_class_mappings(file_path):
    with open(file_path, 'r') as f:
        class_to_idx = json.load(f)
    # Invert the mapping to get index-to-class
    return {v: k for k, v in class_to_idx.items()}



def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Transform the PIL image to a tensor and prepare it for the model."""
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension
    return image_tensor


@app.get("/")
def hello():
    return "done"


@app.post("/predict/")
async def predict(request: ImageRequest):
    try:
        # Convert the base64 image to a PIL image
        pil_image = base64_to_pil_image(request.image)

        # Preprocess the image
        image_tensor = preprocess_image(pil_image)

        # Run the model and get the prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)

        # Convert the prediction to a readable format
        predicted_index = predicted.item()  # Assuming your classes are 0, 1, 2, etc.
        predicted_label = get_label_from_index(predicted_index)

        return {"predicted_index": predicted_index,"label":predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
