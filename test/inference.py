
import torch
from PIL import Image
from torchvision import transforms
import timm

# Load model
model = timm.create_model("convnext_base", pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 3)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  # Adjust path
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = "sample_image.jpg"  # Change path to your test image
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax().item()

classes = ["Normal", "Adenocarcinoma", "Squamous"]
print(f"Predicted Class: {classes[predicted_class]}")
