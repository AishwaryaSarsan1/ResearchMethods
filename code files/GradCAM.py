
import torch
import torchvision.transforms as transforms
from torchvision import models
import timm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam

def show_cam_on_image(img_path, cam):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255.0
    overlay = heatmap * 0.4 + img

    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# Example usage (load model and generate CAM)
model = timm.create_model("convnext_base", pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 3)
model.eval()

target_layer = model.stages[-1][-1].block[-1].dwconv

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = "sample_image.jpg"  # Change this to your image path
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

gradcam = GradCAM(model, target_layer)
cam = gradcam.generate(input_tensor)
show_cam_on_image(img_path, cam)
