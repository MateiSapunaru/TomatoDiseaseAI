import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor, class_index=None):
        self.model.eval()
        self.model.zero_grad()

        input_tensor.requires_grad_(True)

        outputs = self.model(input_tensor)

        if class_index is None:
            class_index = torch.argmax(outputs, dim=1).item()

        score = outputs[0, class_index]
        score.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam

    def close(self):
        self.hook.remove()


def create_gradcam_overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    image = image.convert("RGB")
    original = np.array(image)

    cam = cv2.resize(cam, (original.shape[1], original.shape[0]))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(overlay)


def pil_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")