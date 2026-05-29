"""
GradCAM visualization for electrolyte defect localization.

Overlays attention heatmap on the original image to show
which regions the model focused on when making a prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ResNet-family models."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.activations = {}
        self.gradients = {}

        # Find the last convolutional layer
        target = None
        if target_layer_name:
            for name, module in model.named_modules():
                if name == target_layer_name:
                    target = module
                    break
        else:
            # Auto-find last conv layer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target = module
                    target_layer_name = name

        if target is None:
            raise ValueError("No convolutional layer found in model")

        self.target_name = target_layer_name
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations["value"] = out

    def _save_gradient(self, module, inp, out):
        self.gradients["value"] = out[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) normalized tensor
            class_idx: Target class index (None = predicted class)

        Returns:
            Heatmap (H, W) numpy array, values 0-1
        """
        # Forward
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(1).item()

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get activations and gradients
        acts = self.activations["value"][0]  # (C, H, W)
        grads = self.gradients["value"][0]   # (C, H, W)

        # Global average pooling of gradients
        weights = grads.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activations
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()

    def overlay(self, image: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay GradCAM heatmap on the original image.

        Args:
            image: Original BGR image (H, W, 3)
            heatmap: GradCAM heatmap (H, W), values 0-1
            alpha: Blend weight (0 = original, 1 = heatmap)
            colormap: OpenCV colormap

        Returns:
            Overlay image (H, W, 3) BGR
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), colormap)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay


def generate_gradcam_report(predictor, image: Image.Image,
                            save_dir: str = None) -> dict:
    """
    Full GradCAM analysis: prediction + heatmap + overlay.

    Args:
        predictor: Predictor instance
        image: PIL Image
        save_dir: Directory to save output images

    Returns:
        dict with prediction, heatmap_path, overlay_path
    """
    import os
    from src.detection.transforms import get_val_transforms

    device = predictor.device
    result = predictor.predict(image)

    # Get input tensor for GradCAM
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    # Generate GradCAM
    gradcam = GradCAM(predictor.model)
    ng_prob = result["probability"]
    target_class = 1 if ng_prob >= 0.5 else 0
    heatmap = gradcam.generate(tensor, target_class)

    # Convert PIL to numpy for overlay
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = gradcam.overlay(img_np, heatmap)

    output = {"prediction": result}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        heatmap_path = os.path.join(save_dir, "heatmap.png")
        overlay_path = os.path.join(save_dir, "overlay.png")
        cv2.imwrite(heatmap_path, (heatmap * 255).astype(np.uint8))
        cv2.imwrite(overlay_path, overlay)
        output["heatmap_path"] = heatmap_path
        output["overlay_path"] = overlay_path

    return output
