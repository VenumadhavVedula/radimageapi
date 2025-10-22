import torchxrayvision as xrv
import skimage, torch, torchvision
import numpy as np
import pydicom
import threading

# Global model cache with thread lock
_model = None
_model_lock = threading.Lock()

def _get_model():
    """Load model once and reuse across requests"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-check pattern
                _model = xrv.models.DenseNet(weights="densenet121-res224-all")
                _model.eval()  # set to evaluation mode
    return _model

def analyze_xray(image_path):
    """
    Analyzes a chest X-ray image and returns probabilities of various conditions
    
    Args:
        image_path (str): Path to the X-ray image file
        
    Returns:
        dict: Dictionary of pathology probabilities
    """
    img = skimage.io.imread(image_path)
    
    # Handle both RGB and grayscale images
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    
    img = xrv.datasets.normalize(img, 255.0)
    img = img[None, ...]  # Add channel dimension

    transformer = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
    ])

    img = transformer(img)
    img = torch.from_numpy(img)

    model = _get_model()  # reuse cached model
    with torch.no_grad():  # disable gradients for inference
        outputs = model(img[None, ...])

    return dict(zip(model.pathologies, outputs[0].detach().numpy()))

# Example usage when run as script
if __name__ == "__main__":
    results = analyze_xray("C:/Users/venuv/OneDrive/Desktop/test1.png")
    for condition, probability in results.items():
        print(f"{condition}: {probability:.3f}")