import os

import numpy as np
from PIL import Image

from gradcam import AgroVisionGradCAM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "data", "class_to_index.json")
HEATMAP_MAX_SIZE = 800  # pixels, longest side of the saved heatmap

# Load once when the app starts. Afiqah's module loads the model and prepares Grad-CAM.
explainer = AgroVisionGradCAM(
    model_path=MODEL_PATH,
    class_mapping_path=CLASS_MAPPING_PATH,
)


def predict(image_path, top=3):
    """Return the top predictions and the Grad-CAM overlay (or None)."""
    result = explainer.explain(image_path)

    probs = result["probabilities"]
    best = np.argsort(probs)[::-1][:top]
    predictions = [(explainer.class_names[i], round(float(probs[i]), 2)) for i in best]

    overlay = None
    if result["has_positive_activation"]:
        overlay = Image.fromarray(result["overlay_uint8"])
        overlay.thumbnail((HEATMAP_MAX_SIZE, HEATMAP_MAX_SIZE))

    return predictions, overlay


if __name__ == "__main__":
    test_path = os.path.join(BASE_DIR, "static", "uploads", "test.jpg")
    predictions, overlay = predict(test_path)
    print(predictions)
    if overlay is not None:
        overlay.save(os.path.join(BASE_DIR, "static", "uploads", "test_heatmap.jpg"))
        print("Heatmap saved as test_heatmap.jpg")