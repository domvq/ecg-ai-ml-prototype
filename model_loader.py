
import os
import torch

from ecg_model import ECGClassifier


MODEL_PATH = os.path.join(
    "models",
    "ecg_classifier.pt"
)


def load_model():
    """
    Load the trained ECG research model.

    Returns only the PyTorch model so it is
    compatible with the existing app.py.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    num_classes = int(
        checkpoint["num_classes"]
    )

    model = ECGClassifier(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model

