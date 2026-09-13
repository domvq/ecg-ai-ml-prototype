
import os

import numpy as np
import torch

from model_loader import load_model


MODEL_PATH = os.path.join(
    "models",
    "ecg_classifier.pt"
)

TARGET_LEADS = 12
TARGET_LENGTH = 1000


def prepare_signal(signal):
    """
    Convert extracted ECG waveforms into
    a fixed 12 x 1000 numeric array.
    """

    leads = []

    for lead in signal:

        values = np.asarray(
            lead,
            dtype=np.float32
        ).reshape(-1)

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        if len(values) >= TARGET_LENGTH:

            values = values[:TARGET_LENGTH]

        else:

            padded = np.zeros(
                TARGET_LENGTH,
                dtype=np.float32
            )

            padded[:len(values)] = values

            values = padded

        leads.append(values)

    while len(leads) < TARGET_LEADS:

        leads.append(
            np.zeros(
                TARGET_LENGTH,
                dtype=np.float32
            )
        )

    leads = leads[:TARGET_LEADS]

    return np.stack(
        leads,
        axis=0
    )


def predict(model, signal):
    """
    Run the ECG research classifier.

    This is a prototype and is NOT
    clinically validated.
    """

    if model is None:
        model = load_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    prepared = prepare_signal(
        signal
    )

    tensor = torch.from_numpy(
        prepared
    ).unsqueeze(0)

    if tuple(tensor.shape) != (
        1,
        TARGET_LEADS,
        TARGET_LENGTH
    ):
        raise ValueError(
            f"Invalid ECG tensor shape: "
            f"{tuple(tensor.shape)}"
        )

    model.eval()

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    label_to_index = checkpoint.get(
        "label_to_index",
        {}
    )

    index_to_label = {
        int(index): label
        for label, index
        in label_to_index.items()
    }

    results = []

    for i in range(
        probabilities.shape[1]
    ):

        finding = index_to_label.get(
            i,
            f"class_{i}"
        )

        probability = float(
            probabilities[
                0,
                i
            ].item()
        )

        results.append(
            {
                "finding": finding,
                "probability": probability
            }
        )

    results.sort(
        key=lambda item: item["probability"],
        reverse=True
    )

    return results

