
import os
import ast
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "ptb-xl"
CSV_PATH = os.path.join(
    DATA_DIR,
    "ptbxl_database.csv"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ecg_classifier.pt"
)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

class PTBXLPrototypeDataset(Dataset):

    def __init__(
        self,
        dataframe,
        label_to_index
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.label_to_index = (
            label_to_index
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        record_path = os.path.join(
            DATA_DIR,
            row["filename_lr"]
        )

        record = wfdb.rdrecord(
            record_path
        )

        signal = record.p_signal

        # PTB-XL low-resolution records
        # normally contain 12 leads.
        signal = signal.T

        # Fixed length.
        target_length = 1000

        if signal.shape[1] >= target_length:

            signal = signal[
                :,
                :target_length
            ]

        else:

            padded = np.zeros(
                (12, target_length),
                dtype=np.float32
            )

            padded[
                :,
                :signal.shape[1]
            ] = signal

            signal = padded

        signal = signal.astype(
            np.float32
        )

        # Per-lead normalization.
        mean = signal.mean(
            axis=1,
            keepdims=True
        )

        std = signal.std(
            axis=1,
            keepdims=True
        )

        signal = (
            signal - mean
        ) / (
            std + 1e-6
        )

        # Use the first diagnostic
        # superclass as a prototype label.
        labels = ast.literal_eval(
            row["scp_codes"]
        )

        superclass = None

        for code in labels:

            if code in self.label_to_index:

                superclass = code

                break

        if superclass is None:

            superclass = list(
                self.label_to_index
            )[0]

        target = self.label_to_index[
            superclass
        ]

        return (
            torch.tensor(signal),
            torch.tensor(target)
        )


# ============================================================
# SIMPLE 1-D CNN
# ============================================================

class ECGClassifier(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                12,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = x.squeeze(
            -1
        )

        return self.classifier(x)


# ============================================================
# LOAD METADATA
# ============================================================

print(
    "Loading PTB-XL metadata..."
)

df = pd.read_csv(
    CSV_PATH
)

# Only use records that actually
# exist locally.
available = []

for _, row in df.iterrows():

    path = os.path.join(
        DATA_DIR,
        row["filename_lr"] + ".hea"
    )

    if os.path.exists(path):

        available.append(row)


df = pd.DataFrame(
    available
).reset_index(
    drop=True
)


print(
    f"Found {len(df)} local ECG records."
)


if len(df) < 2:

    raise RuntimeError(
        "Not enough local ECG records."
    )


# ============================================================
# PROTOTYPE LABELS
# ============================================================

# For this tiny local prototype we use
# the most common available SCP codes.

counts = {}

for codes in df["scp_codes"]:

    parsed = ast.literal_eval(
        codes
    )

    for code in parsed:

        counts[code] = (
            counts.get(code, 0) + 1
        )


sorted_codes = sorted(
    counts,
    key=counts.get,
    reverse=True
)

# Keep only labels that occur at least
# twice in the tiny dataset.
labels = [
    code
    for code in sorted_codes
    if counts[code] >= 2
][:3]


if len(labels) < 2:

    raise RuntimeError(
        "The downloaded sample does not "
        "contain enough repeated diagnostic "
        "labels to train a classifier. "
        "Download more PTB-XL records."
    )


label_to_index = {
    label: index
    for index, label
    in enumerate(labels)
}


print(
    "Prototype classes:",
    label_to_index
)


# Keep only records containing one
# of our prototype labels.
selected = []

for _, row in df.iterrows():

    codes = ast.literal_eval(
        row["scp_codes"]
    )

    if any(
        code in label_to_index
        for code in codes
    ):

        selected.append(row)


df = pd.DataFrame(
    selected
).reset_index(
    drop=True
)


print(
    f"Training with {len(df)} records."
)


# ============================================================
# DATA LOADER
# ============================================================

dataset = PTBXLPrototypeDataset(
    df,
    label_to_index
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


model = ECGClassifier(
    num_classes=len(
        label_to_index
    )
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# TRAIN
# ============================================================

print()
print(
    "WARNING: This is a tiny research "
    "prototype and is NOT clinically "
    "validated."
)
print()

for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for signals, targets in loader:

        signals = signals.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            signals
        )

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += len(
            targets
        )

    accuracy = (
        correct / total
        if total
        else 0
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- loss: "
        f"{total_loss / len(loader):.4f} "
        f"- accuracy: "
        f"{accuracy:.2%}"
    )


# ============================================================
# SAVE
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "num_classes":
            len(label_to_index),

        "label_to_index":
            label_to_index,

        "input_channels":
            12,

        "input_length":
            1000,
    },
    MODEL_PATH
)


print()
print(
    f"Model saved to: {MODEL_PATH}"
)
print()
print(
    "IMPORTANT: This model is a "
    "research prototype only. It has "
    "not been clinically validated."
)

