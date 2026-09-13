import torch
import torch.nn as nn


class ECGClassifier(nn.Module):
    """
    Research ECG classifier.

    Input:
        [batch, 12, 1000]

    Output:
        [batch, num_classes]
    """

    def __init__(self, num_classes):
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
        x = x.squeeze(-1)
        return self.classifier(x)

