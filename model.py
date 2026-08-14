import torch
from torch import nn

class CurrencyAutoencoder(nn.Module):
    """
    Convolutional autoencoder for anomaly detection.
    Trained ONLY on genuine notes. At inference, high reconstruction
    error = image doesn't look like the genuine notes it learned = flag.

    This replaces the PyTorch tutorial's flatten+linear stack with conv
    layers, since spatial texture (microprint, watermark, thread) is the
    signal we care about for currency.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 56 -> 28
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class CurrencyClassifier(nn.Module):
    """
    For LATER, once you have genuine + counterfeit examples: a transfer-
    learning classifier head on top of a frozen MobileNetV2 backbone.
    Swap to this once Option B data exists.
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        if freeze_backbone:
            for p in backbone.features.parameters():
                p.requires_grad = False
        backbone.classifier[1] = nn.Linear(backbone.last_channel, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)