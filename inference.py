import json
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

from model import CurrencyAutoencoder

BASE_DIR = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# IMPORTANT: these two transforms are intentionally different, and must each
# match the transform used when their respective model was trained.
# - The classifier is a fine-tuned pretrained MobileNetV2 -> needs ImageNet
#   normalization to match what it was trained with.
# - The autoencoder was trained from scratch on raw [0,1] pixels (its
#   decoder ends in Sigmoid) -> normalizing it here would break it.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

autoencoder_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Below this denomination confidence, don't trust Stage 1's guess enough to
# run Stage 2 against it -- return "not recognized" instead. Tune this
# against your own validation images; 0.85 is a reasonable starting point.
DENOMINATION_CONFIDENCE_FLOOR = 0.85
NON_CURRENCY_BUCKETS = {"_other", "other", "others", "unknown"}


class CurrencyVerifier:
    def __init__(self,
                 denom_model_path=None,
                 class_names_path=None,
                 thresholds_path=None,
                 autoencoder_dir=None):

        if denom_model_path is None:
            denom_model_path = BASE_DIR / "denomination_classifier.pth"
        if class_names_path is None:
            class_names_path = BASE_DIR / "class_names.txt"
        if thresholds_path is None:
            thresholds_path = BASE_DIR / "models" / "thresholds.json"
        if autoencoder_dir is None:
            autoencoder_dir = BASE_DIR / "models"

        self.base_dir = BASE_DIR
        self.autoencoder_dir = Path(autoencoder_dir)

        # --- Stage 1: denomination classifier ---
        with open(class_names_path, encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.denom_model = mobilenet_v2(weights=None)
        self.denom_model.classifier[1] = nn.Linear(
            self.denom_model.last_channel, len(self.class_names)
        )
        self.denom_model.load_state_dict(torch.load(str(denom_model_path), map_location=device))
        self.denom_model.to(device).eval()

        # --- Stage 2: per-class autoencoders + thresholds ---
        with open(thresholds_path, encoding="utf-8") as f:
            self.stats = json.load(f)   # class_name -> {threshold, model_path, ...}

        for entry in self.stats.values():
            if "model_path" in entry:
                normalized = str(entry["model_path"]).replace("\\", "/")
                entry["model_path"] = str((self.base_dir / Path(normalized)).resolve())

        self.autoencoders = {}          # lazy-loaded on first use

    def _load_autoencoder(self, class_name):
        if class_name not in self.autoencoders:
            model = CurrencyAutoencoder().to(device)
            state = torch.load(self.stats[class_name]["model_path"], map_location=device)
            model.load_state_dict(state)
            model.eval()
            self.autoencoders[class_name] = model
        return self.autoencoders[class_name]

    @staticmethod
    def _normalize_class_name(class_name):
        return str(class_name).strip()

    def is_supported_class(self, class_name):
        normalized = self._normalize_class_name(class_name)
        return normalized not in NON_CURRENCY_BUCKETS and normalized in self.stats

    def should_continue_with_prediction(self, class_name, confidence):
        normalized = self._normalize_class_name(class_name)
        if normalized in NON_CURRENCY_BUCKETS:
            return False
        if confidence < DENOMINATION_CONFIDENCE_FLOOR:
            return False
        return True

    def preload_all(self):
        """Optional: load every autoencoder up front (e.g. at app startup)
        instead of on first request, so first-use latency doesn't spike."""
        for class_name in self.stats:
            self._load_autoencoder(class_name)

    @torch.no_grad()
    def verify(self, image_path):
        image = Image.open(image_path).convert("RGB")

        # Stage 1: what is it?
        classifier_tensor = classifier_transform(image).unsqueeze(0).to(device)
        logits = self.denom_model(classifier_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        denom_confidence = float(probs[0, pred_idx].item())
        predicted_class = self.class_names[pred_idx]

        if self._normalize_class_name(predicted_class) in NON_CURRENCY_BUCKETS:
            return {
                "predicted_class": predicted_class,
                "denomination_confidence": round(denom_confidence, 4),
                "verdict": "not recognized",
                "reason": "The classifier matched a non-currency bucket instead of a supported denomination.",
            }

        if not self.should_continue_with_prediction(predicted_class, denom_confidence):
            return {
                "predicted_class": predicted_class,
                "denomination_confidence": round(denom_confidence, 3),
                "verdict": "not recognized",
                "reason": (
                    f"Confidence {denom_confidence:.2f} is below the "
                    f"{DENOMINATION_CONFIDENCE_FLOOR} floor -- likely not a "
                    "currency image, or an unclear/unsupported photo."
                ),
            }

        if not self.is_supported_class(predicted_class):
            return {
                "predicted_class": predicted_class,
                "denomination_confidence": round(denom_confidence, 4),
                "verdict": "unknown",
                "reason": f"No trained authenticity model for '{predicted_class}' yet",
            }

        # Stage 2: does it look genuine for that class?
        autoencoder_tensor = autoencoder_transform(image).unsqueeze(0).to(device)
        autoencoder = self._load_autoencoder(predicted_class)
        recon = autoencoder(autoencoder_tensor)
        error = torch.mean((recon - autoencoder_tensor) ** 2).item()
        threshold = self.stats[predicted_class]["threshold"]

        is_genuine = error <= threshold
        margin = (threshold - error) / threshold if threshold > 0 else 0.0

        return {
            "predicted_class": predicted_class,
            "denomination_confidence": round(denom_confidence, 4),
            "reconstruction_error": round(error, 6),
            "threshold": round(threshold, 6),
            "verdict": "likely genuine" if is_genuine else "flagged - inspect further",
            "margin": round(margin, 4),
        }


if __name__ == "__main__":
    import sys
    verifier = CurrencyVerifier()
    verifier.preload_all()

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    result = verifier.verify(image_path)
    print(json.dumps(result, indent=2))