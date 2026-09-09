import json
import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image, ImageOps

from model import CurrencyAutoencoder

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
# against your own validation images; 0.50 is a reasonable starting point.
DENOMINATION_CONFIDENCE_FLOOR = 0.50


class CurrencyVerifier:
    def __init__(self,
                 denom_model_path="denomination_classifier.pth",
                 class_names_path="class_names.txt",
                 thresholds_path="models/thresholds.json",
                 autoencoder_dir="models"):

        # --- Stage 1: denomination classifier ---
        with open(class_names_path) as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.denom_model = mobilenet_v2(weights=None)
        self.denom_model.classifier[1] = nn.Linear(
            self.denom_model.last_channel, len(self.class_names)
        )
        self.denom_model.load_state_dict(torch.load(denom_model_path, map_location=device))
        self.denom_model.to(device).eval()

        # --- Stage 2: per-class autoencoders + thresholds ---
        with open(thresholds_path) as f:
            self.stats = json.load(f)   # class_name -> {threshold, model_path, ...}

        self.autoencoders = {}          # lazy-loaded on first use
        self.autoencoder_dir = autoencoder_dir

    def _load_autoencoder(self, class_name):
        if class_name not in self.autoencoders:
            model = CurrencyAutoencoder().to(device)
            # Reconstruct the path from the known naming convention rather
            # than trusting the "model_path" field in thresholds.json --
            # that field was written using whatever OS ran training (e.g.
            # Windows backslashes), which breaks when this JSON is deployed
            # to a Linux server. os.path.join here uses the *server's* path
            # rules, so this works regardless of the training machine's OS.
            model_path = os.path.join(self.autoencoder_dir, f"{class_name}_autoencoder.pth")
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            self.autoencoders[class_name] = model
        return self.autoencoders[class_name]

    def preload_all(self):
        """Optional: load every autoencoder up front (e.g. at app startup)
        instead of on first request, so first-use latency doesn't spike."""
        for class_name in self.stats:
            self._load_autoencoder(class_name)

    @torch.no_grad()
    def verify(self, image_path):
        image = Image.open(image_path)
        original_size = image.size  # (width, height) before any correction -- useful for debugging
        had_exif_orientation = image.getexif().get(0x0112, 1) != 1  # 0x0112 = Orientation tag

        image = ImageOps.exif_transpose(image)   # rotate/flip based on EXIF, THEN treat as canonical
        image = image.convert("RGB")

        # Stage 1: what is it?
        classifier_tensor = classifier_transform(image).unsqueeze(0).to(device)
        logits = self.denom_model(classifier_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        denom_confidence = probs[0, pred_idx].item()
        predicted_class = self.class_names[pred_idx]

        if denom_confidence < DENOMINATION_CONFIDENCE_FLOOR:
            return {
                "predicted_class": predicted_class,
                "denomination_confidence": round(denom_confidence, 4),
                "verdict": "not recognized",
                "reason": (
                    f"Confidence {denom_confidence:.2f} is below the "
                    f"{DENOMINATION_CONFIDENCE_FLOOR} floor -- likely not a "
                    "currency image, or an unclear/unsupported photo."
                ),
                "debug_original_size": original_size,
                "debug_had_exif_orientation": had_exif_orientation,
            }

        if predicted_class == "_other":
            return {
                "predicted_class": predicted_class,
                "denomination_confidence": round(denom_confidence, 4),
                "verdict": "not_currency",
                "reason": "This doesn't appear to be currency.",
                "debug_original_size": original_size,
                "debug_had_exif_orientation": had_exif_orientation,
            }

        if predicted_class not in self.stats:
            return {
                "predicted_class": predicted_class,
                "denomination_confidence": round(denom_confidence, 4),
                "verdict": "unsupported_denomination",
                "reason": (
                    f"Recognized as '{predicted_class}', but authenticity "
                    "checking isn't available for this denomination yet."
                ),
                "debug_original_size": original_size,
                "debug_had_exif_orientation": had_exif_orientation,
            }

        # Stage 2: does it look genuine for that class?
        autoencoder_tensor = autoencoder_transform(image).unsqueeze(0).to(device)
        autoencoder = self._load_autoencoder(predicted_class)
        recon = autoencoder(autoencoder_tensor)
        error = torch.mean((recon - autoencoder_tensor) ** 2).item()
        threshold = self.stats[predicted_class]["threshold"]

        is_genuine = error <= threshold
        # how far past/under the threshold, as a rough confidence signal
        margin = (threshold - error) / threshold if threshold > 0 else 0.0

        return {
            "predicted_class": predicted_class,
            "denomination_confidence": round(denom_confidence, 4),
            "reconstruction_error": round(error, 6),
            "threshold": round(threshold, 6),
            "verdict": "likely genuine" if is_genuine else "flagged - inspect further",
            "margin": round(margin, 4),
            "debug_original_size": original_size,
            "debug_had_exif_orientation": had_exif_orientation,
        }


if __name__ == "__main__":
    import sys
    verifier = CurrencyVerifier()
    verifier.preload_all()

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    result = verifier.verify(image_path)
    print(json.dumps(result, indent=2))