import os
import json
import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from model import CurrencyAutoencoder

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

DATA_ROOT = "data/genuine"          # same root as train_denomination.py -- one copy of your images, no duplication
OUTPUT_ROOT = "models"              # per-class .pth + threshold files go here
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Folders that exist for Stage 1 (denomination classification) but should
# NOT get their own autoencoder -- "normal" is meaningless for a bucket of
# unrelated non-currency objects. Add any other non-currency folder names here.
SKIP_FOLDERS = {"_other"}

# How many std devs above the mean reconstruction error counts as "genuine".
# This is a direct precision/recall tradeoff:
#   higher (e.g. 3.0) -> fewer genuine notes flagged, but more fakes slip through
#   lower  (e.g. 1.5) -> catches more fakes, but flags more genuine notes too
# 3.0 (the previous default) is too permissive for a fraud-detection use
# case -- start stricter and loosen only if false-positive-on-genuine rate
# becomes a real problem in testing.
THRESHOLD_SIGMA_MULTIPLIER = 2.0

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])


class SingleClassImageDataset(Dataset):
    """Loads every image file directly inside one folder (no subfolders/labels needed)."""
    def __init__(self, folder, transform):
        self.paths = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def train_one_class(class_name, folder_path, epochs=30, min_images=8):
    dataset = SingleClassImageDataset(folder_path, transform)
    if len(dataset) < min_images:
        print(f"  Skipping '{class_name}': only {len(dataset)} images (need >= {min_images})")
        return None

    val_frac = 0.15
    val_size = max(1, int(len(dataset) * val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=min(16, train_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(16, val_size))

    model = CurrencyAutoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            recon = model(images)
            loss = loss_fn(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon = model(images)
                loss = loss_fn(recon, images)
                val_loss += loss.item() * images.size(0)
        val_loss /= val_size

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  [{class_name}] epoch {epoch+1}/{epochs} train={train_loss:.5f} val={val_loss:.5f}")

    model.load_state_dict(best_state)

    # Per-image reconstruction error on validation set -> threshold
    model.eval()
    errors = []
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            recon = model(images)
            per_image_err = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
            errors.extend(per_image_err.cpu().tolist())

    if len(errors) >= 2:
        mean_err = statistics.mean(errors)
        std_err = statistics.stdev(errors)
    else:
        mean_err = errors[0]
        std_err = mean_err * 0.5   # fallback for tiny val sets

    threshold = mean_err + THRESHOLD_SIGMA_MULTIPLIER * std_err

    model_path = os.path.join(OUTPUT_ROOT, f"{class_name}_autoencoder.pth")
    torch.save(best_state, model_path)

    return {
        "class_name": class_name,
        "num_images": len(dataset),
        "best_val_loss": best_val,
        "mean_reconstruction_error": mean_err,
        "std_reconstruction_error": std_err,
        "threshold": threshold,
        "model_path": model_path,
    }


if __name__ == "__main__":
    class_folders = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d not in SKIP_FOLDERS
    )
    print(f"Found {len(class_folders)} classes: {class_folders}")
    if SKIP_FOLDERS & set(os.listdir(DATA_ROOT)):
        skipped = SKIP_FOLDERS & set(os.listdir(DATA_ROOT))
        print(f"Skipping non-currency folder(s): {sorted(skipped)}")

    summary = {}
    for class_name in class_folders:
        print(f"\nTraining autoencoder for '{class_name}'...")
        folder_path = os.path.join(DATA_ROOT, class_name)
        result = train_one_class(class_name, folder_path)
        if result:
            summary[class_name] = result
            print(f"  -> threshold={result['threshold']:.5f}  (from {result['num_images']} images)")

    with open(os.path.join(OUTPUT_ROOT, "thresholds.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {len(summary)} models saved to '{OUTPUT_ROOT}/'.")
    print(f"Thresholds and stats saved to '{OUTPUT_ROOT}/thresholds.json'.")