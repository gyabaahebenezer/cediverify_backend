import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import CurrencyAutoencoder

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data ---
# ImageFolder expects data/genuine/<class_name>/*.jpg (class_name can be
# denomination, e.g. "GHS_20" -- irrelevant to the autoencoder, but keeps
# things organized. We only use the images, not the folder labels.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),        # light augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder("data/genuine", transform=transform)
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

# --- Model / loss / optimizer ---
model = CurrencyAutoencoder().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(loader):
    model.train()
    total_loss = 0.0
    for images, _ in loader:               # ignore folder labels
        images = images.to(device)
        recon = model(images)
        loss = loss_fn(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate(loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            recon = model(images)
            loss = loss_fn(recon, images)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    epochs = 30
    best_val = float("inf")
    for epoch in range(epochs):
        train_loss = train_epoch(train_loader)
        val_loss = validate(val_loader)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "currency_autoencoder.pth")

    # --- Establish the anomaly threshold from genuine reconstruction errors ---
    # Collect per-image error on validation set to set a cutoff.
    model.eval()
    errors = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            recon = model(images)
            per_image_err = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
            errors.extend(per_image_err.cpu().tolist())

    import statistics
    mean_err = statistics.mean(errors)
    std_err = statistics.stdev(errors)
    threshold = mean_err + 3 * std_err   # 3-sigma cutoff, tune this
    print(f"Suggested anomaly threshold: {threshold:.5f}")
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))