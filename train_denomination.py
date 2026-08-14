import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ImageNet mean/std -- REQUIRED because MobileNetV2's pretrained weights
# expect inputs normalized this way. Skipping this significantly degrades
# the pretrained features and is a common cause of poor accuracy.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    # Kept mild: currency denominations are partly distinguished by color,
    # so aggressive color jitter can teach the model to ignore a real signal.
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Expects: data/genuine/1_cedi_coin/*.jpg, data/genuine/1_cedi_note/*.jpg, data/genuine/_other/*.jpg, ...
full_dataset = datasets.ImageFolder("data/genuine", transform=transform)
class_names = full_dataset.classes          # e.g. ['1_cedi_coin', '1_cedi_note', ...]
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze most of the backbone, but unfreeze the last couple of blocks so the
# model can adapt fine-grained features to currency (not just reuse generic
# ImageNet features as-is). Only do this once you have a reasonable amount
# of data per class (50+) -- with very little data, fully frozen is safer.
FINE_TUNE_LAST_N_BLOCKS = 2
for p in model.features.parameters():
    p.requires_grad = False
for block in list(model.features.children())[-FINE_TUNE_LAST_N_BLOCKS:]:
    for p in block.parameters():
        p.requires_grad = True

model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
# Lower LR for the fine-tuned backbone layers than for the fresh classifier
# head -- the backbone already has useful weights, it just needs nudging.
backbone_params = [p for p in model.features.parameters() if p.requires_grad]
optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},
])

def train_epoch():
    model.train()
    correct, total, total_loss = 0, 0, 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def validate():
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    best_acc = 0.0
    for epoch in range(15):
        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate()
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "denomination_classifier.pth")

    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))