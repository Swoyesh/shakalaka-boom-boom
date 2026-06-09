import os
import numpy as np
import torch
import random
import torch.nn as nn
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "quickDraw_model_new.pth")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories_new.txt")

BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
VAL_SPLIT = 0.2
NUM_WORKERS = 0

KEEP_CATEGORIES = {
    "house", "tree", "sun", "moon", "star", "snake", "fish", "elephant",
    "butterfly", "airplane", "bicycle", "guitar", "umbrella", "mushroom",
    "banana", "pizza", "key", "flower", "cloud", "mountain", "eye",
    "smiley face", "octopus", "cactus", "crown", "candle", "campfire",
    "sailboat", "cat", "car"
}

class BorderWidth():
    def __init__(self, kernel_size = 2):
        self.kernel_size = kernel_size

    def __call__(self, img):
        arr = np.array(img)
        choice = random.choice(['dilate', 'erode', 'none'])
        if choice == 'dilate':
            arr = cv2.dilate(arr, np.ones((self.kernel_size, self.kernel_size), np.uint8))
        elif choice == 'erode':
            arr = cv2.erode(arr, np.ones((self.kernel_size, self.kernel_size), np.uint8))
        else:
            return img

        return Image.fromarray(arr)


class QuickDrawDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.categories = []

        npy_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npy') and f.replace('.npy', '') in KEEP_CATEGORIES)
        for idx, fname in enumerate(npy_files):
            category = fname.replace('.npy', '')
            self.categories.append(category)
            arr = np.load(os.path.join(data_dir, fname))
            self.samples.append(arr)
            self.labels.extend([idx] * len(arr))

        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = Image.fromarray(self.samples[idx].reshape(28, 28), mode='L')
        return x, int(self.labels[idx])


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), torch.tensor(y)

# CNN Model from Scratch

class QuickDNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# EfficientNet Model

def build_model(num_classes):
    model = efficientnet_b0(weights = EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model

def train():
    print("Starting QuickDNN training...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        BorderWidth(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = QuickDrawDataset(DATA_DIR)
    num_classes = len(dataset.categories)
    print(f"Categories: {num_classes} | Samples: {len(dataset)}")

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_set = TransformSubset(train_subset, train_transform)
    val_set   = TransformSubset(val_subset, val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = QuickDNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} | val_acc={correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    with open(CATEGORIES_PATH, "w") as f:
        for cat in dataset.categories:
            f.write(cat + "\n")
    print(f"Categories saved → {CATEGORIES_PATH}")


if __name__ == "__main__":
    train()
