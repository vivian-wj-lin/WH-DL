import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

IMG_SIZE = 28
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

TRAIN_DIR = "data/handwriting/augmented_images/augmented_images1"
TRAIN_CSV = "data/handwriting/image_labels.csv"
TEST_DIR = (
    "data/handwriting/handwritten-english-characters-and-digits/combined_folder/test"
)
MODEL_PATH = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HandwritingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.unique_labels = sorted(self.data["label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # print(f"- 圖片數量: {len(self.data)}")
        # print(f"- 類別數量: {len(self.unique_labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]

        img_name = os.path.normpath(os.path.join(self.root_dir, img_path))
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.label_to_idx[label_str]

        return image, label


class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_mapping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)

            if not os.path.isdir(label_path):
                continue

            for img_name in os.listdir(label_path):
                if img_name.lower().endswith((".png")):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_name)

        if label_mapping is not None:
            self.label_to_idx = label_mapping
        else:
            unique_labels = sorted(set(self.labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # print(f"- 圖片數量: {len(self.image_paths)}")
        # print(f"- 類別數量: {len(self.label_to_idx)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label_str = self.labels[idx]
        label = self.label_to_idx[label_str]

        return image, label


transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


train_dataset = HandwritingDataset(
    csv_file=TRAIN_CSV, root_dir=TRAIN_DIR, transform=transform
)

test_dataset = FolderDataset(
    root_dir=TEST_DIR,
    transform=transform,
    label_mapping=train_dataset.label_to_idx,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # (batch, 1, 28, 28) -> (batch, 16, 28, 28)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch, 16, 28, 28) -> (batch, 16, 14, 14)

        x = self.conv2(x)  # (batch, 16, 14, 14) -> (batch, 32, 14, 14)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch, 32, 14, 14) -> (batch, 32, 7, 7)

        x = self.flatten(x)  # (batch, 32, 7, 7) -> (batch, 1568)

        x = self.fc1(x)  # (batch, 1568) -> (batch, 128)
        x = self.relu3(x)
        x = self.fc2(x)  # (batch, 128) -> (batch, 62)

        return x


model = SimpleCNN(num_classes=len(train_dataset.unique_labels))
model = model.to(device)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n參數總數: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    avg_loss = running_loss / len(train_loader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="evaluating", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

    avg_loss = running_loss / len(test_loader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc


train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nbest acc: {best_acc:.2f}%")
