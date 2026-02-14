import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BATCH_SIZE = 4
LEARNING_RATE = 0.005
EPOCHS = 15
SCORE_THRESHOLD = 0.4
NUM_VIS_IMAGES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "vehicles-data/r-cnn-data/vehicles_images"
OUTPUT_DIR = "output"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
RESULT_DIR = os.path.join(OUTPUT_DIR, "detection_results")
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_labels.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

CATEGORIES = ["Bus", "Car", "Motorcycle", "Pickup", "Truck"]
NUM_CLASSES = len(CATEGORIES) + 1
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


class VehicleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.grouped = self.data.groupby("filename")
        self.image_files = list(self.grouped.groups.keys())

        print(
            f"Loaded {len(self.image_files)} images with {len(self.data)} annotations"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_data = self.grouped.get_group(filename)
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for _, row in img_data.iterrows():
            boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            labels.append(CATEGORY_TO_IDX[row["class"]] + 1)  # +1 因為 0 是背景

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(data_loader, desc="Training")

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, device, score_threshold):
    model.eval()
    all_scores = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")

        for images, _ in progress_bar:
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred in predictions:
                scores = pred["scores"]
                filtered_scores = scores[scores > score_threshold]
                all_scores.extend(filtered_scores.cpu().tolist())

    if len(all_scores) > 0:
        avg_score = sum(all_scores) / len(all_scores)
    else:
        avg_score = 0.0
        print("Warning: No detections found!")

    print(f"\nTotal detections (score > {score_threshold}): {len(all_scores)}")
    print(f"Average confidence score: {avg_score:.4f}")

    return avg_score, all_scores


def visualize_predictions(model, dataset, device, num_images, score_threshold):
    model.eval()
    idx_to_category = {idx + 1: cat for idx, cat in enumerate(CATEGORIES)}
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, _ = dataset[idx]
            filename = dataset.image_files[idx]

            image_for_model = image.to(device)
            predictions = model([image_for_model])
            pred = predictions[0]

            keep = pred["scores"] > score_threshold
            boxes = pred["boxes"][keep].cpu().numpy()
            labels = pred["labels"][keep].cpu().numpy()
            scores = pred["scores"][keep].cpu().numpy()

            img_path = os.path.join(dataset.img_dir, filename)
            original_image = Image.open(img_path).convert("RGB")

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(original_image)

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

                category_name = idx_to_category.get(label, f"Class {label}")
                text = f"{category_name}: {score:.2f}"
                ax.text(
                    x1,
                    y1 - 5,
                    text,
                    bbox=dict(facecolor="red", alpha=0.5),
                    fontsize=10,
                    color="white",
                )

            ax.axis("off")
            plt.title(f"Detection Results - {filename}\nDetections: {len(boxes)}")

            result_path = os.path.join(RESULT_DIR, f"result_{i+1}.jpg")
            plt.savefig(result_path, bbox_inches="tight", dpi=150)
            plt.close()

            print(f"Saved: {result_path} ({len(boxes)} detections)")

    print(f"\nAll {len(indices)} result images saved to {RESULT_DIR}")


def create_data_loaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 轉成 Tensor 並 normalize 到 [0, 1]
        ]
    )

    print("Loading datasets...")
    train_dataset = VehicleDataset(
        csv_file=TRAIN_CSV, img_dir=TRAIN_IMG_DIR, transform=transform
    )

    test_dataset = VehicleDataset(
        csv_file=TEST_CSV, img_dir=TEST_IMG_DIR, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    return train_loader, test_loader, test_dataset


def create_model(device):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=3,  # 解凍後 3 層
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.roi_heads.nms_thresh = 0.3
    model = model.to(device)
    return model


def train_model(model, train_loader, test_loader, device):
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )

    print("Starting training...")

    best_score = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        avg_score, _ = evaluate(model, test_loader, device, SCORE_THRESHOLD)

        if avg_score > best_score:
            best_score = avg_score
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model with score: {best_score:.4f}")

    return best_score


def final_evaluation(model, test_loader, test_dataset, device):
    print("Starting evaluating")

    model.load_state_dict(torch.load(MODEL_PATH))
    final_score, all_scores = evaluate(model, test_loader, device, SCORE_THRESHOLD)
    print(f"\nBest average score: {final_score:.4f}")

    visualize_predictions(
        model,
        test_dataset,
        device,
        num_images=NUM_VIS_IMAGES,
        score_threshold=SCORE_THRESHOLD,
    )
    return final_score


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    train_loader, test_loader, test_dataset = create_data_loaders()
    model = create_model(device)

    train_model(model, train_loader, test_loader, device)
    final_evaluation(model, test_loader, test_dataset, device)
