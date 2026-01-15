import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATA_DIR = "data"
MODEL_DIR = "../Deploy/app/model"
BOARDS = [
    "Baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def load_data(input_file):
    df = pd.read_csv(input_file)
    df["tokens_list"] = df["tokens"].apply(
        lambda x: x.split(",") if pd.notna(x) and x.strip() else []
    )
    df = df[df["tokens_list"].apply(len) > 0]
    print(f"Loaded {len(df)} samples from {input_file}")
    return df


def create_label_encoder(boards):
    board_to_idx = {board: idx for idx, board in enumerate(boards)}
    idx_to_board = {idx: board for idx, board in enumerate(boards)}
    return board_to_idx, idx_to_board


def encode_labels(df, board_to_idx):
    df["label"] = df["board"].map(board_to_idx)
    return df


def extract_features(df, doc2vec_model):
    vectors = []
    for tokens in tqdm(df["tokens_list"], desc="Converting to vectors"):
        vector = doc2vec_model.infer_vector(tokens)
        vectors.append(vector)
    vectors = np.array(vectors)
    print(f"Feature extraction completed. Shape: {vectors.shape}")
    return vectors


def calculate_class_weights(y_train, num_classes):
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.FloatTensor(class_weights)


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=128):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        predictions = torch.max(outputs, 1)[1].cpu().numpy()
    return predictions


def generate_test_sample(df, y, output_file="data/test_sample.csv", sample_size=5000):
    _, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y
    )
    test_df = df.iloc[test_indices]

    sample_df = (
        test_df.groupby("board", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=max(1, int(sample_size * len(x) / len(test_df))), random_state=SEED
            )
        )
        .sample(frac=1, random_state=SEED)
    )

    sample_df[["board", "title", "tokens"]].to_csv(
        output_file, index=False, encoding="utf-8-sig"
    )
    return len(sample_df)


class ArticleClassifier(nn.Module):
    def __init__(
        self, input_size=300, hidden1=128, hidden2=64, num_classes=9, dropout=0.2
    ):
        super(ArticleClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="Evaluating", leave=False):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            _, predicted = torch.max(outputs, 1)

            total_loss += loss.item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy, all_predictions, all_labels


def plot_confusion_matrix(y_true, y_pred, idx_to_board):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[idx_to_board[i] for i in range(len(idx_to_board))],
        yticklabels=[idx_to_board[i] for i in range(len(idx_to_board))],
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Confusion matrix saved.")


if __name__ == "__main__":
    board_to_idx, idx_to_board = create_label_encoder(BOARDS)
    df = encode_labels(load_data(f"{DATA_DIR}/tokenized_titles.csv"), board_to_idx)

    doc2vec_model = Doc2Vec.load(f"{MODEL_DIR}/doc2vec_model.bin")
    X, y = extract_features(df, doc2vec_model), df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"test_sample.csv: {generate_test_sample(df, y)} samples")

    class_weights = calculate_class_weights(y_train, len(BOARDS))
    # print("Class weights:")
    # for idx, weight in enumerate(class_weights):
    #     print(f"  {idx_to_board[idx]:15s}: {weight:.4f}")
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, 128
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArticleClassifier(
        input_size=doc2vec_model.vector_size,
        hidden1=128,
        hidden2=64,
        num_classes=len(BOARDS),
        dropout=0.2,
    ).to(device)

    # print(model)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"\nTotal parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    EPOCHS, best_test_acc, best_epoch = 50, 0.0, 0
    epoch_bar = tqdm(range(EPOCHS), desc="Training Progress", unit="epoch")

    for epoch in epoch_bar:
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc, _, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        epoch_bar.set_postfix(
            {
                "Train_Loss": f"{train_loss:.4f}",
                "Train_Acc": f"{train_acc:.4f}",
                "Test_Loss": f"{test_loss:.4f}",
                "Test_Acc": f"{test_acc:.4f}",
                "Best": f"{best_test_acc:.4f}",
            }
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_classifier.pth")
            epoch_bar.write(f"Epoch {epoch+1}: New best model saved")

        scheduler.step()

    epoch_bar.close()
    print(f"\nBest accuracy: {best_test_acc:.4f} (epoch {best_epoch})")

    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/best_classifier.pth", weights_only=True)
    )
    test_loss, test_acc, y_pred, y_true = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"\nFinal Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_board[i] for i in range(len(idx_to_board))],
        digits=4,
    )
    print(report)
    plot_confusion_matrix(y_true, y_pred, idx_to_board)

    # Test on test_sample.csv
    print("\nTesting test_sample.csv...")
    sample_df = encode_labels(load_data("data/test_sample.csv"), board_to_idx)
    X_sample = extract_features(sample_df, doc2vec_model)
    y_pred_sample = predict(model, X_sample, device)
    print(
        f"Sample Accuracy: {accuracy_score(sample_df['label'].values, y_pred_sample):.4f}"
    )
