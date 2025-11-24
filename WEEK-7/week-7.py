import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MyData(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)


class RegressionNetwork(nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ClassificationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


# =======================================
print("------ Task 1 ------")
# architecture：2  → 16 (relu) → 8 (relu) → 1 (linear)

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv("gender-height-weight.csv")
df["Gender"] = df["Gender"].map({"Male": -1, "Female": 1})

indices = np.random.permutation(len(df))
df_shuffled = df.iloc[indices].reset_index(drop=True)

split_idx = int(len(df_shuffled) * 0.8)
train_df = df_shuffled[:split_idx].copy()
eval_df = df_shuffled[split_idx:].copy()

height_mean = train_df["Height"].mean()
height_std = train_df["Height"].std()
weight_mean = train_df["Weight"].mean()
weight_std = train_df["Weight"].std()

train_df["Height_std"] = (train_df["Height"] - height_mean) / height_std
eval_df["Height_std"] = (eval_df["Height"] - height_mean) / height_std

train_df["Weight_std"] = (train_df["Weight"] - weight_mean) / weight_std
eval_df["Weight_std"] = (eval_df["Weight"] - weight_mean) / weight_std

xs_train = train_df[["Gender", "Height_std"]].values
es_train = train_df["Weight_std"].values.reshape(-1, 1)

xs_eval = eval_df[["Gender", "Height_std"]].values
es_eval = eval_df["Weight_std"].values.reshape(-1, 1)

train_dataset = MyData(xs_train, es_train)
eval_dataset = MyData(xs_eval, es_eval)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model = RegressionNetwork()
loss_fn = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 500

print("------ Start Training ------")
progress_bar = tqdm(range(epochs), desc="Training")

for epoch in progress_bar:
    model.train()
    total_loss = 0

    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_features)

    avg_loss = total_loss / len(train_dataset)
    progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})

print("------ Start Evaluating ------")
model.eval()

train_loss_sum = 0
with torch.no_grad():
    for batch_features, batch_targets in train_loader:
        outputs = model(batch_features)
        loss = loss_fn(outputs, batch_targets)
        train_loss_sum += loss.item() * len(batch_features)

train_mse = train_loss_sum / len(train_dataset)
train_rmse = np.sqrt(train_mse) * weight_std

eval_loss_sum = 0
with torch.no_grad():
    for batch_features, batch_targets in eval_loader:
        outputs = model(batch_features)
        loss = loss_fn(outputs, batch_targets)
        eval_loss_sum += loss.item() * len(batch_features)

eval_mse = eval_loss_sum / len(eval_dataset)
eval_rmse = np.sqrt(eval_mse) * weight_std

print(f"Training Set:")
print(f"RMSE: {train_rmse:.2f} lbs")

print(f"Evaluation Set:")
print(f"RMSE: {eval_rmse:.2f} lbs")

# =======================================
print("------ Task 2 ------")
np.random.seed(42)
torch.manual_seed(42)
# Survived: 0 = No, 1 = Yes
# architecture: 6 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)

df = pd.read_csv("titanic.csv")
# print(f"Missing Age count: {df['Age'].isna().sum()}")
df = df.dropna(subset=["Age"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

indices = np.random.permutation(len(df))
split_idx = int(len(df) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

df_train = df.iloc[train_indices].copy()
df_test = df.iloc[test_indices].copy()

age_mean = df_train["Age"].mean()
age_std = df_train["Age"].std()
fare_mean = df_train["Fare"].mean()
fare_std = df_train["Fare"].std()

features_train = df_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
features_train["Age"] = (features_train["Age"] - age_mean) / age_std
features_train["Fare"] = (features_train["Fare"] - fare_mean) / fare_std

xs_train = features_train.values
es_train = df_train["Survived"].values.reshape(-1, 1)

features_test = df_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
features_test["Age"] = (features_test["Age"] - age_mean) / age_std
features_test["Fare"] = (features_test["Fare"] - fare_mean) / fare_std

xs_test = features_test.values
es_test = df_test["Survived"].values.reshape(-1, 1)

train_dataset = MyData(xs_train, es_train)
test_dataset = MyData(xs_test, es_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ClassificationNetwork()
loss_fn = nn.BCELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 350

print("------ Start Training ------")
progress_bar = tqdm(range(epochs), desc="Training")

for epoch in progress_bar:
    model.train()
    total_loss = 0

    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_features)

    avg_loss = total_loss / len(train_dataset)
    progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})

print("------ Start Evaluating ------")
model.eval()
threshold = 0.5

train_correct = 0
with torch.no_grad():
    for batch_features, batch_targets in train_loader:
        outputs = model(batch_features)
        predictions = (outputs > threshold).float()
        train_correct += (predictions == batch_targets).sum().item()

train_accuracy = train_correct / len(train_dataset)

test_correct = 0
with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        outputs = model(batch_features)
        predictions = (outputs > threshold).float()
        test_correct += (predictions == batch_targets).sum().item()

test_accuracy = test_correct / len(test_dataset)

print(f"Training Set:")
print(f"Correct: {train_correct}/{len(train_dataset)}")
print(f"Accuracy: {train_accuracy * 100:.2f}%")

print(f"Test Set:")
print(f"Correct: {test_correct}/{len(test_dataset)}")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
