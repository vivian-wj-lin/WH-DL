import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BLOCK_SIZE = 128
BATCH_SIZE = 64
N_EMBD = 256
N_HEAD = 8
N_LAYER = 4
DROPOUT = 0.1
EPOCHS = 30
LR = 3e-4
PATIENCE = 5

with open("data.json", encoding="utf-8") as f:
    data = json.load(f)

text = ""
for chapter in data:
    for paragraph in chapter["paragraphs"]:
        text += paragraph

chars = sorted(set(text))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [char_to_idx[ch] for ch in s]


def decode(indices):
    return "".join(idx_to_char[i] for i in indices)


class TextDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids) - BLOCK_SIZE

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + BLOCK_SIZE + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class DecoderOnlyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=N_EMBD,
            nhead=N_HEAD,
            dim_feedforward=N_EMBD * 4,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYER)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        emb = self.token_emb(x) + self.pos_emb(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        out = self.transformer(emb, mask=causal_mask, is_causal=True)
        return self.head(out)


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), y.view(B * T))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate(model, prompt, device, max_new_tokens=40, temperature=0.8):
    model.eval()
    ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        ids_cropped = ids[:, -BLOCK_SIZE:]
        logits = model(ids_cropped)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_ids = torch.tensor(encode(text), dtype=torch.long)
    split = int(len(all_ids) * 0.9)
    train_loader = DataLoader(
        TextDataset(all_ids[:split]), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(all_ids[split:]), batch_size=BATCH_SIZE, shuffle=False
    )

    model = DecoderOnlyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("開始訓練...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, loss_fn, device, train=True
        )
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        print(
            f"Epoch {epoch:2d}/{EPOCHS}  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}",
            end="",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            print()

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(torch.load("best_model.pt", weights_only=True))

    prompts = ["子曰", "君子", "仁者", "天下", "道也"]
    lines = []

    for prompt in prompts:
        result = generate(model, prompt, device, max_new_tokens=40, temperature=0.8)
        lines.append(f"[開頭：{prompt}]")
        lines.append(result)
        lines.append("")

    with open("generated.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        print(lines)


if __name__ == "__main__":
    main()
