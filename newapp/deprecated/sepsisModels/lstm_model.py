# sepsis_models/lstm_model.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from .config import FEATURES, LABEL_COL, SEQ_LEN, TIME_COL
from .data_utils import encode_and_scale, patient_wise_split, create_sequences_per_stay


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out  # logits


def build_lstm_dataloaders(df, batch_size=32):
    df, scaler = encode_and_scale(df, FEATURES, LABEL_COL)
    df_train, df_val = patient_wise_split(df)

    X_train, y_train = create_sequences_per_stay(
        df_train, FEATURES, LABEL_COL, SEQ_LEN, TIME_COL
    )
    X_val, y_val = create_sequences_per_stay(
        df_val, FEATURES, LABEL_COL, SEQ_LEN, TIME_COL
    )

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


def evaluate_lstm(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels, losses = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            losses.append(loss.item())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype("float32")

    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(all_labels, preds)
    return np.mean(losses), auc, acc


def train_lstm(df, epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, scaler = build_lstm_dataloaders(df, batch_size=batch_size)
    model = LSTMClassifier(input_size=len(FEATURES)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_auc, val_acc = evaluate_lstm(model, val_loader, device)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return model, scaler
