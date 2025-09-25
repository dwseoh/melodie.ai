import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from dataset2 import MelodyDatasetV2
from model2 import MelodyLSTMV2

# =======================
# Hyperparameters
# =======================
batch_size = 4
seq_length = 50
num_epochs = 60
learning_rate = 0.001
subset_size = 500
max_grad_norm = 1.0  # gradient clipping

# loss weights
weight_pitch = 1.0
weight_duration = 0.5
weight_rest = 0.3
weight_tempo = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =======================
# Checkpoints
# =======================
CHECKPOINT_DIR = "checkpoints2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

def save_checkpoint(model, optimizer, epoch, loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, checkpoint_path)
    print(f"✅ Saved checkpoint at epoch {epoch+1}")

def load_checkpoint(model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resuming from epoch {start_epoch}")
        return start_epoch
    return 0

# =======================
# Training
# =======================
def train():
    dataset = MelodyDatasetV2(seq_length=seq_length, max_files=subset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MelodyLSTMV2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()  # ⏱️ start timing this epoch

        total_loss = 0
        for x, y in dataloader:
            x = {k: v.to(device) for k, v in x.items()}
            y = {k: v.to(device) for k, v in y.items()}

            optimizer.zero_grad()
            out, _ = model(x)

            loss = 0
            loss += weight_pitch * criterion(out["pitch"].transpose(1, 2), y["pitch"])
            loss += weight_duration * criterion(out["duration"].transpose(1, 2), y["duration"])
            loss += weight_rest * criterion(out["rest"].transpose(1, 2), y["rest"])
            loss += weight_tempo * criterion(out["tempo"].transpose(1, 2), y["tempo"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start  # ⏱️ how long one epoch took

        # ETA calculation
        epochs_left = num_epochs - (epoch + 1)
        eta_seconds = epochs_left * epoch_time
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60

        if eta_hours >= 1:
            eta_str = f"{eta_hours:.1f}h"
        elif eta_minutes >= 1:
            eta_str = f"{eta_minutes:.1f}m"
        else:
            eta_str = f"{eta_seconds:.1f}s"

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, ETA: {eta_str}")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss)

if __name__ == "__main__":
    train()
