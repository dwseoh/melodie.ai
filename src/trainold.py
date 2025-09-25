""" import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import music21 as m21

# =======================
# 1. Hyperparameters
# =======================
batch_size = 4          # keep small to prevent crashing
seq_length = 50         # length of training chunks
num_epochs = 60        # you can run multiple times
learning_rate = 0.001
subset_size = 1000       # how many MIDI files to load per run (adjust!)

# =======================
# 2. Device setup
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =======================
# 3. MIDI utilities
# =======================
def midi_to_sequence(midi_file):
    """Convert MIDI file into list of pitch integers."""
    try:
        score = m21.converter.parse(midi_file)
    except Exception as e:
        print(f"⚠️ Skipping {midi_file}: {e}")
        return []

    notes = []
    for el in score.flat.notes:
        if isinstance(el, m21.note.Note):
            notes.append(el.pitch.midi)
        elif isinstance(el, m21.note.Rest):
            notes.append(0)  # special token for rest
    return notes

# =======================
# 4. Dataset
# =======================
class MelodyDataset(Dataset):
    def __init__(self, folder="data/", seq_length=50, max_files=None):
        self.sequences = []
        files = [f for f in os.listdir(folder) if f.endswith(".mid")]

        if max_files:
            files = files[:max_files]  # only take subset

        for file in files:
            notes = midi_to_sequence(os.path.join(folder, file))
            if len(notes) < seq_length:
                continue
            for i in range(0, len(notes) - seq_length):
                chunk = notes[i:i+seq_length+1]
                self.sequences.append(chunk)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# =======================
# 5. Model
# =======================
class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size=128, embed_size=64, hidden_size=128, num_layers=2):
        super(MelodyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# =======================
# 6. Checkpoint utils
# =======================
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

def save_checkpoint(model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"✅ Saved checkpoint at epoch {epoch+1}")

def load_checkpoint(model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔄 Resuming from epoch {start_epoch}")
        return start_epoch
    return 0

# =======================
# 7. Training
# =======================
def train():
    dataset = MelodyDataset(folder="data/", seq_length=seq_length, max_files=subset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MelodyLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss)

if __name__ == "__main__":
    train()
 """