import torch
import os
import random
import music21 as m21

from model2 import MelodyLSTMV2
from dataset2 import dur_to_idx, idx_to_dur, tempo_to_idx, idx_to_tempo, CHECKPOINT_DIR, REST_VALUES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

# =======================
# Load model
# =======================
def load_model():
    model = MelodyLSTMV2().to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found. Train the model first.")
    model.eval()
    return model

# =======================
# Generate sequence
# =======================
def generate(model, length=100, temperature=1.0, rest_bias=0.2):
    """
    rest_bias: float between 0-1, adds probability to generating a rest
    """
    # Random starting note
    pitch = random.randint(48, 72)  # C3-B4
    duration = random.choice(list(dur_to_idx.values()))
    rest = random.choice(REST_VALUES)
    tempo = random.choice(list(tempo_to_idx.values()))

    sequence = [(pitch, duration, rest, tempo)]
    hidden = None

    for _ in range(length):
        x_pitch = torch.tensor([[pitch]], dtype=torch.long, device=device)
        x_duration = torch.tensor([[duration]], dtype=torch.long, device=device)
        x_rest = torch.tensor([[rest]], dtype=torch.long, device=device)
        x_tempo = torch.tensor([[tempo]], dtype=torch.long, device=device)

        x = {"pitch": x_pitch, "duration": x_duration, "rest": x_rest, "tempo": x_tempo}
        out, hidden = model(x, hidden)

        # Probabilistic sampling with temperature
        pitch_logits = out["pitch"][:, -1, :] / temperature
        dur_logits = out["duration"][:, -1, :] / temperature
        rest_logits = out["rest"][:, -1, :] / temperature
        tempo_logits = out["tempo"][:, -1, :] / temperature

        # Apply rest bias
        rest_probs = torch.softmax(rest_logits, dim=-1)
        rest_probs[0, 1] += rest_bias         # increase rest probability
        rest_probs = rest_probs / rest_probs.sum()  # renormalize

        pitch = torch.multinomial(torch.softmax(pitch_logits, dim=-1), 1).item()
        duration = torch.multinomial(torch.softmax(dur_logits, dim=-1), 1).item()
        rest = torch.multinomial(rest_probs, 1).item()
        tempo = torch.multinomial(torch.softmax(tempo_logits, dim=-1), 1).item()

        sequence.append((pitch, duration, rest, tempo))

    return sequence

# =======================
# Convert sequence to MIDI
# =======================
def sequence_to_midi(sequence, out_file="generated2.mid"):
    stream = m21.stream.Stream()
    for pitch, dur_idx, rest, tempo_idx in sequence:
        dur_val = idx_to_dur.get(dur_idx, 1.0)
        tempo_val = idx_to_tempo.get(tempo_idx, 120)

        if rest == 1 or pitch == 0:
            n = m21.note.Rest(quarterLength=dur_val)
        else:
            n = m21.note.Note(pitch, quarterLength=dur_val)
        stream.append(n)

    # Add tempo
    stream.insert(0, m21.tempo.MetronomeMark(number=tempo_val))
    stream.write("midi", fp=out_file)
    print(f"🎶 Saved MIDI: {out_file}")

# =======================
# Main
# =======================
if __name__ == "__main__":
    model = load_model()
    seq = generate(model, length=200, temperature=1.0, rest_bias=0.2)  # 20% extra rest chance
    sequence_to_midi(seq, "generated2.mid")
