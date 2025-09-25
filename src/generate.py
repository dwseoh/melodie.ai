import torch
import torch.nn as nn
import music21 as m21
import os
from train import MelodyLSTM, checkpoint_path, device

# =======================
# 1. Generation function
# =======================
def generate_melody(start_note=60, length=100, temperature=1.0, output_file="generated.mid"):
    """
    Generate a melody using the trained model.
    - start_note: MIDI pitch integer (60 = Middle C)
    - length: how many notes to generate
    - temperature: randomness (higher = more creative, lower = safer)
    """
    # Load model
    model = MelodyLSTM().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Start sequence
    input_seq = torch.tensor([[start_note]], dtype=torch.long).to(device)
    generated = [start_note]

    hidden = None
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)

            # Sample from distribution
            next_note = torch.multinomial(probabilities, num_samples=1).item()

            generated.append(next_note)
            input_seq = torch.tensor([[next_note]], dtype=torch.long).to(device)

    # Convert to MIDI
    stream = m21.stream.Stream()
    for pitch in generated:
        if pitch == 0:
            n = m21.note.Rest(quarterLength=0.25)
        else:
            n = m21.note.Note(pitch, quarterLength=0.25)
        stream.append(n)

    stream.write("midi", fp=output_file)
    print(f"🎶 Generated melody saved as {output_file}")


# =======================
# 2. Run script
# =======================
if __name__ == "__main__":
    generate_melody(start_note=60, length=100, temperature=1.0, output_file="output.mid")
