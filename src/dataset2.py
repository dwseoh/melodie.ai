import os
import music21 as m21

# =======================
# 1. Checkpoint folder
# =======================
CHECKPOINT_DIR = os.path.join("checkpoints2")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =======================
# 2. Duration mappings
# =======================
DURATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]  # sixteenth, eighth, quarter, half, whole
dur_to_idx = {dur: i for i, dur in enumerate(DURATIONS)}
idx_to_dur = {i: dur for i, dur in enumerate(DURATIONS)}

# =======================
# 3. Tempo mappings
# =======================
TEMPOS = [60, 80, 100, 120, 140]
tempo_to_idx = {tempo: i for i, tempo in enumerate(TEMPOS)}
idx_to_tempo = {i: tempo for i, tempo in enumerate(TEMPOS)}

# =======================
# 4. Rest mapping
# =======================
REST_VALUES = [0, 1]  # 0 = note, 1 = rest

# =======================
# 5. MIDI utilities
# =======================
def midi_to_sequence(midi_file):
    """Convert MIDI file into list of tuples: (pitch, duration, rest, tempo)"""
    try:
        score = m21.converter.parse(midi_file)
    except Exception as e:
        print(f"⚠️ Skipping {midi_file}: {e}")
        return []

    notes = []
    for el in score.flat.notes:
        if isinstance(el, m21.note.Note):
            pitch = el.pitch.midi
            dur = el.quarterLength
            dur_idx = min(DURATIONS, key=lambda x: abs(x - dur))  # closest matching duration
            notes.append((pitch, dur_to_idx[dur_idx], 0, tempo_to_idx[120]))
        elif isinstance(el, m21.note.Rest):
            dur = el.quarterLength
            dur_idx = min(DURATIONS, key=lambda x: abs(x - dur))
            notes.append((0, dur_to_idx[dur_idx], 1, tempo_to_idx[120]))
    return notes
