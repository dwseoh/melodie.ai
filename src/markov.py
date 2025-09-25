import random, collections, pickle
import music21 as m21
from pretty_midi import PrettyMIDI, Instrument, Note

def build_markov_chain(sequences, order=2):
    chain = collections.defaultdict(list)
    for seq in sequences:
        for i in range(len(seq) - order):
            prefix = tuple(seq[i:i+order])
            next_note = seq[i+order]
            chain[prefix].append(next_note)
    return chain

def generate(chain, length=50, order=2):
    start = random.choice(list(chain.keys()))
    melody = list(start)
    for _ in range(length):
        melody.append(random.choice(chain[tuple(melody[-order:])]))
    return melody

def melody_to_midi(melody, filename="results/samples/markov.mid"):
    pm = PrettyMIDI()
    piano = Instrument(program=0)
    time = 0
    for token in melody:
        if token == "REST":
            time += 0.5
            continue
        pitch = m21.pitch.Pitch(token).midi
        note = Note(velocity=80, pitch=pitch, start=time, end=time+0.5)
        piano.notes.append(note)
        time += 0.5
    pm.instruments.append(piano)
    pm.write(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    melodies = pickle.load(open("dataset.pkl", "rb"))
    chain = build_markov_chain(melodies)
    melody = generate(chain, length=100)
    melody_to_midi(melody)
