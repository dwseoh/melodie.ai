import music21 as m21
import glob
import pickle

def load_melodies(path="data/*.mid"):
    melodies = []
    for file in glob.glob(path):
        stream = m21.converter.parse(file)

        # Get key and transpose to C major / A minor
        key = stream.analyze("key")
        i = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        transposed = stream.transpose(i)

        # Extract only melody notes
        notes = []
        for n in transposed.flat.notes:
            if isinstance(n, m21.note.Note):
                notes.append(str(n.pitch))
            elif isinstance(n, m21.note.Rest):
                notes.append("REST")
        melodies.append(notes)
    return melodies

def save_dataset(melodies, filename="dataset.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(melodies, f)

if __name__ == "__main__":
    melodies = load_melodies()
    save_dataset(melodies)
    print(f"Saved {len(melodies)} melodies")
