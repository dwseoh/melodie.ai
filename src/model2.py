import torch
import torch.nn as nn

class MelodyLSTMV2(nn.Module):
    def __init__(self, pitch_size=128, duration_size=64, tempo_size=300, embed_size=64, hidden_size=128, num_layers=2):
        super(MelodyLSTMV2, self).__init__()

        self.pitch_emb = nn.Embedding(pitch_size, embed_size)
        self.duration_emb = nn.Embedding(duration_size, embed_size)
        self.rest_emb = nn.Embedding(2, embed_size)  # rest flag (0 or 1)
        self.tempo_emb = nn.Embedding(tempo_size, embed_size)

        self.lstm = nn.LSTM(embed_size * 4, hidden_size, num_layers, batch_first=True)

        self.fc_pitch = nn.Linear(hidden_size, pitch_size)
        self.fc_duration = nn.Linear(hidden_size, duration_size)
        self.fc_rest = nn.Linear(hidden_size, 2)
        self.fc_tempo = nn.Linear(hidden_size, tempo_size)

    def forward(self, x, hidden=None):
        pitch_e = self.pitch_emb(x["pitch"])
        duration_e = self.duration_emb(x["duration"])
        rest_e = self.rest_emb(x["rest"])
        tempo_e = self.tempo_emb(x["tempo"])

        combined = torch.cat([pitch_e, duration_e, rest_e, tempo_e], dim=-1)
        out, hidden = self.lstm(combined, hidden)

        return {
            "pitch": self.fc_pitch(out),
            "duration": self.fc_duration(out),
            "rest": self.fc_rest(out),
            "tempo": self.fc_tempo(out),
        }, hidden
