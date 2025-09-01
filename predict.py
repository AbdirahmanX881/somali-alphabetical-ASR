import torch
import torch.nn as nn
import numpy as np
import librosa
import sys

# ---------- Somali Alphabet ----------
labels = [
    'b','t','j','x','kh','d','r','s','sh','dh',
    'c','g','f','q','k','l','m','n','w','h','y',
    'a','e','i','o','u'
]

label2idx = {lab: i for i, lab in enumerate(labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

# ---------- Neural Network (same as training) ----------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ---------- Load Model ----------
model = SimpleNN(input_dim=13, hidden_dim=64, num_classes=len(labels))
model.load_state_dict(torch.load("somali_alphabet_asr.pth"))
model.eval()

# ---------- Load and preprocess audio ----------
wav_path = sys.argv[1]  # provide the path when running the script
y, sr = librosa.load(wav_path, sr=16000, mono=True)

# Extract MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)
input_tensor = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0)  # add batch dim

# ---------- Predict ----------
with torch.no_grad():
    output = model(input_tensor)
    pred_idx = torch.argmax(output, dim=1).item()
    pred_label = idx2label[pred_idx]

print(f"Predicted Somali letter: {pred_label}")
