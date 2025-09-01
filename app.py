from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from pydub import AudioSegment
from io import BytesIO

# ---------- Flask App ----------
app = Flask(__name__)
UPLOAD_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Somali Alphabet ----------
labels = [
    'b','t','j','x','kh','d','r','s','sh','dh',
    'c','g','f','q','k','l','m','n','w','h','y',
    'a','e','i','o','u'
]
label2idx = {lab: i for i, lab in enumerate(labels)}
idx2label = {i: lab for i, lab in enumerate(labels)}

# ---------- Neural Network ----------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ---------- Load model ----------
model = SimpleNN(input_dim=13, hidden_dim=64, num_classes=len(labels))
model.load_state_dict(torch.load("somali_alphabet_asr.pth"))
model.eval()

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio_data" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio_data"]

    # Load audio with pydub
    audio = AudioSegment.from_file(BytesIO(file.read()))
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Export to raw audio for librosa
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    # Load with librosa
    y, sr = librosa.load(audio_bytes, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    input_tensor = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = idx2label[pred_idx]

    return jsonify({"prediction": pred_label})

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
