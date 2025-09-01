import librosa
import numpy as np
import os

input_dir = "processed"
output_dir = "features"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".wav"):
        path = os.path.join(input_dir, file)

        # Load audio (16kHz mono from Step 2)
        y, sr = librosa.load(path, sr=16000)

        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Save as numpy file
        out_path = os.path.join(output_dir, file.replace(".wav", ".npy"))
        np.save(out_path, mfcc)

        print(f"Saved MFCC: {out_path}")
