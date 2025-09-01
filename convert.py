from pydub import AudioSegment
import os

input_dir = "data"
output_dir = "processed"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".wav"):
        audio = AudioSegment.from_file(os.path.join(input_dir, file))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(os.path.join(output_dir, file), format="wav")
        print(f"Processed: {file}")
