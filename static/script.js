let recordButton = document.getElementById("record");
let stopButton = document.getElementById("stop");
let statusText = document.getElementById("status");
let predictionText = document.getElementById("prediction");

let mediaRecorder;
let audioChunks = [];

recordButton.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
        audioChunks.push(e.data);
    }

    mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("audio_data", blob, "recording.wav");

        statusText.innerText = "Uploading...";
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        predictionText.innerText = data.prediction;
        statusText.innerText = "Done!";
    }

    mediaRecorder.start();
    recordButton.disabled = true;
    stopButton.disabled = false;
    statusText.innerText = "Recording...";
}

stopButton.onclick = () => {
    mediaRecorder.stop();
    recordButton.disabled = false;
    stopButton.disabled = true;
}
