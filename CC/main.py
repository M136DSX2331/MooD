import tensorflow as tf
import numpy as np
import librosa
import os
import uvicorn
import requests
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydub import AudioSegment

new_model = tf.keras.models.load_model('./Mood.h5')
app = FastAPI()

# Konversi Audio
def convert_to_wav(input_file, output_file):
    file_ext = input_file.split('.')[-1].lower()

    if file_ext == 'mp3':
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_file, format='wav')
    elif file_ext == 'opus':
        data, sr = sf.read(input_file)
        sf.write(output_file, data, sr, format='wav')
    elif file_ext == 'wav':
        pass
    else:
        raise ValueError('Format file tidak didukung.')

#ChatGPT
def chat_with_gpt(prompt):
    # Definisikan URL endpoint API ChatGPT
    url = "https://api.openai.com/v1/chat/completions"

    # Definisikan headers dengan token autentikasi API Anda
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }

    # Definisikan payload permintaan dengan prompt yang diberikan
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    # Kirim permintaan POST ke API ChatGPT
    response = requests.post(url, headers=headers, json=payload)

    # Parse respon sebagai JSON
    response_json = response.json()

    # Ambil teks respon dari JSON
    if "choices" in response_json:
        choices = response_json["choices"]
        if choices:
            return choices[0]["message"]["content"]

    return response_json

@app.get("/")
async def index():
    content = """
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mood Detector</title>
        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            .container {
                max-width: 400px;
                margin: 50px auto;
            }

            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Mood Detector</h1>
            <form action="/result" enctype="multipart/form-data" method="post">
                <div class="form-group">
                    <label for="file">Upload Audio File</label>
                    <input type="file" class="form-control-file" id="file" name="file">
                </div>
                <button type="submit" class="btn btn-primary">Upload</button>
            </form>
            <div id="result" style="display: none;">
                <h2>Hasil:</h2>
                <div class="form-group">
                    <label for="filename">Nama File:</label>
                    <p id="filename"></p>
                </div>
                <div class="form-group">
                    <label for="emotion">Emosi:</label>
                    <p id="emotion"></p>
                </div>
                <div class="form-group">
                    <label for="recommendation">Rekomendasi:</label>
                    <p id="recommendation"></p>
                </div>
            </div>
        </div>

        <!-- Bootstrap JS -->
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script>
            const form = document.querySelector('form');
            const resultSection = document.getElementById('result');
            const filenameElement = document.getElementById('filename');
            const emotionElement = document.getElementById('emotion');
            const recommendationElement = document.getElementById('recommendation');

            form.addEventListener('submit', (event) => {
                event.preventDefault();

                const formData = new FormData(form);
                fetch('/result', {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        filenameElement.textContent = data.filename;
                        emotionElement.textContent = data.emotion;
                        recommendationElement.textContent = data.recommend;
                        resultSection.style.display = 'block';
                    })
                    .catch(error => console.error('Error:', error));
            });
        </script>
    </body>

    </html>
    """
    return HTMLResponse(content=content)

@app.post("/result")
async def upload(file: UploadFile = File(...)):
    base_path = os.path.dirname(os.path.realpath(__file__))
    upload_folder = os.path.join(base_path, "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    convert_to_wav(file.filename, "uploads/"+file.filename)

    audio_path = './uploads/'+file.filename
    dat, sr = librosa.load(audio_path)

    def stretch(dat):
        return librosa.effects.time_stretch(dat, rate=0.75)

    def pitch(dat, sr):
        return librosa.effects.pitch_shift(y=dat, sr=sr, n_steps=4)
    
    def extract_features(dat):
        result = np.array([])

        mfcc = np.mean(librosa.feature.mfcc(y=dat, sr=sr).T, axis=0) 
        result = np.hstack((result, mfcc))

        rms = np.mean(librosa.feature.rms(y=dat).T, axis=0) 
        result = np.hstack((result, rms))

        mel = np.mean(librosa.feature.melspectrogram(y=dat, sr=sr).T, axis=0) 
        result = np.hstack((result, mel))

        return result

    def get_features(path):
        dat, sr = librosa.load(path, duration=2.5, offset=0.6)
        
        res1 = extract_features(dat)
        result = np.array(res1)

        new_data = stretch(dat)
        data_stretch_pitch = pitch(new_data, sr)
        res3 = extract_features(data_stretch_pitch)
        result = np.vstack((result, res3))

        return result
    
    test = get_features(audio_path)
    predictions = new_model.predict(test)
    predicted_classes = tf.argmax(predictions, axis=1)
    predicted_classes_np = predicted_classes.numpy()
    class_mapping = {0: 'angry', 1: 'disgust', 2: 'happy', 3: 'sad'}
    predicted_labels = [class_mapping[cls] for cls in predicted_classes_np]
    os.remove('./uploads/'+file.filename)
    gpt = chat_with_gpt("Bagaimana cara menyikapi emosi "+predicted_labels[0])
    return {"filename": file.filename,
            "emotion": predicted_labels[0],
            "recommend": gpt}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)