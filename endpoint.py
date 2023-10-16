import os.path

from fastapi import FastAPI, UploadFile, File
from voice_verification import person_verification

app = FastAPI()


@app.post("/upload/")
async def upload_files(audio_to_verify: UploadFile = File(...)):
    try:
        if not (audio_to_verify.filename.endswith(('.mp3', '.wav'))):
            return {"error": "Формат файла не поддерживается"}
        if not os.path.exists('uploads'):
            # Если папка не существует, создать ее
            os.makedirs('uploads')
        to_verify = await audio_to_verify.read()
        with open('uploads/' + audio_to_verify.filename, 'wb') as f1:
            f1.write(to_verify)
            f1.close()

        audio1 = os.path.join('uploads', audio_to_verify.filename)

        preds = person_verification(audio1)
        return f"Audio {audio_to_verify.filename} - {preds}"

    except Exception as e:
        return {"error": str(e)}

