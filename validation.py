import librosa
import torch
from speechbrain.pretrained import SpeakerRecognition
import os
from data_preparation import get_wav_from_mp3

verificated_voice = librosa.load('verificated_voice/Голос.mp3', sr=16000)

valid_data = []

for filename in os.listdir('test_data'):
    wav, sr = librosa.load(f'test_data/{filename}', sr=16000)
    valid_data.append(wav)

classifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir="pretrained_models/spkrec-ecapa-voxceleb")

predictions = []

for wav in valid_data:
    score, pred = classifier.verify_batch(verificated_voice, wav)
    if pred.item() is True:
        predictions.append(1)
    else:
        predictions.append(0)

file = open('preds.txt', 'w')
for item in predictions:
    file.write(str(item) + "\n")
file.close()
