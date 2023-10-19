import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
import os
from data_preparation import get_wav_from_mp3
from scipy.spatial.distance import cosine

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":device})

voice = os.path.join('verificated_voice', 'Голос.mp3')


def person_verification(data_path: str, verificated_voice_path=voice):
    emb_1 = classifier.encode_batch(get_wav_from_mp3(f"{data_path}"))
    emb_2 = classifier.encode_batch(get_wav_from_mp3(verificated_voice_path))
    proba = cosine(emb_1.squeeze(0).squeeze(0).to('cpu').numpy(), emb_2.squeeze(0).squeeze(0).to('cpu').numpy())
    if proba <= 0.5:
        preds = True
    else:
        preds = False
    return preds
