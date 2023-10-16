import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
import os
from data_preparation import get_wav_from_mp3

if torch.cuda.is_available():
    device = "cuda"
else:
    gitdevice = "cpu"

classifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir="pretrained_models/spkrec-ecapa-voxceleb")

voice = os.path.join('verificated_voice', 'Голос.mp3')


def person_verification(data_path: str, verificated_voice_path=voice):
    prediction_list = []
    score, prediction = classifier.verify_batch(get_wav_from_mp3(f"{data_path}"),
                                                get_wav_from_mp3(verificated_voice_path))
    return prediction.item()
