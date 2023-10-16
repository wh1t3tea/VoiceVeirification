import torch
import librosa


def get_wav_from_mp3(dir_filename) -> torch.Tensor:
    wav_form, sample_rate = librosa.load(dir_filename, sr=16000)

    return torch.tensor(wav_form)
