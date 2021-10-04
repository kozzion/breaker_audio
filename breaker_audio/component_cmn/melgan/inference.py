from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile
import argparse
import librosa
import torch
import numpy as np
import traceback
import time
import torch

from breaker_audio.component_cmn.melgan.mel2wav.interface import MelVocoder, get_default_device


_melgan_vocoder = None


def load_vocoder_melgan(load_path):
    global _melgan_vocoder
    if _melgan_vocoder is None:
        _melgan_vocoder = MelVocoder(load_path, github=True)
    return _melgan_vocoder


def infer_waveform_melgan(mel, load_path=None):
    global _melgan_vocoder
    if _melgan_vocoder is None:
        _melgan_vocoder = MelVocoder(load_path, github=True)

    mel = torch.from_numpy(mel[np.newaxis].astype(np.float32))
    wav = _melgan_vocoder.inverse(mel).squeeze().cpu().numpy()
    return wav


_net_generator = None


def mel2wav_melgan(mel, load_path=None, device=get_default_device()):
    global _net_generator
    if _net_generator is None:
        _net_generator = torch.load(load_path, map_location=device)
    with torch.no_grad():
        return _net_generator(mel.to(device)).squeeze(1)


def save_model(model: MelVocoder, outpath):
    torch.save(model.mel2wav_model, outpath)


if __name__ == "__main__":
    print(__file__)
