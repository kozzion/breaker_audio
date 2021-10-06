import os
import json
import numpy as np
import torch

from aukit.audio_griffinlim import mel_spectrogram
from aukit import Dict2Obj


from breaker_audio.component.vocoder.vocoder_melgan import VocoderMelgan


class VocoderSynthesizer(VocoderMelgan):
    def __init__():
        pass


    def signal_to_melspec(self, src):
        _pad_len = (self.synthesizer_hparams.n_fft - self.synthesizer_hparams.hop_size) // 2
        wavs = src.cpu().numpy()
        mels = []
        for wav in wavs:
            wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
            mel = mel_spectrogram(wav, self.synthesizer_hparams)
            mel = mel / 20
            mels.append(mel)
        mels = torch.from_numpy(np.array(mels).astype(np.float32))
        return mels

