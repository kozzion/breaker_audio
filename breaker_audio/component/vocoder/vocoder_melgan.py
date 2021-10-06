import os
import json
import numpy as np
from pathlib import Path
import yaml
import torch

from aukit.audio_griffinlim import mel_spectrogram, default_hparams
from aukit import Dict2Obj


from breaker_audio.component_cmn.melgan.mel2wav.modules import Generator, Audio2Mel


class VocoderMelgan:
    def __init__(
            self,
            device,
    ):
        self.device = device
        self._model = None
        my_hp = {
            "n_fft": 1024,  # 800
            "hop_size": 256,  # 200
            "win_size": 1024,  # 800
            "sample_rate": 22050,  # 16000
            "fmin": 0,  # 55
            "fmax": 11025,  # 7600 # sample_rate // 2
            "preemphasize": False,  # True
            'symmetric_mels': True,  # True
            'signal_normalization': False,  # True
            'allow_clipping_in_normalization': False,  # True
            'ref_level_db': 0,  # 20
            'center': False,  # True
            '__file__': __file__
        }

        self.synthesizer_hparams = {k: v for k, v in default_hparams.items()}
        self.synthesizer_hparams = {**self.synthesizer_hparams, **my_hp}
        self.synthesizer_hparams = Dict2Obj(self.synthesizer_hparams)


    def load_model_args(self, mel2wav_path, path_file_yaml):
        """
        Args:
            mel2wav_path (str or Path): path to the root folder of dumped text2mel
            device (str or torch.device): device to load the model
        """
        if str(path_file_yaml).endswith('.yml'):
            with open(path_file_yaml, "r") as f:
                args = yaml.load(f, Loader=yaml.FullLoader)
        else:
            args = json.load(open(path_file_yaml, encoding='utf8'))

        ratios = [int(w) for w in args['ratios'].split()]
        self._model = Generator(args['n_mel_channels'], args['ngf'], args['n_residual_layers'], ratios=ratios).to(self.device)
        self._model.load_state_dict(torch.load(mel2wav_path, map_location=self.device))
        self._model.eval()

    # def load_model_net(self, path_dir_model:Path):
    #     raise Exception("not implemented")
    #     # """
    #     # Args:
    #     #     mel2wav_path (str or Path): path to the root folder of dumped text2mel
    #     #     device (str or torch.device): device to load the model
    #     # """
    #     # with open(root / "args.yml", "r") as f:
    #     #     args = yaml.load(f, Loader=yaml.FullLoader)
    #     # netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(self.device)
    #     # netG.load_state_dict(torch.load(root / "best_netG.pt", map_location=self.device))
    #     # return netG

    def load_model(self, path_dir_model:Path):
        path_file_model = path_dir_model.joinpath('model.pt')
        self._model = Generator(80, 32, 3).to(self.device)
        self._model.load_state_dict(torch.load(path_file_model, map_location=self.device))
        self._model.eval()        

    def melspec_to_signal(self, array_melspec):#TODO convert from torch array
        with torch.no_grad():
            return self._model(array_melspec.to(self.device)).squeeze(1)

    def signal_to_melspec(self, src):
        src = src.unsqueeze(1)
        mel = Audio2Mel()(src)
        return mel
