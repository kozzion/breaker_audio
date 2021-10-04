# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/9/22
"""
demo_cli
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import torch
import numpy as np
from .model import Tacotron2, load_model
from .text import cmudict, text_to_sequence
from .utils import inv_linear_spectrogram, load_wav

from .hparams import create_hparams
from .data_utils import transform_data_train

from .data_utils import transform_mel, transform_text, transform_f0, transform_embed, transform_speaker

import json



class MellotronSynthesizer():
    def __init__(self, model_path, speakers_path, hparams_path):
        args_hparams = open(hparams_path, encoding='utf8').read()
        self.hparams = create_hparams(args_hparams)

        self.model = load_model(self.hparams).to(_device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=_device)['state_dict'])

        self.speakers = json.load(open(speakers_path, encoding='utf8'))

    def synthesize(self, text, speaker):
        text_encoded = torch.LongTensor(transform_text(text, text_cleaners='hanzi'))[None, :].to(_device)
        speaker_id = torch.LongTensor(transform_speaker(speaker, speaker_ids=self.speakers)).to(_device)

        style_input = 0
        pitch_contour = None
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = _mellotron.inference(
                (text_encoded, style_input, speaker_id, pitch_contour))
        out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
        return out_mel

############################################ 以下计划弃用 ###################################################

_mellotron = None
_device = 'cpu'


def load_model_mellotron(model_path, hparams=None):
    global _mellotron
    _mellotron = load_model(hparams).to(_device).eval()
    _mellotron.load_state_dict(torch.load(model_path, map_location=_device)['state_dict'])


def synthesize_one(text, speaker='Aiyue', model_path='', with_alignment=False, hparams=None, encoder_fpath=''):
    if _mellotron is None:
        load_model_mellotron(model_path)

    text_encoded = torch.LongTensor(transform_text(text, text_cleaners='hanzi'))[None, :].to(_device)

    speaker_id = torch.LongTensor(transform_speaker('', speaker_ids={})).to(_device)
    style_input = 0

    # pitch_contour = torch.ones(1, _hparams.prenet_f0_dim, text_encoded.shape[1] * 5, dtype=torch.float) * np.random.random()
    # pitch_contour = None

    wav = load_wav(str(speaker), sr=hparams.sampling_rate)
    embed = transform_embed(wav, encoder_fpath)
    embed = embed[::embed.shape[0] // hparams.prenet_f0_dim]
    embed = embed if embed.shape[0] == hparams.prenet_f0_dim else embed[:hparams.prenet_f0_dim]
    f0 = np.tile(embed, (text_encoded.shape[1] * 5, 1)).T
    pitch_contour = torch.from_numpy(f0[None])

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = _mellotron.inference(
            (text_encoded, style_input, speaker_id, pitch_contour))

    out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
    if with_alignment:
        return out_mel, alignments[0]
    else:
        return out_mel


def griffinlim_vocoder(spec):
    wav = inv_linear_spectrogram(spec)
    return wav


def save_model(model, outpath):
    torch.save(model, outpath)
