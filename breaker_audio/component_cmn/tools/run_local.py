#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/20
"""
"""
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
from tqdm import tqdm
import collections as clt
import os
import re
import json
import numpy as np
import shutil

import aukit
from aukit.audio_griffinlim import default_hparams, mel_spectrogram
from hparams import hparams

my_hp = {
    "n_fft": 1024, "hop_size": 256, "win_size": 1024,
    "sample_rate": 22050, "max_abs_value": 4.0,
    "fmin": 0, "fmax": 8000,
    "preemphasize": True,
    'symmetric_mels': True,
}

default_hparams.update(hparams.values())
# default_hparams.update(my_hp)

a = {(k, v) for k, v in hparams.values().items() if type(v) in {str, int, float, tuple, bool, type(None)}}
b = {(k, v) for k, v in default_hparams.items() if type(v) in {str, int, float, tuple, bool, type(None)}}
print(a - b)
print(b - a)

_pad_len = (default_hparams.n_fft - default_hparams.hop_size) // 2


def wavs2mels(indir: Path, outdir: Path):
    for fpath in tqdm(indir.glob("*.wav")):
        wav = aukit.load_wav(fpath, sr=16000)
        wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
        mel = mel_spectrogram(wav, default_hparams)
        np.save(outdir.joinpath(fpath.stem + ".npy"), mel, allow_pickle=False)


def get_train_files(indir: Path):
    others = []
    names = []
    for fpath in tqdm(sorted(indir.glob("**/*.wav"))):
        s = os.path.getsize(fpath)
        if s < 32000:
            print(s, fpath)
            others.append(fpath)
            continue
        name = "/".join(fpath.relative_to(indir).parts)
        names.append(name)

    with open(indir.joinpath("train_files.txt"), "w", encoding="utf8") as fout:
        for name in names:
            fout.write(name + "\n")


if __name__ == "__main__":
    print(__file__)
    indir = Path(r"E:\lab\melgan\data\aliexamples")
    outdir = Path(r"E:\lab\melgan\data\aliexamples_mel")
    outdir.mkdir(exist_ok=True)
    # wavs2mels(indir=indir, outdir=outdir)

    indir = Path(r"E:\data\aliaudio\alijuzi")
    get_train_files(indir=indir)
