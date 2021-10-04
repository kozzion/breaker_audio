import numpy as np
# from scipy.io.wavfile import read
import torch
import librosa
import os

############################ 用aukit的默认参数 #############################

import numpy as np
from aukit.audio_spectrogram import linear_spectrogram as linearspectrogram
from aukit.audio_spectrogram import mel_spectrogram as melspectrogram
from aukit.audio_io import Dict2Obj
from aukit.audio_griffinlim import load_wav, save_wav, save_wavenet_wav, preemphasis, inv_preemphasis
from aukit.audio_griffinlim import start_and_end_indices, get_hop_size
from aukit.audio_griffinlim import inv_linear_spectrogram, inv_mel_spectrogram
from aukit.audio_griffinlim import librosa_pad_lr
from aukit.audio_griffinlim import default_hparams


def melspectrogram_torch(wav, hparams=None):
    """mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)"""
    mel = melspectrogram(wav, hparams)
    mel_output = torch.from_numpy(mel).type(torch.FloatTensor)
    return mel_output


def linearspectrogram_torch(wav, hparams=None):
    """spec_output: torch.FloatTensor of shape (B, n_spec_channels, T)"""
    spec = linearspectrogram(wav, hparams)
    spec_output = torch.from_numpy(spec).type(torch.FloatTensor)
    return spec_output


def inv_melspectrogram(spec):
    return inv_mel_spectrogram(spec)


def inv_linearspectrogram(spec):
    return inv_linear_spectrogram(spec)


########################### 用aukit的默认参数 #############################

_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def read(fpath):
    wav, sr = librosa.load(fpath, sr=None)
    out = np.clip(wav, -1, 1) * (2 ** 15 - 1)
    return sr, out.astype(int)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(_device))  # out=torch.cuda.LongTensor(max_len)
    mask = (ids < lengths.unsqueeze(1))  # .bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    curdir = os.path.dirname(os.path.abspath(filename))
    filepaths_and_text = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(split)
            if len(tmp) == 2:
                tmp.append('0')
            tmp[0] = os.path.join(curdir, tmp[0])
            filepaths_and_text.append(tmp)
    return filepaths_and_text


def load_filepaths_and_text_train(filename, split="|"):
    curdir = os.path.dirname(os.path.abspath(filename))
    filepaths_and_text = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(split)
            # 以下适用于先生成频谱数据的跑程序方法
            # dirname, basename = os.path.split(tmp[0])
            # tmp[0] = os.path.join(curdir, dirname, 'npy', basename)
            filepaths_and_text.append(tmp)
    return filepaths_and_text


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


if __name__ == "__main__":
    import aukit

    inpath = r"E:\data\temp\01.wav"
    wav = load_wav(inpath, sr=16000)
    mel = melspectrogram(wav)
    out = inv_melspectrogram(mel)
    aukit.play_audio(wav)
    aukit.play_audio(out)
