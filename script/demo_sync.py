import numpy as np
import librosa
import argparse
import time
import torch
import sys
import shutil
import json

import aukit

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from synthesizer import hparams
from synthesizer.utils import audio
from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path




path_file_encoder = Path('C:\\project\\voice\\zhrtvc-master\\models\\encoder\\saved_models\\ge2e_pretrained.pt')
path_dir_synthesizer = Path('C:\\project\\voice\\zhrtvc-master\\models\\synthesizer\\saved_models\\logs-syne\\checkpoints')
path_file_synthesizer_meta = Path('C:\\project\\voice\\zhrtvc-master\\models\\synthesizer\\saved_models\\logs-syne\\metas\\hparams.json')
path_file_vocoder = Path('C:\\project\\voice\\zhrtvc-master\\models\\saved_models\\melgan\\melgan_multi_speaker.pt')

path_file_source = Path('C:\\project\\voice\\zhrtvc-master\\data\\samples\\stcmds\P00001A\\20170001P00001A0091.mp3')  
path_file_target =  Path('C:\\project\data\\data_breaker\\20170001P00001A0091.mp3')   

low_mem = False
no_sound = False


if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))

encoder.load_model(path_file_encoder, device='cpu')

json_hparams = json.load(open(path_file_synthesizer_meta, encoding="utf8"))
hparams = aukit.Dict2Obj(json_hparams)
synthesizer = Synthesizer(path_dir_synthesizer, low_mem=low_mem, hparams=hparams)



dict_text = {}
dict_text['text_0'] = "谁敢拐你跟团吧要不"

for id_text, text in dict_text.items():
    preprocessed_wav = encoder.preprocess_wav(path_file_source)
    print("Loaded file succesfully")

    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")

    ## Generating the spectrogram
    spec_source = synthesizer.make_spectrogram(preprocessed_wav, hparams=synthesizer.hparams)
    spec_target = synthesizer.synthesize_spectrograms([text], [embed])[0]

    ## Generating the waveform
    print("Synthesizing the waveform:")
    wav_generated_source = synthesizer.griffin_lim(spec_source, hparams=synthesizer.hparams)
    wav_generated_target = synthesizer.griffin_lim(spec_target, hparams=synthesizer.hparams)
    # generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")


    import sounddevice as sd
    if not no_sound:
        sd.stop()
        sd.play(wav_generated_target, synthesizer.sample_rate)
        sd.wait()
        sd.stop()
        sd.play(wav_generated_source, synthesizer.sample_rate)
        sd.wait()
        sd.stop()
        sd.play(preprocessed_wav, synthesizer.sample_rate)
        sd.wait()
        exit()
    audio.save_wav(wav_generated_target, path_file_target, synthesizer.sample_rate)  # save
dict_result = {}