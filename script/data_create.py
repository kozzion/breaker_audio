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
#path_file_source = Path('C:\\project\data\\data_breaker\\shu-voice_short.mp3')     
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

# text = "谁敢拐你跟团吧要不"
text = "谁敢拐你跟团吧要" #speaker swallows last sylable



from melgan.inference import MelVocoder, get_default_device
_device = get_default_device()

vocoder = MelVocoder(Path("../models/vocoder/saved_models/melgan/melgan_multi_speaker.pt"), github='default', args_path= '', device=_device, mode='default')


array_output_true = encoder.preprocess_wav(path_file_source)
print("Loaded file succesfully")

embed = encoder.embed_utterance(array_output_true)
print("Created the embedding")

## Generating the spectrogram
array_melspec_reverse = vocoder(torch.from_numpy(array_output_true[None]))

array_melspec_generated = synthesizer.synthesize_spectrograms([text], [embed])[0]

## Generating the waveform
print("Synthesizing the waveform:")
array_input_reverse = vocoder.inverse(array_melspec_reverse.to(_device)).squeeze().cpu().numpy()
array_input_generated = vocoder.inverse(torch.from_numpy(np.expand_dims(array_melspec_generated, axis=0)).to(_device)).squeeze().cpu().numpy()

dict_result = {}
dict_data = {}
dict_data['array_output_true'] = array_output_true
dict_data['array_input_reverse'] = array_input_reverse
dict_data['array_input_generated'] = array_input_generated
dict_data['array_melspec_reverse'] = array_melspec_reverse.cpu().detach().numpy()[0,:,:]
dict_data['array_melspec_generated'] = array_melspec_generated  


import pickle as pkl
with open('data.pkl', 'wb') as file:
    pkl.dump(dict_data, file)
exit()

import sounddevice as sd
if not no_sound:
    sd.stop()
    sd.play(array_output_true, synthesizer.sample_rate)
    sd.wait()
    sd.stop()
    sd.play(array_input_reverse, synthesizer.sample_rate)
    sd.wait()
    sd.stop()
    sd.play(array_input_generated, synthesizer.sample_rate)
    sd.wait()
    exit()