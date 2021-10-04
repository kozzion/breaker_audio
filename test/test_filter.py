import sys
import json
import pickle as pkl
import numpy as np
from sklearn.metrics import mean_squared_error

from breaker_audio.tools_signal import ToolsSignal
from breaker_audio.tools_audio_io import ToolsAudioIO
from breaker_audio.filter_convolution_linear import FilterConvolutionLinear
path_file_data = 'data.pkl'
with open(path_file_data, 'rb') as file:
    dict_data = pkl.load(file)   

array_output_true = dict_data['array_output_true']
array_input_reverse = dict_data['array_input_reverse']
array_input_generated = dict_data['array_input_generated'] 

array_melspec_reverse = dict_data['array_melspec_reverse']
array_melspec_generated = dict_data['array_melspec_generated'] 




# error = mean_squared_error(array_output_true, array_input_reverse)
# print(error)
array_input_generated_0 = ToolsSignal.align_signal_0(array_output_true, array_input_generated)
print(len(array_input_generated_0))
array_input_generated_1 = ToolsSignal.align_signal_1(array_output_true, array_input_generated)
print(len(array_input_generated_1))
array_input_generated_2 = ToolsSignal.align_signal_2(array_output_true, array_input_generated)
print(len(array_input_generated_2))


print(array_melspec_reverse.shape)
array_melspec_generated_3 = ToolsSignal.align_melspec_0(array_melspec_reverse, array_melspec_generated)
print(array_melspec_generated_3.shape)

array_melspec_generated_4 = ToolsSignal.align_melspec_1(array_melspec_reverse, array_melspec_generated)
print(array_melspec_generated_4.shape)

from pathlib import Path
import torch
from melgan.inference import MelVocoder, get_default_device
_device = get_default_device()
vocoder = MelVocoder(Path("../models/vocoder/saved_models/melgan/melgan_multi_speaker.pt"), github=True, args_path= '', device=_device, mode='default')

array_input_generated_3 = vocoder.inverse(torch.from_numpy(np.expand_dims(array_melspec_generated_3, axis=0)).to(_device)).squeeze().cpu().numpy()
array_input_generated_4 = vocoder.inverse(torch.from_numpy(np.expand_dims(array_melspec_generated_4, axis=0)).to(_device)).squeeze().cpu().numpy()


# filter_0 = FilterConvolutionLinear(size_kernel=20)
# array_input_generated_4, error = filter_0.fit_transform(array_input_generated_3, array_output_true[:len(array_input_generated_3)])
# print(error)
ToolsAudioIO.play(array_output_true,sample_rate=16000)
# ToolsAudioIO.play(array_input_reverse, sample_rate=16000)
# ToolsAudioIO.play(array_input_generated, sample_rate=16000)
# ToolsAudioIO.play(array_input_generated_0, sample_rate=16000)
# ToolsAudioIO.play(array_input_generated_1, sample_rate=16000)
# ToolsAudioIO.play(array_input_generated_2, sample_rate=16000)
ToolsAudioIO.play(array_input_generated_3, sample_rate=16000)
ToolsAudioIO.play(array_input_generated_4, sample_rate=16000)
#44100