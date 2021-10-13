import sys
import os
import json
import time
import numpy as np

from pathlib import Path

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# from vocoder import inference as vocoder

sys.path.append('..')

from breaker_audio.voice_cloner import VoiceClonerDefault
from breaker_audio.tools_audio_io import ToolsAudioIO


path_dir_request = Path('C:\\project\\breaker\\breaker_discord\\request')
path_dir_response = Path('C:\\project\\breaker\\breaker_discord\\response')
path_dir_data = Path('C:\\project\\data\\data_breaker\\')

def get_request():
    list_name_file_request  = os.listdir(path_dir_request)
    list_name_file_response  = os.listdir(path_dir_response)
    for name_file_request in list_name_file_request:
        path_file_request = path_dir_request.joinpath(name_file_request)
        path_file_response = path_dir_response.joinpath(name_file_request[:-4] + 'wav')
        if not  path_file_response.is_file():
            return path_file_request, path_file_response
    return None, None


dict_cloner = {}
dict_cloner['eng'] = VoiceClonerDefault(path_dir_data, 'eng')
dict_cloner['cmn'] = VoiceClonerDefault(path_dir_data, 'cmn')

while True:
    path_file_request, path_file_response = get_request()
    if path_file_request == None:
        time.sleep(0.1)
        print('sleep')
        sys.stdout.flush()
    else:
        # with open(path_file_request, mode='r', encoding='utf-8') as file:
        #     dict_request = json.load(file)
        with open(path_file_request, mode='r') as file:
            dict_request = json.load(file)

        language_code_639_3 = dict_request['language_code_639_3']
        path_file_voice_toclone = Path(dict_request['path_file_voice_toclone'])
        text = dict_request['text']
        if language_code_639_3 in dict_cloner:
            cloner = dict_cloner[language_code_639_3]

            signal_voice_toclone, sampling_rate_toclone = ToolsAudioIO.load_signal_wav(path_file_voice_toclone, mode_preprocessing=True)
            cloner.clone_voice(signal_voice_toclone)
            signal_voice_cloned, sampling_rate_cloned = cloner.synthesize(text)
            ToolsAudioIO.save_wav(path_file_response, signal_voice_cloned, sampling_rate_cloned)
        else:
            ToolsAudioIO.save_wav(path_file_response, np.zeros(1000), 1000)