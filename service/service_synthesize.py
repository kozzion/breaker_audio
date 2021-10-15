import sys
import os
import json
import time
import numpy as np

from pathlib import Path

import tensorflow as tf

from breaker_core.datasource.jsonqueue import Jsonqueue
from breaker_core.datasource.bytearraysource import Bytearraysource

sys.path.append('..')
from breaker_audio.voice_cloner import VoiceClonerDefault
from breaker_audio.tools_audio_io import ToolsAudioIO

if __name__ == '__main__':


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    path_file_config = Path('config_aws.cfg')
    path_dir_data = Path('C:\\project\\data\\data_breaker\\')

    with path_file_config.open('r', encoding='utf-8') as file:
        dict_config = json.load(file)

    jsonqueue_request = Jsonqueue.from_dict(dict_config['queue_request'])
    if not jsonqueue_request.exists():
        jsonqueue_request.create()
        
    dict_cloner = {}
    dict_cloner['eng'] = VoiceClonerDefault(path_dir_data, 'eng')
    dict_cloner['cmn'] = VoiceClonerDefault(path_dir_data, 'cmn')

    count = 0
    while True:
        dict_request = jsonqueue_request.dequeue()
        count += 1
        if dict_request == None:
            time.sleep(0.1)
            if count % 10 == 0:
                print('sleep ' + str(count))
                sys.stdout.flush()
        else:
            print('request!')
            sys.stdout.flush()

            language_code_639_3 = dict_request['language_code_639_3']
            bytearraysource_voice = Bytearraysource.from_dict(dict_request['bytearraysource_voice'])
            text = dict_request['text']
            bytearraysource_output = Bytearraysource.from_dict(dict_request['bytearraysource_output'])

            if language_code_639_3 in dict_cloner:
                cloner = dict_cloner[language_code_639_3]
                bytearray_voice = bytearraysource_voice.load()
                signal_voice_toclone, sampling_rate_toclone = ToolsAudioIO.bytearray_wav_to_signal(bytearray_voice, mode_preprocessing=True)
                
                cloner.clone_voice(signal_voice_toclone)
                signal_voice_cloned, sampling_rate_cloned = cloner.synthesize(text)
                
                bytearray_output = ToolsAudioIO.signal_to_bytearray_wav(signal_voice_cloned, sampling_rate_cloned)
                bytearraysource_output.save(bytearray_output)
            else:
                bytearray_output = ToolsAudioIO.signal_to_bytearray_wav(np.zeros(100), 44100)
                bytearraysource_output.save(bytearray_output)