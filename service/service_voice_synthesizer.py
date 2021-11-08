import sys
import os
import json

from pathlib import Path

import tensorflow as tf

from breaker_core.datasource.jsonqueue import Jsonqueue
from breaker_core.datasource.bytessource import Bytessource
from breaker_core.common.service_jsonqueue import ServiceJsonqueue

from breaker_audio.voice_cloner import VoiceClonerDefault
from breaker_audio.tools_audio_io import ToolsAudioIO

class ServiceVoiceSynthesizer(ServiceJsonqueue):


    def __init__(self, config_breaker:dict, queue_request, mode_debug, path_dir_data) -> None:
        super().__init__(config_breaker, queue_request, mode_debug)
        self.path_dir_data = path_dir_data
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        self.dict_cloner = {}
        self.dict_cloner['eng'] = VoiceClonerDefault(path_dir_data, 'eng')
        self.dict_cloner['cmn'] = VoiceClonerDefault(path_dir_data, 'cmn')

    def process_request(self, request:dict) -> 'dict':
        type_request = request['type_request']
        if type_request == 'synthesize':
            print('synthesize')
            sys.stdout.flush()

            language_code_639_3 = request['language_code_639_3']
            bytessource_voice = Bytessource.from_dict(self.config_breaker, request['bytessource_voice'])
            text = request['text']
            bytessource_output = Bytessource.from_dict(self.config_breaker, request['bytessource_output'])

            if not language_code_639_3 in self.dict_cloner:
                return {
                        'was_processed':False,
                        'message':'language_code_639_3: ' + language_code_639_3 + ' was has no synthesizer'
                }

            cloner = self.dict_cloner[language_code_639_3]
            bytes_voice = bytessource_voice.read()
            signal_voice_toclone, sampling_rate_toclone = ToolsAudioIO.bytearray_wav_to_signal(bytes_voice, mode_preprocessing=True)
            
            cloner.clone_voice(signal_voice_toclone)
            signal_voice_cloned, sampling_rate_cloned = cloner.synthesize(text)
            
            bytes_output = ToolsAudioIO.signal_to_bytearray_wav(signal_voice_cloned, sampling_rate_cloned)
            bytessource_output.write(bytes_output)
            return {'was_processed':True}
        else:
            return {'was_processed':False, 'message':'Unknown type_request: ' + type_request}



if __name__ == '__main__':
    path_file_config_breaker = Path(os.getenv('PATH_FILE_CONFIG_BREAKER', '/config/config.cfg'))
    path_dir_data =  Path(os.getenv('PATH_DIR_DATA_BREAKER', '/data/data_breaker/' ))
    mode_debug = True

    with open(path_file_config_breaker, 'r') as file:
        config_breaker = json.load(file)


    jsonqueue_request = Jsonqueue.from_dict(config_breaker, config_breaker['queue_request_voice_synthesizer'])
    if not jsonqueue_request.exists():
        jsonqueue_request.create()

    service = ServiceVoiceSynthesizer(jsonqueue_request, mode_debug, path_dir_data)  
    service.run()