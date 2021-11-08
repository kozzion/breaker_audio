import sys
import os
import json
import time

from pathlib import Path


from breaker_core.datasource.jsonqueue import Jsonqueue
from breaker_core.datasource.bytessource import Bytessource
from breaker_core.common.service_jsonqueue import ServiceJsonqueue

from breaker_audio.voice_authenticator import VoiceAuthenticator
from breaker_audio.tools_audio_io import ToolsAudioIO

class ServiceVoiceAuthenticator(ServiceJsonqueue):

    def __init__(self, config_breaker, queue_request, mode_debug, path_dir_data) -> None:
        super().__init__(config_breaker, queue_request, mode_debug)
        self.path_dir_data = path_dir_data
        self.authenticator = VoiceAuthenticator(path_dir_data)
        
    def process_request(self, request:dict) -> 'dict':
        type_request = request['type_request']
        if type_request == 'encode':
            print('encode')
            bytessource_sound = Bytessource.from_dict(self.config_breaker, request['bytessource_voice_sound'])
            bytessource_encoding = Bytessource.from_dict(self.config_breaker, request['bytessource_voice_encoding'])

            signal_voice, sampling_rate_voice = ToolsAudioIO.bytearray_wav_to_signal(bytessource_sound.read())
            array_encoding = self.authenticator.encode(signal_voice, sampling_rate_voice)
            bytessource_encoding.write_pickle(array_encoding)
            return {'was_processed':True}

        elif type_request == 'authenticate':
            print('authenticate')
            bytessource_sound = Bytessource.from_dict(self.config_breaker, request['bytessource_voice_sound'])
            bytessource_encoding_dir = Bytessource.from_dict(self.config_breaker, request['bytessource_voice_encoding_dir'])
            
            signal_voice, sampling_rate_voice = ToolsAudioIO.bytearray_wav_to_signal(bytessource_sound.read())
            encoding_a = self.authenticator.encode(signal_voice, sampling_rate_voice)
        
            list_list_key = bytessource_encoding_dir.list_shallow()
            list_encoding_b = []
            for list_key in list_list_key:
                list_encoding_b.append(bytessource_encoding_dir.join(list_key).read_pickle())
            authentication_report = self.authenticator.authenticate(encoding_a, list_encoding_b)
            return {'was_processed':True, 'authentication_report':authentication_report}
        else:
            return {'was_processed':False, 'message':'Unknown type_request: ' + type_request}


if __name__ == '__main__':
    path_file_config_breaker = Path(os.getenv('PATH_FILE_CONFIG_BREAKER', 'config.cfg'))
    path_dir_data =  Path(os.getenv('PATH_DIR_DATA_BREAKER', '/data/data_breaker/' ))
    mode_debug = True

    with open(path_file_config_breaker, 'r') as file:
        config_breaker = json.load(file)


    jsonqueue_request = Jsonqueue.from_dict(config_breaker, config_breaker['queue_request_voice_authenticator'])
    if not jsonqueue_request.exists():
        jsonqueue_request.create()

    service = ServiceVoiceAuthenticator(config_breaker, jsonqueue_request, mode_debug, path_dir_data)  
    service.run()

    
  