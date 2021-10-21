import sys
import os
import json
import time

from pathlib import Path


from breaker_core.datasource.jsonqueue import Jsonqueue
from breaker_core.datasource.bytessource import Bytessource

sys.path.append('..')
from breaker_audio.voice_authenticator import VoiceAuthenticator
from breaker_audio.tools_audio_io import ToolsAudioIO

if __name__ == '__main__':
    path_file_config_breaker = Path(os.environ['PATH_FILE_CONFIG_BREAKER_DEV'])
    path_dir_data =  Path(os.environ['PATH_DIR_DATA_BREAKER'])
    
    with open(path_file_config_breaker, 'r') as file:
        dict_config = json.load(file)


    jsonqueue_request = Jsonqueue.from_dict(dict_config['queue_request_voice_authenticator'])
    if not jsonqueue_request.exists():
        jsonqueue_request.create()

    authenticator = VoiceAuthenticator(path_dir_data)
    
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
            try:
                bytessource_response = Bytessource.from_dict(dict_request['bytessource_response'])
                dict_request = dict_request['payload_request']
                print('request!')
                sys.stdout.flush()
                type_request = dict_request['type_request']
                if type_request == 'encode':
                    print('encode')
                    bytessource_sound = Bytessource.from_dict(dict_request['bytessource_voice_sound'])
                    bytessource_encoding = Bytessource.from_dict(dict_request['bytessource_voice_encoding'])
                    # bytessource_summary = Bytessource.from_dict(dict_request['bytessource_voice_summary'])


                    signal_voice, sampling_rate_voice = ToolsAudioIO.bytearray_wav_to_signal(bytessource_sound.read())
                    array_encoding = authenticator.encode(signal_voice, sampling_rate_voice)
                    bytessource_encoding.write_pickle(array_encoding)
                    # summary = bytessource_summary.read_pkl()
                    # authenticator.update_summary(summary)
                    bytessource_response.write_json({'is_succes':True})

                elif type_request == 'authenticate':
                    print('authenticate')
                    bytessource_sound = Bytessource.from_dict(dict_request['bytessource_voice_sound'])
                    bytessource_encoding_dir = Bytessource.from_dict(dict_request['bytessource_voice_encoding_dir'])
                    # bytessource_summary = Bytessource.from_dict(dict_request['bytessource_voice_summary'])

                    
                    signal_voice, sampling_rate_voice = ToolsAudioIO.bytearray_wav_to_signal(bytessource_sound.read())
                    encoding_a = authenticator.encode(signal_voice, sampling_rate_voice)
               
                    list_list_key = bytessource_encoding_dir.list_shallow()
                    list_encoding_b = []
                    for list_key in list_list_key:
                        bs = bytessource_encoding_dir.join(list_key)
                        print(bs.path)
                        list_encoding_b.append(bs.read_pickle())

                    authentication_report = authenticator.authenticate(encoding_a, list_encoding_b)
                    bytessource_response.write_json({'is_succes':True, 'authentication_report':authentication_report})

                    
                elif type_request == 'indentify':
                    print('indentify')
                    bytessource_sound = Bytessource.from_dict(dict_request['bytessource_voice_sound'])
                    # bytessource_summary = Bytessource.from_dict(dict_request['bytessource_voice_summary_parent'])    

                elif type_request == 'umap':
                    bytessource_sound = Bytessource.from_dict(dict_request['bytessource_voice_encoding_parent'])
                    bytessource_response.write_json(
                        {
                            'is_succes':False,
                            'umap':'unknown type_request: ' + type_request 
                    })
                else:
                    bytessource_response.write_json(
                        {
                            'is_succes':False,
                            'message':'unknown type_request: ' + type_request 
                    })
            except Exception as e:
                # print(e)
                raise e

