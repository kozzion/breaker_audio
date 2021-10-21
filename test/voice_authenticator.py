import os
import sys
import pickle

# from vocoder import inference as vocoder
from pathlib import Path
from breaker_audio.voice_authenticator import VoiceAuthenticator
from breaker_audio.tools_audio_io import ToolsAudioIO
from breaker_audio.tools_signal import ToolsSignal

path_dir_data = Path('C:\\project\\data\\data_breaker\\')
authenticator = VoiceAuthenticator(path_dir_data)

path_dir_voice_sound = path_dir_data.joinpath('breaker_discord','bot_dev', 'voice_sound')
path_dir_voice_encoding = path_dir_data.joinpath('breaker_discord','bot_dev', 'voice_encoding')
# C:\project\data\data_breaker\breaker_discord\bot_dev\voice_sound

id_user_from = '102797880246419456'
id_sound_from = '1634757908'
id_user_as = '3331'

print(path_dir_voice_sound.is_dir())
list_id_user = os.listdir(path_dir_voice_sound)
dict_list_encoding = {}
for id_user in list_id_user:
    dict_list_encoding[id_user] = []
    path_dir_user_sound = path_dir_voice_sound.joinpath(id_user)
    path_dir_user_encoding = path_dir_voice_encoding.joinpath(id_user)
    if not path_dir_user_encoding.is_dir():
        os.makedirs(path_dir_user_encoding)
    list_id_sound = os.listdir(path_dir_user_sound)
    for id_sound in list_id_sound:
        path_file_sound = path_dir_user_sound.joinpath(id_sound)
        path_file_encoding = path_dir_user_encoding.joinpath(id_sound)
        if not path_file_encoding.is_file():
            signal, sampling_rate = ToolsAudioIO.load_signal_wav(path_file_sound, mode_preprocessing=True)
            encoding = authenticator.encode(signal, sampling_rate)
            with path_file_encoding.open('wb') as file:
                pickle.dump(encoding, file)
        else:
            with path_file_encoding.open('rb') as file:
                encoding = pickle.load(file)

        dict_list_encoding[id_user].append(encoding)


path_file_to_auth = path_dir_data.joinpath('testdata', 'eng_0000.wav')
signal, sampling_rate = ToolsAudioIO.load_signal_wav(path_file_sound, mode_preprocessing=True)
encoding = authenticator.encode(signal, sampling_rate)
list_encoding = dict_list_encoding[id_user_as]


#for 
authentication_report = authenticator.authenticate(encoding, list_encoding)
print(authentication_report)