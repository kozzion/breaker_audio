import sys
sys.path.append('..')

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# from vocoder import inference as vocoder
from pathlib import Path
from breaker_audio.voice_cloner import VoiceClonerDefault
from breaker_audio.tools_audio_io import ToolsAudioIO
from breaker_audio.tools_signal import ToolsSignal

path_dir_data = Path('C:\\project\\data\\data_breaker\\')

path_file_voice_toclone = path_dir_data.joinpath('testdata', 'eng_0000.wav')

cloner = VoiceClonerDefault(path_dir_data, 'eng')
signal_voice_toclone, sampling_rate_toclone = ToolsAudioIO.load_signal_wav(path_file_voice_toclone, mode_preprocessing=True)
text = "the president is a very smart guy"

cloner.clone_voice(signal_voice_toclone)
signal_voice_cloned, sampling_rate_cloned = cloner.synthesize(text)


ToolsAudioIO.play(signal_voice_toclone, sampling_rate_toclone)
ToolsAudioIO.play(signal_voice_cloned, sampling_rate_cloned)
