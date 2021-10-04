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

path_file_voice_toclone = path_dir_data.joinpath('testdata', 'cmn_0000.mp3')
# text = "谁敢拐你跟团吧要不"
text = "谁敢拐你跟团吧要" #speaker swallows last sylable

cloner = VoiceClonerDefault(path_dir_data, 'cmn')
signal_voice_toclone, sampling_rate = ToolsAudioIO.load_signal_wav(path_file_voice_toclone, mode_preprocessing=True)


cloner.clone_voice(signal_voice_toclone)
signal_voice_cloned = cloner.synthesize(text)
#signal_voice_cloned_aligned = ToolsSignal.align_signal_melspec_energy(signal_voice_toclone, signal_voice_cloned, cloner._vocoder)

ToolsAudioIO.play(signal_voice_toclone, sampling_rate)
ToolsAudioIO.play(signal_voice_cloned, sampling_rate)
#ToolsAudioIO.play(signal_voice_cloned_aligned, sampling_rate)
