
import librosa
import sounddevice as sd
import soundfile
import io 
import numpy as np
from pathlib import Path

from breaker_audio.tools_signal import ToolsSignal

class ToolsAudioIO:

    @staticmethod
    def play(array_signal:np.ndarray, sampling_rate:int=44100):
        sd.stop()
        sd.play(array_signal, sampling_rate)
        sd.wait()


    @staticmethod
    def load_signal_wav(path_file_wav: Path, *, mode_preprocessing:bool=True):
        signal, sampling_rate = librosa.load(path_file_wav, sr=None)
        if mode_preprocessing:
            signal, sampling_rate = ToolsSignal.preprocess_signal(signal, sampling_rate)
        return signal, sampling_rate



    @staticmethod
    def save_mp3(path_file: Path, singal:np.ndarray, sampling_rate:int=44100):
        #TODO ffmpeg?
        raise NotImplementedError()

    staticmethod
    def save_wav(path_file: Path, singal:np.ndarray, sampling_rate:int=44100):
        # import soundfile as sf
# >>> sf.write('stereo_file.wav', np.random.randn(10, 2), 44100, 'PCM_24')
        librosa.output.write_wav(path_file, singal, sampling_rate)


    @staticmethod
    def bytearray_wav_to_signal(bytearray_wav: bytearray, *, mode_preprocessing:bool=True):
        bytesio = io.BytesIO()
        bytesio.write(bytearray_wav)
        bytesio.seek(0)
        signal, sampling_rate = librosa.load(bytesio, sr=None)
        if mode_preprocessing:
            signal, sampling_rate = ToolsSignal.preprocess_signal(signal, sampling_rate)
        return signal, sampling_rate

    @staticmethod
    def signal_to_bytearray_wav(array_singal:np.ndarray, sampling_rate:int=44100):
        bytesio = io.BytesIO()
        soundfile.write(bytesio, array_singal, sampling_rate, format='WAV', subtype='PCM_16') #TODO maybe other pcm
        bytesio.seek(0)
        return bytesio.read()
 