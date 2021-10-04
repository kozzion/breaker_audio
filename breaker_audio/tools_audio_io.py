from pathlib import Path
import librosa
import sounddevice as sd

from breaker_audio.tools_signal import ToolsSignal
class ToolsAudioIO:

    @staticmethod
    def play(array_signal, sampling_rate=44100):
        sd.stop()
        sd.play(array_signal, sampling_rate)
        sd.wait()


    @staticmethod
    def load_signal_wav(path_file_wav: Path, *, mode_preprocessing:bool=True):
        signal, sampling_rate = librosa.load(path_file_wav, sr=None)
        if mode_preprocessing:
            signal, sampling_rate = ToolsSignal.preprocess_signal(signal, sampling_rate)
        return signal, sampling_rate



    # def load_wav(path, sample_rate) :
    #     return librosa.load(str(path), sr=hp.sample_rate)[0]


    # def save_wav(x, path) :
    #     sf.write(path, x.astype(np.float32), hp.sample_rate)