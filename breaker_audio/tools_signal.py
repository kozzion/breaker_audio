
import numpy as np
import sklearn
from scipy import signal
from scipy import interpolate
from scipy import ndimage
import librosa

import webrtcvad # TODO not happy with these
import struct # TODO not happy with these

from breaker_audio.component.vocoder.vocoder_melgan import VocoderMelgan

class ToolsSignal:

    @staticmethod
    def error_abs(signal_true, signal_pred):
        return np.sum(np.abs(signal_true, signal_pred))

    
    @staticmethod
    def error_mse(signal_true, signal_pred):
        return sklearn.metrics.mean_squared_error(signal_true, signal_pred)

    @staticmethod
    def align_signal_0(signal_true, signal_pred):
        return signal.resample(signal_pred, len(signal_true))
        

    @staticmethod
    def align_signal_1(signal_true, signal_pred):
        sr2 =  14000 * (len(signal_true)/ len(signal_pred)) 
        return librosa.resample(signal_pred, 14000, sr2)
        

    @staticmethod
    def align_signal_2(signal_true, signal_pred):
        speed = len(signal_pred) / len(signal_true)
        return librosa.effects.time_stretch(signal_pred, speed)

    @staticmethod
    def align_signal_melspec_energy(signal_true, signal_pred, vocoder:VocoderMelgan):
        import torch #TODO get rid of melvocoder and use somthign like an fft
        melspec_true = vocoder.signal_to_melspec(torch.from_numpy(signal_true[None])).squeeze().cpu().numpy()
        melspec_pred = vocoder.signal_to_melspec(torch.from_numpy(signal_pred[None])).squeeze().cpu().numpy()
        melspec_pred_aligned = ToolsSignal.align_melspec_energy(melspec_true, melspec_pred)
        signal_pred_aligned = vocoder.melspec_to_signal(torch.from_numpy(np.expand_dims(melspec_pred_aligned, axis=0))).squeeze().cpu().numpy()
        return signal_pred_aligned        

    @staticmethod
    def align_melspec_0(melspec_true, melspec_pred):
        # see https://pytorch.org/audio/stable/transforms.html
        melspec_aligned = ToolsSignal.rescale_melspec(melspec_pred, melspec_true.shape)
        # import matplotlib.pyplot as plt
        # min = np.min(melspec_true)
        # max = np.max(melspec_true)
        # plt.figure()
        # plt.subplot(3,2,1)
        # plt.imshow(melspec_true, vmin=min, vmax=max)
        # plt.subplot(3,2,3)
        # plt.imshow(melspec_aligned, vmin=min, vmax=max)
        # plt.subplot(3,2,5)
        # plt.imshow(melspec_aligned, vmin=min, vmax=max)
        # plt.show()
        return melspec_aligned


    @staticmethod
    def align_melspec_energy(melspec_true, melspec_pred):
        # see https://pytorch.org/audio/stable/transforms.html
        min = np.min(melspec_true)
        sum_true = np.cumsum(np.sum(melspec_true - min, axis=0))
        sum_true /= sum_true[-1]
        return ToolsSignal.rescale_melspec(melspec_pred, melspec_true.shape, target_1_mod=sum_true)

    @staticmethod
    def rescale_melspec(array_source, shape_target, mode='linear', *, target_1_mod=None):
        # https://docs.scipy.org/doc/scipy/reference/interpolate.html
        if mode not in {'linear', 'cubic', 'quintic'}:
              raise Exception('unknown mode: ' + mode)

        array_target = np.zeros(shape_target, dtype=array_source.dtype)
        y_source = np.arange(0, array_source.shape[0], 1)
        x_source = np.arange(0, array_source.shape[1], 1)
        target_0 = np.arange(0, array_target.shape[0], 1) * (array_source.shape[0] / shape_target[0])
        if target_1_mod is None:
            target_1 = np.arange(0, array_target.shape[1], 1) * (array_source.shape[1] / shape_target[1])
        else:
            target_1 = target_1_mod * array_source.shape[1]      
        f = interpolate.interp2d(x_source, y_source, array_source[:,:], kind=mode)
        array_target[:,:] = f(target_1, target_0)

        return array_target

    @staticmethod
    def normalize_volume(signal, target_dbfs, increase_only=False, decrease_only=False):    
        int16_max = (2 ** 15) - 1
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((signal * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dbfs - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return signal
        return signal * (10 ** (dBFS_change / 20))



    def preprocess_signal(
            signal: np.ndarray,
            sampling_rate: int,
            normalize: bool = True,
            trim_silence: bool = True):
        """
        Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
        either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

        :param signal: the waveform as a numpy array of floats.
        :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
        preprocessing. After preprocessing, the waveform's sampling rate will match the data 
        hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
        this argument will be ignored.
        """

        # Resample the wav if needed
        if 16000 != sampling_rate:
            #TODO use spectral resampling here, the encoder just runs on 16000
            signal = librosa.resample(signal, sampling_rate, 16000) #TODO make this work for other stuff
            sampling_rate = 16000
            
        # Apply the preprocessing: normalize volume and shorten long silences 
        if normalize:
            target_dbfs = -30
            signal = ToolsSignal.normalize_volume(signal, target_dbfs, increase_only=True)
        if webrtcvad and trim_silence:
            signal = ToolsSignal.trim_long_silences(signal, sampling_rate)
        
        return signal, sampling_rate
        
    def trim_long_silences(signal, sampling_rate):
        int16_max = (2 ** 15) - 1
        print(sampling_rate)

        """
        Ensures that segments without voice in the waveform remain no longer than a 
        threshold determined by the VAD parameters in params.py.

        :param wav: the raw waveform as a numpy array of floats 
        :return: the same waveform with silences trimmed away (length <= original wav length)
        """
        
        ## Voice Activation Detection
        # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
        # This sets the granularity of the VAD. Should not need to be changed.
        vad_window_length = 30  # In milliseconds
        # Number of frames to average together when performing the moving average smoothing.
        # The larger this value, the larger the VAD variations must be to not get smoothed out. 
        vad_moving_average_width = 8
        # Maximum number of consecutive silent frames a segment can have.
        vad_max_silence_length = 6

        # Compute the voice detection window size
        desired_samples_per_window = (vad_window_length * sampling_rate) // 1000 
        samples_per_window = (vad_window_length * sampling_rate) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = signal[:len(signal) - (len(signal) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                            sample_rate=sampling_rate))
        voice_flags = np.array(voice_flags)

        audio_mask = ToolsSignal.moving_average(voice_flags, vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = ndimage.morphology.binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]

    # Smooth the voice detection with a moving average
    @staticmethod
    def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width