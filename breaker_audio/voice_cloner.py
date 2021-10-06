import json
from sys import path
import torch
import numpy as np
from pathlib import Path

from breaker_audio.tools_signal import ToolsSignal
from breaker_audio.component.encoder.encoder import Encoder
from breaker_audio.component.synthesizer.synthesizer import Synthesizer
from breaker_audio.component.vocoder.vocoder_wavernn import VocoderWavernn
from breaker_audio.component.vocoder.vocoder_melgan import VocoderMelgan


from breaker_audio.component_cmn.synthesizer.inference import Synthesizer as SynthesizerCmn

class VoiceClonerDefault:
 
    def __init__(self, path_dir_data:Path, language_code_639_3, *, low_mem=False) -> None:
        
        self.path_dir_data = path_dir_data

        # check language
        self.dict_language_supported = {
            'eng':'English',
            'nld':'Dutch',
            'cmn':'Mandarin'
        }
        if not language_code_639_3 in self.dict_language_supported:
            raise Exception()
        self.language_code_639_3 = language_code_639_3

        self.low_mem = low_mem

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
            print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
                    "%.1fGb total memory.\n" %
                    (torch.cuda.device_count(),
                    device_id,
                    gpu_properties.name,
                    gpu_properties.major,
                    gpu_properties.minor,
                    gpu_properties.total_memory / 1e9))

            self._device = 'cude'
        else:
            self._device = 'cpu'
        
        if self.language_code_639_3 == 'cmn':
            path_dir_model_encoder =      self.path_dir_data.joinpath('model', 'audio_cmn_ge2e_pretrained')
            path_dir_model_synthesizer =  self.path_dir_data.joinpath('model', 'audio_cmn_logs_syne')
            path_dir_model_vocoder =      self.path_dir_data.joinpath('model', 'audio_cmn_melgan_multi_speaker')


            self._encoder = Encoder(device='cpu')
            self._encoder.load_model(path_dir_model_encoder)
 
            self._synthesizer = SynthesizerCmn(path_dir_model_synthesizer)
            # self._synthesizer = Synthesizer(device='cpu')
            # self._synthesizer.load_model(path_dir_model_synthesizer)

            self._vocoder = VocoderMelgan(device='cpu')
            self._vocoder.load_model(path_dir_model_vocoder)

        elif self.language_code_639_3 == 'eng':
            path_dir_model_encoder =     self.path_dir_data.joinpath('model', 'audio_eng_rtv_pretrained_encoder')
            path_dir_model_synthesizer = self.path_dir_data.joinpath('model', 'audio_eng_rtv_pretrained_synthesizer')
            path_dir_model_vocoder =     self.path_dir_data.joinpath('model', 'audio_eng_rtv_pretrained_vocoder')

            self._encoder = Encoder(device='cpu')
            self._encoder.load_model(path_dir_model_encoder)
            self._synthesizer = Synthesizer(device='cpu')
            self._synthesizer.load_model(path_dir_model_synthesizer)
            self._vocoder = VocoderWavernn(device='cpu')
            self._vocoder.load_model(path_dir_model_vocoder)
            
        elif self.language_code_639_3 == 'nld':
            path_dir_model_encoder =     self.path_dir_data.joinpath('model', 'audio_nld_encoder')
            path_dir_model_synthesizer = self.path_dir_data.joinpath('model', 'audio_nld_synthesizer')
            path_dir_model_vocoder =     self.path_dir_data.joinpath('model', 'audio_nld_vocoder')

            self._encoder = Encoder(device='cuda')
            self._encoder.load_model(path_dir_model_encoder)
            self._synthesizer = Synthesizer(device='cuda')
            self._synthesizer.load_model(path_dir_model_synthesizer)
            self._vocoder = VocoderWavernn(device='cpu')
            self._vocoder.load_model(path_dir_model_vocoder)

        else:
            raise Exception('unimplemented language code: ' + self.language_code_639_3)


    def clone_voice(self, signal_voice_toclone):
        self.embeding = self._encoder.embed_utterance(signal_voice_toclone)

    def synthesize(self, text):
        if self.language_code_639_3 == 'cmn':
            array_melspec = self._synthesizer.synthesize_spectrograms([text], [self.embeding])[0]
            array_signal = self._vocoder.melspec_to_signal(torch.from_numpy(np.expand_dims(array_melspec, axis=0)).to(self._device)).squeeze().cpu().numpy()
            return array_signal, 16000

        elif self.language_code_639_3 == 'eng':
            array_melspec = self._synthesizer.synthesize_spectrograms([text], [self.embeding])[0]
            array_signal = self._vocoder.melspec_to_signal(array_melspec)
            array_signal, sampling_rate = ToolsSignal.preprocess_signal(array_signal, 16000)
            return array_signal, sampling_rate
        else:
            raise Exception('unimplemented language code: ' + self.language_code_639_3)
