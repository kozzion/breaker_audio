import json
from sys import path
import torch
import numpy as np
from pathlib import Path

from breaker_audio.tools_signal import ToolsSignal
from breaker_audio.component_cmn.encoder_voice import EncoderVoice
from breaker_audio.component_cmn.synthesizer.inference import Synthesizer as SynthesizerCmn
from breaker_audio.component_cmn.melgan.inference import MelVocoder as MelVocoderCmn
from breaker_audio.component_cmn.melgan.inference import get_default_device

from breaker_audio.component_eng.synthesizer.inference import Synthesizer as SynthesizerEng
from breaker_audio.component_eng.vocoder import inference as VocoderEng

class VoiceClonerDefault:

    
    def __init__(self, path_dir_data:Path, language_code_639_3, *, low_mem=False) -> None:
        
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

        self._device = get_default_device()

        
        if self.language_code_639_3 == 'cmn':
            path_file_model_encoder =    Path('C:\\project\\data\\data_breaker\\model\\audio_cmn_ge2e_pretrained\\model.pt')
            path_dir_synthesizer =       Path('C:\\project\\data\\data_breaker\\model\\audio_cmn_logs_syne\\')
            path_file_model_vocoder =    Path('C:\\project\\data\\data_breaker\\model\\audio_cmn_melgan_multi_speaker\\model.pt')
            


            self._encoder = EncoderVoice()
            self._encoder.load_model(path_file_model_encoder, device='cpu')
 
            self._synthesizer = SynthesizerCmn(path_dir_synthesizer)
            self._vocoder = MelVocoderCmn(
                path_file_model_vocoder, 
                github=True, 
                args_path= '', 
                device=self._device, 
                mode='default')

        elif self.language_code_639_3 == 'eng':
            path_file_model_encoder =     Path('C:\\project\\data\\data_breaker\\model\\audio_eng_rtv_pretrained_encoder\\model.pt')
            path_file_model_synthesizer = Path('C:\\project\\data\\data_breaker\\model\\audio_eng_rtv_pretrained_synthesizer\\model.pt')
            path_file_model_vocoder =     Path('C:\\project\\data\\data_breaker\\model\\audio_eng_rtv_pretrained_vocoder\\model.pt')

            self._encoder = EncoderVoice()
            self._encoder.load_model(path_file_model_encoder)
            self._synthesizer = SynthesizerEng(path_file_model_synthesizer)
            VocoderEng.load_model(path_file_model_vocoder)


        
        elif self.language_code_639_3 == 'nld':
            path_file_model_encoder =     Path()
            path_file_model_synthesizer = Path()
            path_file_model_vocoder =     Path()

            self._encoder = EncoderVoice()
            self._encoder.load_model(path_file_model_encoder)
            self._synthesizer = SynthesizerEng(path_file_model_synthesizer)
            VocoderEng.load_model(path_file_model_vocoder)


            
        else:
            raise Exception('unimplemented language code: ' + self.language_code_639_3)


    def clone_voice(self, signal_voice_toclone):
        self.embeding = self._encoder.embed_utterance(signal_voice_toclone)

    def synthesize(self, text):
        if self.language_code_639_3 == 'cmn':
            array_melspec = self._synthesizer.synthesize_spectrograms([text], [self.embeding])[0]
            array_signal = self._vocoder.inverse(torch.from_numpy(np.expand_dims(array_melspec, axis=0)).to(self._device)).squeeze().cpu().numpy()
            return array_signal, 16000

        elif self.language_code_639_3 == 'eng':
            array_melspec = self._synthesizer.synthesize_spectrograms([text], [self.embeding])[0]
            array_signal = VocoderEng.infer_waveform(array_melspec)
            array_signal, sampling_rate = ToolsSignal.preprocess_signal(array_signal, 16000)
            return array_signal, sampling_rate
        else:
            raise Exception('unimplemented language code: ' + self.language_code_639_3)
