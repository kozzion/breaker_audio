import torch
import numpy as np
from typing import List
from scipy.stats import norm

from breaker_audio.component.encoder.encoder import Encoder


class VoiceAuthenticator(object):
    
    def __init__(self, path_dir_data) -> None:
        super().__init__()
        self.path_dir_data = path_dir_data

        #TODO do somethin here that makes this device stuff work better and not repeat everywhere
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

            self._device = 'cuda'
        else:
            self._device = 'cpu'

        # path_dir_model = self.path_dir_data.joinpath('model', 'audio_eng_rtv_pretrained_encoder')
        path_dir_model = self.path_dir_data.joinpath('model', 'audio_eng_resemblyzer')

        self._encoder = Encoder(device='cpu')
        self._encoder.load_model(path_dir_model)
 
    
    def encode(self, signal:np.ndarray, sampling_rate:int) -> np.ndarray:
        return self._encoder.embed_utterance(signal)

    def authenticate(self, encoding_a:np.ndarray, list_encoding_b:List[np.ndarray]) -> 'dict':
        authentication_report = {}
        if len(list_encoding_b) < 3:
            authentication_report['size_challange'] = len(list_encoding_b)
            authentication_report['core_match'] = 0 
            authentication_report['comment'] = 'need at least 3 samples in the challange'
            authentication_report['list_val_a_b'] = []
            authentication_report['list_val_b_b'] = []
            return authentication_report 


        array_encoding_b = np.vstack(list_encoding_b)

        # select the upper triangle of the covariance matrix (excluding the diagonal)
        list_val_b_b = (array_encoding_b @ array_encoding_b.transpose())[np.triu_indices(len(list_encoding_b), k = 1)].tolist()
        print(type(list_val_b_b[0]))
        # just iterate over the lis for the second one
        list_val_a_b = np.array([encoding_a @ encoding_b for encoding_b in list_encoding_b]).tolist()
        print(type(list_val_a_b[0]))
        std = np.mean([np.std(list_val_b_b), np.std(list_val_a_b)])
        #TODO we can do a proper test here
        z = np.abs(np.mean(list_val_b_b) - np.mean(list_val_a_b)) / std
        
        score_match = 1 - norm.cdf(z)    
        authentication_report['size_challange'] = len(list_encoding_b)
        authentication_report['score_match'] = score_match
        authentication_report['comment'] = ''
        authentication_report['list_val_a_b'] = list_val_a_b
        authentication_report['list_val_b_b'] = list_val_b_b
        return authentication_report