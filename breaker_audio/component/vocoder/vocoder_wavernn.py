import torch

from breaker_audio.component.vocoder.models.fatchord_version import WaveRNN
from breaker_audio.component.vocoder import hparams as hp

class VocoderWavernn:
        
    def __init__(self, device:str) -> None:
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            if not device in ['cpu', 'cuda']:
                raise Exception('unkown device: ' + device)
            self._device = torch.device(device)
        self._model = None

    def load_model(self, path_dir_model):
        path_file_model = path_dir_model.joinpath('model.pt')
        self._model = WaveRNN(
            rnn_dims=hp.voc_rnn_dims,
            fc_dims=hp.voc_fc_dims,
            bits=hp.bits,
            pad=hp.voc_pad,
            upsample_factors=hp.voc_upsample_factors,
            feat_dims=hp.num_mels,
            compute_dims=hp.voc_compute_dims,
            res_out_dims=hp.voc_res_out_dims,
            res_blocks=hp.voc_res_blocks,
            hop_length=hp.hop_length,
            sample_rate=hp.sample_rate,
            mode=hp.voc_mode
        )
        
        checkpoint = torch.load(path_file_model, self._device)
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()


    def is_loaded(self):
        return self._model is not None

    def ensure_loaded(self):
        if self._model is None:
            raise Exception("Please load Wave-RNN in memory before using it")

    def melspec_to_signal(self, mel, normalize=True,  batched=True, target=8000, overlap=800, 
                    progress_callback=None):
        self.ensure_loaded()

        
        if normalize:
            mel = mel / hp.mel_max_abs_value
        mel = torch.from_numpy(mel[None, ...])
        wav = self._model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
        return wav
