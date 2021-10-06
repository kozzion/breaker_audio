
import librosa
import numpy as np

from typing import Tuple
from multiprocess.pool import ThreadPool
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from functools import partial
from itertools import chain


from breaker_audio.component.utils import logmmse
from breaker_audio.component.synthesizer import audio
from breaker_audio.component.encoder.encoder import Encoder
from breaker_audio.component.encoder import audio
from breaker_audio.component.encoder import params_data
from breaker_audio.component.encoder.params_data import *
from breaker_audio.component.encoder.config import librispeech_datasets, anglophone_nationalites

class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """
    def __init__(self, root:Path, name:str):
        print(root)
        print(name)
        print("log_"  + name.replace("/", "_") + ".txt")
        path_file_log = root.joinpath("Log_"  + name.replace("/", "_") + ".txt")
        self.text_file = open(path_file_log, "w")
        self.sample_data = dict()
        
        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()
        
    def _log_params(self):
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")
    
    def write_line(self, line):
        self.text_file.write("%s\n" % line)
        
    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)
            
    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()
       
class ToolsDataset: 
    
    @staticmethod
    def _init_preprocess_dataset(dataset_name, datasets_root, out_dir):
        dataset_root = datasets_root.joinpath(dataset_name)
        if not dataset_root.exists():
            print("Couldn\'t find %s, skipping this dataset." % dataset_root)
            return None, None
        return dataset_root, DatasetLog(out_dir, dataset_name)

    @staticmethod
    def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension,
                                skip_existing, logger):
        print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))
        
        # Function to preprocess utterances for one speaker
        def preprocess_speaker(speaker_dir: Path):
            # Give a name to the speaker that includes its dataset
            speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
            
            # Create an output directory with that name, as well as a txt file containing a 
            # reference to each source file.
            speaker_out_dir = out_dir.joinpath(speaker_name)
            speaker_out_dir.mkdir(exist_ok=True)
            sources_fpath = speaker_out_dir.joinpath("_sources.txt")
            
            # There's a possibility that the preprocessing was interrupted earlier, check if 
            # there already is a sources file.
            if sources_fpath.exists():
                try:
                    with sources_fpath.open("r") as sources_file:
                        existing_fnames = {line.split(",")[0] for line in sources_file}
                except:
                    existing_fnames = {}
            else:
                existing_fnames = {}
            
            # Gather all audio files for that speaker recursively
            sources_file = sources_fpath.open("a" if skip_existing else "w")
            for in_fpath in speaker_dir.glob("**/*.%s" % extension):
                # Check if the target output file already exists
                out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
                out_fname = out_fname.replace(".%s" % extension, ".npy")
                if skip_existing and out_fname in existing_fnames:
                    continue
                    
                # Load and preprocess the waveform
                wav = audio.preprocess_wav(in_fpath)
                if len(wav) == 0:
                    continue
                
                # Create the mel spectrogram, discard those that are too short
                frames = audio.wav_to_mel_spectrogram(wav)
                if len(frames) < partials_n_frames:
                    continue
                
                out_fpath = speaker_out_dir.joinpath(out_fname)
                np.save(out_fpath, frames)
                logger.add_sample(duration=len(wav) / sampling_rate)
                sources_file.write("%s,%s\n" % (out_fname, in_fpath))
            
            sources_file.close()
        
        # Process the utterances for each speaker
        with ThreadPool(8) as pool:
            list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                    unit="speakers"))
        logger.finalize()
        print("Done preprocessing %s.\n" % dataset_name)

    @staticmethod
    def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing=False):
        for dataset_name in librispeech_datasets["train"]["other"]:
            # Initialize the preprocessing
            dataset_root, logger = ToolsDataset._init_preprocess_dataset(dataset_name, datasets_root, out_dir)
            if not dataset_root:
                return 
            
            # Preprocess all speakers
            speaker_dirs = list(dataset_root.glob("*"))
            ToolsDataset._preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac",
                                    skip_existing, logger)

    @staticmethod
    def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False):
        # Initialize the preprocessing
        dataset_name = "VoxCeleb1"
        dataset_root, logger = ToolsDataset._init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Get the contents of the meta file
        with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
            metadata = [line.split("\t") for line in metafile][1:]
        
        # Select the ID and the nationality, filter out non-anglophone speakers
        nationalities = {line[0]: line[3] for line in metadata}
        keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
                            nationality.lower() in anglophone_nationalites]
        print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." % 
            (len(keep_speaker_ids), len(nationalities)))
        
        # Get the speaker directories for anglophone speakers only
        speaker_dirs = dataset_root.joinpath("wav").glob("*")
        speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                        speaker_dir.name in keep_speaker_ids]
        print("VoxCeleb1: found %d anglophone speakers on the disk, %d missing (this is normal)." % 
            (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))

        # Preprocess all speakers
        ToolsDataset._preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                skip_existing, logger)


    def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False):
        # Initialize the preprocessing
        dataset_name = "VoxCeleb2"
        dataset_root, logger = ToolsDataset._init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return
        
        # Get the speaker directories
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
        ToolsDataset._preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "m4a",
                                skip_existing, logger)

    #
    # secton synthesiser
    #
    
    @staticmethod
    def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int,
                            skip_existing: bool, hparams, no_alignments: bool,
                            datasets_name: str, subfolders: str):
        # Gather the input directories
        dataset_root = datasets_root.joinpath(datasets_name)
        input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
        print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
        assert all(input_dir.exists() for input_dir in input_dirs)
        
        # Create the output directories for each output file type
        out_dir.joinpath("mels").mkdir(exist_ok=True)
        out_dir.joinpath("audio").mkdir(exist_ok=True)
        
        # Create a metadata file
        metadata_fpath = out_dir.joinpath("train.txt")

        with metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8") as file:
            # Preprocess the dataset
            speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
            func = partial(ToolsDataset.preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, 
                        hparams=hparams, no_alignments=no_alignments)
            job = Pool(n_processes).imap(func, speaker_dirs)
            for speaker_metadata in tqdm(job, datasets_name, len(speaker_dirs), unit="speakers"):
                for metadatum in speaker_metadata:
                    file.write("|".join(str(x) for x in metadatum) + "\n")


        # Verify the contents of the metadata file
        with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        mel_frames = sum([int(m[4]) for m in metadata])
        timesteps = sum([int(m[3]) for m in metadata])
        sample_rate = hparams.sample_rate
        hours = (timesteps / sample_rate) / 3600
        print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
            (len(metadata), mel_frames, timesteps, hours))
        print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
        print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
        print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))

    @staticmethod
    def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
        metadata = []
        for book_dir in speaker_dir.glob("*"):
            if no_alignments:
                # Gather the utterance audios and texts
                # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
                extensions = ["*.wav", "*.flac", "*.mp3"]
                for extension in extensions:
                    wav_fpaths = book_dir.glob(extension)

                    for wav_fpath in wav_fpaths:
                        # Load the audio waveform
                        wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
                        if hparams.rescale:
                            wav = wav / np.abs(wav).max() * hparams.rescaling_max

                        # Get the corresponding text
                        # Check for .txt (for compatibility with other datasets)
                        text_fpath = wav_fpath.with_suffix(".txt")
                        if not text_fpath.exists():
                            # Check for .normalized.txt (LibriTTS)
                            text_fpath = wav_fpath.with_suffix(".normalized.txt")
                            assert text_fpath.exists()
                        with text_fpath.open("r") as text_file:
                            text = "".join([line for line in text_file])
                            text = text.replace("\"", "")
                            text = text.strip()

                        # Process the utterance
                        metadata.append(ToolsDataset.process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
                                                        skip_existing, hparams))
            else:
                # Process alignment file (LibriSpeech support)
                # Gather the utterance audios and texts
                try:
                    alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                    with alignments_fpath.open("r") as alignments_file:
                        alignments = [line.rstrip().split(" ") for line in alignments_file]
                except StopIteration:
                    # A few alignment files will be missing
                    continue

                # Iterate over each entry in the alignments file
                for wav_fname, words, end_times in alignments:
                    wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                    assert wav_fpath.exists()
                    words = words.replace("\"", "").split(",")
                    end_times = list(map(float, end_times.replace("\"", "").split(",")))

                    # Process each sub-utterance
                    wavs, texts = ToolsDataset.split_on_silences(wav_fpath, words, end_times, hparams)
                    for i, (wav, text) in enumerate(zip(wavs, texts)):
                        sub_basename = "%s_%02d" % (wav_fname, i)
                        metadata.append(ToolsDataset.process_utterance(wav, text, out_dir, sub_basename,
                                                        skip_existing, hparams))

        return [m for m in metadata if m is not None]

    @staticmethod
    def split_on_silences(wav_fpath, words, end_times, hparams):
        # Load the audio waveform
        wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        
        words = np.array(words)
        start_times = np.array([0.0] + end_times[:-1])
        end_times = np.array(end_times)
        assert len(words) == len(end_times) == len(start_times)
        assert words[0] == "" and words[-1] == ""
        
        # Find pauses that are too long
        mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
        mask[0] = mask[-1] = True
        breaks = np.where(mask)[0]

        # Profile the noise from the silences and perform noise reduction on the waveform
        silence_times = [[start_times[i], end_times[i]] for i in breaks]
        silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
        noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
        if len(noisy_wav) > hparams.sample_rate * 0.02:
            profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
            wav = logmmse.denoise(wav, profile, eta=0)
        
        # Re-attach segments that are too short
        segments = list(zip(breaks[:-1], breaks[1:]))
        segment_durations = [start_times[end] - end_times[start] for start, end in segments]
        i = 0
        while i < len(segments) and len(segments) > 1:
            if segment_durations[i] < hparams.utterance_min_duration:
                # See if the segment can be re-attached with the right or the left segment
                left_duration = float("inf") if i == 0 else segment_durations[i - 1]
                right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
                joined_duration = segment_durations[i] + min(left_duration, right_duration)

                # Do not re-attach if it causes the joined utterance to be too long
                if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                    i += 1
                    continue

                # Re-attach the segment with the neighbour of shortest duration
                j = i - 1 if left_duration <= right_duration else i
                segments[j] = (segments[j][0], segments[j + 1][1])
                segment_durations[j] = joined_duration
                del segments[j + 1], segment_durations[j + 1]
            else:
                i += 1
        
        # Split the utterance
        segment_times = [[end_times[start], start_times[end]] for start, end in segments]
        segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
        wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
        texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]
        
        # # DEBUG: play the audio segments (run with -n=1)
        # import sounddevice as sd
        # if len(wavs) > 1:
        #     print("This sentence was split in %d segments:" % len(wavs))
        # else:
        #     print("There are no silences long enough for this sentence to be split:")
        # for wav, text in zip(wavs, texts):
        #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
        #     # when playing them. You shouldn't need to do that in your parsers.
        #     wav = np.concatenate((wav, [0] * 16000))
        #     print("\t%s" % text)
        #     sd.play(wav, 16000, blocking=True)
        # print("")
        
        return wavs, texts
        

    @staticmethod
    def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str, 
                        skip_existing: bool, hparams):
        ## FOR REFERENCE:
        # For you not to lose your head if you ever wish to change things here or implement your own
        # synthesizer.
        # - Both the audios and the mel spectrograms are saved as numpy arrays
        # - There is no processing done to the audios that will be saved to disk beyond volume  
        #   normalization (in split_on_silences)
        # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
        #   is why we re-apply it on the audio on the side of the vocoder.
        # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
        #   without extra padding. This means that you won't have an exact relation between the length
        #   of the wav and of the mel spectrogram. See the vocoder data loader.
        
        
        # Skip existing utterances if needed
        mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
        wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
        if skip_existing and mel_fpath.exists() and wav_fpath.exists():
            return None

        # Trim silence
        if hparams.trim_silence:
            wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
        
        # Skip utterances that are too short
        if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
            return None
        
        # Compute the mel spectrogram
        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        mel_frames = mel_spectrogram.shape[1]
        
        # Skip utterances that are too long
        if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
            return None
        
        # Write the spectrogram, embed and audio to disk
        np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
        np.save(wav_fpath, wav, allow_pickle=False)
        
        # Return a tuple describing this training example
        return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text
    
    @staticmethod
    def embed_utterance(fpaths, encoder_model_fpath):
        if not encoder.is_loaded():
            encoder.load_model(encoder_model_fpath)

        # Compute the speaker embedding of the utterance
        wav_fpath, embed_fpath = fpaths
        wav = np.load(wav_fpath)
        wav = encoder.preprocess_wav(wav)
        embed = encoder.embed_utterance(wav)
        np.save(embed_fpath, embed, allow_pickle=False)
        
    @staticmethod
    def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
        wav_dir = synthesizer_root.joinpath("audio")
        metadata_fpath = synthesizer_root.joinpath("train.txt")
        assert wav_dir.exists() and metadata_fpath.exists()
        embed_dir = synthesizer_root.joinpath("embeds")
        embed_dir.mkdir(exist_ok=True)
        
        # Gather the input wave filepath and the target output embed filepath
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
            fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]
            
        # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
        # Embed the utterances in separate threads
        func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
        job = Pool(n_processes).imap(func, fpaths)
        list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
        
