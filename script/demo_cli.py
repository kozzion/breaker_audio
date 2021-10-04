import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from synthesizer import hparams
from synthesizer.utils import audio
from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import time
import torch
import sys
import shutil
import json

import aukit
from toolbox.sentence import xinqing_texts

example_texts = xinqing_texts

sample_dir = Path(r"../files")
reference_paths = [w for w in sorted(sample_dir.glob('*.wav'))]

if __name__ == '__main__':
    # ## Info & args
    # parser = argparse.ArgumentParser(
    #     description="命令行执行的Demo。",
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument("-e", "--enc_model_fpath", type=Path,
    #                     default="../models/encoder/saved_models/ge2e_pretrained.pt",
    #                     help="Path to a saved encoder")
    # parser.add_argument("-s", "--syn_model_dir", type=Path,
    #                     default="../models/synthesizer/saved_models/logs-syne/checkpoints",  # pretrained
    #                     help="Directory containing the synthesizer model")
    # parser.add_argument("-v", "--voc_model_fpath", type=Path,
    #                     default="../models/vocoder/saved_models/melgan/melgan_multi_speaker.pt",
    #                     help="Path to a saved vocoder")
    # parser.add_argument("-o", "--out_dir", type=Path,
    #                     default="../data/outs",
    #                     help="Path to a saved vocoder")
    # parser.add_argument("--low_mem", action="store_true", help= \
    #     "If True, the memory used by the synthesizer will be freed after each use. Adds large "
    #     "overhead but allows to save some GPU memory for lower-end GPUs.")
    # parser.add_argument("--no_sound", action="store_true", help= \
    #     "If True, audio won't be played.")

    path_file_encoder = Path('C:\\project\\voice\\zhrtvc-master\\models\\encoder\\saved_models\\ge2e_pretrained.pt')
    path_dir_synthesizer = Path('C:\\project\\voice\\zhrtvc-master\\models\\synthesizer\\saved_models\\logs-syne\\checkpoints')
    path_file_synthesizer_meta = Path('C:\\project\\voice\\zhrtvc-master\\models\\synthesizer\\saved_models\\logs-syne\\metas\\hparams.json')
    path_file_vocoder = Path('C:\\project\\voice\\zhrtvc-master\\models\\saved_models\\melgan\\melgan_multi_speaker.pt')
    
    # path_file_base = Path('C:\\project\\voice\\zhrtvc-master\\data\\samples\\stcmds\P00001A\\20170001P00001A0091.mp3')   # text spoken 谁敢拐你跟团吧要不
    path_file_base = Path('C:\project\data\data_breaker\\shu-voice_short.mp3')
  
  
 
    out_dir = Path('C:\\project\data\\data_breaker\\voice_out\\')
    path_file_text = Path('C:\\project\data\\data_breaker\\voice_out\\text.json')
    low_mem = False
    no_sound = False

    # if not args.no_sound:
    #     import sounddevice as sd

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
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

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(path_file_encoder, device='cpu')

    # 从模型目录导入hparams
    # hp_path = path_dir_synthesizer args.syn_model_dir.parent.joinpath("metas", "hparams.json")    # load from trained models
    if path_file_synthesizer_meta.exists():
        hparams = aukit.Dict2Obj(json.load(open(path_file_synthesizer_meta, encoding="utf8")))
        print('hparams: Yes')
    else:
        hparams = None
        print('hparams: No')

    synthesizer = Synthesizer(path_dir_synthesizer, low_mem=low_mem, hparams=hparams)

    # vocoder.load_model(args.voc_model_fpath)

    ## Run a test
    # print("Testing your configuration with small inputs.")
    # print("\tTesting the encoder...")
    # encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    # embed = np.random.rand(speaker_embedding_size)
    # embed /= np.linalg.norm(embed)
    # embeds = [embed, np.zeros(speaker_embedding_size)]

    dict_text = {}
    dict_text['text_0'] = "你好"
    dict_text['text_1'] = "欢迎使用语音克隆工具"
    dict_text['text_2'] = "谁敢拐你跟团吧要不"

    list_text = [
        '高婷',
        '游靜如',
        '刘紫薇',
        '張彩萍',
        '闫宝心',
        '樂怡',
        '贺茜',
        '耿芳',
        '李虹亭',
        '陈韵潼',
        '杨丽美',
        '金蘭',
        '黃燕玲',
        '卓羨華',
        '安鈺珍',
        '戴向阳',
        '朱新菊',
        '朱新菊',
        '关奕琳',
        '陈华金',
        '韓肖佳',
        '曾欣怡',
        '李照櫻',
        '林苑廷',
        '尹幼峰',
        '宋美枝',
        '朱慧玲',
        '顏小惠',
        '林芳如']
    dict_text = {}
    for i, text in enumerate(list_text):
        dict_text['text_' + str(i).rjust(3).replace(' ', "0")] = text
    
    with open(path_file_text, 'w') as file:
        json.dump(dict_text, file)
    # print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    # mels = synthesizer.synthesize_spectrograms(list(dict_text.values()), embeds)

    # mel = np.concatenate(mels, axis=1)
    # no_action = lambda *args: None

    # generated_wav = audio.inv_melspectrogram(mel, hparams=audio.melgan_hparams)
    # print("All test passed! You can now synthesize speech.\n\n")

    print("Loop")
    num_generated = 0
    out_dir.mkdir(exist_ok=True, parents=True)
    for id_text, text in dict_text.items():
        print('Reference audio: {}'.format(path_file_base))
        # - Directly load from the filepath:
        preprocessed_wav = encoder.preprocess_wav(path_file_base)
        print("Loaded file succesfully")

        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")

        ## Generating the spectrogram
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]

        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")

        ## Generating the waveform
        print("Synthesizing the waveform:")

        # generated_wav = audio.inv_melspectrogram(spec, hparams=audio.melgan_hparams)
        generated_wav = synthesizer.griffin_lim(spec, hparams=synthesizer.hparams)
        # generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Play the audio (non-blocking)
        # if not args.no_sound:
        #     sd.stop()
        #     sd.play(generated_wav, synthesizer.sample_rate)
        #     sd.wait()

        # Save it on the disk
        cur_time = time.strftime('%Y%m%d_%H%M%S')
        path_file_output = out_dir.joinpath(id_text + ".wav")
        # librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
        audio.save_wav(generated_wav, path_file_output, synthesizer.sample_rate)  # save

        num_generated += 1
        print("\nSaved output as %s\n\n" % path_file_output)
