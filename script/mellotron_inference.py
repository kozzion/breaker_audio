#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/11/28
"""
mellotron_inference
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import aukit
import time
import json
import traceback
import argparse

from mellotron.inference import *

xinqing_texts = """生活岂能百般如意
正因有了遗漏和缺憾
我们才会有所追寻
功成莫自得
或许下一步就是陷阱
败后勿卑微
没有谁一直紧锁冬寒
哪怕再平凡
平常
平庸
都不能让梦想之地荒芜
人生最可怕的
不是你置身何处
而是不知走向哪里
人有很多面具，上班一个，下班一个；工作一个，娱乐一个；面对老板一个，面对客户一个；面对同事一个，面对朋友一个……
能看到你不戴面具的人，是你最亲近最值得珍惜的人。
不管发生了什么事，不要抱怨生命中的任何一天，真正的大心境，在于放弃后的坦然，也在于放下后的无意。
人生反复，总会让我们悲喜交织，遇时请珍重，路过请祝福。拥有一个良好心态，生活就是一片艳阳天！
人生就像不停在用的铅笔，开始很尖，但慢慢地就磨得圆滑了。不过，太过圆滑了也不好，那就意味着差不多该挨削了。所以幸福是修来的，不是求来的。
假如你很感性就不必去模仿别人的理性
假如你很率真就不必去模仿别人的世故
假如你很随性就不必去模仿别人的拘谨
知道自己是什么样的人
要做什么
无需活在别人非议或期待里
你勤奋充电努力工作
对人微笑
是为了扮靓自己照亮自己的心
告诉自己
我是一股独立向上的力量
一个人的自愈的能力越强，才越有可能接近幸福。做一个寡言，却心有一片海的人，不伤人害己，于淡泊中，平和自在。
年轻不是资本，健康才是；年轻也不是值得炫耀的谈资，健康的生活方式才是。在病痛和死亡面前，年轻不堪一击。
经历越多就越不想说话，环境的不同，想说的话别人未必能懂，也就慢慢学会了自己默默承受。
你冷落了我
我也会慢慢地冷落你
能习惯有你的陪伴
也能习惯没有你的孤单
慢慢地
都淡了
渐渐地
都忘了
世上的事就是这样
无论友情还是爱情
当一个人忽略你时
不要伤心
每个人都有自己的生活
谁都不可能一直陪你
二三十岁的时候，是最艰苦的一段岁月，承担着渐重的责任，拿着与工作量不匹配的薪水，艰难地权衡事业和感情……
但你总得学着坚强，毕竟你不扬帆，没人替你起航。
不要只因一次挫败，就迷失了最初想抵达的远方。
人生没有假设，当下即是全部。总是看到比自己优秀的人，说明你正在走上坡路；总是看到不如自己的人，说明你正在走下坡路。
与其埋怨，不如思变。
人总要学会长大，学会原谅，学会倾听，学会包容，学会坚强。这段路，是逼自己撑过来的。
人生的退步有时比进步更重要。因为回归到内心和本质。
年龄越长，越不肯交出真心了。退到最初，守着朴素和贞淑，与时间化干戈为玉帛了。
心情不是人生的全部
却能左右人生的全部
心情好
什么都好
心情不好
一切都乱了
我们常常不是输给了别人
而是坏心情贬低了我们的形象
降低了我们的能力
扰乱了我们的思维
从而输给了自己
控制好心情
生活才会处处祥和
好的心态塑造好心情
好心情塑造最出色的你
人成熟的过程，其实是学会与自我相处的过程。
这个过程必然伴随着从热闹到安静，从慌张到淡定，从迷茫到自知，从有人陪伴到泰然独处。
生活就是这样，你所做的一切不能让每个人都满意，不要为了讨好别人而丢失自己的本性，因为每个人都有原则和自尊，
没有不被评说的事，没有不被猜测的人，做最真实最漂亮的自己，依心而行，无憾今生。
只要以开阔的心胸，包容各种不同的人事物，自然能看到最美好的一面，领悟到世界上真正美妙、可爱之处。
你坚强了太久
偶尔发个累的状态想发泄一下
就会有人说你真矫情
你晒了几张玩乐的照片
就会有人说你怎么只知道玩
他们哪知道这是你自己给自己放的假而已
你平时嘻嘻哈哈
但这从来不代表你没心没肺
不代表你没有不想理人的时候
愿你不求被所有人理解
愿有人能够理解你
有人说要感谢前任让你成长，让你变得更好，
我觉得不是这样，那些痛不欲生撕心裂肺的日子，都是你咬着牙一天一天熬过来的，凭什么要谢别人，你要谢谢你自己。
别让自己活得太累。应该学着想开、看淡，学着不强求，学着深藏。适时放松自己，寻找宣泄，给疲惫的心灵解解压。
人之所以会烦恼，就是记性太好，记住了那些不该记住的东西。所以，记住快乐的事，忘记令你悲伤的事。""".split("\n")

aliaudio_fpaths = [str(w) for w in sorted(Path(r'../data/samples/aliaudio').glob('*/*.mp3'))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--checkpoint_path', type=str,
                        default=r"../models/mellotron/samples/checkpoint-000000.pt",
                        help='模型路径。')
    parser.add_argument('-s', '--speakers_path', type=str,
                        default=r"../models/mellotron/samples/metadata/speakers.json",
                        help='发音人映射表路径。')
    parser.add_argument("-o", "--out_dir", type=Path, default=r"../models/mellotron/samples/test",
                        help='保存合成的数据路径。')
    parser.add_argument("-p", "--play", type=int, default=0,
                        help='是否合成语音后自动播放语音。')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--hparams_path', type=str,
                        default=r"../models/mellotron/samples/metadata/hparams.json",
                        required=False, help='comma separated name=value pairs')
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path your trained encoder model.")

    args = parser.parse_args()
    args_hparams = open(args.hparams_path, encoding='utf8').read()
    _hparams = create_hparams(args_hparams)

    model_path = args.checkpoint_path
    load_model_mellotron(model_path, hparams=_hparams)

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

    msyner = MellotronSynthesizer(model_path=args.checkpoint_path, speakers_path=args.speakers_path,
                                  hparams_path=args.hparams_path)

    spec = msyner.synthesize(text='你好，欢迎使用语言合成服务。', speaker='speaker')

    ## Run a test
    # spec, align = synthesize_one('你好，欢迎使用语言合成服务。', aliaudio_fpaths[0], with_alignment=True,
    #                              hparams=_hparams, encoder_fpath=args.encoder_model_fpath)
    print("Spectrogram shape: {}".format(spec.shape))
    # print("Alignment shape: {}".format(align.shape))
    wav = griffinlim_vocoder(spec)
    print("Waveform shape: {}".format(wav.shape))

    print("All test passed! You can now synthesize speech.\n\n")

    print("Interactive generation loop")
    num_generated = 0
    args.out_dir.mkdir(exist_ok=True, parents=True)
    speaker_index_dict = json.load(open(args.speakers_path, encoding='utf8'))
    speaker_names = list(speaker_index_dict.keys())
    example_texts = xinqing_texts
    example_fpaths = aliaudio_fpaths
    while True:
        try:
            # Get the reference audio filepath
            speaker = input("Speaker:\n")
            if not speaker.strip():
                speaker = np.random.choice(speaker_names)
            print('Speaker: {}'.format(speaker))

            ## Generating the spectrogram
            text = input("Text:\n")
            if not text.strip():
                text = np.random.choice(example_texts)
            print('Text: {}'.format(text))
            # The synthesizer works in batch, so you need to put your data in a list or numpy array

            print("Creating the spectrogram ...")
            spec = msyner.synthesize(text=text, speaker=speaker)
            # spec, align = synthesize_one(text, speaker=speaker, with_alignment=True,
            #                              hparams=_hparams, encoder_fpath=args.encoder_model_fpath)
            print("Spectrogram shape: {}".format(spec.shape))
            # print("Alignment shape: {}".format(align.shape))
            ## Generating the waveform
            print("Synthesizing the waveform ...")
            wav = griffinlim_vocoder(spec)
            print("Waveform shape: {}".format(wav.shape))

            # Save it on the disk
            cur_time = time.strftime('%Y%m%d_%H%M%S')
            fpath = args.out_dir.joinpath("demo_out_{}.wav".format(cur_time))
            # librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
            aukit.save_wav(wav, fpath, sr=_hparams.sampling_rate)  # save

            txt_path = args.out_dir.joinpath("info_dict.txt".format(cur_time))
            with open(txt_path, 'at', encoding='utf8') as fout:
                dt = dict(text=text, audio_path=str(fpath), speaker=speaker, time=cur_time)
                out = json.dumps(dt, ensure_ascii=False)
                fout.write('{}\n'.format(out))

            num_generated += 1
            print("\nSaved output as %s\n\n" % fpath)
            if args.play:
                aukit.play_audio(fpath, sr=_hparams.sampling_rate)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            traceback.print_exc()
