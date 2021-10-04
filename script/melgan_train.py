#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
"""
import argparse
from utils.argutils import print_args
from melgan.train import train_melgan

if __name__ == "__main__":
    try:
        from setproctitle import setproctitle

        setproctitle('zhrtvc-melgan-train')
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="训练MelGAN声码器模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--data_path", type=str, default=r"../data/samples/metadata.csv",
                        help='metadata path')
    parser.add_argument("-o", "--save_path", type=str, default='../models/melgan/samples',
                        help=r"your model save dir")
    parser.add_argument("--load_path", type=str, default=None,
                        help=r"pretrained generator model path")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=10)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=4800)  # 8192

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=4)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--mode", type=str, default='mellotron')
    parser.add_argument("--ratios", type=str, default='5 5 4 2')  # '8 8 2 2'

    args = parser.parse_args()

    print_args(args, parser)
    train_melgan(args)
