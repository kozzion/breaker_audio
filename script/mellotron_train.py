import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mellotron'))

import argparse
import torch

from mellotron.hparams import create_hparams
from mellotron.train import train, json_dump

if __name__ == '__main__':
    try:
        from setproctitle import setproctitle

        setproctitle('zhrtvc-mellotron-train')
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, default=r'../data/samples/metadata.csv',
                        help='directory to save checkpoints')
    parser.add_argument('-o', '--output_directory', type=str, default=r"../models/mellotron/samples",
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default='tensorboard',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        default='{"batch_size":4,"iters_per_checkpoint":10,"learning_rate":0.001,"dataloader_num_workers":1}',
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    meta_folder = os.path.join(args.output_directory, 'metadata')
    os.makedirs(meta_folder, exist_ok=True)

    path = os.path.join(meta_folder, "args.json")
    obj = args.__dict__
    json_dump(obj, path)

    train(args.input_directory, args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
