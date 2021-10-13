from os import path
import sys
import json
from pathlib import Path

sys.path.append('../..')
from breaker_audio.tools_dataset import ToolsDataset


if __name__ == '__main__':   
    path_file_config_train = 'config_train.cfg'
    with open(path_file_config_train, 'r') as file:
        dict_config_train = json.load(file)   

    path_dir_dataset_source = Path(dict_config_train['path_dir_dataset_source'])
    path_dir_dataset_target = Path(dict_config_train['path_dir_dataset_librispeech_target'])
    path_dir_dataset_target.mkdir(exist_ok=True)
    ToolsDataset.preprocess_librispeech(path_dir_dataset_source, path_dir_dataset_target, skip_existing=True)


