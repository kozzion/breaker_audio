import sys
import json
from pathlib import Path

sys.path.append('../..')
from breaker_audio.component.encoder.train import TrainerEncoder

path_file_config_train = 'config_train.cfg'
with open(path_file_config_train, 'r') as file:
    dict_config_train = json.load(file)   
    

# Process the arguments
if __name__ == '__main__':    

    run_id = 'run_test'
    clean_data_root = Path(dict_config_train['path_dir_dataset_librispeech_target'])
    path_dir_model_output = Path("C:\\project\\data\\data_breaker\\model")
    path_dir_model_output.mkdir(exist_ok=True)
    save_every = 500
    backup_every = 7500
    TrainerEncoder.train(
        run_id=run_id,
        clean_data_root=clean_data_root, 
        models_dir=path_dir_model_output, 
        save_every=500,
        backup_every=7500, 
        force_restart=False)