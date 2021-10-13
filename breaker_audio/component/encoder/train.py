
import sys
import torch
from pathlib import Path
# from multiprocessing import freeze_support

from breaker_audio.component.encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader 
from breaker_audio.component.encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from breaker_audio.component.encoder.params_model import *
from breaker_audio.component.encoder.model_speaker import ModelSpeaker


class TrainerEncoder:
    @staticmethod
    def sync(device: torch.device):
        # For correct profiling (cuda operations are async)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
    @staticmethod
    def train(run_id: str, 
        clean_data_root: Path, 
        models_dir: Path,  
        save_every: int,
        backup_every: int, 
        force_restart: bool):

        # freeze_support() #TODO this seems silly
        
        # Create a dataset and a dataloader
        dataset = SpeakerVerificationDataset(clean_data_root)
        loader = SpeakerVerificationDataLoader(
            dataset,
            speakers_per_batch,
            utterances_per_speaker,
            num_workers=8,
        )
        
        # Setup the device on which to run the forward pass and the loss. These can be different, 
        # because the forward pass is faster on the GPU whereas the loss is often (depending on your
        # hyperparameters) faster on the CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(device)
        # FIXME: currently, the gradient is None if loss_device is cuda
        loss_device = torch.device("cpu")
        
        # Create the model and the optimizer
        model = ModelSpeaker(device, loss_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
        init_step = 1
        
        # Configure file path for the model
        state_fpath = models_dir.joinpath(run_id + ".pt")
        backup_dir = models_dir.joinpath(run_id + "_backups")

        # Load any existing model
        if not force_restart:
            if state_fpath.exists():
                print("Found existing model \"%s\", loading it and resuming training." % run_id)
                checkpoint = torch.load(state_fpath)
                init_step = checkpoint["step"]
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                optimizer.param_groups[0]["lr"] = learning_rate_init
            else:
                print("No model \"%s\" found, starting training from scratch." % run_id)
        else:
            print("Starting the training from scratch.")
        model.train()
        
        # Training loop

        # for speaker_batch in loader:
        for step, speaker_batch in enumerate(loader, init_step):            
            # Forward pass
            print('step: ' + str(step))
            sys.stdout.flush()
            inputs = torch.from_numpy(speaker_batch.data).to(device)

            TrainerEncoder.sync(device) #TODO what is this?
            embeds = model(inputs)
            
            TrainerEncoder.sync(device)
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            
            print('loss: ' + str(loss))
            print('eer: ' + str(eer))
            sys.stdout.flush()

            TrainerEncoder.sync(loss_device)

            # Backward pass
            model.zero_grad()
            loss.backward()
            model.do_gradient_ops()
            optimizer.step()
            
        

            # Overwrite the latest version of the model
            if save_every != 0 and step % save_every == 0:
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)
                
            # Make a backup
            if backup_every != 0 and step % backup_every == 0:
                print("Making a backup (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)
 