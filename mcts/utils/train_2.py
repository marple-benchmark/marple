import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import wandb

from datetime import datetime

from MarpleLongModels.data.dataset import BasicDataset
from MarpleLongModels.models.pl_models import BasicModel, LowBasicModel
from MarpleLongModels.models.models import PolicyModel, TransformerHeadPolicyModel, HistAwareTransformerHeadPolicyModel, AudioConditionedTransformerHeadPolicyModel, LowPolicyModel, SubgoalConditionedLowPolicyModel

import os 

from arguments import Arguments # arguments, defaults overwritten by config/config.yaml

os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(config_path="config", config_name="train.yaml")
def main(args: DictConfig) -> None:
    #wandb.login(key='409d576b1e20724351b01a9d45b006f36972d20f')
    os.environ['WANDB_API_KEY'] = '409d576b1e20724351b01a9d45b006f36972d20f' 
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")    
    args.wandb.project = f"marple_long_{args.experiment.experiment_name}"
    args.wandb.name = f"2-{args.model.model_name}-{timestamp}"
    args.wandb.save_dir = f"/vision/u/emilyjin/marple_long/experiments/data_level_2/{args.experiment.experiment_name}/{args.model.model_name}"  

    # args.trainer.devices = num_gpus
    args.trainer.accelerator = 'gpu'
    args.trainer.devices = 1#find_usable_cuda_devices(2)
    accumulate = 1 #if args.data.data_level == 'low' else 2

    torch.cuda.empty_cache()

    num_cpus = os.cpu_count()

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        num_gpus = 1
    else:
        num_gpus = 0

    mission_1, mission_1_pref, mission_2, mission_2_pref = args.experiment.experiment_name.split('-')
    print(mission_1, mission_1_pref, mission_2, mission_2_pref)

    # get data
    args.data.data_path = f"/vision/u/emilyjin/marple_long/data_2/{mission_1}_{mission_2}"

    args.data.mission_dict = {
        mission_1: float(mission_1_pref),
        mission_2: float(mission_2_pref),  
    }

    print(f'model: {args.model.model_name}')
    print(f'experiment name: {args.experiment.experiment_name}')
    print(f'mission_dict: {args.data.mission_dict}')

    # global seed
    pl.seed_everything(args.experiment.random_seed)

    # model selection
    is_policy_model = "policy" == args.model.model_name.lower() 
    is_transformer_model = "transformer" == args.model.model_name.lower()
    is_hist_aware_transformer_model = "hist_aware_transformer" == args.model.model_name.lower()
    is_audio_transformer_model = "audio_transformer" == args.model.model_name.lower()
    is_low_policy_model = "low_policy" == args.model.model_name.lower()
    is_goal_low_policy_model = "subgoal_low_policy" == args.model.model_name.lower()

    # args.optimizer.lr = 1e-3 if is_low_policy_model else 5e-3
    args.optimizer.lr = 1e-4 if is_low_policy_model else 1e-5
    args.optimizer.weight_decay = 0
    args.optimizer.use_lr_scheduler = True
    args.optimizer.warmup = 7 if is_low_policy_model else 20

    # set args for model
    args.model.dirpath = f"/vision/u/emilyjin/marple_long/checkpoints/data_level_2/{args.experiment.experiment_name}/{args.model.model_name}"
    args.model.n_checkpoints = 1 
    print('save checkpoint every', args.model.n_checkpoints)

    # set args for data
    args.data.historical = is_hist_aware_transformer_model
    args.data.with_audio = is_audio_transformer_model
    args.data.goal_conditioned = is_goal_low_policy_model
    args.data.data_level = "low" if is_low_policy_model or is_goal_low_policy_model else "mid"

    args.dataloader.batch_size = 4 if args.data.historical else 16  # Increase batch size
    if args.data.goal_conditioned:
        args.data.end_state_flag = True
        
    print('args.data', args.data)

    # load dataset of traj    
    train_dataset = BasicDataset(args.data)
    test_args = args.data
    test_args.split = 'test'
    valid_dataset = BasicDataset(test_args)
    args.data.split = 'train'

    args.dataloader.num_workers = num_cpus // 2
    args.dataloader.pin_memory = True
    print(args.dataloader)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **args.dataloader)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **args.dataloader)

    print('num train batches', len(train_dataloader))
    print('num val batches', len(valid_dataloader))

    total_steps = 1.5e5 if args.data.data_level == 'low' else 1.5e4
    args.trainer.max_epochs = int(total_steps / len(train_dataloader)) + args.optimizer.warmup if args.data.data_level == 'low' else 20
    args.trainer.max_epochs = 7 if args.data.data_level == 'low' else 20

    # model 
    if is_low_policy_model:
        model = LowBasicModel(args=args, model=LowPolicyModel, model_name="low_policy")
    elif is_goal_low_policy_model:
        model = LowBasicModel(args=args, model=SubgoalConditionedLowPolicyModel, model_name="subgoal_low_policy")
    elif is_policy_model:
        model = BasicModel(args=args, model=PolicyModel, model_name="policy")
    elif is_transformer_model: 
        model = BasicModel(args=args, model=TransformerHeadPolicyModel, model_name="transformer")
    elif is_hist_aware_transformer_model:
        model = BasicModel(args=args, model=HistAwareTransformerHeadPolicyModel, model_name="hist_aware_transformer")
    elif is_audio_transformer_model:
        model = BasicModel(args=args, model=AudioConditionedTransformerHeadPolicyModel, model_name="audio_transformer")
        
    # trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model.dirpath,
        filename=f"lr-{args.optimizer.lr}-wd-{args.optimizer.weight_decay}-{{epoch}}-embedding_dim-{args.model.embedding_dim}-warmup-{args.optimizer.warmup}-lr_restart-{args.optimizer.lr_restart}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    earlystopping_callback = EarlyStopping(monitor="loss/val_action_loss", mode="min", patience=5)#, check_val_every_n_epoch=5)

    # wandb setup
    print(args.wandb)
    wandb_logger = WandbLogger(**args.wandb, settings=wandb.Settings(_service_wait=300))
    
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=20,
                        **args.trainer, check_val_every_n_epoch=1, accumulate_grad_batches=accumulate)#, strategy="ddp")# devices=find_usable_cuda_devices(1), strategy="auto") # Pass optimizer and scheduler


    num_gpus_used = trainer.num_devices
    print(f"Number of GPUs used: {num_gpus_used}")
    # train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    main()
