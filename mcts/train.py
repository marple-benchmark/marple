import os
from datetime import datetime

import hydra
from omegaconf import DictConfig
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.accelerators import find_usable_cuda_devices

from src.simulator.dataset import BasicDataset
from src.models.pl_models import BasicModel, LowBasicModel
from src.models.models import PolicyModel, LowPolicyModel, SubgoalConditionedLowPolicyModel
from arguments import Arguments


@hydra.main(config_path="config", config_name="train.yaml", version_base="1.1")
def main(args: DictConfig) -> None:  
    
    mission_1, mission_2 = args.experiment.mission_1, args.experiment.mission_2
    mission_1_pref, mission_2_pref = args.experiment.mission_1_pref, args.experiment.mission_2_pref
    
    args.experiment.experiment_name = f'{mission_1}-{mission_1_pref}-{mission_2}-{mission_2_pref}'
    args.data.data_path = f"{args.data.data_path}/train_{args.data.generalization}/{mission_1}-{mission_2}"

    args.data.mission_dict = {
        mission_1: float(mission_1_pref),
        mission_2: float(mission_2_pref),  
    }

    print(f'model: {args.model.model_name}')
    print(f'mission_dict: {args.data.mission_dict}')

    # global seed
    pl.seed_everything(args.experiment.random_seed)

    # model selection
    is_policy_model = "policy" == args.model.model_name.lower() 
    is_transformer_model = "transformer" == args.model.model_name.lower()
    is_low_policy_model = "low_policy" == args.model.model_name.lower()
    is_goal_low_policy_model = "subgoal_low_policy" == args.model.model_name.lower()

    # set args for model
    args.model.dirpath = os.path.join(args.model.dirpath, args.data.generalization, f'{mission_1}-{mission_2}', args.experiment.experiment_name, args.model.model_name)

    # set args for data
    args.data.goal_conditioned = is_goal_low_policy_model
    args.data.data_level = "low" if is_low_policy_model or is_goal_low_policy_model else "mid"

    if args.data.goal_conditioned:
        args.data.end_state_flag = True

    # load dataset of traj    
    train_dataset = BasicDataset(args.data)
    test_args = args.data
    test_args.split = 'test'
    valid_dataset = BasicDataset(test_args)
    args.data.split = 'train'
    
    args.dataloader.num_workers = os.cpu_count() // 2
    train_dataloader = DataLoader(train_dataset, shuffle=True, **args.dataloader)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **args.dataloader)

    print('num train batches', len(train_dataloader))
    print('num val batches', len(valid_dataloader))

    # model 
    if is_low_policy_model:
        model = LowBasicModel(args=args, model=LowPolicyModel, model_name="low_policy")
    elif is_goal_low_policy_model:
        model = LowBasicModel(args=args, model=SubgoalConditionedLowPolicyModel, model_name="subgoal_low_policy")
    elif is_policy_model:
        model = BasicModel(args=args, model=PolicyModel, model_name="policy")
        
    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model.dirpath,
        filename=f"lr-{args.optimizer.lr}-wd-{args.optimizer.weight_decay}-{{epoch}}-embedding_dim-{args.model.embedding_dim}-warmup-{args.optimizer.warmup}-lr_restart-{args.optimizer.lr_restart}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    earlystopping_callback = EarlyStopping(monitor="loss/val_action_loss", mode="min", patience=5)
    wandb_logger = WandbLogger(**args.wandb, settings=wandb.Settings(_service_wait=300))

    # define trainer and train
    trainer = pl.Trainer(
        logger=wandb_logger, 
        callbacks=[checkpoint_callback, lr_monitor], 
        log_every_n_steps=20,
        **args.trainer, 
        check_val_every_n_epoch=1, 
        accumulate_grad_batches=1
        )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    main()
