import os
import json
import pickle as pkl
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from src.models.dataset import BasicDataset
from src.models.pl_models import BasicModel, LowBasicModel
from arguments import Arguments


@hydra.main(config_path="config", config_name='eval.yaml')
def main(args: DictConfig) -> None: 
    mission_1, mission_2 = args.experiment.mission_1, args.experiment.mission_2
    mission_1_pref, mission_2_pref = args.experiment.mission_1_pref, args.experiment.mission_2_pref
    
    args.experiment.experiment_name = f'{mission_1}-{mission_1_pref}-{mission_2}-{mission_2_pref}'

    # get data
    args.data.data_path = os.path.join(args.data.data_path, f"{mission_1}_{mission_2}")
    print('loading data from', args.data.data_path)
        
    args.data.mission_dict = {
        mission_1: float(mission_1_pref),
        mission_2: float(mission_2_pref)
    }
    
    print(f'model: {args.model.model_name}')
    print(f'experiment name: {args.experiment.experiment_name}')
    print(f'mission_dict: {args.data.mission_dict}')

    # global seed
    pl.seed_everything(args.experiment.random_seed)

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # model selection
    is_transformer_model = "transformer" in args.model.model_name.lower()
    is_low_policy_model = "low_policy" in args.model.model_name.lower()
    is_goal_low_policy_model = "subgoal_low_policy" in args.model.model_name.lower()

    # data
    args.data.goal_conditioned = is_goal_low_policy_model
    args.data.data_level = "low" if is_low_policy_model or is_goal_low_policy_model else "mid"
    if args.data.goal_conditioned:
        args.data.end_state_flag = True

    test_dataset = BasicDataset(args.data)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **args.dataloader)
    checkpoint_path = os.path.join(args.model.dirpath, args.model.checkpoint_name) 

    # test
    if args.data.data_level == 'low':
        policy_model = LowBasicModel.load_from_checkpoint(checkpoint_path)
    else:
        policy_model = BasicModel.load_from_checkpoint(checkpoint_path)

    if torch.cuda.is_available(): 
        policy_model.to('cuda')
        args.trainer.accelerator = 'gpu'

    policy_model.eval()
    trainer = pl.Trainer(**args.trainer)

    metrics = trainer.test(model=policy_model, dataloaders=test_dataloader)

    print(metrics) 

    if args.experiment.results_dir:
        results_path = os.path.join(args.experiment.results_dir, f"{args.experiment.experiment_name}-{args.model.model_name}.json")
        os.makedirs(args.experiment.results_dir, exist_ok=True)        
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f'results at {results_path}')


if __name__ == "__main__":
    main()

