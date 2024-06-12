import hydra
from omegaconf import DictConfig

import torch
import os
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datetime import datetime

from MarpleLongModels.data.dataset import BasicDataset
from MarpleLongModels.models.pl_models import BasicModel, LowBasicModel
from MarpleLongModels.models.models import PolicyModel, TransformerHeadPolicyModel, HistAwareTransformerHeadPolicyModel, AudioConditionedTransformerHeadPolicyModel
import MarpleLongModels.models.vocab as vocab

from arguments import Arguments # arguments, defaults overwritten by config/config.yaml
from tqdm import tqdm
import pdb  # for debugging
import numpy as np

import json
import pickle as pkl

@hydra.main(config_path="config", config_name='eval.yaml')
def main(args: DictConfig) -> None:
    data_level = args.model.dirpath
    args.model.dirpath = os.path.join('/vision/u/emilyjin/marple_long/final_checkpoints', data_level)
    mission_1, mission_1_pref, mission_2, mission_2_pref = args.experiment.experiment_name.split('-')
    print(mission_1, mission_1_pref, mission_2, mission_2_pref)    

    # get data
    args.data.data_path = f"/vision/u/emilyjin/marple_long/data/{mission_1}_{mission_2}"
    print('loading data from', args.data.data_path)

    args.model.dirpath = os.path.join(args.model.dirpath, f"{mission_1}_{mission_2}/{args.experiment.experiment_name}")
    args.model.model_name = '_'.join(args.model.checkpoint_name.split('_')[:-1])
        
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

    # is_policy_model = "policy" in args.model.model_name.lower() 
    is_transformer_model = "transformer" in args.model.model_name.lower()
    is_hist_aware_transformer_model = "hist_aware_transformer" in args.model.model_name.lower()
    is_audio_transformer_model = "audio_transformer" in args.model.model_name.lower()
    is_low_policy_model = "low_policy" in args.model.model_name.lower()
    is_goal_low_policy_model = "subgoal_low_policy" in args.model.model_name.lower()

    # data
    args.data.historical = is_hist_aware_transformer_model
    args.data.with_audio = is_audio_transformer_model
    args.data.goal_conditioned=is_goal_low_policy_model
    args.data.data_level = "low" if is_low_policy_model or is_goal_low_policy_model else "mid"
    args.dataloader.batch_size = 64 if is_low_policy_model else 32
    if args.data.goal_conditioned:
        args.data.end_state_flag = True

    print(args.dataloader)
    print(args.data)

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

    results_path = os.path.join("/vision/u/emilyjin/marple_long/results", data_level, f"{args.experiment.experiment_name}.json")
    os.makedirs(os.path.join("/vision/u/emilyjin/marple_long/results", data_level), exist_ok=True)

    if os.path.exists(results_path):
        results = json.load(open(results_path, 'r'))
    else:
        results = {}

    with open(results_path, 'w') as f:
        results[args.model.model_name] = metrics
        json.dump(results, f, indent=4)

    print(f'results at {results_path}')


if __name__ == "__main__":
    main()

