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

import pandas as pd

def gen_table(results):
    results_ = []

    for room, mission_results in results.items():
        inference_step = mission_results['inference_step']
        for mission, model_results in mission_results.items():
            if mission == 'inference_step':
                continue
            for model, metrics in model_results.items(): 
                result = [room, mission, inference_step, model, results[room][mission][model]['overall']] 
                results_.append(result) 

    temp = pd.DataFrame(results_, columns=['room', 'mission', 'inference_step', 'model', 'accuracy'])
    
    pivot_table = pd.pivot_table(temp, index=['room', 'inference_step', 'mission'], columns='model', values='accuracy')

    print(pivot_table)
    pivot_table.to_csv(os.path.join(results_dir, f"{file.split('.')[0]}.csv"))

@hydra.main(config_path="config", config_name='eval.yaml')
def main(args: DictConfig) -> None:
    data_level = args.model.dirpath 

    missions = [args.rollout.a_mission, args.rollout.b_mission]

    final_experiments = json.load(open('/vision/u/emilyjin/mini-behavior-llm-baselines/final_experiments.json', 'r'))    
    
    # results = {
    #     f"{room}-{traj}": 
    #     {
    #         missions[0]: {}, 
    #         missions[1]: {}
    #     } 
    #     for room, traj in final_experiments[f'init_config_{missions[0]}_{missions[1]}']['trajs']
    # }

    results = {missions[0]: {}, missions[1]: {}}
    
    target_missions = ['watch_movie_cozily', 'take_shower', 'get_snack', 'move_plant_at_night', 'do_laundry']
    
    args.data.split = 'inference'

    model_names = ['low_policy', 'subgoal_low_policy', 'transformer']
    model_names = ['low_policy']

    for model_name in model_names:
        args.model.model_name = model_name
        args.model.checkpoint_name = f'{model_name}_1024.ckpt' 

        # model selection
        is_transformer_model = "transformer" in args.model.model_name.lower()
        is_hist_aware_transformer_model = "hist_aware_transformer" in args.model.model_name.lower()
        is_low_policy_model = "low_policy" in args.model.model_name.lower()
        is_goal_low_policy_model = "subgoal_low_policy" in args.model.model_name.lower()

        # data
        args.data.historical = is_hist_aware_transformer_model
        args.data.goal_conditioned=is_goal_low_policy_model
        args.data.data_level = "low" if is_low_policy_model or is_goal_low_policy_model else "mid"
        args.dataloader.batch_size = 64 if is_low_policy_model else 64
        if args.data.goal_conditioned:
            args.data.end_state_flag = True

        # load in data 
        args.data.mission_dict = {m: 1 for m in missions}
        args.data.data_path = f"/vision/u/emilyjin/mini-behavior-llm-baselines/data/init_config_{missions[0]}_{missions[1]}"
        print('loading data from', args.data.data_path)

        test_dataset = BasicDataset(args.data)   

#       for mission_1_pref in [args.rollout.a_pref, args.rollout.b_pref]:
        #cur_mission = missions[0] if float(mission_1_pref) > 0 else missions[1]
        mission_1_pref = args.rollout.a_pref
        mission_2_pref = args.rollout.b_pref
        cur_mission = missions[0] if float(args.rollout.a_pref) > 0 else missions[1]
            # load in model
        args.experiment.experiment_name = f"{missions[0]}_{missions[1]}"
        args.model.dirpath = f"/vision/u/emilyjin/marple_long/final_checkpoints/{data_level}/{missions[0]}_{missions[1]}/{missions[0]}-{mission_1_pref}-{missions[1]}-{mission_2_pref}"
        #args.model.dirpath = f"/vision/u/emilyjin/marple_long/final_checkpoints/{data_level}/{missions[0]}_{missions[1]}/{missions[0]}-{mission_1_pref}-{missions[1]}-{str(round(1-float(mission_1_pref), 1))}"
        print(f'model: {args.model.model_name}')
        print(f'experiment name: {args.experiment.experiment_name}')
        checkpoint_path = os.path.join(args.model.dirpath, args.model.checkpoint_name) 

        if os.path.exists(checkpoint_path):
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

            """
                room_results = {
                    room_num: {
                        traj:
                            mission:
                                model: 
                                    results
                    }
                }
            """

                # for room, traj in final_experiments[f'init_config_{missions[0]}_{missions[1]}']['trajs']:
                #     # get data
                #     traj_path = os.path.join(args.data.data_path, room, traj, cur_mission)
                #     test_dataset.data = test_dataset.path_to_data[traj_path]
                #     test_dataloader = DataLoader(test_dataset, shuffle=False, **args.dataloader)

                #     if cur_mission in target_missions:
                #         inference_step = json.load(open(os.path.join(traj_path, 'timesteps.json'), 'r'))[-1] # TODO: get inference stpes for target mission
                #         results[f"{room}-{traj}"]['inference_step'] = inference_step

                #     # get metrics
                #     metrics = trainer.test(model=policy_model, dataloaders=test_dataloader)[0]

                #results[f"{room}-{traj}"][cur_mission][args.model.model_name] = {
                #    "overall": metrics["accuracy/test"],
                #    "obj": metrics["accuracy/test_obj"],
                #    "action": metrics["accuracy/test_action"],
                #}

    # results_path = os.path.join("/vision/u/emilyjin/marple_long/results/by_traj", data_level, f"{missions[0]}-{missions[1]}.json")
    # breakpoint()
    # os.makedirs(os.path.join("/vision/u/emilyjin/marple_long/results/by_traj", data_level), exist_ok=True)
    # with open(results_path, 'w') as f:
    #     json.dump(results, f, indent=4)

    # print(f'results at {results_path}')

    # gen_table(results)
            test_dataloader = DataLoader(test_dataset, shuffle=False, **args.dataloader) 

            # get metrics
            metrics = trainer.test(model=policy_model, dataloaders=test_dataloader)[0]
            results[cur_mission][args.model.model_name] = {
                        "overall": metrics["accuracy/test"],
                        "obj": metrics["accuracy/test_obj"],
                        "action": metrics["accuracy/test_action"],
            }

    results_path = os.path.join("/vision/u/emilyjin/marple_long/results/inference", data_level, f"{missions[0]}-{args.rollout.a_pref}-{missions[1]}-{args.rollout.b_pref}.json")
    os.makedirs(os.path.join("/vision/u/emilyjin/marple_long/results/inference", data_level), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'results at {results_path}')

    
if __name__ == "__main__":
    main()

