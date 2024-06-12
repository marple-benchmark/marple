import os 
import numpy as np
import json
import shutil
import argparse
import os

from mini_behavior.missions import *
DATA_PATH = '/vision/u/emilyjin/marple_long/data'

def count_data(mission_1, mission_2):
    configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
    configs = os.listdir(configs_dir)

    for config in configs:
        data_path = os.path.join(configs_dir, config)
        a_trajs = os.listdir(os.path.join(data_path, 'A', mission_1))
        b_trajs = os.listdir(os.path.join(data_path, 'B', mission_2))

        trajs = [traj for traj in a_trajs if traj in b_trajs]
        print('num data in ', data_path, len(trajs))


# def clean_data(mission_1, mission_2, config=None):
#     configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
    
#     if config is None:
#         configs = os.listdir(configs_dir)
#         for config in configs:
#             remove_trajs(configs_dir,config,mission_1,mission_2)
    
#     else:
#         config = os.path.splitext(config)[0]
#         remove_trajs(configs_dir,config,mission_1,mission_2)
#         save_data_info(mission_1, mission_2, data_path, config)

        if data_level == 'low':
            state_files = sorted([file for file in os.listdir(directory) if file.endswith(state_suffix)])[:-1]
        elif data_level == 'mid':
            state_files = sorted([file for file in os.listdir(directory) if file.endswith(state_suffix)])
        action_files = sorted([file for file in os.listdir(directory) if file.endswith(action_suffix)])
        # print(state_files)
        # print(action_files)
        # assert len(state_files) == len(action_files) 
        return state_files, action_files

def clean_train_data(mission, config_dir):
    for traj in os.listdir(os.path.join(config_dir, mission)):
        num_subgoals = len(MISSION_TO_SUBGOALS[mission])

        traj_path = os.path.join(config_dir, mission, traj)
        low_state_files = [f for f in os.listdir(traj_path) if f.endswith('states.npy')]
        low_action_files = [f for f in os.listdir(traj_path) if f.endswith('action.json')]

        mid_state_files = [f for f in os.listdir(os.path.join(traj_path, 'midlevel')) if f.endswith('final_states.npy')]
        mid_action_files = [f for f in os.listdir(os.path.join(traj_path, 'midlevel')) if f.endswith('subgoal.json')]        

        if len(low_state_files) != len(low_action_files) or len(mid_state_files) != len(mid_action_files) or len(mid_action_files) != num_subgoals:
            print('removing ', traj_path)
            shutil.rmtree(traj_path)

def remove_trajs(configs_dir,config,mission_1,mission_2):
    data_path = os.path.join(configs_dir, config)
    a_trajs = os.listdir(os.path.join(data_path, 'A', mission_1))
    b_trajs = os.listdir(os.path.join(data_path, 'B', mission_2))

    trajs = [traj for traj in a_trajs if traj in b_trajs] 
    print('num data in Both ', data_path, len(trajs))

    a_other_trajs = [traj for traj in a_trajs if traj not in b_trajs]
    print('num data in A only ', data_path, len(a_other_trajs))

    b_other_trajs = [traj for traj in b_trajs if traj not in a_trajs]
    print('num data in B only', data_path, len(b_other_trajs))

    for traj in a_other_trajs:
        traj_path = os.path.join(data_path, 'A', mission_1, traj)
        print('removing ', traj_path)
        shutil.rmtree(traj_path) 
    for traj in b_other_trajs:
        traj_path = os.path.join(data_path, 'B', mission_2, traj)
        print('removing ', traj_path)
        shutil.rmtree(traj_path)


def save_data_info(mission_1, mission_2, data_path, config):
    file_path = os.path.join(DATA_PATH, 'dataset_info.json')    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            dataset_info = json.load(f)
            dataset_dict = dataset_info.get(f'{mission_1}_{mission_2}', {})
    else:
        dataset_info = {}
        dataset_dict = {}

    dataset_dict[config] = len(os.listdir(os.path.join(data_path, 'A', mission_1)))
    dataset_info[f'{mission_1}_{mission_2}'] = dataset_dict
    
    with open(file_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)        


def get_test_data(mission_1, mission_2, num=None, frac=None):
    configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
    configs = os.listdir(configs_dir)

    for config in configs:
        data_path = os.path.join(configs_dir, config)
        trajs = os.listdir(os.path.join(data_path, 'A', mission_1)) 

        # print(len(trajs))
        
        num = int(num) if num is not None else int(frac * len(trajs))
        traj_dirs = np.random.choice(trajs, num, replace=False)
        # write traj_dirs to a new directory with same name as mission + _test
        print("Sampled {} trajectories for missions {}, {}".format(len(traj_dirs), mission_1, mission_2))
        a_test_dir = os.path.join(data_path, 'A', mission_1 + "_test")
        b_test_dir = os.path.join(data_path, 'B',  + "_test")
        
        for test_dir in [a_test_dir, b_test_dir]:
            os.makedirs(test_dir, exist_ok=True)
            #  write names of traj to a json file and add to the new test directory
            with open(os.path.join(test_dir, "test_traj.json"), "w") as f:
                json.dump(traj_dirs.tolist(), f)
            print('saved test data to ', test_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate room configurations')
    parser.add_argument('--mission_1', type=str, help='First mission')
    parser.add_argument('--mission_2', type=str, help='Second mission')
    parser.add_argument('--config', type=str, help='Data config')

    args = parser.parse_args()
 
    print('clean data')
    clean_data(args.mission_1, args.mission_2, args.config)    
 

if __name__ == '__main__':
    main()
