import os 
import numpy as np
import json
import shutil
import argparse
import pdb
from mini_behavior.missions import MISSION_TO_SUBGOALS
from mini_behavior.inference_states import MISSION_TO_INFERENCE_STATE

DATA_PATH = '/vision/u/emilyjin/marple_long/data_2'
#TEST_DATA_PATH = '/vision/u/emilyjin/marple_long/test_data'


def get_inference_state(mission, data_dir, traj_name):
    traj_path = os.path.join(data_dir, mission, traj_name)

    subgoals = {
        json.load(open(os.path.join(traj_path, 'midlevel', f), 'r'))[0]: f for f in sorted(os.listdir(os.path.join(traj_path, 'midlevel'))) if f.endswith('subgoal.json')
    }
    # print(subgoals)
    state = MISSION_TO_INFERENCE_STATE[mission][0]
    # print(state)
    if state in subgoals.keys():
        inference_step = int(subgoals[state].split('_')[0]) + 1
        inference_state = f'{inference_step:05d}_states.json'
        state_dict = json.load(open(os.path.join(traj_path, inference_state), 'r'))

        return inference_step, state_dict
    else: 
        return None, None

def clean_current_state(state_dict):
    new_dict = {
        'Grid': {
            'height': state_dict['Grid']['height'],
            'width': state_dict['Grid']['width'],
            'agents': {
                'num': 1,
                'initial': [agent for agent in state_dict['Grid']['agents']['initial'] if agent['cur_mission'] is not None]
            },
            'rooms': state_dict['Grid']['rooms'],
            'doors': state_dict['Grid']['doors'],
        }
    }

    new_dict['Grid']['agents']['initial'][0] = {
        'name': new_dict['Grid']['agents']['initial'][0]['name'],
        'id': new_dict['Grid']['agents']['initial'][0]['id'],
        'gender': new_dict['Grid']['agents']['initial'][0]['gender'],
        'pos': new_dict['Grid']['agents']['initial'][0]['pos'],
        'dir': new_dict['Grid']['agents']['initial'][0]['dir'],
    }

    return new_dict

def clean_inference_state(state_dict):
    new_dict = {
        'Grid': {
            'height': state_dict['Grid']['height'],
            'width': state_dict['Grid']['width'],
            'agents': {
                'num': 1,
                'initial': [agent for agent in state_dict['Grid']['agents']['initial'] if agent['cur_mission'] is not None]
            },
            'rooms': state_dict['Grid']['rooms'],
            'doors': state_dict['Grid']['doors'],
        }
    }

    new_dict['Grid']['agents']['initial'][0] = {
        'name': None,
        'id': None,
        'gender': None,
        'pos': new_dict['Grid']['agents']['initial'][0]['pos'],
        'dir': new_dict['Grid']['agents']['initial'][0]['dir']
    }

    return new_dict


def clean_data(mission_1, mission_2):
    configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
    configs = os.listdir(configs_dir)

    total_trajs = 0

    for config in configs:
        data_path = os.path.join(configs_dir, config)
        a_trajs = os.listdir(os.path.join(data_path, 'A', mission_1))
        b_trajs = os.listdir(os.path.join(data_path, 'B', mission_2))

        trajs = [traj for traj in a_trajs if traj in b_trajs] 
        print('num data in Both ', data_path, len(trajs))
        total_trajs += len(trajs)

        a_other_trajs = [traj for traj in a_trajs if traj not in b_trajs]
        print('num data in A only ', data_path, len(a_other_trajs))

        b_other_trajs = [traj for traj in b_trajs if traj not in a_trajs]
        print('num data in B only', data_path, len(b_other_trajs))

        # check that all trajs are the full traj
        for traj in trajs:
            for agent, mission in [['A', mission_1], ['B', mission_2]]:
                traj_path = os.path.join(data_path, agent, mission, traj)

                if os.path.exists(os.path.join(traj_path, 'midlevel')):
                    subgoals = [json.load(open(os.path.join(traj_path, 'midlevel', f), 'r'))[0] for f in sorted(os.listdir(os.path.join(traj_path, 'midlevel'))) if f.endswith('subgoal.json')]
                    if set(subgoals) != set([subgoal[0] for subgoal in MISSION_TO_SUBGOALS[mission]]):
                        print(f'traj {traj} is incomplete')
                        a_other_trajs.append(traj)
                        b_other_trajs.append(traj)
                        total_trajs -= 1
                        break
                else:
                    print(f'traj {traj} is incomplete')
                    a_other_trajs.append(traj)
                    b_other_trajs.append(traj)
                    total_trajs -= 1
                    break

        for traj in a_other_trajs:
            traj_path = os.path.join(data_path, 'A', mission_1, traj)
            print('removing ', traj_path)
            shutil.rmtree(traj_path) 
        for traj in b_other_trajs:
            traj_path = os.path.join(data_path, 'B', mission_2, traj)
            print('removing ', traj_path)
            shutil.rmtree(traj_path)

def clean_current_state(state_dict):
    new_dict = {
        'Grid': {
            'height': state_dict['Grid']['height'],
            'width': state_dict['Grid']['width'],
            'agents': {
                'num': 1,
                'initial': [agent for agent in state_dict['Grid']['agents']['initial'] if agent['cur_mission'] is not None],
            },
            'rooms': state_dict['Grid']['rooms'],
            'doors': state_dict['Grid']['doors'],
        }
    }

    new_dict['Grid']['agents']['initial'][0] = {
        'name': new_dict['Grid']['agents']['initial'][0]['name'],
        'id': new_dict['Grid']['agents']['initial'][0]['id'],
        'gender': new_dict['Grid']['agents']['initial'][0]['gender'],
        'pos': new_dict['Grid']['agents']['initial'][0]['pos'],
        'dir': new_dict['Grid']['agents']['initial'][0]['dir'],
        'carrying': new_dict['Grid']['agents']['initial'][0]['carrying']
    }

    return new_dict


def clean_inference_state(state_dict):
    new_dict = {
        'Grid': {
            'height': state_dict['Grid']['height'],
            'width': state_dict['Grid']['width'],
            'agents': {
                'num': 1,
                'initial': [agent for agent in state_dict['Grid']['agents']['initial'] if agent['cur_mission'] is not None],
            },
            'rooms': state_dict['Grid']['rooms'],
            'doors': state_dict['Grid']['doors'],
        }
    }
    new_dict['Grid']['agents']['initial'][0] = {
        'name': None,
        'id': None,
        'gender': None,
        'pos': new_dict['Grid']['agents']['initial'][0]['pos'],
        'dir': new_dict['Grid']['agents']['initial'][0]['dir'],
        'carrying': new_dict['Grid']['agents']['initial'][0]['carrying']
    }

    return new_dict


# def clean_data(mission_1, mission_2):
#     configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
#     configs = os.listdir(configs_dir)

#     total_trajs = 0

#     for config in configs:
#         data_path = os.path.join(configs_dir, config)
#         a_trajs = os.listdir(os.path.join(data_path, 'A', mission_1))
#         b_trajs = os.listdir(os.path.join(data_path, 'B', mission_2))

#         trajs = [traj for traj in a_trajs if traj in b_trajs] 
#         print('num data in Both ', data_path, len(trajs))
#         total_trajs += len(trajs)

#         a_other_trajs = [traj for traj in a_trajs if traj not in b_trajs]
#         print('num data in A only ', data_path, len(a_other_trajs))

#         b_other_trajs = [traj for traj in b_trajs if traj not in a_trajs]
#         print('num data in B only', data_path, len(b_other_trajs))

#         # check that all trajs are the full traj
#         for traj in trajs:
#             for agent, mission in [['A', mission_1], ['B', mission_2]]:
#                 traj_path = os.path.join(data_path, agent, mission, traj)

#                 if os.path.exists(os.path.join(traj_path, 'midlevel')):
#                     subgoals = [json.load(open(os.path.join(traj_path, 'midlevel', f), 'r'))[0] for f in sorted(os.listdir(os.path.join(traj_path, 'midlevel'))) if f.endswith('subgoal.json')]
#                     if set(subgoals) != set([subgoal[0] for subgoal in MISSION_TO_SUBGOALS[mission]]):
#                         print(f'traj {traj} is incomplete')
#                         a_other_trajs.append(traj)
#                         b_other_trajs.append(traj)
#                         total_trajs -= 1
#                         break
#                 else:
#                     print(f'traj {traj} is incomplete')
#                     a_other_trajs.append(traj)
#                     b_other_trajs.append(traj)
#                     total_trajs -= 1
#                     break

#         for traj in a_other_trajs:
#             traj_path = os.path.join(data_path, 'A', mission_1, traj)
#             print('removing ', traj_path)
#             shutil.rmtree(traj_path) 
#         for traj in b_other_trajs:
#             traj_path = os.path.join(data_path, 'B', mission_2, traj)
#             print('removing ', traj_path)
#             shutil.rmtree(traj_path)

#         # #check final num traj for both agents
#         # a_trajs = os.listdir(os.path.join(data_path, 'A', mission_1))
#         # b_trajs = os.listdir(os.path.join(data_path, 'B', mission_2))
#         # assert len(a_trajs) == len(b_trajs) 

def get_test_data(mission_1, mission_2, num=None, frac=None):
    configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
    configs = os.listdir(configs_dir)
    configs = [f for f in configs if os.path.isdir(os.path.join(configs_dir, f))]
    for config in configs:
        data_path = os.path.join(configs_dir, config)
        missions = os.listdir(data_path)
        missions = [f for f in missions if not f.endswith('test')]
        print('data path ', data_path)
        for mission in missions:
            # sample trajectories
            mission_path = os.path.join(data_path, mission)
            trajs = os.listdir(mission_path)

            num = int(num) if num is not None else int(frac * len(trajs))
            traj_dirs = np.random.choice(trajs, num, replace=False)
            
            # write traj_dirs to a new directory with same name as mission + _test
            print("Sampled {} trajectories for missions {}, {}".format(len(traj_dirs), mission_1, mission_2))

            test_dir = os.path.join(data_path, f'{mission}_test')
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)            
            os.makedirs(test_dir)

            print('created ', test_dir)
                         
            with open(os.path.join(test_dir, "test_traj.json"), "w") as f:
                json.dump(traj_dirs.tolist(), f)
            print('saved test data to ', test_dir)


def reorganize_data(mission_1, mission_2):
    configs_dir = os.path.join(DATA_PATH, f'{mission_1}_{mission_2}')
    configs = os.listdir(configs_dir)
    configs = [f for f in os.listdir(configs_dir) if os.path.isdir(os.path.join(configs_dir, f))]

    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_1}'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_2}'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_1}_test'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_2}_test'), exist_ok=True)

    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_1}'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_2}'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_1}_test'), exist_ok=True)
    # os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}/{mission_2}_test'), exist_ok=True)


    # print('created ', os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}'))

    count = 0
    
    configs = [f for f in configs if os.path.isdir(os.path.join(configs_dir, f))]
    for config in configs:
        data_path = os.path.join(configs_dir, config)
        print('data path ', data_path)
        os.makedirs(os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}'), exist_ok=True)
        for mission in os.listdir(data_path):
            old_dir = os.path.join(data_path, mission)
            new_dir = os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}', mission)
            
            print('move from', old_dir, 'to', new_dir)

            os.makedirs(new_dir, exist_ok=True)
            if mission.endswith('test'):
                # load in test set so far
                if 'test_traj.json' in os.listdir(new_dir):
                    test_traj = json.load(open(os.path.join(new_dir, "test_traj.json"), "r"))
                else:
                    test_traj = []
                
                # update test set
                new_test_traj = json.load(open(os.path.join(old_dir, "test_traj.json"), "r"))
                test_traj.extend(new_test_traj)
                test_traj = list(set(test_traj))

                # save test set
                with open(os.path.join(new_dir, "test_traj.json"), "w") as f:
                    json.dump(test_traj, f) 

                print('saved test set length', len(test_traj))
            else:
                trajs = os.listdir(os.path.join(data_path, mission))

                for traj in trajs:
                    traj_path = os.path.join(data_path, mission, traj)
                    new_traj_path = os.path.join(DATA_PATH, f'init_config_{mission_1}_{mission_2}', mission, traj)

                    if not os.path.exists(new_traj_path):
#                        print('copy ', traj)
                        shutil.copytree(traj_path, new_traj_path, dirs_exist_ok=True)
                    # else:
                        # print('alr exists')
                count += len(trajs)
                print('total trajs so far: ', count)


def main():
    mission_pairs = [
        ['watch_movie_cozily', 'watch_news_on_tv'],
        ['feed_dog', 'take_shower'],
        ['get_snack', 'clean_living_room_table'],
         ['move_plant_at_night', 'get_night_snack'],
        ['change_outfit', 'do_laundry']
    ]

    for mission_1, mission_2 in mission_pairs:
        print('get test data')
        # args.frac = 0.8 if args.frac is None else args.frac
    #get_test_data(args.mission_1, args.mission_2, num=args.num, frac=args.frac)
        get_test_data(mission_1, mission_2, frac=0.2)
    #    reorganize_data(mission_1, mission_2)


if __name__ == '__main__':
    main()
