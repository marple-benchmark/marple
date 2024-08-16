import random 
import json
import os 
import numpy as np
import json
from PIL import Image
import sys
sys.path.append('/vision/u/emilyjin/marple_long/src/marple_mini_behavior')

import argparse
from gym_minigrid.wrappers import *
from marple_mini_behavior.mini_behavior.window import Window
from marple_mini_behavior.mini_behavior.minibehavior import *
from marple_mini_behavior.mini_behavior.planner import NpEncoder
from marple_mini_behavior.mini_behavior.envs import *
from marple_mini_behavior.mini_behavior.utils.save import get_state_dict, get_cur_arrays
from marple_mini_behavior.auto_control import check_reachability, get_trajectory
from marple_mini_behavior.bddl import *

from init_conditions import MISSION_TO_INIT_CONDITIONS

def redraw(env, window, img):
    img = env.render('rgb_array', tile_size=16)

    window.no_closeup()
    window.set_inventory(env)
    window.show_img(img)

    return img

def reset(env, window):
    obs = env.reset()

    img = redraw(env, window, obs)

    return img

def save_train_state(env, config_dir, config_name):
    # save_name = os.path.splitext(config_name)[0]
 
    json_path = os.path.join(config_dir, f"{config_name}.json")

    state_dict = get_state_dict(env)
    
    for agent in state_dict['Grid']['agents']['initial']:
        agent['pos'] = None
        agent['dir'] = None 
    
    for door in state_dict['Grid']['doors']['initial']:
        door['state'] = 'open'

    with open(json_path, "w") as outfile:
        outfile.write(json.dumps(state_dict, cls=NpEncoder, indent=4))
    
    print('save state to: ', json_path) 


def gen_config(missions, base_config_path, min_size=20, max_size=25):
    config_dir = os.path.join('/vision/u/emilyjin/marple_long/generate_data/configs')    
    config = json.load(open(base_config_path, 'r'))

    config['Grid']['width'] = random.randint(min_size, max_size)
    config['Grid']['height'] = random.randint(min_size, max_size)

    rooms = config['Grid']['rooms']['initial']
    random.shuffle(rooms)

    init_conditions = {room['type']: {} for room in rooms}
 
    for mission in missions:
        print(mission)
        mission_conditions = MISSION_TO_INIT_CONDITIONS[mission]

        for room, fur, fur_states, max_objs, req_objs in mission_conditions:
            room_dict = init_conditions[room]

            if fur in room_dict.keys():
                fur_conditions = room_dict[fur]

                if fur_states is not None:
                    assert type(fur_states) == dict
                    for state, value in fur_states.items():
                        fur_conditions['states'][state] = value
                else:
                    fur_states = {}

                fur_conditions['max_objs'] = min(max_objs, fur_conditions['max_objs'])
                fur_conditions['req_objs'] += req_objs
                
            else:
                fur_conditions = {'states': fur_states, 'max_objs': max_objs, 'req_objs': req_objs}

            room_dict[fur] = fur_conditions

    print('done creating init conditions')

    # print(init_conditions)
    for room in rooms:
        room_type = room['type']
        room_dict = init_conditions[room_type]

        assert type(len(room_dict.keys())) == int
        assert  room['furnitures']['num'] is not None, f"room[furnitures][num] is None. room['furnitures'] is {room['furnitures']}" 
        room['furnitures']['initial'] = []

        for fur_name, fur_conditions in room_dict.items():
            fur_states = fur_conditions['states']
            fur_num = random.randint(len(fur_conditions['req_objs']), fur_conditions['max_objs'])
            fur_objs = [{'type': obj, 'pos': None, 'state': None} for obj in fur_conditions["req_objs"]]

            room['furnitures']['initial'].append(
                {
                    "type": fur_name,  
                    'pos': None, 
                    'state': fur_states, 
                    'objs': {
                        'num': fur_num, 
                        'initial': fur_objs
                    }
                }
            )

    return config

def get_env_img(config): 
    # breakpoint()
    env = gym.make('MiniGrid-AutoGenerate-16x16-N2-v1', initial_dict=config)
    window = Window('mini_behavior - ' + 'MiniGrid-AutoGenerate-16x16-N2-v1')

    reset(env, window) 
    while not check_reachability(env):
        seed = random.randint(0, 1000)
        print("---------------seed", seed)
        env.seed(seed)
        reset(env, window)

    obs = env.gen_obs()
    img = redraw(env, window, obs) 
    img = Image.fromarray(img)

    return env, img, window


def close_handler(window):
    window.closed = True

def save_img(img, config_dir, config_name):
    img.save(os.path.join(config_dir, f"{config_name}.jpeg"))
    print('saved img to', os.path.join(config_dir, f"{config_name}.jpeg"))

def gen_room(mission_1, mission_2, config_type, split='train', min_size=20, max_size=24, num_traj=500, config_dir='data/config'):
    missions = [mission_1, mission_2] 

    save_path = os.path.join(config_dir, f'{missions[0]}_{missions[1]}')
    os.makedirs(save_path, exist_ok=True)    

    base_config_path = os.path.join('/vision/u/emilyjin/marple_long/generate_data/base_configs', split, f'init_config_base_{config}.json')#_type}.json')

    # get the config num
    config_num = int(len([f for f in os.listdir(save_path) if f.startswith(f'init_config_{missions[0]}_{missions[1]}_{config_type}')]) / 2)
    config_name = f'init_config_{missions[0]}_{missions[1]}_{config_type}_{str(config_num)}'
    
    # generate config for necessary fur / obj
    config = gen_config(missions, base_config_path, min_size=min_size, max_size=max_size)


        # generate env and room
    print('get env img, save')
    env, img, window = get_env_img(config)

    save_img(img, save_path, config_name)
    
    # for train
    for mission in missions: 
        save_train_state(env, save_path, config_name)

    # window.fig.canvas.mpl_connect('close_event', close_handler)
    print('saved config to: ', os.path.join(config_dir, config_name))

    return env, window

def main():
    parser = argparse.ArgumentParser(description='Generate room configurations')
    parser.add_argument('mission_1', type=str, help='First mission')
    parser.add_argument('mission_2', type=str, help='Second mission')
    parser.add_argument('config_type', type=str, default='simple', help='Config')
    parser.add_argument('split', type=str, default='train', help='Config')
    parser.add_argument('--min_size', type=int, default=14, help='Minimum size of the room')
    parser.add_argument('--max_size', type=int, default=18, help='Maximum size of the room')
    parser.add_argument('--config_dir', type=str, default='data/config', help='Directory to save config file')
    args = parser.parse_args()

    print(f'generating room config for missions {args.mission_1}, {args.mission_2}')
    gen_room(args.mission_1, args.mission_2, split=args.split, config_type=args.config_type, min_size=args.min_size, max_size=args.max_size, config_dir=args.config_dir)


if __name__ == '__main__':
    main()
