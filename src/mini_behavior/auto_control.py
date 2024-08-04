#!/isr/bin/env python3

import argparse
from gym_minigrid.wrappers import *
from mini_behavior.configs import *
from mini_behavior.planner import *
from mini_behavior.window import Window
from mini_behavior.utils.save import save_demo
from mini_behavior.grid import GridDimension
from mini_behavior.minibehavior import *
from mini_behavior.envs import *
from bddl import *
import numpy as np
import json
import calendar
import time
import sys
from tqdm import tqdm
from collections import deque
from pydub import AudioSegment
from array2gif import write_gif
import copy


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False
show_objects = False
show_room_types = False
show_fur_states = False
show_obj_states = False


def redraw(env, window, img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.no_closeup()
    window.set_inventory(env)
    window.show_img(img)

    return img


def render_furniture(env, window, frames, audios):
    global show_furniture
    show_furniture = not show_furniture
    audio_file, audio_idx = get_audio_file_idx(empty=True)
    if show_furniture:
        env.furniture_view = env.grid.render_furniture(
            tile_size=TILE_PIXELS, obj_instances=env.obj_instances)
        img = np.copy(env.furniture_view)
        for agent in env.agents:
            agent = env.agents[env.cur_agent_id]
            i, j = agent.agent_pos
            ymin = j * TILE_PIXELS
            ymax = (j + 1) * TILE_PIXELS
            xmin = i * TILE_PIXELS
            xmax = (i + 1) * TILE_PIXELS
            img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
                img[ymin:ymax, xmin:xmax, :], agent.agent_dir, agent.agent_color)
        # img, audio = env.render_furniture_states(
        #     img), 
        img = np.transpose(img, (2,0,1))
        frames.append(img)
        # window.show_img(img)
    else:
        obs = env.gen_obs()
        img = redraw(env, window, obs)
        img = np.transpose(img, (2,0,1))
        frames.append(img)
    audios.append(AudioSegment.from_wav(audio_file))
    return frames, audios


def show_states(frames, audios):
    imgs = env.render_states()
    window.show_closeup(imgs)
    print(len(imgs), imgs)
    frames += imgs
    audios.append(AudioSegment.from_wav(AUDIO_MAP['empty']))
    return frames, audios


def switch_agent(env, i=None):
    print("switch agent")
    env.switch_agent(i)


def reset(env, window):
    
    env.seed(args.seed)
    obs = env.reset()

    redraw(env, window, obs)


def load(env, window):
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    obs = env.load_state(args.load)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(env, window, obs)


def step(env, window, action):
    obs, reward, done, info = env.step(
        action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        if args.save:
            save_demo(all_steps, args.env, env.episode)
        reset(env, window)
    else:
        redraw(env, window, obs)


# def switch_dim(env, dim):
#     env.switch_dim(dim)
#     print(f'switching to dim: {env.render_dim}')
#     obs = env.gen_obs()
#     redraw(env, obs)


def key_handler(event, env, window, frames, audios):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset(keep=True)
        return
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return
    # Spacebar
    if event.key == ' ':
        render_furniture(env, window, frames, audios)
        return
    if event.key == 'c':
        step('choose')
        return
    if event.key == 's':
        env.save_state()
        return
    if event.key == 'r':
        show_states(frames, audios)
        return
    if event.key == 'a':
        switch_agent(env)
        return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-AutoGenerate-16x16-N2-v1'
    )

    parser.add_argument(
        '-c',
        '--config',
        help='Path to config JSON file',
        default='mini_behavior/configs/envs/init_config_agent_a_b.json'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )
    parser.add_argument(
        '--split',
        default='train',
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )    
    # NEW
    parser.add_argument(
        "--save",
        default=True,
        help="whether or not to save the demo_16"
    )
    # NEW
    parser.add_argument(
        "--load",
        default=None,
        help="path to load state from"
    )
    parser.add_argument(
        "--gif",
        type=str,
        default='save.gif',
        help="store output as gif with the given filename"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000000,
        help="number of episodes to visualize"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="pause duration between two consequent actions of the agent (default: 0.seed 0_2)"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default='/vision/u/emilyjin/marple_long/data',
        help="folder to store data"
    )

    parser.add_argument(
        "--num_data",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--mission",
        type=str,
        default='watch_news_on_tv'
    )
    return parser.parse_args()


def check_all_empty_cell_reachable(env):
    # check all empty cell reachable
    cur_agent = env.agents[env.cur_agent_id]

    #for j in range(env.grid.height):
    #    row = []
    #    for i in range(env.grid.width):
    #        row.append(env.grid.get_all_items(i, j))
    #    print(f'row {i}, {row}')

    for i in range(env.grid.width):
        for j in range(env.grid.height):
            if env.grid.is_empty(i, j) and (i, j) != cur_agent.agent_pos:
                action_list, _, _ = cur_agent.navigate(
                    env, cur_agent.agent_pos, 0, (i, j), exact=True)
                if not action_list:
                    print(
                        f"{i}, {j} empty cell not reachable from {cur_agent.agent_pos}")
                    return False
    print("empty cells reachable")
    return True


    #         if env.grid.is_empty(i, j):
    #             action_list, _, _ = cur_agent.navigate(
    #                 env, cur_agent.agent_pos, 0, (i, j))
    #             if not action_list:
    #                 print(f"{i}, {j} empty cell not reachable")
    #                 return False 

def check_all_empty_cell_connected(env):
    rows = env.grid.width
    cols = env.grid.height

    visitedGrid = [[False] * cols for _ in range(rows)]
    empty_cells = []

    for i in range(rows):
        for j in range(cols):
            if env.grid.is_empty(i, j):
                empty_cells.append((i, j))

    start = empty_cells.pop(0)
    if not visitedGrid[start[0]][start[1]]:
        queue = deque([start])
        visitedGrid[start[0]][start[1]] = True

        while queue:
            current = queue.popleft()
            row, col = current

            # Check neighbors (up, down, left, right)
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]

            for neighbor in neighbors:
                n_row, n_col = neighbor
                if 0 <= n_row < rows and 0 <= n_col < cols:
                    if env.grid.is_empty(n_row, n_col) and not visitedGrid[n_row][n_col]:
                        queue.append((n_row, n_col))
                        visitedGrid[n_row][n_col] = True

    # Check if any empty cells are unvisited
    for i in range(rows):
        for j in range(cols):
            if env.grid.is_empty(i, j) and not visitedGrid[i][j]:
                return False

    return True

def check_all_furniture_reachable(env):
    for room in env.room_instances:
        for fur in room.furnitures:
            fur_flag = False
            # check if any of the pos have an empty cell surrounding it
            for pos in fur.surround_pos:
                if  0 <= pos[0] < env.grid.width \
                        and 0 <= pos[1] < env.grid.height \
                        and env.grid.is_empty(*pos):
                        fur_flag = True
                        break
                if fur_flag:
                    break
            if not fur_flag:
                return False
    return True


def check_reachability(env):
    print("check reachability")
    return check_all_empty_cell_reachable(env) and check_all_furniture_reachable(env)


def get_trajectory(args, num_data, data_folder, mission, multimodal=False, save_gif=False, config=None):
    for episode in tqdm(range(num_data)):
        # 1. create environment
        if config is not None:
            initial_dict = config
            initial_dict["Grid"]["agents"]["initial"][0]["mission_preference_initial"] = {mission: 1}            
            initial_dict["Grid"]["agents"]["initial"][0]["pos"] = config["Grid"]["agents"]["initial"][0]["pos"]
            initial_dict["Grid"]["agents"]["initial"][0]["dir"] = config["Grid"]["agents"]["initial"][0]["dir"]
            env = gym.make(args.env, initial_dict=initial_dict)
        elif args.config is not None:
            with open(args.config, 'r') as f:
                initial_dict = json.load(f)
                initial_dict["Grid"]["agents"]["initial"][0]["mission_preference_initial"] = {mission: 1}
                env = gym.make(args.env, initial_dict=initial_dict)
        else:
            env = gym.make(args.env)
        all_steps = {}
        
        if args.agent_view:
            env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)

        window = Window('mini_behavior - ' + args.env)
        window.reg_key_handler(key_handler)

        args.seed = random.randint(0, 1000)
        print("---------------seed", args.seed)
        reset(env, window)
        # window.show(block=True)

        # Create a window to view the environment
        # import pdb; pdb.set_trace()
        env.render('human')

        while not check_reachability(env):
            args.seed = random.randint(0, 1000)
            print("---------------seed", args.seed)
            reset(env, window)

        # 2. run planner
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        env.render('human')
        env_original = copy.deepcopy(env)
        cur_data_folder = f"{data_folder}/{args.config.split('/')[-1].split('.')[0]}"
        cur_data_folder = f"{data_folder}"
        os.makedirs(cur_data_folder, exist_ok=True)

        print('cur data folder', cur_data_folder)
        # traj_folder, traj_folder_midlevel = None, None
        
        
        for agent_iter in range(len(env.agents)):
            # for every agent, get a planner
            switch_agent(env, agent_iter)
            step_count = 0

            frames = []
            audios = []
            texts = []
            audios_symbolic = []
            
            if multimodal or save_gif:
                frames, audios = render_furniture(env, window, frames, audios)
            

            # start planner
            planner = Planner(env, env.cur_agent_id)
            planner.data_folder = cur_data_folder + '/' #+ str(env.agents[env.cur_agent_id].name)

            # conduct actions
            obs, reward, done, frames, audios, audios_symbolic, texts, step_count = planner.high_level_planner(
                args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=multimodal, save_gif=save_gif)
                
            planner.traj_folder = f'{planner.mission_folder}/{time_stamp}'
            planner.traj_folder_midlevel = os.path.join(planner.traj_folder, 'midlevel')                
            print("check planner save folder", planner.traj_folder, planner.traj_folder_midlevel)
            # traj_folder = planner.traj_folder
            # traj_folder_midlevel = planner.traj_folder_midlevel

            if multimodal or save_gif:
                frames, audios = render_furniture(env, window, frames, audios) 
            
            if save_gif:
                print("Saving gif... ", end="")
                print("Saving to ", planner.traj_folder)
                breakpoint()
                write_gif(np.array(frames),
                            os.path.join(planner.traj_folder, "visual.gif"), fps=1/args.pause)
                print("Done.")

            if multimodal:
                print("Saving text... ", end="")
                with open(os.path.join(planner.traj_folder, "language.txt"), 'w') as fp:
                    for i, item in enumerate(texts):
                        # write each item on a new line
                        fp.write(f"{item}\n")
                print("Done.")
                print("Saving sound... ", end="")
                # combined_sounds = sum(audios)
                # velocidad_X = 2
                # combined_sounds = combined_sounds.speedup(velocidad_X, 150, 25)
                # combined_sounds.export(
                #     os.path.join(planner.data_folder, f"{time_stamp}/audio.wav"), format="wav")
            
                with open(os.path.join(planner.traj_folder, "audio.txt"), 'w') as fp:
                    for i, item in enumerate(audios_symbolic):
                        # write each item on a new line
                        fp.write(f"{item}\n")

                    print(len(frames), len(audios), len(texts))
                    # assert len(frames) == len(audios), f"Frame no not equal audio no: {time_stamp}"
                    print("Done.")

            env = copy.deepcopy(env_original)


if __name__ == '__main__':
    args = parse_arguments() 
    #args.save = True

    args.seed = random.randint(0, 1000)
    print("---------------seed", args.seed)
    print(args.data_folder)
    print(args.mission)
    
    args.split = 'test'
    if args.split == 'train':
        get_trajectory(args=args, num_data=args.num_data, data_folder=args.data_folder, mission=args.mission)
    else:
        print('save gif')
        get_trajectory(args=args, num_data=args.num_data, data_folder=args.data_folder, mission=args.mission, multimodal=False, save_gif=True)

    # env = run_subgoal(args, load_state_json, load_subgoal_json, save_state_folder, step_count=step_count, episode=episode)

    

    
