#!/usr/bin/env python3

import argparse
from gym_minigrid.wrappers import *
from mini_behavior.window import Window
from mini_behavior.utils.save import get_step, save_demo
from mini_behavior.grid import GridDimension
import numpy as np
import json

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.no_closeup()
    window.set_inventory(env)
    window.show_img(img)


def render_furniture():
    global show_furniture
    show_furniture = not show_furniture

    if show_furniture:
        img = np.copy(env.furniture_view)

        # i, j = env.agent.cur_pos
        for agent in env.agents:
            i, j = agent.agent_pos
            ymin = j * TILE_PIXELS
            ymax = (j + 1) * TILE_PIXELS
            xmin = i * TILE_PIXELS
            xmax = (i + 1) * TILE_PIXELS
            img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
                img[ymin:ymax, xmin:xmax, :], agent.agent_dir, agent.agent_color)
        img = env.render_furniture_states(img)

        window.show_img(img)
    else:
        obs = env.gen_obs()
        redraw(obs)


def show_states():
    imgs = env.render_states()
    window.show_closeup(imgs)


def switch_agent():
    print("switch agent")
    env.switch_agent()


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def load():
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    obs = env.load_state(args.load)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if args.save:
        step_count, step = get_step(env)
        all_steps[step_count] = step

    if done:
        print('done!')
        if args.save:
            save_demo(all_steps, args.env, env.episode)
        reset()
    else:
        redraw(obs)


def switch_dim(dim):
    env.switch_dim(dim)
    print(f'switching to dim: {env.render_dim}')
    obs = env.gen_obs()
    redraw(obs)


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
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
        render_furniture()
        return
    if event.key == 'c':
        step('choose')
        return
    if event.key == 's':
        env.save_state()
        return
    if event.key == 'r':
        show_states()
        return
    if event.key == 'a':
        switch_agent()
        return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    # default='MiniGrid-ThrowLeftoversFourRooms-8x8-N2-v1'
    # default='MiniGrid-FloorPlanEnv-16x16-N1-v0'
    # default='MiniGrid-TwoRoomNavigation-8x8-N2-v0'
    # default='MiniGrid-ThrowLeftoversSceneEnv-0x0-N2-v0'
    # default='MiniGrid-ThrowLeftovers-16x16-N2-v1'
    # default='MiniGrid-InstallingAPrinter-16x16-N2-v1'
    # default='MiniGrid-ThawingFrozenFood-16x16-N2-v1'
    default='MiniGrid-WhoLeftTheLightOnInKitchen-16x16-N2-v1'
)

parser.add_argument(
    '-c',
    '--config',
    help='Path to config JSON file',
    default='mini_behavior/configs/init_config_default.json'
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
# NEW
parser.add_argument(
    "--save",
    default=False,
    help="whether or not to save the demo_16"
)
# NEW
parser.add_argument(
    "--load",
    default=None,
    help="path to load state from"
)

args = parser.parse_args()

if args.config is not None:
    with open(args.config, 'r') as f:
        initial_dict = json.load(f)
        env = gym.make(args.env, initial_dict=initial_dict)
else:
    env = gym.make(args.env)

all_steps = {}

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('mini_behavior - ' + args.env)
window.reg_key_handler(key_handler)

if args.load is None:
    reset()
else:
    load()

# Blocking event loop
window.show(block=True)
