import os
import sys
import json
import hydra
import calendar
import time
import numpy as np
from os import getpid
import torch
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import psutil


import copy
from array2gif import write_gif
sys.path.append('/Users/emilyjin/Code/marple_long/src')
import MarpleLongModels.models.vocab as vocab
from gym_minigrid.wrappers import *
from MarpleLongModels.models.pl_models import BasicModel, LowBasicModel
from arguments import Arguments # arguments, defaults overwritten by config/config.yaml
from rollout import Rollout
from marple_mini_behavior.mini_behavior.minibehavior import *
from marple_mini_behavior.mini_behavior.planner import *
from marple_mini_behavior.mini_behavior.envs import *
from marple_mini_behavior.mini_behavior.utils import *
from marple_mini_behavior.mini_behavior.actions import *
from marple_mini_behavior.mini_behavior.missions import *

import pdb
import multiprocessing



def reset(args, env):
    obs = env.reset()

    redraw(args, env, obs)

def redraw(args, env, img):
    img = env.render('rgb_array', tile_size=args.tile_size)

    return img


def pad(state, target_size=(20,20,8)):
    pad_width = [(0, max(0, target_size[i] - state.shape[i])) for i in range(len(target_size))]
    state = np.pad(state, pad_width, mode='constant', constant_values=0)
    return state

def get_inference_prob(args, init_step, last_step, extra_steps, num_samples, target_action, rollout, temp=1):
    rollouts_actions = []

    envs_list = [copy.deepcopy(rollout.env) for _ in range(num_samples)]        
    cur_state_arrays = [pad(get_cur_arrays(env)) for env in envs_list]
    step_counts = [init_step for i in range(num_samples)]
    
    envs, cur_state_arrays, step_counts, actions = rollout.sample(envs_list, cur_state_arrays, step_counts, num_workers=args.rollout.n_jobs, num_samples=100, temp=temp, use_audio=rollout.audio)

    for i in range(1, rollout.rollout_depth):
        print('rollout_depth', i) 
        torch.cuda.empty_cache()
        envs, cur_state_arrays, step_counts, actions = rollout.sample(envs_list, cur_state_arrays, step_counts, num_workers=args.rollout.n_jobs, num_samples=100, temp=temp, use_audio=None)
        envs_list = envs
        rollouts_actions.append(actions)

    rollouts_actions = np.array(rollouts_actions) # rollout_depth x samples x action shape
    successes = np.any(np.all(rollouts_actions == target_action, axis=2), axis=0)

#    for i in range(num_samples):

#        cur_actions = []
#        env = envs_list[i]
#        cur_state_array = cur_state_arrays[i]
#        step_count = step_counts[i]
#        for i in range(rollout.rollout_depth):
#            print('rollout_depth', i) 
#            env, cur_state_array, step_count, action = rollout.one_sample(env, cur_state_array, step_count)
#            cur_actions.append(action)
#        rollouts_actions.append(cur_actions)
    
#    rollouts_actions = np.array(rollouts_actions) # rollout_depth x samples x action shape
#    successes = np.any(np.all(rollouts_actions == target_action, axis=2), axis=1)    
    return successes, rollout


# @timing_decorator
def inference_probs(args, missions, agent, mission_name, agent_prefs, target_action, target_step, steps, num_samples=100, extra_steps=10, temp=1):
    args.rollout.verbose = False

    agent_model = f'{missions[0]}-{agent_prefs[missions[0]]}-{missions[1]}-{agent_prefs[missions[1]]}'
    args.model.dirpath = os.path.join('/vision/u/emilyjin/marple_long/final_checkpoints', args.model.dirpath, args.experiment.experiment_name, agent_model)
    args.rollout.policy_model_checkpoint = os.path.join(args.model.dirpath, args.rollout.policy_model_checkpoint_short)
    args.experiment.agent = agent
    args.experiment.mission_name = mission_name  

    cpu_count = os.cpu_count()
    num_workers = cpu_count // 6 if cpu_count is not None else 1 
    print('num cpus: ', num_workers)

    args.rollout.n_jobs = num_workers
    print('start inference with args: ', args)
    all_successes = [] 

    last_step = target_step
    args.rollout.rollout_depth = last_step + extra_steps 
    args.rollout.from_step_count = 0 
    rollout = Rollout(args)

    print('inference from steps', steps) 

    for i in steps:
        if last_step + extra_steps - i > 40:
            successes = np.array([False for i in range(num_samples)])
        else:
            print(f'inference from state: {i}') 
            rollout.re_load(i, last_step + extra_steps - i)
            torch.cuda.empty_cache() 
            successes, rollout = get_inference_prob(args, i, last_step, extra_steps, num_samples, target_action, rollout, temp=temp)
        all_successes.append(successes)

    all_successes = np.vstack(all_successes)
    probs = np.sum(all_successes, axis=1) / num_samples
    stds = np.std(all_successes, axis=1)

    # print('successes: ',  np.sum(successes), 'prob: ', prob, 'std: ', std)
    # return prob, std, rollout

    print('probs', probs)
    print('stds', stds)
    return probs, stds


def get_save_to(args, target_mission, target_agent, target_step, data_level, temp):
    time_stamp = calendar.timegm(time.gmtime())
    inference_scenario = '-'.join([target_mission, target_agent])
    agent_pref = f'{args.rollout.a_mission}-{args.rollout.a_pref}-{args.rollout.b_mission}-{args.rollout.b_pref}-temp_{str(temp)}'
    path = os.path.join("/vision/u/emilyjin/marple_long/inference", f'{args.experiment.experiment_name}-{target_agent}', f'{args.rollout.room_config}-{args.rollout.traj_name}', agent_pref)
    os.makedirs(path, exist_ok=True)      

    if args.model.model_name == 'low_policy':
        filename = 'low_policy'
    elif args.model.model_name == 'subgoal_low_policy' and args.rollout.simulator:
        filename = 'subgoal_low_policy_simulator'
    else:
        filename = 'subgoal_low_policy_predicted'

    if args.rollout.audio:
        filename += '_audio'
    if data_level == 'data_level_2':
        filename += '_2'

    return path, filename


def get_normalized_probs(probs_a, probs_b):
    """
    probs_a, probs_b: list of probs P(final state = final agent state | current agent state)
    """

    # Make probs_a and probs_b the same size
    max_len = max(len(probs_a), len(probs_b))

    if len(probs_a) < max_len:
        last_a = probs_a[-1]
        probs_a += [last_a] * (max_len - len(probs_a))
    elif len(probs_b) < max_len:
        last_b = probs_b[-1]
        probs_b += [last_b] * (max_len - len(probs_b))
        
    probs_a = np.array(probs_a)
    probs_b = np.array(probs_b)

    # assert len(probs_a) == len(probs_b) 

    total = probs_a + probs_b
    total = np.where(total == 0, 1, total)
    probs_a_ = np.where(probs_a + probs_b == 0, 0.5, probs_a)
    probs_b_ = np.where(probs_a + probs_b == 0, 0.5, probs_b)

    normalized_a = list(probs_a_ / total)
    normalized_b = list(probs_b_ / total)
    print(normalized_a)
    print(normalized_b)
    return normalized_a, normalized_b



def get_inference_plot(probs_a, probs_b, stds_a, stds_b, times, save_path, save_filename, prob_type='probs'):    
    plt.plot(times, probs_a, label='Agent A')
    plt.errorbar(times, probs_a, yerr=stds_a, alpha=0.3, fmt='-o')
    plt.plot(times, probs_b, label='Agent B')
    plt.errorbar(times, probs_a, yerr=stds_b, alpha=0.3, fmt='-o')
    plt.ylim(0, 1) 
    plt.xlabel('Time')

    
    ylabel = 'Probability of Target State' if prob_type == 'probs' else 'Probability of Target State (normalized)'
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(os.path.join(save_path, f'{save_filename}_{prob_type}.png'))
    print(f'inference plot saved at {save_path}/{save_filename}_{prob_type}.png')

    plt.clf() 


def save_all_results(save_path, save_filename, normalized_a, normalized_b, probs_a, probs_b, stds_a, stds_b):
    with open(os.path.join(save_path, f'{save_filename}.pkl'), 'wb') as f:
        probs = {'normalized_a': list(normalized_a),
                 'normalized_b': list(normalized_b),
                 'probs_a': list(probs_a),
                 'probs_b': list(probs_b),
                 'stds_a': list(stds_a),
                 'stdss_b': list(stds_b)                 
                 }
        pkl.dump(probs, f) 

    print(f'inference probs saved at {save_path}/{save_filename}.pkl')


def get_unique_subgoals(missions_list):
    mission_subgoals = {mission: MISSION_TO_INFERENCE[mission] for mission in missions_list}

    other_mission_subgoals = {mission_key: 
        {subgoal for mission, subgoals in mission_subgoals.items() if mission != mission_key for subgoal in subgoals}
        for mission_key in mission_subgoals.keys()
    }

    unique_subgoals = {
        mission: [subgoal for subgoal in subgoals if subgoal not in other_mission_subgoals[mission]]
        for mission, subgoals in mission_subgoals.items()
    }
    
    for mission, subgoals in unique_subgoals.items():
        print(mission, subgoals)
    
    return unique_subgoals


def main_inference(args, data_path, num_samples=100, extra_steps=5, temp=1, only_agent=None):
    data_level = args.model.dirpath
    multiprocessing.set_start_method('spawn')

    target_agent = args.experiment.agent
    target_mission = args.rollout.a_mission if target_agent == 'A' else args.rollout.b_mission
    agent_prefs = {
        'A': {
            args.rollout.a_mission: str(round(float(args.rollout.a_pref), 1)),
            args.rollout.b_mission: str(round(1 - float(args.rollout.a_pref), 1)),
            },
        'B': {
            args.rollout.a_mission: str(round(1 - float(args.rollout.b_pref), 1)),
            args.rollout.b_mission: str(round(float(args.rollout.b_pref), 1))
            }            
    }

    print('agent info')
    print('agent A: ', args.rollout.a_mission, agent_prefs['A'])
    print('agent B: ', args.rollout.b_mission, agent_prefs['B'])

    args.data.data_path = os.path.join(data_path, f'init_config_{args.experiment.experiment_name}', args.rollout.room_config)    
    args.rollout.traj_name = sorted(args.data.data_path)[0] if args.rollout.traj_name is None else args.rollout.traj_name
    
    traj_folder_target = os.path.join(args.data.data_path, args.rollout.traj_name, target_mission)
    inference_file = [f for f in os.listdir(os.path.join(traj_folder_target)) if f.endswith('inference_states.json')][0]

    inference_state_file = f'{inference_file[:5]}_states.json'
    target_step = int(inference_file.split('_')[0])

    target_action = json.load(open(os.path.join(traj_folder_target, f"{target_step-1:05d}_action.json"), 'r'))  
    target_action = [target_action['object_type'], target_action['action_type']]

    steps = json.load(open(os.path.join(traj_folder_target, 'timesteps.json'), 'r'))[:-1] # for vision model, we do not do the last state.
    if target_step - 1 not in steps:
        steps.append(target_step - 1)

    missions = [args.rollout.a_mission, args.rollout.b_mission]

    save_path, save_filename = get_save_to(args, target_mission, target_agent, target_step, data_level, temp)
    print(save_path, save_filename)

    if only_agent is not None:
        temp_mission = args.rollout.a_mission if only_agent == 'A' else args.rollout.b_mission

        print('get inference_probs for agent ', only_agent)
        probs, stds = inference_probs(args, missions, only_agent, temp_mission, agent_prefs[only_agent], target_action, target_step, steps, num_samples, extra_steps, temp)

        a_probs = probs if only_agent == 'A' else [0 for i in probs]
        b_probs = probs if only_agent == 'B' else [0 for i in probs]
        a_stds = stds if only_agent == 'A' else [0 for i in probs]
        b_stds = stds if only_agent == 'B' else [0 for i in probs] 
        
    elif (float(args.rollout.a_pref) < 1.0 or float(args.rollout.b_pref) < 1.0) and float(args.rollout.a_pref) > 0 and float(args.rollout.b_pref) > 0:
        print('get inference_probs for agent A')
        a_probs, a_stds = inference_probs(args, missions, 'A', args.rollout.a_mission, agent_prefs['A'], target_action, target_step, steps, num_samples, extra_steps, temp)

        with open(os.path.join(save_path, f'{save_filename}.pkl'), 'wb') as f:
            probs = {'probs_a': list(a_probs), 'stds_a': list(a_stds)}
            pkl.dump(probs, f) 
        args.model.dirpath = data_level

        print('get inference_probs for agent B')
        b_probs, b_stds = inference_probs(args, missions, 'B', args.rollout.b_mission, agent_prefs['B'], target_action, target_step, steps, num_samples, extra_steps, temp)
    else:
        print('get inference_probs for agent ', target_agent)
        probs, stds = inference_probs(args, missions, target_agent, target_mission, agent_prefs[target_agent], target_action, target_step, steps, num_samples, extra_steps, temp)

        a_probs = probs if target_agent == 'A' else [0 for i in probs]
        b_probs = probs if target_agent == 'B' else [0 for i in probs]
        a_stds = stds if target_agent == 'A' else [0 for i in probs]
        b_stds = stds if target_agent == 'B' else [0 for i in probs]    

    print('final results w/ inference answer ', target_agent, 'and mission ', target_mission)

    print('Agent A ', args.rollout.a_mission, agent_prefs['A'])
    print(f'probs: {a_probs}')
    print(f'stds: {a_stds}') 

    print('Agent B ', args.rollout.b_mission, agent_prefs['B'])
    print(f'probs: {b_probs}')
    print(f'stds: {b_stds}') 

    # # get and save inference plot results
    # save_path, save_filename = get_save_to(args, target_mission, target_agent, target_step, data_level)

    # print(save_path, save_filename)
    # get normalized probs

    a_probs = list(a_probs)
    b_probs = list(b_probs)
    normalized_a, normalized_b = get_normalized_probs(a_probs, b_probs)

    print('save all results')
    save_all_results(save_path, save_filename, normalized_a, normalized_b, a_probs, b_probs, list(a_stds), list(b_stds))
    
    print('get inference plots')
    get_inference_plot(a_probs, b_probs, a_stds, b_stds, steps, save_path, save_filename, prob_type='probs') 
    get_inference_plot(normalized_a, normalized_b, a_stds, b_stds, steps, save_path, save_filename, prob_type='normalized') 

    # a_probs, a_stds = inference_probs(args, missions, 'A', args.rollout.a_mission, agent_prefs['A'], target_action, target_step, steps, num_samples, extra_steps)
    # b_probs, b_stds = inference_probs(args, missions, 'B', args.rollout.b_mission, agent_prefs['B'], target_action, target_step, steps, num_samples, extra_steps)

    # print('final results w/ inference answer ', target_agent, 'and mission ', target_mission)
    # print('Agent A ', args.rollout.a_mission, agent_prefs['A'])
    # print(f'probs: {a_probs}')
    # print(f'stds: {a_stds}') 

    # print('Agent B ', args.rollout.b_mission, agent_prefs['B'])
    # print(f'probs: {b_probs}')
    # print(f'stds: {b_stds}') 

    # # get and save inference plot results
    # save_path, save_filename = get_save_to(args, target_mission, target_agent, target_step)

    # print(save_path, save_filename)
    # # get normalized probs

    # a_probs = list(a_probs)
    # b_probs = list(b_probs)
    # normalized_a, normalized_b = get_normalized_probs(a_probs, b_probs)

    # print('save all results')
    # save_all_results(save_path, save_filename, normalized_a, normalized_b, a_probs, b_probs, list(a_stds), list(b_stds))
    
    # print('get inference plots')
    # get_inference_plot(a_probs, b_probs, a_stds, b_stds, steps, save_path, save_filename, prob_type='probs') 
    # get_inference_plot(normalized_a, normalized_b, a_stds, b_stds, steps, save_path, save_filename, prob_type='normalized') 

