import os
import json
import time
import copy
import calendar

import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *

from src.mini_behavior import *
from src.models.pl_models import BasicModel, LowBasicModel
from src.models.rollout import Rollout


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
    return successes, rollout


def get_inference_probs(args, missions, agent, mission_name, agent_prefs, target_action, target_step, steps, num_samples=100, extra_steps=10, temp=1):
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
    stds = np.std(all_successes, axis=1) t

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
