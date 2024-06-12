import argparse
from gym_minigrid.wrappers import *
from mini_behavior.minibehavior import *
from auto_control import plan_actions, conduct_actions
import numpy as np
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser("marple argument parser")

    parser.add_argument('--env', type=str,
        default='MiniGrid-WhoLeftTheLightOnInKitchen-16x16-N2-v1',
        help='gym environment to load')
    parser.add_argument('-c', '--config', type=str,
        default='mini_behavior/configs/init_config_default.json',
        help='path to config JSON file')
    parser.add_argument('--mission_config', type=str,
        default='mini_behavior/configs/toggle_light_in_kitchen_config.json',
        help='path to mission config JSON file')
    parser.add_argument('--trajectory', type=str,
        default='demos/seed1',
        help='path to directory containing actual trajectory')
    parser.add_argument('--num-simulations', type=int, default=1,
        help='number of times to simulate each agent')

    parser.add_argument('--seed', type=int, default=1,
        help='set pseudorandom seed')

    return parser.parse_args()


def reset(env):
    env.seed(args.seed)
    env.reset()
    

def get_evidence(initial_state, final_state):
    # return objects that changed between initial and final states 
    changes = []

    for room in final_state['Grid']['rooms']['initial']:
        initial_rooms = list(filter(lambda r: r['name'] == room['name'],
                                    initial_state['Grid']['rooms']['initial']))
        assert len(initial_rooms) > 0, f"room {room['name']} doesn't match"
        initial_room = initial_rooms[0]
        for furn in room['furnitures']['initial']:
            initial_furns = list(filter(lambda f: f['name'] == furn['name'],
                                        initial_room['furnitures']['initial']))
            assert len(initial_furns) > 0, f"furniture {furn['name']} doesn't match"
            initial_furn = initial_furns[0]

            # check if own state changed
            if furn['state'] != initial_furn['state']:
                changes.append(furn)
                continue

            # check if own position changed
            if furn['pos'] != initial_furn['pos']:
                changes.append(furn)

            # check if state or position of any objs in it has changed
            for obj in furn['objs']['initial']:
                initial_objs = list(filter(lambda o: o['name'] == obj['name'],
                                           initial_furn['objs']['initial']))
                assert len(initial_objs) > 0, f"object {obj['name']} doesn't match"
                initial_obj = initial_objs[0]

                if obj['state'] != initial_obj['state']:
                    changes.append(obj)
                    continue
                if obj['pos'] != initial_obj['pos']:
                    changes.append(obj)
                    continue

    # check if any door states have changed
    for door in final_state['Grid']['doors']['initial']:
        initial_doors = list(filter(lambda d: d['pos'] == door['pos'],
                                    initial_state['Grid']['doors']['initial']))
        assert len(initial_doors) > 0, f"door at {door['pos']} doesn't match"
        initial_door = initial_doors[0]
        if door['state'] != initial_door['state']:
            changes.append(door)

    return changes


def contains_evidence(env, evidence):
    # check whether environment contains evidence (objects match)
    objs = env.get_state()['objs']
    for ev in evidence:
        actual_evs = list(filter(lambda e: e.name == ev['name'],
                                 objs[ev['type']]))
        assert len(actual_evs) > 0, f"object {ev['name']} doesn't match"
        actual_ev = actual_evs[0]
        if eval(ev['pos']) != actual_ev.cur_pos:
            return False
        for state in ev['state']:
            if not actual_ev.states[state].value:
                return False
    return True


def simulate_agent(env, agent_id, mission, evidence, args):
    print(f'\n{"-"*30}\nstart simulating {env.agents[agent_id].name}\n{"-"*30}')
    num_evidence = 0
    for i in range(args.num_simulations):
        reset(env)
        env.set_agent(agent_id)
        step_count = 0
        actions = plan_actions(mission, env)
        _, _, _, _, _ = conduct_actions(
            args, actions, env)
        if contains_evidence(env, evidence):
            num_evidence += 1
    return num_evidence/args.num_simulations


if __name__ == '__main__':
    args = parse_arguments()

    with open(args.config, 'r') as f:
        initial_dict = json.load(f)
    with open(args.mission_config, 'r') as f:
        mission_dict = json.load(f)
    
    env = gym.make(args.env, initial_dict = initial_dict)

    # extract evidence from saved trajectory
    states = sorted(os.listdir(args.trajectory))
    with open(f'{args.trajectory}/{states[0]}', 'r') as f:
        initial_state = json.load(f)
    with open(f'{args.trajectory}/{states[-1]}', 'r') as f:
        final_state = json.load(f)
    evidence = get_evidence(initial_state, final_state)
    print(f'\nevidence: {evidence}')

    # sample agents carrying out mission and compare evidence
    args.save, args.gif = False, False
    evidence_probs = {}
    for i in range(len(env.agents)):
        prob = simulate_agent(env, i, mission_dict, evidence, args)
        evidence_probs[env.agents[i].name] = prob
        print(f'prob: {prob}')

    # normalize to get posteriors
    factor = 1.0/sum(evidence_probs.values())
    posterior = {k: v * factor for k, v in evidence_probs.items()}
    print(f'\nposterior: {posterior}')