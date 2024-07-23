import os
import json
import hydra
import multiprocessing
from arguments import Arguments
from inference import get_save_to, get_normalized_probs, save_all_results, get_inference_plot


@hydra.main(version_base='1.1', config_path="config", config_name="inference.yaml")
def main(args: DictConfig) -> None:
    if args.rollout.rollout_level == 'low':
        args.data.data_level = 'low'
        args.model.model_name = 'low_policy'
        args.data.goal_conditioned = False
    else:
        args.data.data_level = 'mid'
        args.model.model_name = 'subgoal_low_policy'
        args.data.goal_conditioned = True
        args.rollout.simulator = True

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

    args.data.data_path = os.path.join(args.data.data_path, f'init_config_{args.experiment.experiment_name}', args.rollout.room_config)    
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

    save_path, save_filename = get_save_to(args, target_mission, target_agent, target_step, data_level, args.rollout.temp)
    print(save_path, save_filename)

    print('get inference_probs for agent A')
    a_probs, a_stds = get_inference_probs(args, missions, 'A', args.rollout.a_mission, agent_prefs['A'], target_action, target_step, steps, args.rollout.num_samples, args.rollout.extra_steps, args.rollout.temp)

    with open(os.path.join(save_path, f'{save_filename}.pkl'), 'wb') as f:
        probs = {'probs_a': list(a_probs), 'stds_a': list(a_stds)}
        pkl.dump(probs, f) 
    args.model.dirpath = data_level

    print('get inference_probs for agent B')
    b_probs, b_stds = get_inference_probs(args, missions, 'B', args.rollout.b_mission, agent_prefs['B'], target_action, target_step, steps, args.rollout.num_samples, args.rollout.extra_steps, args.rollout.temp)

    print('final results w/ inference answer ', target_agent, 'and mission ', target_mission)

    print('Agent A ', args.rollout.a_mission, agent_prefs['A'])
    print(f'probs: {a_probs}')
    print(f'stds: {a_stds}') 

    print('Agent B ', args.rollout.b_mission, agent_prefs['B'])
    print(f'probs: {b_probs}')
    print(f'stds: {b_stds}') 

    a_probs = list(a_probs)
    b_probs = list(b_probs)
    normalized_a, normalized_b = get_normalized_probs(a_probs, b_probs)

    print('save all results')
    save_all_results(save_path, save_filename, normalized_a, normalized_b, a_probs, b_probs, list(a_stds), list(b_stds))
    
    print('get inference plots')
    get_inference_plot(a_probs, b_probs, a_stds, b_stds, steps, save_path, save_filename, prob_type='probs') 
    get_inference_plot(normalized_a, normalized_b, a_stds, b_stds, steps, save_path, save_filename, prob_type='normalized') 


if __name__ == '__main__':
    main()