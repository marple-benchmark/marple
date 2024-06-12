import os
import sys
import json 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na") 

from plot_utils import get_paths, get_prob_correct, FINAL_EXPERIMENTS, make_same_shape, get_plot, gen_avg_plot, get_all_model_data


prefs = ['0.8', '0.6']

def main():
    for pref in prefs:
        experiment = FINAL_EXPERIMENTS[1]
        experiment_name = experiment['experiment_name']
        target_mission = experiment['target_mission']
        target_agent = experiment['target_agent'] 
        
        mission_1 = experiment['mission_1'] 
        mission_2 = experiment['mission_2'] 

            inference_path = os.path.join('/vision/u/emilyjin/marple_long/inference/final_results', f'{experiment_name}-{target_agent}')

    #         all_correct = {model: [] for model in models}
            all_correct = {}
    #         all_wrong = {model: [] for model in models}

            all_results = {}

            if os.path.exists(inference_path):
                for room_traj in os.listdir(inference_path):
                    room_config, traj_name = room_traj.split('-')
                    data_path = os.path.join('/vision/u/emilyjin/mini-behavior-llm-baselines', 'data', f'init_config_{experiment_name}', room_config, traj_name, target_mission)

                    print(os.path.join(inference_path, room_traj))

                    agent_pref = f'{mission_1}-{pref}-{mission_2}-{pref}-temp_1.0'
                    load_path, save_path = get_paths(experiment_name, agent_pref, room_config, traj_name, target_mission, target_agent)

                    inference_scenario = '-'.join([target_mission, target_agent])

                    load_path = os.path.join(f'/vision/u/emilyjin/marple_long/inference/{experiment_name}-{target_agent}', f'{room_config}-{traj_name}', agent_pref)

                    if os.path.exists(load_path):
                        print(load_path)

    #                     model_probs = get_all_model_data(data_path, load_path, models, target_agent)
                        model_probs = get_all_model_data(data_path, load_path, target_agent)

                        # add to all_probs
                        for model in model_probs.keys(): 
                            cur_prob = make_same_shape(model_probs[model]['prob_correct'], [0] * 11)
                            cur_model_prob = all_correct.get(model, [])
                            cur_model_prob.append(cur_prob)
                            all_correct[model] = cur_model_prob

    #                         cur_prob = make_same_shape(model_probs[model]['prob_wrong'], [0] * 11)
    #                         all_wrong[model].append(cur_prob)

                for model in all_correct.keys(): 
                    correct_probs = np.array(all_correct[model]) 
                    wrong_probs = np.zeros_like(correct_probs)
                    probs = np.exp(5 * np.dstack([correct_probs, wrong_probs]))
                    softmax = probs / np.sum(probs, axis=2, keepdims=True)

                    softmax = softmax[:, :, 0]
                    prob_correct = np.mean(softmax, axis=0)
                    std_correct = np.std(softmax, axis=0)
                    all_results[model] = prob_correct

                    if os.path.exists(os.path.join('/vision/u/emilyjin/marple_long/final_results', f'{experiment_name}_{pref}_probs.json')):
                        probs_so_far = json.load(open(os.path.join('/vision/u/emilyjin/marple_long/final_results', f'{experiment_name}_{pref}_probs.json'), 'r'))
                    else:
                        probs_so_far = {}
                    probs_so_far[model] = prob_correct.tolist()

                    with open(os.path.join('/vision/u/emilyjin/marple_long/final_results', f'{experiment_name}_{pref}_probs.json'), 'w') as f:
                        json.dump(probs_so_far, f, indent=4)

                    if os.path.exists(os.path.join('/vision/u/emilyjin/marple_long/final_results', f'{experiment_name}_{pref}_vars.json')):
                        stds_so_far = json.load(open(os.path.join('/vision/u/emilyjin/marple_long/final_results', f'{experiment_name}_{pref}_vars.json'), 'r'))
                    else:
                        stds_so_far = {}

                    stds_so_far[model] = std_correct.tolist()

                    with open(os.path.join('/vision/u/emilyjin/marple_long/final_results', f'{experiment_name}_{pref}_vars.json'), 'w') as f:
                        json.dump(stds_so_far, f, indent=4) 

if __name__ == "__main__":
    main()
