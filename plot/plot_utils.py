import os
import sys
import json 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl



FINAL_EXPERIMENTS = [
    {
        'experiment_name': "watch_movie_cozily_watch_news_on_tv",
        'target_mission': "watch_movie_cozily", 
        'target_agent': 'A',
        'mission_1': 'watch_movie_cozily',
        'mission_2': 'watch_news_on_tv'
    },
    {
        'experiment_name': "feed_dog_take_shower",
        'target_mission': "take_shower", 
        'target_agent': 'B',
        'mission_1': 'feed_dog',
        'mission_2': 'take_shower'
    },
    {
        'experiment_name': "get_snack_clean_living_room_table",
        'target_mission': "clean_living_room_table", 
        'target_agent': 'A',
        'mission_1': 'get_snack',
        'mission_2': 'clean_living_room_table'
    },
    {
        'experiment_name': "move_plant_at_night_get_night_snack",
        'target_mission': "move_plant_at_night", 
        'target_agent': 'A',
        'mission_1': 'move_plant_at_night',
        'mission_2': 'get_night_snack'
    },
    {   
        'experiment_name': "change_outfit_do_laundry",
        'target_mission': "do_laundry", 
        'target_agent': 'B',
        'mission_1': 'change_outfit',
        'mission_2': 'do_laundry'
    } 
]
 

def make_same_shape(array, target):
    if len(array) < len(target):
        array = [0] * (len(target) - len(array)) + array 
    else:
        array = array[-len(target):]
    
    return array
    
def get_paths(experiment_name, agent_pref, room_config, traj_name, target_mission, target_agent):
    inference_scenario = '-'.join([target_mission, target_agent])
    # agent_pref = f'{args.rollout.a_mission}-{args.rollout.a_pref}-{args.rollout.b_mission}-{args.rollout.b_pref}'

    load_path = '/vision/u/emilyjin/marple_long/inference/final_results'
    save_path = "/vision/u/emilyjin/marple_long/plots"

    dirs = os.path.join(f'{experiment_name}-{target_agent}', f'{room_config}-{traj_name}', agent_pref)

    load_path = os.path.join(load_path, dirs)
    save_path = os.path.join(save_path, f'{experiment_name}-{target_agent}', agent_pref)

    os.makedirs(save_path, exist_ok=True)       
    return load_path, save_path

def get_prob_correct(model_probs, inference_answer, steps):
    probs_correct = {}

    for model, probs in model_probs.items():
        try:
            if model == 'llm':
                probs_correct[model] = {
                    'steps': probs['steps'],
                    'prob_correct': [p / 100. for p in probs['prob_correct']],
                    'stds_correct': [p / 100. for p in probs['stds_correct']]
                }
            else:
                probs_correct[model] = {
                    'steps': steps,
                    'prob_correct': probs['probs_a'] + [1] if inference_answer == 'A' else probs['probs_b'] + [1],
                    'prob_wrong': probs['probs_b'] if inference_answer == 'A' else probs['probs_a']
                }

                if 'stds_a' in probs.keys() and 'stdss_b' in probs.keys():
                    probs_correct[model]['stds_correct'] = probs['stds_a'] + [0] if inference_answer == 'A' else probs['stdss_b'] + [0]
                    probs_correct[model]['stds_wrong'] = probs['stdss_b'] + [0] if inference_answer == 'A' else probs['stds_a'] + [0]
                else:
                    probs_correct[model]['stds_correct'] = [0 for _ in range(len(probs['probs_a']))],
                    probs_correct[model]['stds_wrong'] = [0 for _ in range(len(probs['probs_a']))]
        except Exception as e:
            print(e)
            # breakpoint()
    return probs_correct



def get_plot(save_path, times, model_probs, experiment_name, room_traj, prob_type='prob_correct', filename=None, title=None):
    # plot of P(normalized answer) for each model
    xticks = [round(0.2 * i, 1) for i in range(11)]
    # xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # color = {'low_policy': 'green', 'subgoal_low_policy': 'blue'}
    for model in sorted(model_probs.keys()):
        try: 
            probs = model_probs[model]
            prob = np.array(probs[prob_type])
            if model == 'low_policy' or model == 'subgoal_low_policy':
                prob = prob / 2 + 0.5

            xs = np.array(probs['steps']) / max(probs['steps'])
            std = np.array(probs['stds_correct'])
            std = std / 2
            sns.lineplot(x=xs, y=prob, label=model)

            lower = prob - std / 2
            upper = prob + std / 2

            plt.fill_between(xs, lower, upper, alpha=.3)

            # plt.errorbar(xs, probs[prob_type], yerr=probs['stds_correct'], alpha=0.8, fmt='o')#, ecolor=color[model], color=color[model])
        except Exception as error:
            # xs = [i for i in range(len(probs[prob_type]))]
            # xs = np.array(xs) / max(xs)
            # sns.lineplot(x=xs, y=probs[prob_type], label=model)
            print(error)
            # breakpoint()
        # plt.errorbar(times, probs_a, yerr=stds_a, alpha=0.3, fmt='-o')
    
    plt.xticks(xticks)
    plt.ylim(-0.01, 1.01)
    plt.xlim(0, 1) 
    plt.xlabel('Fraction of Trajectory')
    title = f'{experiment_name}-{room_traj}' if title is None else title
    plt.title(title)
    
    ylabel = 'Probability of Correct Answer (normalized)' if prob_type == 'normalized' else 'Probability of Correct Answer'
    plt.ylabel(ylabel)
    plt.legend()

    filename = f'{room_traj}-{prob_type}.png' if filename is None else filename
    plt.savefig(os.path.join(save_path, filename))
    print(f'inference plot saved at {save_path}/{filename}')

    plt.clf() 


# def get_all_model_data(data_path, load_path, models, target_agent):
def get_all_model_data(data_path, load_path, target_agent):
    target_step = [f for f in os.listdir(os.path.join(data_path)) if f.endswith('inference_states.json')][0]
    target_step = int(target_step.split('_')[0])

    steps = os.path.join(data_path, 'timesteps.json')
    steps = json.load(open(steps, 'r'))
    if target_step - 1 not in steps:
        steps.append(target_step - 1)

    steps = sorted(steps)
 
    if os.path.exists(load_path):
        model_pkls = [f for f in os.listdir(load_path) if f.endswith('.pkl')]
#         model_pkls = [f for f in model_pkls if os.path.splitext(f)[0] in models]

        if len(model_pkls) > 0:
            model_probs = {os.path.splitext(f)[0]: pkl.load(open(os.path.join(load_path, f), 'rb')) for f in model_pkls}
            model_probs = get_prob_correct(model_probs, target_agent, steps)
        else:
            model_probs = {}

    return model_probs


def gen_avg_plot(all_avg_probs, all_vars, experiment_name, save_path, filename=None, title=None, ymin=0, ymax=1, legend=True):
        # plot of P(normalized answer) for each model
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 24
    palette = sns.color_palette('colorblind')


    xs = [round(0.1 * i, 1) for i in range(11)]
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0.4, 0.6, 0.8, 1.0]
    for i, model in enumerate(sorted(all_avg_probs.keys())):
        try: 
            probs = all_avg_probs[model]

            # sns.lineplot(x=xticks, y=probs, data=stds, ci='sd', err_style='bars', label=model)
            if legend:
                sns.lineplot(x=xs, y=probs, label=model)
            else:
                sns.lineplot(x=xs, y=probs)
            
            if all_vars is not None:
                stds = all_vars[model] 

                lower = probs - stds * 1.95 / np.sqrt(10)
                upper = probs + stds * 1.95 / np.sqrt(10)

                plt.fill_between(xs, lower, upper, alpha=.3)

        except Exception as error: 
            print(error) 
            # breakpoint()
    
    plt.xticks(xticks, fontsize=15)
    plt.yticks(yticks, fontsize=15)
    plt.ylim(ymin, ymax)
    plt.xlim(0, 1) 
    plt.xlabel('Fraction of Trajectory', fontsize=15)

    title = experiment_name if title is None else title
    plt.title(title, fontsize=20)
    
    ylabel = 'Probability of Correct Answer'
    plt.ylabel(ylabel, fontsize=15)

    if legend:
        plt.legend()
    plt.tight_layout()

    filename = f'{experiment_name}.svg' if filename is None else f'{filename}.png'
    plt.savefig(os.path.join(save_path, filename), format='png')
    # plt.savefig(os.path.join(save_path, filename), format='png')
    print(f'inference plot saved at {save_path}/{filename}')

    plt.clf() 



def main():

    for experiment in FINAL_EXPERIMENTS:
        experiment_name = experiment['experiment_name']
        target_mission = experiment['target_mission']
        target_agent = experiment['target_agent'] 

        inference_path = os.path.join('/vision/u/emilyjin/marple_long/inference', f'{experiment_name}-{target_agent}')

        if os.path.exists(inference_path):
            for room_traj in os.listdir(inference_path):

                room_config, traj_name = room_traj.split('-')

                data_path = os.path.join('/vision/u/emilyjin/mini-behavior-llm-baselines', 'data', f'init_config_{experiment_name}', room_config, traj_name, target_mission)

                for agent_pref in os.listdir(os.path.join(inference_path, room_traj)):
                    try:
                        target_step = [f for f in os.listdir(os.path.join(data_path)) if f.endswith('inference_states.json')][0]
                        target_step = int(target_step.split('_')[0])

                        steps = os.path.join(data_path, 'timesteps.json')
                        steps = json.load(open(steps, 'r'))
                        if target_step - 1 not in steps:
                            steps.append(target_step - 1)

                        load_path, save_path = get_paths(experiment_name, agent_pref, room_config, traj_name, target_mission, target_agent)

                        model_pkls = [f for f in os.listdir(load_path) if f.endswith('.pkl')]
                        model_pkls = [f for f in model_pkls if f!='llm_normalized.pkl' and f!='llm_probs.pkl']
                        if len(model_pkls) > 0:

                            model_probs = {os.path.splitext(f)[0]: pkl.load(open(os.path.join(load_path, f), 'rb')) for f in model_pkls}
                            model_probs = get_prob_correct(model_probs, target_agent, steps)

                            get_plot(save_path, steps, model_probs, experiment_name, room_traj, prob_type='prob_correct')
                            # get_plot(save_path, steps, model_probs, experiment_name, room_traj,  prob_type='normalized')
                    except:
                        print(1)

if __name__ == "__main__":
    main()
