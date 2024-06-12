import os
import sys
import json 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import warnings
warnings.filterwarnings("ignore") 

from plot_utils import get_paths, get_prob_correct, FINAL_EXPERIMENTS, make_same_shape, get_plot, gen_avg_plot, get_all_model_data

# sns.set_palette('Set2')
# sns.set_style("ticks")
# sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
# plt.rcParams['font.size'] = 20
sns.set_palette('muted')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})


EXPERIMENT_TO_TITLE = {
    "feed_dog_take_shower": "Who Turned On the Shower?", 
}


MODEL_TO_LABEL = { 
    'low_policy': 'Vision Only',
    'low_policy_2': 'Vision Only (2)',
    'low_policy_audio': 'Vision + Audio',
    'low_policy_audio_2': 'Vision + Audio (2)',    
    'subgoal_low_policy': "Vision + Language",
    'subgoal_low_policy_simulator_2': 'Vision + Language (2)',
    'subgoal_low_policy_predicted_2': 'Vision + Predicted Language (2)',
    'subgoal_low_policy_simulator_audio': 'Vision + Language + Audio',
    'subgoal_low_policy_simulator_audio_2': 'Vision + Language + Audio (2)'
}

prefs = ['1.0', '0.8', '0.6']

def gen_ax(model, all_probs, all_vars, ax, xticks, yticks, xs, fontsize=24):
    for data_type in sorted(all_probs.keys()):
        probs = all_probs[data_type]
        stds = all_vars[data_type] 
        lower = probs - stds * 1.95 / np.sqrt(10)
        upper = probs + stds * 1.95 / np.sqrt(10)

        sns.lineplot(x=xs, y=probs, ax=ax)
        ax.fill_between(xs, lower, upper, alpha=.3) 
        
    ax.set_title(model, fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)    
    ax.set_xlabel('Fraction of Trajectory', fontsize=fontsize)
    ax.set_ylabel('Probability Correct', fontsize=fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    
def main():          
    plot_models = {
        'Fixed Rooms': ['low_policy', 'low_policy_audio', 'subgoal_low_policy', 'subgoal_low_policy_simulator_audio'],
        'Procedurally Generated Rooms': ['low_policy_2', 'low_policy_audio_2', 'subgoal_low_policy_simulator_2', 'subgoal_low_policy_simulator_audio_2'],        
    }
    
    plot_names = ['Fixed Rooms', 'Procedurally Generated Rooms']
    
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 20
    palette = sns.color_palette('colorblind')

    xs = [round(0.1 * i, 1) for i in range(11)]
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0.4, 0.6, 0.8, 1.0]

    experiment_name = 'feed_dog_take_shower'
    
    for plot_name in plot_names:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        i = 0
        
        for pref in prefs:
            models = plot_models[plot_name]
            
            all_probs = json.load(open(f'/vision/u/emilyjin/marple_long/final_results/{experiment_name}_{pref}_probs.json', 'r'))
            all_vars = json.load(open(f'/vision/u/emilyjin/marple_long/final_results/{experiment_name}_{pref}_vars.json', 'r'))

            all_probs = {key: value for key, value in all_probs.items() if key in models}
            all_vars = {key: value for key, value in all_vars.items() if key in models}

            plot_probs = {}
            plot_vars = {}

            for key, value in all_probs.items():
                plot_probs[key] = np.array(value)
            for key, value in all_vars.items():
                plot_vars[key] = np.array(value)            
            
            gen_ax(f'Agent Pref {pref}', plot_probs, plot_vars, axs[i], xticks, yticks, xs)
            
            i += 1
            
        save_path = '/vision/u/emilyjin/marple_long/final_plots/other_pref'
        os.makedirs(save_path, exist_ok=True)

        # Set overall title
        fig.suptitle(plot_name, fontsize=32)

        # Adjust layout
        plt.tight_layout()

        filename = f'{plot_name}.png'
        plt.savefig(os.path.join(save_path, filename), format='png')
        print(f'inference plot saved at {save_path}/{filename}')

    plt.clf() 
         


if __name__ == "__main__":
    main()
            


if __name__ == "__main__":
    main()
