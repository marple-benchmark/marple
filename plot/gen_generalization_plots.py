import os
import sys
import json 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")#, "is_categorical_dtype")
# warnings.filterwarnings("ignore", "use_inf_as_na") 
# warnings.filterwarnings("ignore", "use_inf_as_na") 

from plot_utils import get_paths, get_prob_correct, FINAL_EXPERIMENTS, make_same_shape, get_plot, gen_avg_plot, get_all_model_data

# sns.set_palette('Set2')
# sns.set_style("ticks")
# sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
# plt.rcParams['font.size'] = 20
sns.set_palette('muted')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})


EXPERIMENT_TO_TITLE = {
    "watch_movie_cozily_watch_news_on_tv": "Who Picked Up the Pillow?",
    "feed_dog_take_shower": "Who Turned On the Shower?",
    "get_snack_clean_living_room_table": "Who Picked up the Sandwich?",
    "move_plant_at_night_get_night_snack": 'Who Picked up the Plant?',
    "change_outfit_do_laundry": 'Who Turned On the Laundry?'
}
 
pref = '1.0'

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
        'Vision Only': ['low_policy', 'low_policy_2'],
        'Vision + Audio': ['low_policy_audio', 'low_policy_audio_2'],
        'Vision + Language': ['subgoal_low_policy', 'subgoal_low_policy_simulator_2'],
        'Vision + Language + Audio': ['subgoal_low_policy_simulator_audio',  'subgoal_low_policy_simulator_audio_2'],        
    }
    
    plot_names = ['Vision Only', 'Vision + Audio', 'Vision + Language', 'Vision + Language + Audio']
    
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 24
    palette = sns.color_palette('colorblind')

    xs = [round(0.1 * i, 1) for i in range(11)]
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0.4, 0.6, 0.8, 1.0]

        
    for experiment in FINAL_EXPERIMENTS:
        fig, axs = plt.subplots(1, 4, figsize=(25, 6))
        i = 0
        
        for plot_name in plot_names:
            models = plot_models[plot_name]
            experiment_name = experiment['experiment_name']
            
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
            
            gen_ax(plot_name, plot_probs, plot_vars, axs[i], xticks, yticks, xs)
            
            i += 1
            
        save_path = os.path.join('/vision/u/emilyjin/marple_long/final_plots/generalization')#, experiment_name)
        os.makedirs(save_path, exist_ok=True)

        # Set overall title
        fig.suptitle(EXPERIMENT_TO_TITLE[experiment_name], fontsize=42)

        # Adjust layout
        plt.tight_layout()

        filename =f'{experiment_name}-pref_{pref}.png'
        plt.savefig(os.path.join(save_path, filename), format='png')

        print(f'inference plot saved at {save_path}/{filename}')

    plt.clf() 
         


if __name__ == "__main__":
    main()
