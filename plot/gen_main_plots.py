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
    "watch_movie_cozily_watch_news_on_tv": "Who Picked Up the Pillow?",
    "feed_dog_take_shower": "Who Turned On the Shower?",
    "get_snack_clean_living_room_table": "Who Picked up the Sandwich?",
    "move_plant_at_night_get_night_snack": 'Who Picked up the Plant?',
    "change_outfit_do_laundry": 'Who Turned On the Laundry?'
}


MODEL_TO_LABEL = {
    'llm': 'LLM',
    'human': 'Human',
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

pref = '1.0'

def main():
#     'llm', 'vision_only', 'all_3_modalities', 'human_expert'
    plot_models = {
        'main': ['llm', 'low_policy', 'human', 'subgoal_low_policy_simulator_audio'],
        'mcts_1': ['low_policy', 'low_policy_audio', 'subgoal_low_policy', 'subgoal_low_policy_simulator_audio'],
        'mcts_2': ['low_policy_2', 'low_policy_audio_2', 'subgoal_low_policy_simulator_2', 'subgoal_low_policy_simulator_audio_2'],        
        'humans': ['zoey', 'zoey_2', 'hannah', 'hannah_expert']
    }
    
    for plot_name, models in plot_models.items():
        for experiment in FINAL_EXPERIMENTS:
            experiment_name = experiment['experiment_name']
            
            all_probs = json.load(open(f'/vision/u/emilyjin/marple_long/final_results/{experiment_name}_{pref}_probs.json', 'r'))
            all_vars = json.load(open(f'/vision/u/emilyjin/marple_long/final_results/{experiment_name}_{pref}_vars.json', 'r'))

            all_probs = {key: value for key, value in all_probs.items() if key in models}
            all_vars = {key: value for key, value in all_vars.items() if key in models}

            plot_probs = {}
            plot_vars = {}

            for key, value in all_probs.items():
                model_name = MODEL_TO_LABEL.get(key, key)
                plot_probs[model_name] = np.array(value)
            for key, value in all_vars.items():
                model_name = MODEL_TO_LABEL.get(key, key)
                plot_vars[model_name] = np.array(value)            

            save_path = os.path.join('/vision/u/emilyjin/marple_long/final_plots', plot_name)
            os.makedirs(save_path, exist_ok=True)
            
            ymin = 0.2 if plot_name == 'main' else 0.4
            gen_avg_plot(
                plot_probs, plot_vars, experiment_name, save_path, 
                ymin=ymin, 
                ymax=1.0, 
                filename=f'{experiment_name}-pref_{pref}', 
                title=EXPERIMENT_TO_TITLE[experiment_name], 
                legend=True
                # legend=False
            )


if __name__ == "__main__":
    main()
