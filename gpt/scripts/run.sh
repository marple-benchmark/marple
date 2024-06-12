#!/bin/bash

# Define the list of configurations
declare -a configs=(
    "init_config_change_outfit_move_plant_at_night"
    "init_config_do_laundry_feed_dog"
    "init_config_get_snack_get_night_snack"
    # "init_config_watch_movie_cozily_watch_news_on_tv"
    "init_config_take_shower_clean_living_room_table"
)

# Activate conda environment
source /Users/iphilipp/miniforge3/etc/profile.d/conda.sh
conda activate scai-tuning

# Loop over the configurations
for config in "${configs[@]}"
do
    cd /Users/iphilipp/Documents/research/mini-behavior-llm-baselines/src
    
    python main.py data.mission=$config 
done
