from typing import Dict, Any, List

import os
import json


def get_data(data_path, mission, inference_mission, other_mission, room, traj):
    # get data
    traj_path = f"{data_path}{mission}/{room}/{traj}"
    
    inference_traj_path = f"{traj_path}/{inference_mission}"
    other_traj_path = f"{traj_path}/{other_mission}"
    
    timesteps = json.load(open(f"{inference_traj_path}/timesteps.json"))
    
    try:
    
        # get inference state
        inference_traj_states = os.listdir(inference_traj_path)
        inference_traj_states = [state for state in inference_traj_states if "states.json" in state]
        inference_traj_states = sorted(inference_traj_states, key=lambda x: int(x.partition('_')[0]))
        inference_state = [state for state in inference_traj_states if "inference" in state][0]
        inference_traj_states.remove(inference_state)   
        inference_traj_timesteps = [(inference_traj_states[max(0, i - 1)],  inference_traj_states[i]) for i in timesteps]
        
        
        # get other state
        other_traj_states = os.listdir(other_traj_path)
        other_traj_states = [state for state in other_traj_states if "states.json" in state]
        other_traj_states = sorted(other_traj_states, key=lambda x: int(x.partition('_')[0]))
        other_state = [state for state in other_traj_states if "inference" in state][0]
        other_traj_states.remove(other_state)
        other_traj_timesteps = [(other_traj_states[max(0, i - 1)],  other_traj_states[i]) for i in timesteps]
        
        # load jsons 
        inference_traj_timestamp_states = [(
            json.load(open(os.path.join(inference_traj_path, state_0))),
            json.load(open(os.path.join(inference_traj_path, state_1))),
        ) for state_0, state_1 in inference_traj_timesteps]
        
        other_traj_timestamp_states = [(
            json.load(open(os.path.join(other_traj_path, state_0))),
            json.load(open(os.path.join(other_traj_path, state_1))),
        ) for state_0, state_1 in other_traj_timesteps]
    except:
        print(inference_traj_path)
        breakpoint()
        
    
    return {
        "inference_state_name": inference_state,
        "inference_state": json.load(open(os.path.join(inference_traj_path, inference_state))),
        "inference_trajs_names": inference_traj_timesteps,
        "other_trajs_names": other_traj_timesteps,
        "inference_trajs": inference_traj_timestamp_states,
        "other_trajs": other_traj_timestamp_states,
    }
        


def format_response(out: str, vars: List[str]) -> Dict[str, str]:
    var_dict = {var.lower(): None for var in vars}

    try:
        out = out.split('\n')
        out = [l for l in out if ':' in l]
        for line in out:
            elems = line.split(': ')
            for var in vars:
                if var in elems[0]:
                    try:
                        # Attempt to convert elems[1] to float
                        var_dict[var.lower()] = float(elems[1])
                    except ValueError:
                        # If conversion fails, set var_dict value to 50.0
                        var_dict[var.lower()] = 50.0
    except Exception as e:
        print(e)

    return var_dict