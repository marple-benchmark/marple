import os
import logging

import json
import hydra
from itertools import islice
from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig


from gpt4 import GPT4Agent
from azure import AsyncAzureChatLLM

from helpers import *

from prompts import SYSTEM_MESSAGES, PROMPT_TEMPLATES


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:

    # get model 
    args.model.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    llm = AsyncAzureChatLLM(**args.model.azure_api)
    model = GPT4Agent(llm=llm, **args.model.completion_config)
    
    # prompts 
    system_message = SYSTEM_MESSAGES[args.prompts.system_message]
    prompt_template = PROMPT_TEMPLATES[args.prompts.prompt_template]
    
    # get mission config 
    config_file = json.load(open(args.data.config_file))
    
    # keep track of eval results 
    
    # loop over missions in config file
    for mission in config_file.keys():
        results = {}
        # if mission == 'init_config_change_outfit_do_laundry':
        if mission == "init_config_feed_dog_take_shower":
            results[mission] = {}
            inference_mission = config_file[mission]["inference_mission"]
            other_mission = config_file[mission]["other_mission"]
            inference_question = config_file[mission]["inference_question"]
            room_trajs = config_file[mission]["trajs"]
            
            # if mission == "init_config_feed_dog_take_shower":
            print('mission', mission)
        
            # loop over trajs in mission
            for room_traj in room_trajs:
                
                            
                if not results[mission].get(room_traj[0]):
                    results[mission][room_traj[0]] = {}
                results[mission][room_traj[0]][room_traj[1]] = {}
                
                room = room_traj[0]
                traj = room_traj[1]
                
                    
                print(room_traj)
        

                # get data 
                data = get_data(args.data.dir, mission, inference_mission, other_mission, room, traj)
                inference_state = data["inference_state"]
                inference_trajs = data["inference_trajs"]
                other_trajs = data["other_trajs"]
                inference_trajs_names = data["inference_trajs_names"]    
                other_trajs_names = data["other_trajs_names"] 
                
                step = 0  
                
                save_reasoning = os.path.join(f"{args.data.result_dir}/{mission}/{room}/{traj}", 'reasoning')
                os.makedirs(save_reasoning, exist_ok=True)

                for inference_traj_pair, other_traj_pair, inference_traj_name, other_traj_name in zip(inference_trajs, other_trajs, inference_trajs_names, other_trajs_names):
            

                    results[mission][room_traj[0]][room_traj[1]][step] = {}
                    
                    # a_first = inference_traj_pair[0]["Grid"]["agents"]["initial"][0]["name"] == "A"
                    print(inference_traj_name, other_traj_name)
                
                    prompt = prompt_template.format(
                        state_0_agent_target=inference_traj_pair[0], # A = traget, B equals other
                        current_state_target=inference_traj_pair[1],
                        state_0_agent_other=other_traj_pair[0],
                        current_state_other=other_traj_pair[1],
                        final_state=inference_state,
                        question=inference_question,
                    )

                    responses = model.batch_prompt(
                        system_message=system_message,
                        messages=[prompt],
                    )
                    
                    formatted_responses = [
                        format_response(response, ["Answer"]) 
                        for response in responses[0]
                    ]

                    results[mission][room_traj[0]][room_traj[1]][step] = formatted_responses
                    
                    # Save the results in a JSON file
                    result_file_path = f"{args.data.result_dir}/{mission}/{room}/{traj}/results.json"

                    # Check if the directory exists, if not, create it
                    result_dir = os.path.dirname(result_file_path)
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    # Store the results in a JSON file
                    with open(result_file_path, 'w') as file:
                        json.dump(results, file, indent=4)
                    
                    with open(os.path.join(save_reasoning, f"{step}_reasoning.json"), 'w') as f:
                        json.dump(responses, f, indent=4)
                        
                    step += 1

                    if step == len(inference_trajs):
                        breakpoint()
                    

if __name__ == '__main__':
    main()
