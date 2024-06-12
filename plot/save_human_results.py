import os
import json
import numpy as np

humans = ['zoey', 'zoey_2', 'hannah', 'hannah_expert']

base_path = '/vision/u/emilyjin/marple_long/human_results'

experiments = ['change_outfit_do_laundry', 'feed_dog_take_shower', 'get_snack_clean_living_room_table', 'move_plant_at_night_get_night_snack', 'watch_movie_cozily_watch_news_on_tv']

for exp in experiments:
    # all_responses = []
    for human in os.listdir(os.path.join(base_path)):
        all_responses = []
        if human in humans:
            result_jsons = os.listdir(os.path.join(base_path, human, exp))
            for result_json in result_jsons:
                result_path = os.path.join(base_path, human, exp, result_json)
                with open(result_path) as f:
                    data = json.load(f) 
                
                responses = data['responses'] 
                # flip responses if needed
                if responses[-1] < 50:
                    responses = [100 - r for r in responses]

                # convert to probability
                responses = [r/100 for r in responses]

                if len(responses) >= 11:
                    responses = responses[-11:]
                else:
                    responses = [0] * (11 - len(responses)) + responses


                assert len(responses) == 11
                all_responses.append(responses)

        # if there are data points, should be of shape num participants * num traj (10) x num timesteps (11)
        if len(all_responses) > 0: 
            all_responses = np.array(all_responses)
            print(exp, all_responses.shape)

            all_means = np.mean(all_responses, axis=0)
            all_vars = np.std(all_responses, axis=0)

            path = '/vision/u/emilyjin/marple_long/final_results'
            probs_so_far = json.load(open(os.path.join(path, f'{exp}_1.0_probs.json'), 'r'))
            vars_so_far = json.load(open(os.path.join(path, f'{exp}_1.0_vars.json'), 'r'))

            probs_so_far[human] = list(all_means)
            vars_so_far[human] = list(all_vars)
            
            with open(os.path.join(path, f'{exp}_1.0_probs.json'), 'w') as f:
                json.dump(probs_so_far, f, indent=4)

            with open(os.path.join(path, f'{exp}_1.0_vars.json'), 'w') as f:
                json.dump(vars_so_far, f, indent=4)
            
            print('saved to', path, exp)


# # for exp in experiments:
# #     if os.path.isdir(os.path.join(base_path, exp)):
# #         for human in os.listdir(os.path.join(base_path, exp)):
# #             all_responses = []
# #             result_jsons = os.listdir(os.path.join(base_path, exp, human))
# #             for result_json in result_jsons:
# #                 result_path = os.path.join(base_path, exp, result_json)
# #                 with open(result_path) as f:
# #                     data = json.load(f) 
                
# #                 responses = data['responses'] 
# #                 # flip responses if needed
# #                 if responses[-1] < 50:
# #                     responses = [100 - r for r in responses]

# #                 # convert to probability
# #                 responses = [r/100 for r in responses]

# #                 if len(responses) >= 11:
# #                     responses = responses[-11:]
# #                 else:
# #                     responses = [0] * (11 - len(responses)) + responses

# #                 assert len(responses) == 11

# #                 path = '/vision/u/emilyjin/marple_long/final_results'
# #                 probs_so_far = json.load(open(os.path.join(path, f'{exp}_probs.json'), 'r'))
# #                 vars_so_far = json.load(open(os.path.join(path, f'{exp}_vars.json'), 'r'))

# #                 probs_so_far[human] = responses
# #                 vars_so_far[human] = [0 for i in range(11)]

# #                 with open(os.path.join(path, f'{exp}_probs.json'), 'w') as f:
# #                     json.dump(probs_so_far, f, indent=4)

# #                 with open(os.path.join(path, f'{exp}_vars.json'), 'w') as f:
# #                     json.dump(vars_so_far, f, indent=4)
                
# #                 print('saved to', path, exp)




