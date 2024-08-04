import os
import sys
import json
import calendar
import time
import multiprocessing
from copy import deepcopy

import hydra
import numpy as np
import torch
from joblib import Parallel, delayed
from array2gif import write_gif
from gym_minigrid.wrappers import *

import src.simulator.vocab as vocab
from src.models.pl_models import BasicModel, LowBasicModel


class Rollout:
    def __init__(self, args, low_policy_model=None, mid_policy_model=None):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.create_save_rollout_folder()
        self.set_rollout_traj_name() 
        self.set_init_step_count()
        self.set_rollout_depth()
        self.set_inference_type()
        self.set_mission_dict()
        self.load_initial_state()
        
        self.load_policy_model(rollout_level='low', policy_model_checkpoint=os.path.join(self.args.model.dirpath, f'{self.args.model.model_name}_{self.args.model.embedding_dim}.ckpt'))
        if self.args.model.model_name == 'subgoal_low_policy':
            if self.args.rollout.simulator:
                self.load_subgoals()
            else:
                self.env.cur_subgoal = None
                self.load_policy_model(rollout_level='mid', policy_model_checkpoint=os.path.join(self.args.model.dirpath, f'transformer_{self.args.model.embedding_dim}.ckpt'))

        self.env.hist_states = []

    def reset(self):
        self.set_init_step_count()
        self.env.reset()
        if self.args.model.model_name == 'subgoal_low_policy':
            self.load_subgoals()

    def re_load(self, init_step_count, rollout_depth, verbose=True):
        self.set_init_step_count(init_step_count)
        self.set_rollout_depth(rollout_depth)
        self.load_initial_state(verbose=verbose)
        
        if self.args.model.model_name == 'subgoal_low_policy':
            if self.args.rollout.simulator:
                self.load_subgoals()
            else:
                self.env.cur_subgoal = None

    def create_save_rollout_folder(self, save_rollout_folder=None):
        '''
        Create folder to save rollout.

        Args:
            save_rollout_folder (str, optional): The path of the save rollout folder. 
                                                If None, a new folder will be created. 
                                                Defaults to None.

        Returns:
            None
        '''
        if not save_rollout_folder:
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_rollout_folder = os.path.join(self.args.rollout.rollout_dir, 
                                            # self.args.experiment.experiment_name,
                                            self.args.rollout.policy_model_checkpoint_short,
                                            self.args.experiment.mission_name, 
                                            self.args.rollout.traj_name, 
                                            f"rolloutfrom{self.args.rollout.from_step_count:05d}",
                                            time_stamp)

        if self.args.simulator.save:
            os.makedirs(save_rollout_folder, exist_ok=True)
            print("Created rollout folder", save_rollout_folder)

        self.save_rollout_folder = save_rollout_folder
    
    def load_policy_model(self, rollout_level=None, policy_model_checkpoint=None) -> None:
        '''
        Load policy model from checkpoint and set self.policy_model.

        Args:
            policy_model_checkpoint (str, optional): The path of the policy model checkpoint. 
                                                    If None, the default checkpoint will be used. 
                                                    Defaults to None.
        Returns:
            None
        '''
        rollout_level = self.args.rollout.rollout_level if rollout_level is None else rollout_level
        policy_model_checkpoint = self.args.rollout.policy_model_checkpoint if policy_model_checkpoint is None else policy_model_checkpoint

        if rollout_level == 'low':
            model_class = LowBasicModel
        elif rollout_level == 'mid':
            model_class = BasicModel

   
        print(f'loading {rollout_level} policy model from {policy_model_checkpoint}')
        if rollout_level == "low":
            self.policy_model = model_class.load_from_checkpoint(policy_model_checkpoint, map_location=torch.device(self.device))
        else:
            self.mid_policy_model = model_class.load_from_checkpoint(policy_model_checkpoint, map_location=torch.device(self.device))

        self.policy_model.eval()

    def load_initial_state(self, state_json_path=None, mission_preference=None, verbose=True):
        '''
        Generate initial env from json file as the start of rollout.
        '''
        init_filename = "" if self.args.data.data_level == 'low' else "midlevel/"
        init_filename += f"{self.init_step_count :05d}" 
        
        state_json_path = os.path.join(self.args.data.data_path,  
                                       self.args.rollout.traj_name, 
                                       self.args.experiment.mission_name) 

        with open(os.path.join(state_json_path, f"{init_filename}_states.json"), 'r') as f:
            initial_dict = json.load(f)
            if verbose:
                print('load_initial_state from:', state_json_path)

        if self.args.rollout.inference:
            for agent in initial_dict['Grid']['agents']['initial']:
                agent['mission_preference_initial'] = {self.args.experiment.mission_name: 1} 

        if self.args.rollout.audio:
            with open(os.path.join(state_json_path, f"{init_filename}_action.json"), 'r') as f:
                action = json.load(f)
                self.audio = np.expand_dims(np.array(vocab.ACTION_TO_AUDIO_MASK[vocab.IDX_TO_ACTION[action['action_type']]]), axis=0)
        else:
            self.audio = None

        self.env = gym.make(self.args.simulator.env, initial_dict=initial_dict)
        self.env.grid.load_init = True
        self.reset()
        
    def load_subgoals(self, subgoal_json_path=None):
        '''
        Get the subgoal from json file as an extra information of rollout.
        '''
        self.subgoals = [] 
        subgoal_json_path = os.path.join(self.args.data.data_path,  
                                       self.args.rollout.traj_name, 
                                       self.args.experiment.mission_name, 
                                       'midlevel') 

        # subgoal_files = sorted([f for f in os.listdir(subgoal_json_path) if f.endswith('subgoal.json')])
        subgoal_steps = sorted([int(f.split('_')[0]) for f in os.listdir(subgoal_json_path) if f.endswith('subgoal.json')])
        subgoal_steps = [step for step in subgoal_steps if step >= self.init_step_count]

        for subgoal_step in subgoal_steps:
            subgoal_file = f"{subgoal_step:05d}_subgoal.json"
            with open(os.path.join(subgoal_json_path, subgoal_file), 'r') as f:
                subgoal_dict = json.load(f)[1]
                # get symbolic subgoal
                subgoal = self.dict_to_subgoal(subgoal_dict)
                self.subgoals.append(subgoal)
        
        self.env.cur_subgoal_num = 0
        self.env.cur_subgoal = self.subgoals[self.env.cur_subgoal_num]

    def dict_to_subgoal(self, subgoal_dict):
        obj = 0 if subgoal_dict['obj'] is None else vocab.OBJECT_TO_IDX[subgoal_dict['obj']]
        fur = 0 if subgoal_dict['fur'] is None else vocab.OBJECT_TO_IDX[subgoal_dict['fur']]
        room = 0 if subgoal_dict['room'] is None else vocab.ROOM_TO_IDX[subgoal_dict['room']]
        action = 0 if subgoal_dict['action'] is None else vocab.ACTION_TO_IDX[subgoal_dict['action']]

        return obj, fur, room, action

    def load_subgoal(self, subgoal_json_path=None):
        '''
        Get the subgoal from json file as an extra information of rollout.
        '''
        subgoal_json_path = os.path.join(self.args.data.data_path,
                                         self.args.experiment.agent,
                                         self.args.experiment.mission_name,
                                         self.args.rollout.traj_name,
                                         f"midlevel/{self.args.rollout.from_step_count:05d}_subgoal.json")
        with open(subgoal_json_path, 'r') as f:
            subgoal_dict = json.load(f)
            # get symbolic subgoal
            print(subgoal_dict)
            self.subgoal = subgoal_dict[2]

    def set_rollout_traj_name(self, rollout_traj_name=None):
        if not rollout_traj_name:
            self.rollout_traj_name = self.args.rollout.traj_name
        else:
            self.rollout_traj_name = rollout_traj_name

    def set_init_step_count(self, initial_step_count=None):
        if not initial_step_count:
            self.init_step_count = self.args.rollout.from_step_count
        else:
            self.init_step_count = initial_step_count
    
    def set_inference_type(self, inference_type=None):
        if not inference_type:
            self.inference_type = self.args.rollout.inference_type
        else:
            self.inference_type = inference_type
    
    def set_mission_dict(self):
        self.mission_dict = self.args.data.mission_dict

    def set_rollout_depth(self, rollout_depth=None):
        if not rollout_depth:
            self.rollout_depth = self.args.rollout.rollout_depth
        else:
            self.rollout_depth = rollout_depth
    
    def set_rollout_traj_name(self, rollout_traj_name=None):
        if not rollout_traj_name:
            self.rollout_traj_name = self.args.rollout.traj_name
        else:
            self.rollout_traj_name = rollout_traj_name

    def subgoal_to_dict(self, obj, fur, room, action):
        return [
            "subgoal_predicted",
            {
                "obj": None if vocab.IDX_TO_OBJECT[obj] == 'null' else vocab.IDX_TO_OBJECT[obj],
                "fur": None if vocab.IDX_TO_OBJECT[fur] == 'null' else vocab.IDX_TO_OBJECT[fur],
                "room": None if vocab.IDX_TO_ROOM[room] == 'null' else vocab.IDX_TO_ROOM[room],
                "pos": None,
                "action": None if (action not in vocab.IDX_TO_SUBGOAL_ACTION or vocab.IDX_TO_SUBGOAL_ACTION[action] == 'null') else vocab.IDX_TO_SUBGOAL_ACTION[action],
                "state": None,
                "can_skip": False,
                "end_state": False
            },
            {
                "obj": obj,
                "fur": fur,
                "room": room,
                "pos": None,
                "action": action,
                "state": None
            }
        ]

    def get_subgoal(self, env, cur_state_array=None, inference_type='deterministic', temp=1):
        '''
        Get subgoal from current state.
        Carry out the predicted subgoal and its parent state prob and step count
        '''     
        if inference_type == 'deterministic':
            cur_state_array = get_cur_arrays(env)
            cur_state_dict = get_state_dict(env)
            print("---predict subgoals for state---", cur_state_dict)

            pred_obj, pred_fur, pred_room, pred_action = self.policy_model.inference(cur_state_array)

            pred_obj = pred_obj.item()
            pred_fur = pred_fur.item()
            pred_room = pred_room.item()
            pred_action = pred_action.item()

            subgoal = self.subgoal_to_dict(pred_obj, pred_fur, pred_room, pred_action)
            return subgoal
        
        elif inference_type == 'sample':
            if cur_state_array is None:
                cur_state_array = get_cur_arrays(env)
            pred_objs, pred_furs, pred_rooms, pred_actions = self.mid_policy_model.inference_sample(cur_state_array)
            return pred_objs, pred_furs, pred_rooms, pred_actions   
            
            pred_objs = pred_objs.tolist()
            pred_furs = pred_furs.tolist()
            pred_rooms = pred_rooms.tolist()
            pred_actions = pred_actions.tolist()
            pred_probs = pred_probs.tolist()

            subgoals = [self.subgoal_to_dict(pred_obj, pred_fur, pred_room, pred_action) \
                            for pred_obj, pred_fur, pred_room, pred_action in \
                                zip(pred_objs, pred_furs, pred_rooms, pred_actions)]
    
            return subgoals, pred_probs

    def get_action(self, env, inference_type = 'sample', cur_state_array=None, sub_action=None, sub_obj=None, sub_fur=None, sub_room=None, state_mask=None, temp=1, use_audio=None):
        '''
        Get action from current state.
        Returns zipped list of [pred_objs, pred_actions, and pred_heatmaps], along with probs for beam search.
        '''
        if inference_type == 'deterministic':
            cur_state_array = get_cur_arrays(env)
            cur_state_dict = get_state_dict(env)
            print("---predict subgoals for state---", cur_state_dict)
            pred = self.policy_model.inference(cur_state_array) 
            return self.convert_action(env, pred)
        elif inference_type == 'sample':
            if self.args.model.model_name == 'low_policy':
                pred_obj, pred_action, pred_heatmap = self.policy_model.inference_sample(cur_state_array, vocab, temp=temp, use_audio=use_audio) 
                pred_objs = pred_obj.squeeze().tolist()
                pred_actions = pred_action.squeeze().tolist()
                pred_heatmaps = [None for _ in pred_objs]
                return pred_objs, pred_actions, pred_heatmaps
            elif self.args.model.model_name == 'subgoal_low_policy':
                pred_obj, pred_action, pred_heatmap = self.policy_model.inference_sample(cur_state_array, vocab, state_mask, sub_action=sub_action, sub_obj=sub_obj, sub_fur=sub_fur, sub_room=sub_room, temp=temp, use_audio=use_audio)
                pred_objs = pred_obj.squeeze().tolist()
                pred_actions = pred_action.squeeze().tolist()
                pred_heatmaps = [None for _ in pred_objs]  
                return pred_objs, pred_actions, pred_heatmaps                 
            else:
                raise NotImplementedError('no low level policy model') 
 
    def rollout_subgoal(self, args, env, cur_subgoals, last_env_step_count):
        '''
        rollout a list of subgoal sequentially
        '''
        frames = []
        audios = []
        texts = []
        audios_symbolic = []

        # reset(args, env)
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        planner = Planner(env, env.cur_agent_id) 
        planner.traj_folder = f'{self.save_rollout_folder}/{time_stamp}'
        planner.traj_folder_midlevel = f'{planner.traj_folder}/midlevel'
        mission_dict = {}
        mission_dict['Goal'] = []
        for cur_subgoal in cur_subgoals:
            mission_dict['Goal'].append(cur_subgoal)

        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = \
            planner.low_level_planner(mission_dict, env, env.cur_agent_id, args, frames, audios, audios_symbolic, texts, step_count=last_env_step_count, time_stamp=time_stamp)
        
        done = 1
        if done or env.window.closed:
            if args.save:
                visual_path = os.path.join(planner.traj_folder, "visual.gif")
                print(f"Saving gif... to {visual_path}", end="")
                write_gif(np.array(frames),
                            os.path.join(planner.traj_folder, "visual.gif"), fps=1/args.pause)
                print("Done.")
                print("Saving text... ", end="")
                with open(os.path.join(planner.traj_folder, "language.txt"), 'w') as fp:
                    for i, item in enumerate(texts):
                        fp.write(f"{item}\n")
                print("Done.")
                print("Saving sound... ", end="")
                combined_sounds = sum(audios)
                velocidad_X = 2
                combined_sounds = combined_sounds.speedup(velocidad_X, 150, 25)
                combined_sounds.export(
                    os.path.join(planner.traj_folder, "audio.wav"), format="wav")
            
                with open(os.path.join(planner.traj_folder, "audio.txt"), 'w') as fp:
                    for i, item in enumerate(audios_symbolic):
                        fp.write(f"{item}\n")

                print("Done.")
        
        return env, step_count, time_stamp
        
    def rollout_action(self, args, env, cur_actions, last_env_step_count):
        '''
        rollout a list of subgoal sequentially
        input env and cur_actions, where actions is a list of actions, and last_env_step_count
        ''' 
        frames = []
        audios = []
        texts = []
        audios_symbolic = []

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        planner = Planner(env, env.cur_agent_id)
        planner.traj_folder = f'{self.save_rollout_folder}'
        todos = []
        todos = cur_actions
        
        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = planner.conduct_actions(
                args, todos, frames, audios, audios_symbolic, texts, last_env_step_count, time_stamp)
        
        done = 1
        if done or env.window.closed:
            if args.save:
                visual_path = os.path.join(planner.traj_folder, "visual.gif")
                print(f"Saving gif... to {visual_path}", end="")
                write_gif(np.array(frames),
                            os.path.join(planner.traj_folder, "visual.gif"), fps=1/args.pause)
                print("Done.")
                print("Saving text... ", end="")
                with open(os.path.join(planner.traj_folder, "language.txt"), 'w') as fp:
                    for i, item in enumerate(texts):
                        # write each item on a new line
                        fp.write(f"{item}\n")
                print("Done.")
                print("Saving sound... ", end="")
                combined_sounds = sum(audios)
                velocidad_X = 2
                combined_sounds = combined_sounds.speedup(velocidad_X, 150, 25)
                combined_sounds.export(
                    os.path.join(planner.traj_folder, "audio.wav"), format="wav")
            
                with open(os.path.join(planner.traj_folder, "audio.txt"), 'w') as fp:
                    for i, item in enumerate(audios_symbolic):
                        fp.write(f"{item}\n")

                print("Done.")
        
        return env, step_count, time_stamp
    
    def check_action(self, env, pred_action):
        """
        Checks if the action is valid in the current environment

        Args:
            env (Environment): The current environment.
            pred_action (tuple): The predicted action.

        Returns:
            bool: True if the action is valid, False otherwise.
            object: The object involved in the action, if any.
        """
        pred_obj, pred_action, pred_heatmap = pred_action

        pred_obj = vocab.IDX_TO_OBJECT[pred_obj]
        pred_action = vocab.IDX_TO_ACTION[pred_action]
        agent = env.agents[env.cur_agent_id] 
        fwd_pos = (env.front_pos[0], env.front_pos[1])

        fur_ahead, obj_ahead = env.grid.get_all_items(*fwd_pos)
        if pred_action in ["forward", "left", "right", "null", "end"]:
            return self.check_movement(pred_action, fur_ahead, obj_ahead, fwd_pos, env), None
        elif pred_action in ["drop", "pickup", "toggle", "open", "close", "clean", "idle"]:
            if pred_action == 'pickup':
                if fur_ahead:
                    obj_ahead = [obj_ahead] + fur_ahead.objects if obj_ahead is not None else fur_ahead.objects 
                else:
                    obj_ahead = [obj_ahead] if obj_ahead is not None else []  
            return self.check_object_interaction(pred_action, pred_obj, agent, fur_ahead, obj_ahead, pred_heatmap, fwd_pos, env)
        else:
            raise NotImplementedError()

    def check_movement(self, pred_action, fur_ahead, obj_ahead, fwd_pos, env):
        if pred_action == "forward":                
            if (fur_ahead and fur_ahead.type != 'door') or obj_ahead or not (0 <= fwd_pos[0] < env.grid.width and 0 <= fwd_pos[1] < env.grid.height):
                if self.args.rollout.verbose:
                    print("forward not feasible")
                return False
        return True

    def check_object_interaction(self, pred_action, pred_obj, agent, fur_ahead, obj_ahead, pred_heatmap, fwd_pos, env):
        # heatmap_array = np.array(pred_heatmap)
        if pred_action == "drop":
            return self.check_drop(pred_obj, agent, fur_ahead, obj_ahead, env)
        elif pred_action == "pickup":
            return self.check_pickup(pred_obj, fur_ahead, obj_ahead, env)
        elif pred_action in ["toggle", "open", "close", "clean", "idle"]:
            return self.check_manipulate(pred_action, pred_obj, fur_ahead, obj_ahead, env)

    def check_drop(self, pred_obj, agent, fur_ahead, obj_ahead, env=None):
        env = self.env if env is None else env
        for carry_obj in agent.carrying:
            if carry_obj.type == pred_obj:
                if ACTION_FUNC_MAPPING['drop'](env).can(carry_obj): # obj_ahead is None or 
                    return True, carry_obj
        if self.args.rollout.verbose:
            print(
                f"drop not feasible, no object in hand named {pred_obj}, in hand is {[obj.type for obj in agent.carrying]}")
        return False, None

    def check_pickup(self, pred_obj, fur_ahead, obj_ahead, env=None):
        env = self.env if env is None else env
        for obj in obj_ahead: 
            if obj and obj.type == pred_obj and ACTION_FUNC_MAPPING['pickup'](env).can(obj):
                return True, obj
        if self.args.rollout.verbose:
            print(f"pickup not feasible, no object ahead named {pred_obj}, obj_ahead is {obj_ahead}")
        return False, None

    def check_manipulate(self, pred_action, pred_obj, fur_ahead, obj_ahead, env=None): 
        env = self.env if env is None else env
        tgt = None
        if fur_ahead:
            if type(fur_ahead) == list:
                for fur in fur_ahead:
                    if fur is not None and (fur.type == pred_obj or pred_obj == 'null'):
                        tgt = fur
            else:
                if fur_ahead.type == pred_obj or pred_obj == 'null':
                    tgt = fur_ahead
        elif obj_ahead:
            for obj in obj_ahead:
                if obj is not None and obj.type == pred_obj:
                    tgt = obj_ahead
        else:
            if self.args.rollout.verbose:
                print(f"{pred_action} not feasible, no object/fur ahead named {pred_obj}, obj ahead: {obj_ahead}, fur ahead: {fur_ahead}")
            return False, None  

        if tgt and ACTION_FUNC_MAPPING[pred_action](env).can(tgt):    
            return True, tgt 
        
        if self.args.rollout.verbose:
            print(f"{pred_action} not feasible pre can, obj name: {pred_obj}, tgt ahead: {tgt}")

        return False, None        

    def convert_action(self, env, pred_action, tgt=None):
        '''
        args: pred_action, return: simulator action (convert predicted action to action formats same as  todos  in conduct_actions)
        '''
        flag, tgt = self.check_action(env, pred_action)
        obj, action, heatmap = pred_action
        obj = vocab.IDX_TO_OBJECT[obj]
        action = vocab.IDX_TO_ACTION[action]
        if flag:
            if action in ["forward", "left", "right", "null", "end"]:
                if action == "forward":
                    return [env.actions.forward]
                elif action == "left":
                    return [env.actions.left]
                elif action == "right":
                    return [env.actions.right]
                elif action == "null":
                    return ['null']
                elif action == "end":
                    return ['end']
            elif action in ["pickup", "drop", "toggle", "open", "close", "clean", "idle"]:
                if tgt:
                    return [action, tgt]
                else:
                    return None
            else:
                raise NotImplementedError("not a valid action name")
        else:
            return None
        
    def rollout_action_wrapper(self, env, action, step_count):
        try:
            cur_action = self.convert_action(env, action)

            if cur_action is not None and cur_action != ['null'] and cur_action != ['end']:
                env, step_count, time_stamp = self.rollout_action(self.args.simulator, env, cur_actions=[cur_action], last_env_step_count=step_count)
                if len(cur_action) == 1:
                    action = np.array([0, vocab.ACTION_TO_IDX[cur_action[0].name]])
                else:
                    action = np.array([vocab.OBJECT_TO_IDX[cur_action[1].type], vocab.ACTION_TO_IDX[cur_action[0]]])
            else:
                action = np.array([0, 0])

            if self.args.model.model_name == 'subgoal_low_policy':
                done = False
                if action[1] == vocab.ACTION_TO_IDX['end']: 
                    done = True
                elif action[1] == env.cur_subgoal[3]:
                    if env.cur_subgoal[0] != 0 and env.cur_subgoal[1] != 0:
                        # agent interacts w/ obj, obj interacts w/ fur (either drop or pickup)
                        if action[0] == env.cur_subgoal[0]:
                            done = True
                    else:
                        # agent interacts w/ fur (either toggle, open, close, clean, idle)
                        if action[0] == env.cur_subgoal[1]: 
                            done = True

                if done:
                    if self.args.rollout.simulator:
                        env.cur_subgoal_num += 1
                        env.cur_subgoal = self.subgoals[env.cur_subgoal_num]
                    else:
                        env.cur_subgoal = None
            
        except Exception as error:
            print("couldn't rollout", action[:2])
            print(error)
            action = [0, 0]
        
        cur_state_array = get_cur_arrays(env)

        return env, cur_state_array, step_count, action

    def sample(self, envs_list, cur_state_arrays, step_counts, num_workers=10, num_samples=100, temp=1, use_audio=None): 
        if self.args.model.model_name == 'low_policy':
            pred_objs, pred_actions, pred_heatmaps = self.get_action(None, cur_state_array=cur_state_arrays, inference_type=self.inference_type, temp=temp, use_audio=use_audio)
        elif self.args.model.model_name == 'subgoal_low_policy':
            if not self.args.rollout.simulator:
                pred_objs, pred_furs, pred_rooms, pred_actions = self.get_subgoal(None, cur_state_array=cur_state_arrays, inference_type=self.inference_type)

                for i in range(len(envs_list)):
                    if envs_list[i].cur_subgoal is None:
                        envs_list[i].cur_subgoal = [pred_objs[i].item(), pred_furs[i].item(), pred_rooms[i].item(), pred_actions[i].item()] 
            pred_subobjs, pred_furs, pred_rooms, pred_subactions = zip(*[env.cur_subgoal for env in envs_list])
            pred_objs, pred_actions, pred_heatmaps = self.get_action(None, inference_type=self.inference_type, cur_state_array=cur_state_arrays, sub_action=pred_subactions, sub_obj=pred_subobjs, sub_fur=pred_furs, sub_room=pred_rooms, temp=temp, use_audio=use_audio)
        else:
            raise NotImplementedError()
        
        # rollout action 
        all_params = [(env, [pred_obj, pred_action, pred_heatmap], step_count) for \
            env, pred_obj, pred_action, pred_heatmap, step_count in zip(envs_list, pred_objs, pred_actions, pred_heatmaps, step_counts)]

        torch.cuda.empty_cache()

        results = []

        with multiprocessing.Pool(processes=num_workers) as pool: 
            results = pool.starmap(self.rollout_action_wrapper, all_params)
            pool.close()
            pool.join()
        
        return zip(*results)

    def save_rollouts(self, subgoals_record):
        prob_file = f"{self.save_rollout_folder}/prob.txt"
        if os.path.exists(prob_file):
            os.remove(prob_file)
        rollout_agent = self.args.experiment.agent
        rollout_mission = self.args.experiment.mission_name
        rollout_traj = self.rollout_traj_name
        rollout_from_step_count = self.init_step_count
        rollout_inference_type = self.inference_type
        rollout_depth = self.rollout_depth
        rollout_beam_search_width = self.args.rollout.beam_search_width
    
        with open (prob_file, 'w') as f:
            f.write(f"rollout_agent:{rollout_agent}\n")
            f.write(f"rollout_mission:{rollout_mission}\n")
            f.write(f"rollout_traj:{rollout_traj}\n")
            f.write(f"rollout_from_step_count:{rollout_from_step_count}\n")
            f.write(f"rollout_inference_type:{rollout_inference_type}\n")
            f.write(f"rollout_depth:{rollout_depth}\n")
            f.write(f"rollout_beam_search_width:{rollout_beam_search_width}\n")
        
        cache = self.args.simulator.save
        self.args.simulator.save = True
        if self.args.rollout.rollout_level == "low":
            for (prob, prob_list, action_seq) in subgoals_record:
                env, step_count, time_stamp = self.rollout_action(self.args.simulator, self.env, cur_actions=action_seq, \
                                        last_env_step_count=self.init_step_count)
                with open(prob_file, 'a') as f:
                    f.write(f"{time_stamp}:{prob},{prob_list}\n")
                reset(self.args.simulator, self.env)
        elif self.args.rollout.rollout_level == "mid":
            for (prob, prob_list, subgoal_seq) in subgoals_record:
                env, step_count, time_stamp = self.rollout_subgoal(self.args.simulator, self.env, cur_subgoals=subgoal_seq, \
                                        last_env_step_count=self.init_step_count)
                with open(prob_file, 'a') as f:
                    f.write(f"{time_stamp}:{prob},{prob_list}\n")
                reset(self.args.simulator, self.env)
            
        self.args.simulator.save = cache

    def save_rollout(self, action_seq):
        print('resetting rollout for save rollout')
        reset(self.args.simulator, self.env)

        rollout_agent = self.args.experiment.agent
        rollout_mission = self.args.experiment.mission_name
        rollout_traj = self.rollout_traj_name
        rollout_from_step_count = self.init_step_count
        rollout_inference_type = self.inference_type
        rollout_depth = self.rollout_depth
        rollout_beam_search_width = self.args.rollout.beam_search_width

        cache = self.args.simulator.save
        self.args.simulator.save = True
        
        action_seq = [action for action in action_seq if action is not None]
        print('conduct actions:', action_seq)
        if self.args.rollout.rollout_level == "low":
            env, step_count, time_stamp = self.rollout_action(self.args.simulator, self.env, cur_actions=action_seq, \
                last_env_step_count=self.init_step_count)
        elif self.args.rollout.rollout_level == 'mid':
            raise NotImplementedError("not implemented")

        self.args.simulator.save = cache
