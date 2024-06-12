import os
import sys
import json
import hydra
import calendar
import time
import numpy as np
from os import getpid
import torch
from joblib import Parallel, delayed

from array2gif import write_gif
sys.path.append('/Users/emilyjin/Code/marple_long/src')
import MarpleLongModels.models.vocab as vocab
from gym_minigrid.wrappers import *
from MarpleLongModels.models.pl_models import BasicModel, LowBasicModel
from MarpleLongModels.models.models import PolicyModel, TransformerHeadPolicyModel, HistAwareTransformerHeadPolicyModel, AudioConditionedTransformerHeadPolicyModel, LowPolicyModel, SubgoalConditionedLowPolicyModel

from arguments import Arguments # arguments, defaults overwritten by config/config.yaml
import multiprocessing

from marple_mini_behavior.mini_behavior.minibehavior import *
from marple_mini_behavior.mini_behavior.planner import *
from marple_mini_behavior.mini_behavior.envs import *
from marple_mini_behavior.mini_behavior.utils import *
from marple_mini_behavior.mini_behavior.actions import *
import pdb
import traceback

from copy import deepcopy

def reset(args, env):
    obs = env.reset()
    if args.render: 
        redraw(args, env, obs)

def redraw(args, env, img):
    img = env.render('rgb_array', tile_size=args.tile_size)
    return img


class Rollout:
    # @timing_decorator
    def __init__(self, args, low_policy_model=None, mid_policy_model=None):
        self.args = args

        self.create_save_rollout_folder()

        self.set_rollout_traj_name() 
        self.set_init_step_count()
        # the initial state of the rollout trajectory we ask inference question about (e.g. 'traj_0/00000_states')
        self.set_rollout_depth()
        self.set_inference_type()
        self.set_mission_dict()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print('load initial_state')
        self.load_initial_state()
        
        # load policy models
        self.load_policy_model(rollout_level='low', policy_model_checkpoint=os.path.join(self.args.model.dirpath, f'{self.args.model.model_name}_{self.args.model.embedding_dim}.ckpt'))

        # if goal conditioend rollout
        if self.args.model.model_name == 'subgoal_low_policy':
            if self.args.rollout.simulator:
                # simulator provided subgoal
                self.load_subgoals()
            else:
                # predicted subgoal
                self.env.cur_subgoal = None
                self.load_policy_model(rollout_level='mid', policy_model_checkpoint=os.path.join(self.args.model.dirpath, f'transformer_{self.args.model.embedding_dim}.ckpt'))

        self.env.hist_states = []
        # self.set_agent()

    # @timing_decorator
    def reset(self):
        self.set_init_step_count()
        # reset(self.args.simulator, self.env)
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

        # if self.args.model.model_name == 'low_policy':
        #     model_class = LowBasicModel(args=self.args, model=LowPolicyModel, model_name="low_policy")
        # elif self.args.model.model_name == 'subgoal_low_policy':
        #     model_class = LowBasicModel(args=self.args, model=SubgoalConditionedLowPolicyModel, model_name="subgoal_low_policy")
        if rollout_level == 'low':
            model_class = LowBasicModel
        elif rollout_level == 'mid':
            model_class = BasicModel#(args=self.args, model=TransformerHeadPolicyModel, model_name="transformer")

        map_location = None
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
   
        print(f'loading {rollout_level} policy model from {policy_model_checkpoint}')

        if rollout_level == "low":
            self.policy_model = model_class.load_from_checkpoint(policy_model_checkpoint, map_location=map_location)
        else:
            self.mid_policy_model = model_class.load_from_checkpoint(policy_model_checkpoint, map_location=map_location)
            breakpoint()
        if torch.cuda.is_available():
            self.policy_model.to('cuda')
        self.policy_model.eval()

    def load_initial_state(self, state_json_path=None, mission_preference=None, verbose=True):
        '''
        Generate initial env from json file as the start of rollout.
        '''
        init_filename = "" if self.args.data.data_level == 'low' else "midlevel/"
        init_filename += f"{self.init_step_count :05d}"#_states.json" 
        
        state_json_path = os.path.join(self.args.data.data_path,  
                                       self.args.rollout.traj_name, 
                                       self.args.experiment.mission_name)#, 
                                       #init_filename) 

        with open(os.path.join(state_json_path, f"{init_filename}_states.json"), 'r') as f:
            initial_dict = json.load(f)
            if verbose:
                print('load_initial_state from:', state_json_path)

        if self.args.rollout.inference:
            for agent in initial_dict['Grid']['agents']['initial']:
                agent['mission_preference_initial'] = {self.args.experiment.mission_name: 1} #itial_carrying = []

        if self.args.rollout.audio:
            with open(os.path.join(state_json_path, f"{init_filename}_action.json"), 'r') as f:
                action = json.load(f)
                self.audio = np.expand_dims(np.array(vocab.ACTION_TO_AUDIO_MASK[vocab.IDX_TO_ACTION[action['action_type']]]), axis=0)
        else:
            self.audio = None

        self.env = gym.make(self.args.simulator.env, initial_dict=initial_dict)
        self.env.grid.load_init = True
        reset(self.args.simulator, self.env)
        
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
        # state = subgoal_dict['state']

        return obj, fur, room, action#, state

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

    # def set_agent(self):
    #     agent = [agent for agent in self.env.agents if agent.name == self.args.experiment.agent][0]
    #     self.cur_agent_id = agent.id - 1
    #     self.agent_pos = agent.agent_pos
    #     self.agent_dir = agent.agent_dir
    #     self.agent_color = agent.agent_color

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

            # pred_objs = pred_obj.tolist()
            # pred_furs = pred_fur.tolist()
            # pred_rooms = pred_room.tolist()
            # pred_actions = pred_action.tolist() 
        
            return pred_objs, pred_furs, pred_rooms, pred_actions   
            
        elif inference_type == 'beam_search':
            # env, last env prob, last env step count are lists
            # TODO: make sample actions parallel with env batch
            if self.args.model.model_name == 'transformer_model':
                cur_state_array = get_cur_arrays(env)
                pred_objs, pred_furs, pred_rooms, pred_actions, pred_probs = self.policy_model.sample_actions(cur_state_array, vocab, decode_strategy="beam", debug=False)
            elif self.args.model.model_name == 'hist_aware_transformer':
                #TODO: the maximum history length should be the checkpoint's maximum mission length
                maximum_history_length = max([vocab.MISSION_TO_SUBGOAL_NUM[mission] for mission in self.mission_dict])
                cur_state_arrays = [get_cur_arrays(env_i) for env_i in env]
                H, W, C = cur_state_arrays[0].shape
                concatenated_shape = (maximum_history_length, H, W, C)

                # Create an array filled with zeros with the concatenated shape
                states = np.zeros(concatenated_shape, dtype=cur_state_arrays[0].dtype)
                state_mask = np.ones(maximum_history_length, dtype=np.int32)
                
                # Copy the individual arrays into the concatenated array
                for i, array in enumerate(cur_state_arrays):
                    try:
                        states[i, :, :, :] = array.copy()
                        state_mask[i] = 0.
                    except:
                        print("except check array shape", array.shape)
                        print("check states shape", states.shape)
                        print("check index", i)

                pred_objs, pred_furs, pred_rooms, pred_actions, pred_probs = self.policy_model.sample_actions(states, vocab, decode_strategy="beam", debug=False, state_mask=state_mask)

            pred_objs = pred_objs.tolist()
            pred_furs = pred_furs.tolist()
            pred_rooms = pred_rooms.tolist()
            pred_actions = pred_actions.tolist()
            pred_probs = pred_probs.tolist()

            subgoals = [self.subgoal_to_dict(pred_obj, pred_fur, pred_room, pred_action) \
                            for pred_obj, pred_fur, pred_room, pred_action in \
                                zip(pred_objs, pred_furs, pred_rooms, pred_actions)]
    
            return subgoals, pred_probs

    # @timing_decorator
    def get_action(self, env, inference_type = 'beam_search', cur_state_array=None, sub_action=None, sub_obj=None, sub_fur=None, sub_room=None, state_mask=None, temp=1, use_audio=None):
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
            # use model to predict --- using gpu. inference_sample function in pl_models.py
            if self.args.model.model_name == 'low_policy':
                pred_obj, pred_action, pred_heatmap = self.policy_model.inference_sample(cur_state_array, vocab, temp=temp, use_audio=use_audio) # uses gpu

                pred_objs = pred_obj.squeeze().tolist()
                pred_actions = pred_action.squeeze().tolist()
                # pred_heatmaps = pred_heatmap.squeeze().tolist()     
                pred_heatmaps = [None for _ in pred_objs]
                return pred_objs, pred_actions, pred_heatmaps
                        # use model to predict --- using gpu. inference_sample function in pl_models.py
            elif self.args.model.model_name == 'subgoal_low_policy':
                pred_obj, pred_action, pred_heatmap = self.policy_model.inference_sample(cur_state_array, vocab, state_mask, sub_action=sub_action, sub_obj=sub_obj, sub_fur=sub_fur, sub_room=sub_room, temp=temp, use_audio=use_audio)
                pred_objs = pred_obj.squeeze().tolist()
                pred_actions = pred_action.squeeze().tolist()
                # pred_heatmaps = pred_heatmap.squeeze().tolist()   
                pred_heatmaps = [None for _ in pred_objs]  
            
                return pred_objs, pred_actions, pred_heatmaps                 
            else:
                raise NotImplementedError('no low level policy model') 

        elif inference_type == 'beam_search':
            cur_state_arrays = get_cur_arrays(env)
            H, W, C = cur_state_arrays.shape
            if self.args.model.model_name == 'low_policy':
                # pdb.set_trace()
                pred_objs, pred_actions, pred_heatmaps, probs = self.policy_model.inference_list(cur_state_arrays, vocab, num_samples=self.args.rollout.beam_search_width)
            elif self.args.model.model_name == 'subgoal_low_policy':
                pred_objs, pred_actions, pred_heatmaps, probs = self.policy_model.inference_list(
                    cur_state_arrays, vocab, sub_action=sub_action, sub_obj=sub_obj, sub_fur=sub_fur, sub_room=sub_room)
            else:
                raise NotImplementedError("No such policy model")
            pred_objs = pred_objs.tolist()
            pred_actions = pred_actions.tolist()
            pred_heatmaps = pred_heatmaps.tolist()
            probs = probs.tolist()

            # print("check get action", pred_objs, pred_actions, pred_heatmaps)
            return [self.convert_action(env,list(a)) for a in zip(pred_objs, pred_actions, pred_heatmaps)], probs

    def rollout_action_subgoal(self, args, env, cur_actions, cur_subgoals, last_env_step_count, simulator=True): 
        # print(f"PID: {getpid()}, argument: {cur_actions}, {last_env_step_count}")
        # breakpoint()
        frames = []
        audios = []
        texts = []
        audios_symbolic = []

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        planner = Planner(env, env.cur_agent_id)
        planner.traj_folder = f'{self.save_rollout_folder}'
        planner.traj_folder_midlevel = f'{planner.traj_folder}/midlevel'

        todos = []
        todos = cur_actions
        
        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = planner.conduct_actions(
                args, todos, frames, audios, audios_symbolic, texts, last_env_step_count, time_stamp)
        
        done = 1
        if done or env.window.closed:
            if args.save:
                # add time_stamp to planner.data_folder and create that directory
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
                        # write each item on a new line
                        fp.write(f"{item}\n")

                print(len(frames), len(audios), len(texts))
                # assert len(frames) == len(audios) + 1, f"{self.rollout_traj_name}"
                print("Done.")
        
        return env, step_count, time_stamp        

    def rollout_subgoal(self, args, env, cur_subgoals, last_env_step_count):
        '''
        rollout a list of subgoal sequentially
        '''
        print(f"PID: {getpid()}, argument: {cur_subgoals}, {last_env_step_count}")
        frames = []
        audios = []
        texts = []
        audios_symbolic = []

        # reset(args, env)
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        planner = Planner(env, env.cur_agent_id)
        # planner.data_folder = self.save_rollout_folder
        # planner.mission_folder = self.save_rollout_folder
        planner.traj_folder = f'{self.save_rollout_folder}/{time_stamp}'
        planner.traj_folder_midlevel = f'{planner.traj_folder}/midlevel'
        mission_dict = {}
        mission_dict['Goal'] = []
        for cur_subgoal in cur_subgoals:
            mission_dict['Goal'].append(cur_subgoal)

        # os.makedirs(os.path.join(planner.data_folder, f"{time_stamp}"), exist_ok=True)
        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = \
            planner.low_level_planner(mission_dict, env, env.cur_agent_id, args, frames, audios, audios_symbolic, texts, step_count=last_env_step_count, time_stamp=time_stamp)
        
        done = 1
        if done or env.window.closed:
            if args.save:
                # add time_stamp to planner.data_folder and create that directory
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
                        # write each item on a new line
                        fp.write(f"{item}\n")

                print(len(frames), len(audios), len(texts))
                # assert len(frames) == len(audios) + 1, f"{self.rollout_traj_name}"
                print("Done.")
        
        return env, step_count, time_stamp
        
    # @timing_decorator
    def rollout_action(self, args, env, cur_actions, last_env_step_count):
        '''
        rollout a list of subgoal sequentially
        input env and cur_actions, where actions is a list of actions, and last_env_step_count
        '''
        # print(f"PID: {getpid()}, argument: {cur_actions}, {last_env_step_count}")
        # breakpoint()
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
        
        # print('conduct actions')
        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = planner.conduct_actions(
                args, todos, frames, audios, audios_symbolic, texts, last_env_step_count, time_stamp)
        
        done = 1
        if done or env.window.closed:
            if args.save:
                # add time_stamp to planner.data_folder and create that directory
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
                        # write each item on a new line
                        fp.write(f"{item}\n")

                print(len(frames), len(audios), len(texts))
                # assert len(frames) == len(audios) + 1, f"{self.rollout_traj_name}"
                print("Done.")
        
        return env, step_count, time_stamp
    
    def beam_search_rollout(self):
        '''
        Beam Search Rollout.

        Perform beam search at subgoal level, while keeping track of final trajectories.

        Given:
            start_env (env instance): The starting initial env.
            beam_search_depth (int): Maximum depth of the env/subgoal search.
            beam_width (int): Number of candidates to keep at each step.
            prediction (function): Transition probabilities from policy model.
            
        Returns:
            list: List of beam_width final trajectories (s, a, s', a', ...).
        '''

        if self.inference_type == 'deterministic':
            print("******************Deterministic Rollout******************")
            last_env_step_counts = [self.init_step_count]
            last_envs = [self.env]

            subgoals_record = [[]]
            for _ in range(self.rollout_depth):
                print(f"******************rollout depth:{_}, step_count:{last_env_step_counts}******************")
                # get subgoal
                subgoal = self.get_subgoal(last_envs[0], self.inference_type) # TODO: parallel here
                env, step_count, time_stamp = self.rollout_subgoal(self.args.simulator, last_envs[0], cur_subgoals=[subgoal], \
                                                last_env_step_count=last_env_step_counts[0])         
                if step_count == last_env_step_counts[0]:
                    # subgoal not feasible
                    print("subgoal not feasbile or already completed.")
                else:
                    subgoals_record[0].append(subgoal.copy())
                    # get new state after subgoal
                    last_envs[0] = env
                    last_env_step_counts[0] = step_count

        elif self.inference_type == 'beam_search':
            print("******************Beam Search Rollout******************")
            # initialize beam search with accumulated prob, independent prob_list, last_step_count, last_env, subgoal list
            beam = [[0.0, [], self.init_step_count, [self.env], []]]
            subgoals_record = []

            for _ in range(self.rollout_depth):
                print(f"******************rollout depth:{_} ******************")
                new_beam = []

                # TODO: parallel get subgoal by inputing a batch of envs
                for prob, prob_list, last_step_count, env_list, subgoal_seq in beam:
                    # get new subgoal
                    if self.args.model.model_name == 'transformer_model':
                        subgoals, pred_probs = self.get_subgoal(env_list[-1], self.inference_type)
                    elif self.args.model.model_name == 'hist_aware_transformer':
                        subgoals, pred_probs = self.get_subgoal(env_list, self.inference_type)
                    # add to new_beam
                    for subgoal, subgoal_prob in zip(subgoals, pred_probs):
                        missionFSM = MissionFSM([], env_list[-1], env_list[-1].cur_agent_id)
                        flag, _ = missionFSM.check_subgoal(subgoal)
                        if flag == 'feasible':
                            new_prob = prob + np.log(subgoal_prob)
                            new_prob_list = prob_list + [np.log(subgoal_prob)]
                            new_subgoal_seq = subgoal_seq + [subgoal]
                            new_beam.append([new_prob, new_prob_list, last_step_count, env_list, new_subgoal_seq])
                
                # sort new_beam and only keep top 
                beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:self.args.rollout.beam_search_width]
                
                # parallel rollout
                assert len(beam) > 0, "no feasible subgoal"          
                envs, step_counts, time_stamps = zip(*Parallel(n_jobs=self.args.rollout.n_jobs)(
                                    delayed(self.rollout_subgoal)(self.args.simulator, env_list[-1], cur_subgoals=[subgoal_seq[-1]], \
                                                                  last_env_step_count=last_step_count)
                                        for _, _, last_step_count, env_list, subgoal_seq in beam)
                                    )
                # update beam
                for i, ((prob, prob_list, last_step_count, _, subgoal_seq), (step_count, env, _)) in enumerate(zip(beam, zip(step_counts, envs, time_stamps))):
                    if last_step_count == step_count:
                        # subgoal not feasible, don't update this new subgoal
                        print("subgoal not feasbile or already completed.")
                        beam[i][4].pop()
                    else:
                        beam[i][2] = step_count
                        beam[i][3] = beam[i][3] + [env]

            subgoals_record = [(prob, prob_list, subgoal_seq) for prob, prob_list, _, _, subgoal_seq in beam]

        return subgoals_record

    # @timing_decorator
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
            if carry_obj.type == pred_obj:# or pred_obj == 'null': TODO: debug
                if ACTION_FUNC_MAPPING['drop'](env).can(carry_obj): # obj_ahead is None or 
                    return True, carry_obj
                # else:
                #     if self.args.rollout.verbose:
                #         print(f'drop not feasible, object in hand name {pred_obj}, but obj ahead: {obj_ahead}')
                #     return False, None
        if self.args.rollout.verbose:
            print(
                f"drop not feasible, no object in hand named {pred_obj}, in hand is {[obj.type for obj in agent.carrying]}")
        return False, None

    def check_pickup(self, pred_obj, fur_ahead, obj_ahead, env=None):
        env = self.env if env is None else env
        for obj in obj_ahead: 
            # if obj and (obj.type == pred_obj or pred_obj == 'null') and ACTION_FUNC_MAPPING['pickup'](env).can(obj): # TODO: debug
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
        
    def beam_search_rollout_action(self, get_state=False):
        '''
        Beam Search Rollout.

        Perform beam search at low level, while keeping track of final trajectories.

        Given:
            start_env (env instance): The starting initial env.
            beam_search_depth (int): Maximum depth of the env/subgoal search.
            beam_width (int): Number of candidates to keep at each step.
            prediction (function): Transition probabilities from policy model.
            
        Returns:
            list: List of beam_width final trajectories (s, a, s', a', ...).
        ''' 

        if self.inference_type == 'deterministic':
            ## TODO: fix based on sample function
            print(f"******************{self.inference_type} Rollout******************")
            step_count = self.init_step_count

            actions = []
            states = [get_cur_arrays(self.env)]
            for _ in range(self.rollout_depth):
                if self.args.rollout.verbose:
                    print(f"****************** step_count: {step_count} ******************")
                # get most probable action from current state
                action = self.get_action(self.env, inference_type=self.inference_type)
                # action = tuple(pred_obj, pred_action, pred_heatmap)

                # roll out action
                cur_action = self.convert_action(self.env, action)
                env, step_count, time_stamp = self.rollout_action(self.args.simulator, self.env, cur_actions=[cur_action], \
                                                last_env_step_count=step_count)
                
                if self.args.rollout.verbose:
                    print("agent pos at ", env.agents[env.cur_agent_id].agent_pos, env.agents[env.cur_agent_id].agent_dir)

                state = get_cur_arrays(env)
                states.append(state)
                actions.append(action)
                step_count += 1
            
            return actions

        elif self.inference_type == 'beam_search':
            print("******************Beam Search Rollout******************")
            # initialize beam search with accumulated prob, independent prob_list, last_step_count, last_env, subgoal list
            beam = [[0.0, [], self.init_step_count, [self.env], []]]
            # beam = [prob, prob_list, last_step_count, env_list, action_seq]

            actions_record = []

            for _ in range(self.rollout_depth):
                print(f"******************rollout depth:{_} ******************")
                new_beam = []

                # TODO: parallel get action by inputing a batch of envs
                for i, (prob, prob_list, last_step_count, env_list, action_seq) in enumerate(beam):
                    # get new action
                    if self.args.model.model_name == 'low_policy':
                        actions, pred_probs = self.get_action(env_list[-1], self.inference_type)
                        print(
                            f"-----------get action for beam {i}----------", actions, pred_probs)
                    elif self.args.model.model_name == 'subgoal_low_policy':
                        sub_obj, sub_fur, sub_room, sub_action = self.subgoal['obj'] if self.subgoal['obj'] else 0, \
                            self.subgoal['fur'] if self.subgoal['fur'] else 0, \
                            self.subgoal['room'] if self.subgoal['room'] else 0, \
                            self.subgoal['action'] if self.subgoal['action'] else 0
                        actions, pred_probs = self.get_action(env_list[-1], self.inference_type, sub_action=sub_action,
                                                              sub_obj=sub_obj, sub_fur=sub_fur, sub_room=sub_room)
                    else: #TODO: implement for subgoal conditioned policy model
                        raise NotImplementedError()
                    # add to new_beam
                    for action, action_prob in zip(actions, pred_probs):
                        if action is not None:
                            new_prob = prob + np.log(action_prob)
                            new_prob_list = prob_list + [np.log(action_prob)]
                            new_action_seq = action_seq + [action]
                            new_beam.append([new_prob, new_prob_list, last_step_count, env_list, new_action_seq])
                            print("valid pred action", action)
                        else:
                            print("invalid pred action", action)
                
                for i, (prob, _, _, _, action_seq) in enumerate(new_beam):
                    if action_seq[-1][0] in ['open', 'close', 'toggle', 'clean', 'pickup', 'drop', 'idle']:
                        print(f"for beam {i+1} action seq", action_seq, prob)
                        # Give a boost to open door action
                        new_beam[i][0] += 10

                # sort new_beam and only keep top
                beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:self.args.rollout.beam_search_width]
                #TODO: if there are open door action in any of the beam when agent is next to the door
                
                # parallel rollout
                assert len(beam) > 0, "no feasible action"
                envs, step_counts, time_stamps = zip(*Parallel(n_jobs=self.args.rollout.n_jobs)(
                                    delayed(self.rollout_action)(self.args.simulator, env_list[-1], cur_actions=[action_seq[-1]], \
                                                                  last_env_step_count=last_step_count)
                                        for _, _, last_step_count, env_list, action_seq in beam)
                                    )
                # update beam
                for i, ((prob, prob_list, last_step_count, _, action_seq), (step_count, env, _)) in enumerate(zip(beam, zip(step_counts, envs, time_stamps))):
                    if last_step_count == step_count:
                        # action not feasible, don't update this new action
                        print("action not feasbile or already completed.")
                        beam[i][4].pop()
                    else:
                        beam[i][2] = step_count
                        beam[i][3] = beam[i][3] + [env]
                    print(f"for beam {i+1} agent pos at ", envs[0].agents[envs[0].cur_agent_id].agent_pos, envs[0].agents[envs[0].cur_agent_id].agent_dir)
                    print(f"for beam {i+1} beam search action", beam[0][4][-1])

            if get_state:
                for prob, prob_list, last_step_count, env_list, action_seq in beam:
                    state_seq = [get_cur_arrays(cur_env) for env in env_list]
                    actions_record.append((prob, prob_list, action_seq, state_seq))
            else:
                actions_record = [(prob, prob_list, action_seq) for prob, prob_list, _, _, action_seq in beam]

        return actions_record 

    # @timing_decorator
    def rollout_action_wrapper(self, env, action, step_count):
        try:
            cur_action = self.convert_action(env, action) #  [action, tgt]

            if cur_action is not None and cur_action != ['null'] and cur_action != ['end']:
                env, step_count, time_stamp = self.rollout_action(self.args.simulator, env, cur_actions=[cur_action], last_env_step_count=step_count)
                if len(cur_action) == 1:
                    action = np.array([0, vocab.ACTION_TO_IDX[cur_action[0].name]])
                else:
                    action = np.array([vocab.OBJECT_TO_IDX[cur_action[1].type], vocab.ACTION_TO_IDX[cur_action[0]]])
            else:
                action = np.array([0, 0])

            if self.args.model.model_name == 'subgoal_low_policy':
                # action = [obj, action]
                # cur_subgoal = [obj, fur, room, action]
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
                            # for fur in env.obj_instances.values():
                            #     # check that the fur is in the right room
                            #     if vocab.OBJECT_TO_IDX[fur.type] == env.cur_subgoal[1] and fur.in_room.type == env.cur_subgoal[2]:
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
            traceback.print_exc()
            action = [0, 0]
        
        cur_state_array = get_cur_arrays(env)

        return env, cur_state_array, step_count, action

    # @timing_decorator
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

#        for env, action, step_count in all_params:
#            result = self.rollout_action_wrapper(env, action, step_count)
#            results.append(result)
        with multiprocessing.Pool(processes=num_workers) as pool: 
            results = pool.starmap(self.rollout_action_wrapper, all_params)
            pool.close()
            pool.join()
        
        return zip(*results)


    def one_sample(self, env, cur_state_array, step_count):
        if self.args.model.model_name == 'subgoal_low_policy': 
            if not self.args.rollout.simulator:
                pred_subobj, pred_fur, pred_room, pred_subaction = self.get_subgoal(None, cur_state_array=cur_state_array, inference_type=self.inference_type)
                breakpoint()
                env.cur_subgoal = [pred_subobj.item(), pred_fur.item(), pred_room.item(), pred_subaction.item()]

            pred_subobj, pred_fur, pred_room, pred_subaction = env.cur_subgoal
            action = self.get_action(None, inference_type=self.inference_type, cur_state_array=cur_state_array, sub_action=pred_subaction, sub_obj=pred_subobj, sub_fur=pred_fur, sub_room=pred_room)
        else:
            action = self.get_action(None, cur_state_array=cur_state_array, inference_type=self.inference_type)   

        env, cur_state_array, step_count, action = self.rollout_action_wrapper(env, action, step_count) 
        return env, cur_state_array, step_count, action


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
        # self.reset()

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


@hydra.main(version_base='1.1', config_path="config", config_name="rollout_low_policy.yaml")
def main(args):
    '''
    This is an example of:
    policy model checkpoint + state json file -> subgoal prediction
    subgoal prediction + state json file -> one-subgoal step rollout
    # '''
    traj_folder = os.path.join(args.data.data_path, 
                               f'init_config_{args.experiment.experiment_name}',
                               args.experiment.agent, 
                               args.experiment.mission_name)

    traj_names = [traj for traj in sorted(os.listdir(traj_folder))]

    
    with open(os.path.join(traj_folder + "_test", "test_traj.json"), "r") as f:
         test_traj = json.load(f)
    
    if args.data.split == 'train':
        traj_names = [traj for traj in traj_names if traj not in test_traj]
    else:
        traj_names = test_traj
    
    traj_samples = np.random.choice(traj_names, size=args.rollout.n_traj_rollout, replace=False)
    print(f'getting {traj_samples.shape[0]} rollouts')
    print(traj_samples)
    for traj_name in traj_samples:
        args.rollout.traj_name = str(traj_name)
        rollout = Rollout(args)
        if args.rollout.rollout_level == "low":
            if args.rollout.inference_type == "sample":

                rollout_actions, rollout_states = rollout.sample() # actions = list of beam_search_width (prob, prob_list, action_seq) 
                #print(rollout_actions)
                rollout.save_rollout(rollout_actions)
            else:
                actions = rollout.beam_search_rollout_action() # actions = list of beam_search_width (prob, prob_list, action_seq) 
                rollout.save_rollouts(actions)
        else:
            subgoals = rollout.beam_search_rollout()
            rollout.save_rollouts(subgoals)

if __name__ == '__main__':
    # config_name = sys.argv[1]
    # hydra.run(main, config_name=config_name)
    # 
    main()
