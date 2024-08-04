import hydra
from omegaconf import DictConfig

import numpy as np
import os
import torch
from tqdm import tqdm
import json

from torch.utils.data import DataLoader

import src.simulator.vocab as vocab

'''
atomic action
{
    "action_type": 3,
    "coordinate": [
        1,
        6
    ],
    "object_type": -1,
    "object_id": -1
}
'''
class BasicDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.split = args.split
        self.data_path = args.data_path

        self.room = args.room
        # self.data_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path))
        self.idx_traj_map = {}
        self.data_level = args.data_level
        self.historical = args.historical
        self.with_audio = args.with_audio
        self.max_audio_length = args.max_audio_length
        self.max_atomic_length = args.max_atomic_length
        self.end_state_flag = args.end_state_flag
        self.data = []
        self.mission_dict = args.mission_dict
        self.goal_conditioned = args.goal_conditioned
        self.path_to_data = {}
        self.construct_data_idx()
        self.construct_historical_data()
        self.set_max_dims()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = {}
        cur_data = self.data[idx]

        # get state field
        if self.historical:
            if self.data_level == 'low':
                maximum_history_length = self.max_atomic_length
            else:
                maximum_history_length = max([vocab.MISSION_TO_SUBGOAL_NUM[mission] for mission in self.mission_dict])
                # TODO: hardcoded history length
                # maximum_history_length = 2

            history_length = len(cur_data['state'])
            states_list = [np.load(state_file, allow_pickle=True).astype(np.float32) for state_file in cur_data['state']]
            # H, W, C = states_list[0].shape
            H, W, C = self.target_size
            concatenated_shape = (maximum_history_length, H, W, C)

            # Create an array filled with zeros with the concatenated shape
            states = np.zeros(concatenated_shape, dtype=states_list[0].dtype)
            state_mask = np.ones(maximum_history_length, dtype=np.int32)

            # Copy the individual arrays into the concatenated array
            for i, array in enumerate(states_list):
                try:
                    pad_array = self.pad(array)
                    states[i, :, :, :] = pad_array.copy()
                    state_mask[i] = 0.
                except:
                    print("except check array shape", array.shape)
                    print("check states shape", states.shape)
                    print("check index", i)
            data_entry['state'] = states
            data_entry['state_mask'] = state_mask
        else:
            state = np.load(cur_data['state'], allow_pickle=True).astype(np.float32)
            # H, W, C = state.shape
            H, W, C = self.target_size
            data_entry['state'] = self.pad(state)
        
        # get action field for low level
        if self.data_level == 'low':
            # for low level

            if cur_data['action'] is None:
                action = {'action_type': 11, 'coordinate': np.array([-1, -1]), 'object_type': 0, 'object_id': 0}
            else:
                try:
                    with open(cur_data['action'], "r") as action_file:
                        action = json.load(action_file)
                except json.decoder.JSONDecodeError as e: 
                    print(f"action file: {cur_data['action']}")
                    raise json.decoder.JSONDecodeError(f"action file: {cur_data['action']}", "json_data", 0)
            if action['action_type'] is None:
                action['action_type'] = 0
            if action['coordinate'] is None:
                #TODO: what's the null coordinate if -1 -1 not allowed?
                action['coordinate'] = self.coordinates_to_heatmap(np.array([-1, -1]), H, W)
            else:
                action['coordinate'] = self.coordinates_to_heatmap(np.array(action['coordinate']), H, W)
            if action['object_type'] is None or action['object_type'] == -1:
                action['object_type'] = 0
            if action['object_id'] is None or action['object_id'] == -1:
                action['object_id'] = 0

            data_entry['action_type'] = action['action_type']
            data_entry['coordinate'] = action['coordinate']
            data_entry['object_type'] = action['object_type']
            data_entry['object_id'] = action['object_id']
            if action['object_type'] != 0: # deal with drop index
                data_entry['object_carry_index'], data_entry['carry_num'] = self.get_obj_carry_idx(action['object_type'], action['object_id'], cur_data['state'])
            else:
                data_entry['object_carry_index'], data_entry['carry_num'] = 0, 0
            
        # get action field for mid level
        elif self.data_level == 'mid':
            # for mid level

            if cur_data['action'] is None:
                action = {'fur': 0, 'obj': 0, 'room': 0, 'action': 11}
            else:
                with open(cur_data['action'], "r") as action_file:
                    action = json.load(action_file)[1]

            data_entry['obj'] = 0 if action['obj'] is None else vocab.OBJECT_TO_IDX[action['obj']]
            data_entry['fur'] = 0 if action['fur'] is None else vocab.OBJECT_TO_IDX[action['fur']]
            data_entry['room'] = 0 if action['room'] is None else vocab.ROOM_TO_IDX[action['room']]
            data_entry['action'] = 0 if action['action'] is None else vocab.ACTION_TO_IDX[action['action']]

        # get audio field
        if self.with_audio:
            if cur_data['manipulate_audio'] is None:
                cur_data['manipulate_audio'] = vocab.AUDIO_TO_IDX['empty']
            data_entry['manipulate_audio'] = cur_data['manipulate_audio']
            audios = np.array(cur_data['audios'])
            audios_padded = np.zeros(self.max_audio_length, dtype=np.float32)
            audios_padded[:len(audios)] = audios

            audio_mask = np.ones(self.max_audio_length, dtype=np.float32)
            audio_mask[:len(audios)] = 0.
            data_entry['audios'] = audios_padded
            data_entry['audio_mask'] = audio_mask
        if self.goal_conditioned:
            try:
                with open(cur_data['subgoal'], "r") as subgoal_file:
                    subgoal_symbolic = json.load(subgoal_file)[1]

                    data_entry['sub_action'] = vocab.ACTION_TO_IDX[subgoal_symbolic['action']] if subgoal_symbolic['action'] is not None else 0
                    data_entry['sub_room'] = vocab.ROOM_TO_IDX[subgoal_symbolic['room']] if subgoal_symbolic['room'] is not None else 0
                    data_entry['sub_fur'] = vocab.OBJECT_TO_IDX[subgoal_symbolic['fur']] if subgoal_symbolic['fur'] is not None else 0
                    data_entry['sub_obj'] = vocab.OBJECT_TO_IDX[subgoal_symbolic['obj']] if subgoal_symbolic['obj'] is not None else 0
            except:
                print("couldn't load data ", cur_data)
                
        return data_entry
    
    def get_obj_carry_idx(self, obj_type, obj_id, state_files):
        if self.historical:
            last_state_file = state_files[-1]
        else:
            last_state_file = state_files

        parent_path = os.path.dirname(last_state_file)
        state_name = last_state_file.split('/')[-1].split('.')[0]
        with open(f'{parent_path}/{state_name}.json', 'r') as json_file:
            state_json = json.load(json_file)
        
        carrying = state_json['Grid']['agents']['initial'][0]['carrying']['initial']
        for i, item in enumerate(carrying):
            if vocab.OBJECT_TO_IDX[item['type']] == obj_type and item['id'] == obj_id:
                return i+1, len(carrying) # index start from 1
        
        return 0, len(carrying)

    
    def construct_data_idx(self):
        idx = 0
        all_traj = []

        rooms = os.listdir(self.data_path) if self.room is None else [self.room]

        if self.split == 'train' or self.split == 'test':
            for room in rooms:
                for mission in os.listdir(os.path.join(self.data_path, room)):
                    if mission in list(self.mission_dict.keys()):
                        if os.path.exists(os.path.join(self.data_path, room, mission + "_test")):
                            with open(os.path.join(self.data_path, room, mission + "_test", "test_traj.json"), "r") as f:
                                test_traj = json.load(f)
                        else:
                            test_traj = []

                        if self.split == 'train':
                            cur_traj = os.listdir(os.path.join(self.data_path, room, mission))
                            cur_traj = [traj for traj in cur_traj if traj not in test_traj]
                        elif self.split == 'test':
                            cur_traj = test_traj
                        else:
                            raise NotImplementedError
                            
                        num_sample = int(self.mission_dict[mission] * len(cur_traj))
                        sampled_traj = np.random.choice(cur_traj, num_sample, replace=False)
                        for traj_name in sampled_traj:
                            self.idx_traj_map[idx] = os.path.join(self.data_path, room, mission, traj_name)
                            idx += 1 

                        all_traj.extend(sampled_traj)
                        # print(f"loaded {len(sampled_traj)} trajectories from room {room} mission: {mission}")
                        
        elif self.split == 'inference':
            for room in rooms:
                for traj in os.listdir(os.path.join(self.data_path, room)):
                    for mission in os.listdir(os.path.join(self.data_path, room, traj)): 
                        if mission in list(self.mission_dict.keys()):
                            num_sample = int(self.mission_dict[mission] * 1)
                            if num_sample > 0:             
                                self.idx_traj_map[idx] = os.path.join(self.data_path, room, traj, mission)
                                idx += 1 
                                all_traj += os.path.join(self.data_path, room, traj, mission)
        
        print('loaded total trajectories: ', len(all_traj))
        
    def coordinates_to_heatmap(self, coordinates, H, W):
        '''
        coordinates: [w, h]
        '''
        heatmap = np.zeros((H, W))
        if 0 <= coordinates[0] < W and 0 <= coordinates[1] < H:
            heatmap[coordinates[1], coordinates[0]] = 1
        return heatmap
    
    def action_to_audio(self, action_file):
        '''
        action file contains dictionary:
        {
            "action_type": 4,
            "coordinate": [
                5,
                23
            ],
            "object_type": -1,
            "object_id": -1
        }
        to audio:
        index
        '''
        # print("converting action file: ", action_file)
        if os.path.exists(action_file):
            with open(action_file, "r") as action_file:
                action = json.load(action_file)
            action_type = vocab.IDX_TO_ACTION[action['action_type']]
            object_type = vocab.IDX_TO_OBJECT[action['object_type']] if action['object_type'] != -1 else None
            if action_type in ['null', 'left', 'right']:
                return vocab.AUDIO_TO_IDX['empty']
            elif action_type == 'idle':
                return vocab.AUDIO_TO_IDX[f'{action_type}_{object_type}'] if f"{action_type}_{object_type}" in vocab.AUDIO_TO_IDX else vocab.AUDIO_TO_IDX['empty']
            elif action_type == 'forward':
                # print(self.data_path)
                # index = self.data_path.split('/')[-1].split('_')[-2]
                # TODO: currently all forward steps are same
                index = 'a'
                return vocab.AUDIO_TO_IDX[f'{action_type}_{index}']
            else:
                return vocab.AUDIO_TO_IDX[f'{action_type}_{object_type}']
        else:
            return vocab.AUDIO_TO_IDX['empty']
        
    def audio_to_action(self, audio_index):
        '''
        audio is an integer index:
        to
        action file, which contains dictionary (with None for coordinate and object_id):
        {
            "action_type": 4,
            "coordinate": None,
            "object_type": -1,
            "object_id": None
        }
        '''
        audio_name = vocab.IDX_TO_AUDIO[audio_index]
        if audio_name == 'empty':
            return {"action_type": None, "coordinate": None, "object_type": None, "object_id": None}
        else:
            action_file = {}
            action_type, object_type = audio_name.split('_')
            action_file["action_type"] = vocab.ACTION_TO_IDX[action_type]
            action_file["object_type"] = -1 if action_type in ["forward", "left", "right"] else vocab.OBJECT_TO_IDX[object_type]
            action_file["coordinate"] = None # We do not know the coordinate nor the object id of the object for sure
            action_file["object_id"] = None
        return action_file

    def get_sorted_files(self, directory, state_suffix, action_suffix, data_level):
        if data_level == 'low':
            state_files = sorted([file for file in os.listdir(directory) if file.endswith(state_suffix)])[:-1]
        elif data_level == 'mid':
            state_files = sorted([file for file in os.listdir(directory) if file.endswith(state_suffix)])
        action_files = sorted([file for file in os.listdir(directory) if file.endswith(action_suffix)])
        return state_files, action_files

    def append_data(self, traj_path, state_files, action_files, low_action_files, traj_type):
        traj_entry = []

        if len(state_files) == len(action_files) + 2:
            state_files = state_files[:-1]
        action_files.append('dummy_action.json')
        cache_manipulate_action_file_idx = 0
        for i, (state_file, action_file) in enumerate(zip(state_files, action_files)):
            #assert state_file.split('_')[0] == action_file.split('_')[0] if i < len(action_files) - 1 else True

            if action_file == 'dummy_action.json' and self.end_state_flag or action_file != 'dummy_action.json':
                if action_file == 'dummy_action.json':
                    manipulate_action_file = low_action_files[-1]
                else:
                    cur_step_count = int(action_file.split('_')[0])
                    if self.data_level == 'mid':
                        manipulate_step_count = cur_step_count 
                    else:
                        manipulate_step_count = cur_step_count - 1
                    manipulate_action_file = f'{manipulate_step_count:05d}_action.json'

                history_state_files = state_files[:i + 1] if self.historical else []

                data_entry = {}

                if traj_type in ['mid', 'mid_no_audio']:
                    data_entry['state'] = [os.path.join(traj_path, 'midlevel', state_file) for state_file in history_state_files] \
                                        if self.historical else os.path.join(traj_path, 'midlevel', state_file)
                    data_entry['action'] = None if action_file == 'dummy_action.json' else os.path.join(traj_path, 'midlevel', action_file)
                if traj_type in ['low', 'low_no_audio']:
                    data_entry['state'] = [os.path.join(traj_path, state_file) for state_file in history_state_files] \
                                        if self.historical else os.path.join(traj_path, state_file)
                    data_entry['action'] = None if action_file == 'dummy_action.json' else os.path.join(traj_path, action_file)

                if traj_type in ['mid', 'low']: # add audio part
                    if traj_type == 'low':
                        history_action_files = action_files[:i] 
                        # all previous actions
                    elif traj_type == 'mid':
                        manipulate_action_file_idx = low_action_files.index(manipulate_action_file) if manipulate_action_file in low_action_files else -1
                        history_action_files = low_action_files[cache_manipulate_action_file_idx:manipulate_action_file_idx+1] 
                        # only audio from previous mid-action
                        cache_manipulate_action_file_idx = manipulate_action_file_idx + 1

                    data_entry['manipulate_audio'] = self.action_to_audio(os.path.join(traj_path, manipulate_action_file))
                    data_entry['audios'] = [self.action_to_audio(os.path.join(traj_path, action_file)) for action_file in history_action_files]
                if self.goal_conditioned:
                    if traj_type in ['low', 'low_no_audio']:
                        subgoal_files = sorted([file for file in os.listdir(os.path.join(traj_path, 'midlevel')) if file.endswith('subgoal.json')])
                        action_timestep = None if action_file == 'dummy_action.json' else int(action_file.split('_')[0])
                        # data_entry['subgoal'] = None if action_timestep == None else os.path.join(traj_path, 'midlevel', subgoal_files[np.digitize(action_timestep, [int(x.split('_')[0]) for x in subgoal_files]) - 1]) #Bucket into subgoal based on action timestep
                        data_entry['subgoal'] = None if action_timestep == None else os.path.join(traj_path, 'midlevel', subgoal_files[np.digitize(action_timestep, [int(x.split('_')[0]) for x in subgoal_files], right=True)]) #Bucket into subgoal based on action timestep
                self.data.append(data_entry)
                traj_entry.append(data_entry)
            else:
                continue

        self.path_to_data[traj_path] = traj_entry


    def construct_historical_data(self):
        for idx, traj_path in tqdm(self.idx_traj_map.items()):
            low_state_files, low_action_files = self.get_sorted_files(traj_path, 'states.npy', 'action.json', 'low')
            mid_state_files, mid_action_files = self.get_sorted_files(os.path.join(traj_path, 'midlevel'), 'init_states.npy', 'subgoal.json', 'mid')

            check_low_shift = not os.path.exists(os.path.join(traj_path, f'{0:05d}_action.json'))
            check_mid_shift = not os.path.exists(os.path.join(os.path.join(traj_path, 'midlevel'), f'{0:05d}_init_states.json'))

            if check_mid_shift or check_low_shift:
                print(f"the action step count is shifted for {traj_path}")
                continue

            if self.data_level == 'low':
                if len(low_state_files) == len(low_action_files) and len(mid_state_files) == len(mid_action_files):
                    if self.with_audio:
                        self.append_data(traj_path, low_state_files, low_action_files, low_action_files, traj_type='low')
                    else:
                        self.append_data(traj_path, low_state_files, low_action_files, low_action_files, traj_type='low_no_audio')
                else:
                    print(f"number of states and actions don't match for {traj_path}")
                    continue
            elif self.data_level == 'mid':
                if len(mid_state_files) == len(mid_action_files):
                    if self.with_audio:
                        self.append_data(traj_path, mid_state_files, mid_action_files, low_action_files, traj_type='mid')
                    else:
                        self.append_data(traj_path, mid_state_files, mid_action_files, low_action_files, traj_type='mid_no_audio')
                else:
                    print(f"number of states and actions don't match for {traj_path}")
                    continue
            else:
                raise NotImplementedError

        print(f"final loaded {len(self.data)} samples")
            

    def set_max_dims(self):
        # get max dim 
        # if self.
        # states =[np.load(entry['state'], allow_pickle=True).astype(np.float32) for entry in self.data]

        # target_size = [max(state.shape[0] for state in states), max(state.shape[1] for state in states) , 8]
        # print(target_size)

        # self.target_size = target_size
        self.target_size = [20, 20, 8]
    

    def pad(self, state):
        pad_width = [(0, max(0, self.target_size[i] - state.shape[i])) for i in range(len(self.target_size))]
        state = np.pad(state, pad_width, mode='constant', constant_values=0)
        return state