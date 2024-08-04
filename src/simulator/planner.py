# State Machine
import random
import json
import time
import calendar
import shutil
import copy

import numpy as np

from src.mini_behavior.mini_behavior.missions import *
from src.mini_behavior.mini_behavior.minibehavior import *
from src.mini_behavior.mini_behavior.utils.save import *
from src.mini_behavior.mini_behavior.actions import *
from src.mini_behavior.mini_behavior.objects import *
from src.mini_behavior.bddl import *
from src.simulator.vocab import *


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

class StateMachine:

    def __init__(self):
        self.handlers = {}
        self.startState = None
        self.endStates = []
        self.mission_dicts = {} # [idx: [subgoal name, subgoal dict]]

    def add_state(self, idx, name, handler, mission_dict):
        self.handlers[idx] = handler
        self.mission_dicts[idx] = [name, mission_dict]
        if mission_dict and mission_dict['end_state']:
            self.endStates.append(idx)

    def set_start(self, idx=-1):
        self.startState = idx

    def run(self, low_level_planner, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=False, save_gif=False):
        try:
            handler = self.handlers[self.startState]
        except:
            raise NotImplemented("must call .set_start() before .run()")
        if not self.endStates:
            raise NotImplemented("at least one state must be an end_state")

        cur_state = self.startState
        while True:
            newState, object = handler(cur_state)
            if newState == 'impossible':
                print('mission failed')
                return 'mission failed', 0, 0, frames, audios, audios_symbolic, texts, step_count
            mission_dict = {}
            mission_dict['Goal'] = []
            mission_dict['Goal'].append(self.mission_dicts[newState])
            print('start low level planner')
            obs, reward, done, frames, audios, audios_symbolic, texts, step_count = low_level_planner(
                mission_dict, self.agent_id, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=multimodal, save_gif=save_gif)        
            if newState in self.endStates:
                break
            else:
                handler = self.handlers[newState]
                cur_state = newState
        
        # mission_dict = {}
        # mission_dict['Goal'] = []
        # mission_dict['Goal'].append(['back_to_initial', None])
        # obs, reward, done, frames, audios, audios_symbolic, texts, step_count = low_level_planner(
        #     mission_dict, self.agent_id, args, frames, audios, audios_symbolic, texts, step_count, time_stamp)

        return obs, reward, done, frames, audios, audios_symbolic, texts, step_count
        


class MissionFSM(StateMachine):
    def __init__(self, subgoals_order, env=None, agent_id=None):
        self.env = env
        self.agent_id = agent_id
        self.subgoals_order = subgoals_order
        self.idx_to_state = {}
        super().__init__()

    def add_all_states(self):
        self.add_state(-1, 'Start', self.state_transitions, {})
        for idx, subgoal in enumerate(self.subgoals_order):
            self.add_state(idx, subgoal[0], self.state_transitions, subgoal[1])
            self.idx_to_state[idx] = subgoal[0]
    
    def check_subgoal_find_obj(self, subgoal_mission_dict):
        goal_obj_type = subgoal_mission_dict['obj']
        goal_on_fur_type = subgoal_mission_dict['fur']
        goal_in_room = subgoal_mission_dict['room']
        target_objs = []
        target_furs = []

        for room in self.env.room_instances:
            if goal_in_room and room.type != goal_in_room:
                continue

            for furniture in room.furnitures:
                if goal_on_fur_type and furniture.type != goal_on_fur_type:
                    continue

                target_furs.append(furniture)

                if subgoal_mission_dict['action'] != 'drop':
                    for obj in furniture.objects:
                        if obj.type == goal_obj_type:
                            target_objs.append(obj)

            if subgoal_mission_dict['action'] != 'drop':
                for obj in room.floor_objs:
                    if obj.type == goal_obj_type:
                        target_objs.append(obj)

        if subgoal_mission_dict['action'] == 'drop':
            for obj in self.env.agents[self.env.cur_agent_id].carrying:
                if obj.type == goal_obj_type:
                    target_objs.append(obj) 
                    
        return target_objs, target_furs

    def check_subgoal_if_finished(self, target_objs, subgoal_mission_dict):
        need_state = subgoal_mission_dict['state']
        for target_obj in target_objs:
            finished = False
            # check if this target obj is finished
            if need_state:
                state_name, flag = need_state
                cur_states = target_obj.get_ability_values(self.env)
                if state_name in cur_states and cur_states[state_name] == flag:
                    # already finished
                    finished = True

            # based on the check result, return or not
            if not finished:
                # a not finished target obj
                return True, target_obj
            else:
                continue
        # no not finished target obj
        return False, None

    
    def check_subgoal(self, subgoal_mission_dict):
        # 1. Can find the object?
        subgoal_mission_name = subgoal_mission_dict[0]
        subgoal_mission_dict = subgoal_mission_dict[1]
        if subgoal_mission_name == 'random_walk':
            return 'feasible', None
        target_objs, target_furs = self.check_subgoal_find_obj(subgoal_mission_dict)

        # 2. if subgoal has an obj/fur but didn't found the obj/fur? If so the subgoal is also invalid
        if (subgoal_mission_dict['obj'] and not target_objs) or (subgoal_mission_dict['fur'] and not target_furs):
            # print(f"checked subgoal {subgoal_mission_name}, {subgoal_mission_dict} is impossible, can't find obj or fur")
            return 'impossible', None
        if target_objs and target_furs: # has obj and has fur, then the target must be obj
            cando, object = self.check_subgoal_if_finished(target_objs, subgoal_mission_dict)
        elif not target_objs and target_furs: # no obj and has fur, then the target must be fur
            cando, object = self.check_subgoal_if_finished(target_furs, subgoal_mission_dict)
        else:
            # print(f"checked subgoal {subgoal_mission_dict} is impossible, can't find obj or fur")
            return 'impossible', None
        
        # 3. if the subgoal is finished already?
        if cando:
            if not object.possible_action(subgoal_mission_dict['action']):
                # print(f"checked subgoal {subgoal_mission_dict} is impossible, not possible action")
                return 'impossible', None
            if subgoal_mission_dict['action'] in ACTION_FUNC_MAPPING:
                # print("check subgoal, if object", object.type, "can do action", subgoal_mission_dict['action'])
                if subgoal_mission_dict['action'] in ['pickup', 'drop', 'idle']:
                    for fur in target_furs:
                        if ACTION_FUNC_MAPPING[subgoal_mission_dict['action']](self.env).pre_can(object, fur):
                            # print(f"checked subgoal {subgoal_mission_dict} is feasible")
                            return 'feasible', object
                    # print(f"checked subgoal {subgoal_mission_dict} is cannotdo")
                    return 'cannotdo', None
                else:
                    if not ACTION_FUNC_MAPPING[subgoal_mission_dict['action']](self.env).pre_can(object):
                        # print(f"checked subgoal {subgoal_mission_dict} is cannotdo")
                        return 'cannotdo', None
                    else:
                        # print(f"checked subgoal {subgoal_mission_dict} is feasible")
                        return 'feasible', object
            else: # not a manipulatable action
                # print(f"checked subgoal {subgoal_mission_dict} is impossible")
                return 'impossible', None
        else:
            # print(f"checked subgoal {subgoal_mission_dict} is finished")
            return 'finished', object

    def state_transitions(self, cur_state):
        state_id = cur_state + 1
        forgetness = self.env.agents[self.env.cur_agent_id].forgetness
        while True:
            if state_id in self.mission_dicts:
                subgoal_mission_dict = self.mission_dicts[state_id][1]
                # first check if can skip
                if subgoal_mission_dict['can_skip']:
                    skip = random.random()
                    if skip > forgetness: # not skip state_id
                        subgoal_mission_dict = self.mission_dicts[state_id]
                        candoflag, object = self.check_subgoal(subgoal_mission_dict)
                        if candoflag == 'feasible':
                            return state_id, object
                        elif candoflag == 'finished':
                            # skip
                            state_id += 1
                            continue
                        elif candoflag == 'impossible':
                            return 'impossible', None
                        else:
                            raise NotImplementedError 
                    else: # skip state_id
                        state_id += 1
                        continue
                else:
                    subgoal_mission_dict = self.mission_dicts[state_id]
                    candoflag, object = self.check_subgoal(subgoal_mission_dict)
                    if candoflag == 'feasible':
                        return state_id, object
                    elif candoflag == 'finished':
                        # skip
                        state_id += 1
                        continue
                    elif candoflag == 'impossible':
                        return 'impossible', None
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError('The subgoal id is out of index')


class Planner:

    def __init__(self, env, agent_id, episodes=1):
        self.env = env
        self.agent_id = agent_id
        self.episodes = episodes
        self.mission_preference = self.env.agents[self.agent_id].mission_preference
        self.data_folder = None
        self.idle_time = 10
        self.traj_folder = None
        self.traj_folder_midlevel = None
        # a dictionary with {task_name: prob}

    def high_level_planner(self, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=False, save_gif=False):
        print('start high level planner', step_count)
        feasible_mission_preferences = self.check_mission_feasible()
        try:
            random_mission = random.choices(
                list(feasible_mission_preferences.keys()),
                weights=list(feasible_mission_preferences.values()), k=1)[0]
        except:
            return obs, reward, done, frames, audios, audios_symbolic, texts, step_count

        self.env.agents[self.agent_id].cur_mission = random_mission
        self.mission_folder = os.path.join(self.data_folder, random_mission)
        self.traj_folder_midlevel = None
        self.traj_folder = None
        self.env.agents[self.agent_id].init_pos = self.env.agents[self.agent_id].agent_pos
        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.mid_level_planner(
            random_mission, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=multimodal, save_gif=save_gif)

        while obs == 'mission failed':
            if os.path.exists(f'{self.mission_folder}/{time_stamp}'):
                shutil.rmtree(f'{self.mission_folder}/{time_stamp}')
            feasible_mission_preferences[random_mission] = 0
            try:
                random_mission = random.choices(
                list(feasible_mission_preferences.keys()),
                weights=list(feasible_mission_preferences.values()), k=1)[0]
            except:
                return obs, reward, done, frames, audios, audios_symbolic, texts, step_count
            print(f'selected random mission {random_mission}')
            self.env.agents[self.agent_id].cur_mission = random_mission
            self.mission_folder = os.path.join(self.data_folder, random_mission)
            # self.traj_folder_midlevel = None
            # self.traj_folder = None
            step_count = 0
            frames = []
            audios = []
            audios_symbolic = []
            texts = []
            current_GMT = time.gmtime()
            time_stamp = calendar.timegm(current_GMT)
            obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.mid_level_planner(
                random_mission, args, frames, audios, audios_symbolic, texts, step_count, time_stamp)

        return obs, reward, done, frames, audios, audios_symbolic, texts, step_count

    def mid_level_planner(self, mission, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=False, save_gif=False):
        print('start mid level planner', step_count)
        mission_subgoals_order = MISSION_TO_SUBGOALS[mission]
        mission_state_machine = MissionFSM(
            mission_subgoals_order, self.env, self.agent_id)
        mission_state_machine.add_all_states()

        mission_state_machine.set_start()
        obs, reward, done, frames, audios, audios_symbolic, texts, step_count = mission_state_machine.run(
            self.low_level_planner, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=multimodal, save_gif=save_gif)

        return obs, reward, done, frames, audios, audios_symbolic, texts, step_count

    def get_destination(self, goal, target_furs, cur_agent, env, virtual_cur_pos, virtual_cur_dir, object=None):
        if goal['action'] == 'drop':
            tgt_pos = self.get_target_position(
                target_furs, cur_agent, env, virtual_cur_pos, virtual_cur_dir)
        else:
            tgt_fur = object if isinstance(
                object, FurnitureObj) else object.on_fur
            tgt_pos = self.get_target_position(
                [tgt_fur], cur_agent, env, virtual_cur_pos, virtual_cur_dir)

        return tgt_pos

    def get_target_position(self, target_furs, cur_agent, env, virtual_cur_pos, virtual_cur_dir):
        for tgt_fur in target_furs:
            sample_poses = tgt_fur.all_pos
            action_length = np.inf
            print(f"start looking at all empty cells around the furniture {tgt_fur.name}")
            for pos in sample_poses:
                if self.check_at_least_one_empty_around(env, pos):
                    action_list, _, _ = cur_agent.navigate(
                        env, virtual_cur_pos, virtual_cur_dir, pos)
                    if len(action_list) < action_length:
                        tgt_pos = pos
                        action_length = len(action_list)
            if action_length != np.inf:
                return tgt_pos

    def low_level_planner(self, mission_dict, agent_id, args, frames, audios, audios_symbolic, texts, step_count, time_stamp, multimodal=False, save_gif=False):
        # low level planner for one single subgoal
        print("start low level planner", step_count)
        print('mission dict', mission_dict)
        if mission_dict is None:
            todos = []
            cur_agent = self.env.agents[agent_id]
            action_list, virtual_cur_pos, virtual_cur_dir = cur_agent.random_walk(
                self.env)
            todos += action_list
            print(f"-----plan actions for agent {agent_id} with final todos {todos}")

            obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.conduct_actions(
                args, todos, frames, audios, audios_symbolic, texts, step_count, time_stamp, subgoal=subgoal, multimodal=multimodal, save_gif=save_gif)

            return obs, reward, done, frames, audios, audios_symbolic, texts, step_count

        if self.traj_folder is None:
            self.traj_folder = f'{self.mission_folder}/{time_stamp}'
        if self.traj_folder_midlevel is None:
            self.traj_folder_midlevel = os.path.join(self.traj_folder, 'midlevel')
            # self.traj_folder_midlevel = f'{self.mission_folder}/{time_stamp}/midlevel'

        tgt_obj = None
        tgt_pos = None
        tgt_fur = None

        if args.save: 
            print(f'-------saved initial state 0 for low level {step_count} low level planner')
            self.save_state(copy.deepcopy(self.env), self.traj_folder, step_count)


            # save init state
            img = self.env.render('rgb_array', tile_size=16) 
            img = Image.fromarray(img)

            img.save(os.path.join(self.traj_folder, "init_state.jpeg"))
            print('saved img to', os.path.join(self.traj_folder, "init_state.jpeg"))
        
        for subgoal in mission_dict['Goal']:
            print(f"Agent {self.env.agents[self.agent_id].name}: I am trying to {subgoal}.")
            self.env.agents[self.agent_id].cur_subgoal = subgoal

            if args.save:
                print(
                    f"********save state for mid level {step_count}, init state for {subgoal} in low level planner", self.env.agents[self.agent_id].carrying)
                self.save_subgoal_state(copy.deepcopy(self.env),
                                self.traj_folder_midlevel, step_count, 'init')

            cur_agent = self.env.agents[agent_id]
            virtual_cur_pos = (
                copy.deepcopy(self.env.agents[agent_id].agent_pos[0]), copy.deepcopy(self.env.agents[agent_id].agent_pos[1]))
            virtual_cur_dir = copy.deepcopy(
                self.env.agents[agent_id].agent_dir)
            # virtual_carrying = copy.deepcopy(self.env.agents[agent_id].carrying)
            if subgoal[0] == 'random_walk':
                todos = []
                # random walk
                action_list, virtual_cur_pos, virtual_cur_dir = cur_agent.random_walk(
                    self.env)
                todos += action_list
                obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.conduct_actions(
                args, todos, frames, audios, audios_symbolic, texts, step_count, time_stamp, subgoal=subgoal, multimodal=multimodal, save_gif=save_gif)
                if args.save:
                    self.save_subgoal(self.traj_folder_midlevel, step_count, subgoal)                
                continue
            if subgoal[0] == 'back_to_initial':
                # navigate back to initial position
                todos = []
                tgt_pos = self.env.agents[agent_id].init_pos
                # print(f"navigate to {tgt_pos} from {virtual_cur_pos}")
                action_list, virtual_cur_pos, virtual_cur_dir = cur_agent.navigate(
                    self.env, virtual_cur_pos, virtual_cur_dir, tgt_pos, exact=True)
                todos += action_list

                obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.conduct_actions(
                    args, todos, frames, audios, audios_symbolic, texts, step_count, time_stamp, subgoal=subgoal, multimodal=multimodal, save_gif=save_gif)
                if args.save:
                    self.save_subgoal(self.traj_folder_midlevel, step_count, subgoal)                    
                continue


            goal = subgoal[1]
            # for every goal, check if it is feasible:
            missionFSM = MissionFSM([], self.env, self.agent_id)
            flag, _ = missionFSM.check_subgoal(subgoal)
            if flag != 'feasible':
                print("Check subgoal", subgoal, "is not feasible")
                return None, None, None, frames, audios, audios_symbolic, texts, step_count

            print("Check subgoal", subgoal, "is feasible")
 
            todos = []
            # 1. get the object
            target_objs, target_furs = missionFSM.check_subgoal_find_obj(goal)

            if target_objs and target_furs: # has obj and has fur, then the target must be obj
                cando, object = missionFSM.check_subgoal_if_finished(target_objs, goal)
            elif not target_objs and target_furs: # no obj and has fur, then the target must be fur
                cando, object = missionFSM.check_subgoal_if_finished(target_furs, goal)
            else: # no target objects and no target furnitures, then should be random walk
                action_list, virtual_cur_pos, virtual_cur_dir = cur_agent.random_walk(
                    self.env)
                todos += action_list
                # conduct all actions
                obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.conduct_actions(
                    args, todos, frames, audios, audios_symbolic, texts, step_count, time_stamp, subgoal=subgoal, multimodal=multimodal, save_gif=save_gif)
                continue

            # 2. get the position of the destination
            tgt_pos = self.get_destination(
                goal, target_furs, cur_agent, self.env, virtual_cur_pos, virtual_cur_dir, object)

            # 3. Navigate to target
            # print(f"Navigate to {tgt_pos} from {virtual_cur_pos}")
            action_list, virtual_cur_pos, virtual_cur_dir = cur_agent.navigate(
                self.env, virtual_cur_pos, virtual_cur_dir, tgt_pos)
            todos += action_list

            # print(f"Agent currently at {virtual_cur_pos} {virtual_cur_dir}, face to {tgt_pos}")
            # Face to
            action_list, virtual_cur_pos, virtual_cur_dir = cur_agent.face_to(
                self.env, tgt_pos, virtual_cur_pos, virtual_cur_dir)
            todos += action_list

            # 3. Manipulate
            action_name = goal['action']
            todos.append([action_name, object])

            # # TODO: zoey removed the virtual carrying
            # if action_name == 'pickup':
            #     virtual_carrying.add(object)
            # if action_name == 'drop':
            #     virtual_carrying.remove(object)

            # conduct all actions

            obs, reward, done, frames, audios, audios_symbolic, texts, step_count = self.conduct_actions(
                args, todos, frames, audios, audios_symbolic, texts, step_count, time_stamp, subgoal=subgoal, multimodal=multimodal, save_gif=save_gif)

            if args.save:
                print(f'save state after subgoal: {subgoal}, step_count: {step_count}')
                self.save_subgoal(self.traj_folder_midlevel, step_count - 1, subgoal)
                self.save_subgoal_state(copy.deepcopy(self.env), self.traj_folder_midlevel, step_count, 'final')

        if args.save:
            # save init state
            img = self.env.render('rgb_array', tile_size=16) 
            img = Image.fromarray(img)

            img.save(os.path.join(self.traj_folder, "final_state.jpeg"))
            print('saved img to', os.path.join(self.traj_folder, "final_state.jpeg"))

        if save_gif:
            # print(
            #     f"**********save state for mid level {step_count}, after in low level planner", self.env.agents[self.agent_id].carrying)
            # save for mid level
            # self.save_subgoal_state(copy.deepcopy(self.env),
            #                 self.traj_folder_midlevel, step_count)
            if self.traj_folder is None:
                self.traj_folder = f'{self.mission_folder}/{time_stamp}'
            # self.save_state(copy.deepcopy(self.env), self.traj_folder, step_count) # save for low level
            # print(
            #     f'------saved state for low level {step_count} low level planner')
            frames.append(np.moveaxis(
                    self.env.render("rgb_array"), 2, 0))
        #if multimodal:
        #    audio_file, audio_idx = get_audio_file_idx(empty=True)
        #    audios.append(AudioSegment.from_wav(audio_file))
        #    audios_symbolic.append(audio_idx)
            
        return obs, reward, done, frames, audios, audios_symbolic, texts, step_count

    def check_mission_feasible(self):
        feasible_mission_preferences = {}
        return self.mission_preference

    def check_at_least_one_empty_around(self, env, pos):
        base_action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dir in base_action_space:
            neighbor = (pos[0] + dir[0],
                        pos[1] + dir[1])
            if neighbor[0] < 0 or neighbor[0] > env.grid.width \
                    or neighbor[1] < 0 or neighbor[1] > env.grid.height or not env.grid.is_empty(*neighbor):
                continue
            else:
                return True
        return False
        
    def save_state(self, env, folder_dir, step_count):
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        json_path = os.path.join(folder_dir, f"{step_count:05d}_states.json")

        state_dict = get_state_dict(env)
        with open(json_path, "w") as outfile:
            outfile.write(json.dumps(state_dict, cls=NpEncoder, indent=4))
        
        npy_path = os.path.join(folder_dir, f"{step_count:05d}_states.npy")
        state_array = get_cur_arrays(env)
        with open(npy_path, "wb") as outfile:
            np.save(outfile, state_array)

        return state_array 
    
    def save_subgoal_state(self, env, folder_dir, step_count, state_type):
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        json_path = os.path.join(folder_dir, f"{step_count:05d}_{state_type}_states.json")

        state_dict = get_state_dict(env)
        with open(json_path, "w") as outfile:
            outfile.write(json.dumps(state_dict, cls=NpEncoder, indent=4))
        
        npy_path = os.path.join(folder_dir, f"{step_count:05d}_{state_type}_states.npy")
        state_array = get_cur_arrays(env)
        with open(npy_path, "wb") as outfile:
            np.save(outfile, state_array)

        return state_array 
        
    def save_action(self, folder_dir, step_count, action_dict):
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        action_path = os.path.join(folder_dir, f"{step_count:05d}_action.json")
        with open(action_path, "w") as outfile:
            outfile.write(json.dumps(action_dict, cls=NpEncoder, indent=4))
        return

    def save_subgoal(self, folder_dir, step_count, subgoal):
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        
        subgoal_path = os.path.join(folder_dir, f"{step_count:05d}_subgoal.json")
        subgoal_symbolic = {}
        subgoal_symbolic['obj'] = OBJECT_TO_IDX[subgoal[1]['obj']] if (
            subgoal[1] and subgoal[1]['obj']) else None
        subgoal_symbolic['fur'] = OBJECT_TO_IDX[subgoal[1]['fur']] if (
            subgoal[1] and subgoal[1]['fur']) else None
        subgoal_symbolic['room'] = ROOM_TO_IDX[subgoal[1]['room']] if (
            subgoal[1] and subgoal[1]['obj']) else None
        subgoal_symbolic['pos'] = subgoal[1]['pos'] if subgoal[1] else None
        subgoal_symbolic['action'] = ACTION_TO_IDX[subgoal[1]['action']
                                                   ] if subgoal[1] and isinstance(subgoal[1]['action'], str) else None
        subgoal_symbolic['state'] = [ABILITIES.index(
            subgoal[1]['state'][0]), subgoal[1]['state'][1]] if subgoal[1] and subgoal[1]['state'] else None

        subgoal_to_save = subgoal
        subgoal_to_save.append(subgoal_symbolic)

        with open(subgoal_path, "w") as outfile:
            outfile.write(json.dumps(subgoal_to_save, indent=4))
    
    def conduct_actions(self, args, todos, frames=[], audios=[], audios_symbolic=[], texts=[], step_count=0, episode='', subgoal=None, multimodal=False, save_gif=False):
        # print(
        #     f"start conduct actions {todos} for agent {self.env.cur_agent_id}")
        # intent_text = f"Agent {self.env.agents[self.agent_id].name}: I am trying to {subgoal[0] if subgoal else ''} ."
        obs = None
        reward = None
        done = None
        # save before
        if self.traj_folder is None:
            self.traj_folder = f'{self.mission_folder}/{episode}'
        # if args.save:
        #     self.save_state(copy.deepcopy(self.env),
        #                     self.traj_folder, step_count)
        #     print(
        #         f'-------saved state for low level {step_count}, conduct actions before', self.env.agents[self.agent_id].carrying)
        #     frames.append(np.moveaxis(self.env.render("rgb_array"), 2, 0))
        for act_idx, action in enumerate(todos):
            if type(action) == list:
                if len(action) > 1:
                    if isinstance(action[0], str):
                        # 1. action type: idle
                        if action[0] == 'idle':
                            step_count += 1
                            obs = None
                            reward = None
                            done = None
                            action_dict = {
                                 'action_type': ACTION_TO_IDX[action[0]],
                                 'coordinate': [int(action[1].cur_pos[0]), int(action[1].cur_pos[1])],
                                 'object_type': OBJECT_TO_IDX[action[1].type],
                                 'object_id': action[1].id
                            }
                            for _ in range(0, self.idle_time):
                                if args.save:
                                    if save_gif:
                                        frames.append(np.moveaxis(
                                            self.env.render("rgb_array"), 2, 0))
                                    self.save_action(
                                        self.traj_folder, step_count+_-1, action_dict)
                                    self.save_state(copy.deepcopy(
                                        self.env), self.traj_folder, step_count+_)
                                    # print(f'-------saved action for low level {step_count+_-1}: idle')
                                    # print(f'-------saved state for low level {step_count+_} in idle')
                                    # for agent in self.env.agents:
                                    #     print('-------carrying at step', step_count, ': ', agent.carrying) 
#                            if multimodal:                               
#                                if action[1].get_state() and 'idle_on' in OPEN_CLOSE_OBJECT_AUDIO_MAP[action[1].type]:
#                                    audio_file, audio_idx = get_audio_file_idx(action='idle', object=action[1].type, state='on')
#                                    sound = [AudioSegment.from_wav(audio_file) for _ in range(int(self.idle_time))] # the audio file is 2s instead of 1s
#                                    audio_idxs = [audio_idx for _ in range(int(self.idle_time))]
#                                    text = [f'{step_count+_}: {intent_text if act_idx==0 else ""} {action[1].type} is currently idle on.' for _ in range(int(self.idle_time))]
#                                else:
#                                    audio_file, audio_idx = get_audio_file_idx(empty=True)
#                                    sound = [AudioSegment.from_wav(audio_file) for _ in range(int(self.idle_time))]
#                                    audio_idxs = [audio_idx for _ in range(int(self.idle_time))]
#                                    text = [
#                                        f'{step_count+_}: {intent_text if act_idx==0 else ""}' for _ in range(int(self.idle_time))]
                            step_count += self.idle_time - 1
                        # 2. action type: manipulate
                        else:
                            print(
                                f'Conduct actions {action[0]}, {action[1]} at {step_count}')
                            empty_str = ''
#                            text = [f'{step_count}: {intent_text if act_idx==0 else ""} {"Furniture " + action[1].name if action[1].is_furniture() else "Object " + action[1].name} {"on " + action[1].on_fur.name if action[1].on_fur else empty_str} { "in " + action[1].in_room.name if action[1].in_room else empty_str} is currently being {action[0]}.']
                            if action[0] == 'pickup':
                                action_dict = {
                                    'action_type': ACTION_TO_IDX[action[0]],
                                    'coordinate': [int(action[1].cur_pos[0]), int(action[1].cur_pos[1])],
                                    'object_type': OBJECT_TO_IDX[action[1].type],
                                    'object_id': action[1].id
                                    }

                            try:
                                ACTION_FUNC_MAPPING[action[0]](self.env).do(action[1])
                            except Exception as e:
                                print("couldn't do action", action[0], action[1].type)
                                print(e)
                                
                            obs = None
                            reward = None
                            done = None

                            # if multimodal:
                            #     if action[0] == 'toggle':
                            #         if action[1].get_state():
                            #             audio_file, audio_idx = get_audio_file_idx(action=action[0], object=action[1].type, state='off')
                            #         else:
                            #             audio_file, audio_idx = get_audio_file_idx(action=action[0], object=action[1].type, state='on')
                            #     else:
                            #         audio_file, audio_idx = get_audio_file_idx(action=action[0], object=action[1].type)
                            #     sound = [AudioSegment.from_wav(audio_file)]
                            #     audio_idxs = [audio_idx]

                            # # show state
                            # # audios.append(sound)
                            # # frames, sound = show_states(frames, audios)
                            if action[1].cur_pos is None:
                                action[1].cur_pos = (-1. -1)
                            if action[0] != 'pickup':
                                action_dict = {
                                    'action_type': ACTION_TO_IDX[action[0]],
                                    'coordinate': [int(action[1].cur_pos[0]), int(action[1].cur_pos[1])],
                                    'object_type': OBJECT_TO_IDX[action[1].type],
                                    'object_id': action[1].id
                                    }
                            
                    # 3. action type: forward
                elif action[0] == self.env.actions.forward: 
                    obs, reward, done, _ = self.env.step(action[0])

                    # if multimodal:
                    #     audio_file, audio_idx = get_audio_file_idx(action=f'forward_{self.env.cur_agent_id + 1}', object='')
                    #     sound = [AudioSegment.from_wav(audio_file)]
                    #     audio_idxs = [audio_idx]

                    in_rooms = []
                    agent_pos = self.env.agents[self.env.cur_agent_id].agent_pos
                    for room in self.env.room_instances:
                        if room.top[0] - 1 <= agent_pos[0] < room.top[0] + room.size[0] + 1 and \
                        room.top[1] - 1 <= agent_pos[1] < room.top[1] + room.size[1] + 1:
                            in_rooms.append(room.type)
#                    text = [f'{step_count}: {intent_text if act_idx==0 else ""} There are step sounds in {in_rooms[0] if len(in_rooms)==1 else " and ".join(in_rooms)}... Seems like {self.env.agents[self.env.cur_agent_id].name}\' step.']
                    
                    action_dict = {
                        'action_type': ACTION_TO_IDX['forward'],
                        'coordinate': [int(agent_pos[0]), int(agent_pos[1])],
                        'object_type': -1,
                        'object_id': -1,
                    }
                # 3. action type: left right
                elif action[0] in [self.env.actions.left, self.env.actions.right]:
                    obs, reward, done, _ = self.env.step(action[0])
                    # text = [f'{step_count}: {intent_text if act_idx==0 else ""}']

                    # if multimodal:
                    #     audio_file, audio_idx = get_audio_file_idx(empty=True)
                    #     sound = [AudioSegment.from_wav(audio_file)]
                    #     audio_idxs = [audio_idx]
                    # else:
                    #     sound = []
                    #     audio_idxs = []
                    agent_pos = self.env.agents[self.env.cur_agent_id].agent_pos
                    action_dict = {
                            'action_type': ACTION_TO_IDX['left'] if action[0] == self.env.actions.left else ACTION_TO_IDX['right'],
                            'coordinate': [int(agent_pos[0]), int(agent_pos[1])],
                            'object_type': -1,
                            'object_id': -1
                        }
                elif action[0] == 'null': # null, end
                    print("null, no env changes")
                    continue
                elif action[0] == 'end':
                    print("end, no env changes")
                    break
                else:
                    print("action not implemented")
                    raise NotImplementedError
            else:
                print('no action')
                continue

            step_count += 1

            # # save after
            if args.save:
            #     if save_gif:
            #         frames.append(np.moveaxis(self.env.render("rgb_array"), 2, 0))
            #     if multimodal:
            #         audios += sound
            #         texts += text
            #         audios_symbolic += audio_idxs
            #     # folder_dir = f'demos/sample{episode}'
                self.save_action(self.traj_folder, step_count - 1, action_dict)
                self.save_state(copy.deepcopy(self.env),
                                self.traj_folder, step_count)
            #     # print(f'-------saved action for low level {stepW  QX6T ': ', agent.carrying)                                   

        return obs, reward, done, frames, audios, audios_symbolic, texts, step_count
