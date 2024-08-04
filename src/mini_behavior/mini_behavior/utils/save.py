import os
import json
import pickle as pkl
import numpy as np
from marple_mini_behavior.mini_behavior.rooms import *
from marple_mini_behavior.bddl import *



def all_cur_pos(env):
    """
    returns dict with key=obj, value=cur_pos
    """
    pos = {'agent': [int(obj_pos) for obj_pos in env.agent.cur_pos]}

    for obj_name in env.obj_instances:
        obj_instance = env.obj_instances[obj_name]
        pos[obj_name] = [int(obj_pos) for obj_pos in obj_instance.cur_pos]

    return pos


def all_state_values(env):
    """
    returns dict with key=obj_state, value=state value
    """
    states = {}
    for obj_name, obj_instance in env.obj_instances.items():
        obj_states = obj_instance.get_all_state_values(env)
        states.update(obj_states)
    return states

def get_state_dict(env):
    """
    return state dict
    """
    agent_dict = env.state_dict['Grid']['agents']
    agent_dict['num'] = len(env.agents)
    agent_dict['initial'] = []
    for agent in env.agents:
        agent_dict['initial'].append({})
        agent_dict['initial'][-1]['name'] = agent.name
        agent_dict['initial'][-1]['id'] = agent.id
        agent_dict['initial'][-1]['gender'] = agent.gender
        # if env.agents[env.cur_agent_id] == agent:
        #     carrying_set = env.carrying
        #     agent_dict['initial'][-1]['pos'] = [int(env.agent_pos[0]), int(env.agent_pos[1])]
        #     agent_dict['initial'][-1]['dir'] = env.agent_dir
        # else:
        carrying_set = agent.carrying
        agent_dict['initial'][-1]['pos'] = [int(agent.agent_pos[0]), int(agent.agent_pos[1])]
        agent_dict['initial'][-1]['dir'] = agent.agent_dir
        agent_dict['initial'][-1]['color'] = agent.agent_color
        agent_dict['initial'][-1]['step_size'] = agent.step_size
        agent_dict['initial'][-1]['forgetness'] = agent.forgetness
        agent_dict['initial'][-1]['mission_preference_initial'] = agent.mission_preference
        agent_dict['initial'][-1]['cur_mission'] = agent.cur_mission
        agent_dict['initial'][-1]['cur_subgoal'] = agent.cur_subgoal
        agent_dict['initial'][-1]['carrying'] = {}
        agent_dict['initial'][-1]['carrying']['num'] = len(carrying_set)
        agent_dict['initial'][-1]['carrying']['initial'] = []
        for obj in carrying_set:
            agent_dict['initial'][-1]['carrying']['initial'].append({})
            agent_dict['initial'][-1]['carrying']['initial'][-1]['type'] = obj.type
            agent_dict['initial'][-1]['carrying']['initial'][-1]['name'] = obj.name
            agent_dict['initial'][-1]['carrying']['initial'][-1]['id'] = obj.id
            agent_dict['initial'][-1]['carrying']['initial'][-1]['pos'] = [int(obj.cur_pos[0]), int(obj.cur_pos[1])]
            states = obj.get_ability_values(env)
            agent_dict['initial'][-1]['carrying']['initial'][-1]['state'] = states

    room_dict = env.state_dict['Grid']['rooms']
    room_dict['num'] = len(env.rooms)
    room_dict['initial'] = []
    for room in env.room_instances:
        room_dict['initial'].append({})
        room_dict['initial'][-1]['type'] = room.type
        room_dict['initial'][-1]['name'] = room.name
        room_dict['initial'][-1]['id'] = room.id
        room_dict['initial'][-1]['top'] = room.top
        room_dict['initial'][-1]['size'] = room.size
        room_dict['initial'][-1]['furnitures'] = {}

        fur_dict = room_dict['initial'][-1]['furnitures']
        fur_dict['num'] = len(room.furnitures)
        fur_dict['initial'] = []
        for fur in room.furnitures:
            fur_dict['initial'].append({})
            fur_dict['initial'][-1]['type'] = fur.type
            fur_dict['initial'][-1]['name'] = fur.name
            fur_dict['initial'][-1]['id'] = fur.id
            fur_dict['initial'][-1]['pos'] = [int(fur.cur_pos[0]), int(fur.cur_pos[1])]
            states = fur.get_ability_values(env)
            fur_dict['initial'][-1]['state'] = states
            fur_dict['initial'][-1]['objs'] = {}
            obj_dict = fur_dict['initial'][-1]['objs']
            obj_dict['num'] = len(fur.objects)
            obj_dict['initial'] = []
            for obj in fur.objects:
                obj_dict['initial'].append({})
                obj_dict['initial'][-1]['type'] = obj.type
                obj_dict['initial'][-1]['name'] = obj.name
                obj_dict['initial'][-1]['id'] = obj.id
                obj_dict['initial'][-1]['pos'] = [int(obj.cur_pos[0]), int(obj.cur_pos[1])]
                states = obj.get_ability_values(env)
                obj_dict['initial'][-1]['state'] = states

        room_dict['initial'][-1]['floor_objs'] = {}
        floor_obj_dict = room_dict['initial'][-1]['floor_objs']
        floor_obj_dict['num'] = len(room.floor_objs)
        floor_obj_dict['initial'] = []
        for obj in room.floor_objs:
            floor_obj_dict['initial'].append({})
            floor_obj_dict['initial'][-1]['type'] = obj.type
            floor_obj_dict['initial'][-1]['name'] = obj.name
            floor_obj_dict['initial'][-1]['id'] = obj.id
            floor_obj_dict['initial'][-1]['pos'] = [int(obj.cur_pos[0]), int(obj.cur_pos[1])]
            states = obj.get_ability_values(env)
            floor_obj_dict['initial'][-1]['state'] = states

    env.state_dict['Grid']['doors'] = {}
    door_dict = env.state_dict['Grid']['doors']
    door_dict['num'] = len(env.objs['door'])
    door_dict['initial'] = []
    for door in env.objs['door']:
        door_dict['initial'].append({})
        door_dict['initial'][-1]['type'] = door.type
        door_dict['initial'][-1]['name'] = door.name
        door_dict['initial'][-1]['id'] = door.id
        door_dict['initial'][-1]['dir'] = door.dir
        door_dict['initial'][-1]['pos'] = [int(door.cur_pos[0]) , int(door.cur_pos[1])]
        door_dict['initial'][-1]['state'] = 'open' if door.is_open else 'close'

    return env.state_dict

def get_cur_arrays(env):
    """
    return current (grid_w, grid_h, 8) array
    """
    width = env.grid.width
    height = env.grid.height

    array = np.zeros(shape=(height, width, 8), dtype=np.uint8)
    # room tpye
    # obj/fur type
    # obj/fur states
    # agent direction
    # 1. Room type
    room_type_ch = 0
    for room in env.room_instances:
        top = room.top
        size = room.size
        room_type_idx = ROOM_TYPE_TO_IDX[room.type]
        for i in range(top[0], top[0] + size[0]):
            for j in range(top[1], top[1] + size[1]):
                ymin = j
                ymax = (j+1)
                xmin = i
                xmax = (i+1)
                array[ymin:ymax, xmin:xmax, room_type_ch] = room_type_idx
    
    # 2. Furniture type & states
    fur_type_ch = 1
    fur_state_ch = 2
    for obj in list(env.obj_instances.values()):
        if obj.is_furniture():
            if obj.cur_pos is not None:
                try:
                    i, j = obj.cur_pos
                    ymin = j
                    ymax = (j + obj.height)
                    xmin = i
                    xmax = (i + obj.width)
                    # type
                    fur_type_idx = OBJECT_TO_IDX[obj.type]
                    array[ymin:ymax, xmin:xmax, fur_type_ch] = fur_type_idx
                    # state
                    state_values = obj.get_ability_values(env)
                    fur_states_idx = states_to_idx(state_values)
                    array[ymin:ymax, xmin:xmax, fur_state_ch] = fur_states_idx
                except:
                    pass
    
    # 3. Object type & states & ids
    obj_type_ch = 3
    obj_state_ch = 4
    obj_id_ch = 5
    check_obj_ids = set()
    for obj in list(env.obj_instances.values()):
        check_obj_ids.add(obj.id)
        if not obj.is_furniture():
            if obj.cur_pos is not None:
                try:
                    i, j = obj.cur_pos
                    ymin = j
                    ymax = (j + obj.height)
                    xmin = i
                    xmax = (i + obj.width)
                    # type
                    obj_type_idx = OBJECT_TO_IDX[obj.type]
                    array[ymin:ymax, xmin:xmax, obj_type_ch] = obj_type_idx
                    # state
                    state_values = obj.get_ability_values(env)
                    obj_states_idx = states_to_idx(state_values)
                    array[ymin:ymax, xmin:xmax, obj_state_ch] = obj_states_idx
                    # id
                    array[ymin:ymax, xmin:xmax, obj_id_ch] = obj.id
                except:
                    pass
    assert len(check_obj_ids) == len(env.obj_instances), 'lens dont match in get_cur_array'

    # 5. Agent id and dir
    agent_id_ch = 6
    agent_dir_ch = 7
    for agent in env.agents:
        i, j = agent.agent_pos
        ymin = j
        ymax = (j + 1)
        xmin = i
        xmax = (i + 1)
        # id
        array[ymin:ymax, xmin:xmax, agent_id_ch] = agent.id
        # dir
        array[ymin:ymax, xmin:xmax, agent_dir_ch] = agent.agent_dir

    return array


def get_cur_step(env):
    action = 'none' if env.last_action is None else env.last_action.name
    state = env.get_state()
    state_values = all_state_values(env)

# save last action, all states, all obj pos, all door pos


def get_step(env):
    step_count = env.step_count
    action = 'none' if env.last_action is None else env.last_action.name
    state = env.get_state()
    state_values = all_state_values(env)

    # door_pos = []
    # for door in env.doors:
    #     door_pos.append(door.cur_pos)
    step = {'action': action,
            'predicates': state_values,
            'agent_dirs': state['agent_dirs'],
            'agent_poses': state['agent_poses']
            # 'door_pos': door_pos
            }

    return step_count, step


# save demo_16 as a pkl file
def save_demo(all_steps, env_name, episode):
    demo_dir = os.path.join('../../../demos', env_name)
    if not os.path.isdir(demo_dir):
        os.mkdir(demo_dir)
    demo_dir = os.path.join(demo_dir, str(episode))
    if not os.path.isdir(demo_dir):
        os.mkdir(demo_dir)
    all_files = os.listdir(demo_dir)
    demo_num = len(all_files)
    demo_file = os.path.join(demo_dir, '{}_{}'.format(env_name, demo_num))
    assert not os.path.isfile(demo_file)

    print('saving demo_16 to: {}'.format(demo_file))

    with open(demo_file, 'wb') as f:
        pkl.dump(all_steps, f)

    print('saved')


# save demo_16 as a pkl file
def save_snapshots(env_steps, model_name='', date=''):
    dir = '../snapshots'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    snapshot_file = os.path.join(dir, f'{model_name}_{date}')

    print('saving snapshot to: {}'.format(snapshot_file))

    # hf = h5py.File('{demo_file}.h5', 'w')

    with open(snapshot_file, 'wb') as f:
        pkl.dump(env_steps, f)

    print('saved')


def open_demo(demo_file):
    assert os.path.isfile(demo_file)

    with open(demo_file, 'rb') as f:
        demo = pkl.load(f)
        print('num_steps in demo_16: {}'.format(len(demo)))
        return demo


def get_step_num(step_num, demo_file):
    # returns dict with keys: action, grid, agent_carrying, agent_pos
    demo = open_demo(demo_file)
    return demo[step_num]


def get_action_num(step_num, demo_file):
    demo = open_demo(demo_file)
    action = demo[step_num]['action']
    print('action at step {}: {}'.format(step_num, action.name))
    return action


def get_states(step_num, demo_file):
    demo = open_demo(demo_file)
    states = demo[step_num]['states']
    return states


def print_actions_states(demo_file):
    demo = open_demo(demo_file)
    for step_num in demo:
        print('{}: {}'.format(step_num,  demo[step_num]['action']))
        print('true predicates')
        for state in demo[step_num]['predicates']:
            if demo[step_num]['predicates'][state]:
                print(state)


def print_actions(demo_file):
    demo = open_demo(demo_file)
    for step_num in demo:
        print('{}: {}'.format(step_num,  demo[step_num]['action']))



# import os
# import json
# import pickle as pkl
# import numpy as np
# from marple_mini_behavior.mini_behavior.rooms import *
# from marple_mini_behavior.bddl import *



# def all_cur_pos(env):
#     """
#     returns dict with key=obj, value=cur_pos
#     """
#     pos = {'agent': [int(obj_pos) for obj_pos in env.agent.cur_pos]}

#     for obj_name in env.obj_instances:
#         obj_instance = env.obj_instances[obj_name]
#         pos[obj_name] = [int(obj_pos) for obj_pos in obj_instance.cur_pos]

#     return pos


# def all_state_values(env):
#     """
#     returns dict with key=obj_state, value=state value
#     """
#     states = {}
#     for obj_name, obj_instance in env.obj_instances.items():
#         obj_states = obj_instance.get_all_state_values(env)
#         states.update(obj_states)
#     return states

# def get_state_dict(env):
#     """
#     return state dict
#     """

#     agent_dict = {}
#     agent_dict['num'] = len(env.agents)
#     agent_dict['initial'] = []
#     for agent in env.agents:
#         agent_dict['initial'].append({})
#         agent_dict['initial'][-1]['name'] = agent.name
#         agent_dict['initial'][-1]['id'] = agent.id
#         agent_dict['initial'][-1]['gender'] = agent.gender
#         agent_dict['initial'][-1]['pos'] = [int(agent.agent_pos[0]), int(agent.agent_pos[1])]
#         agent_dict['initial'][-1]['dir'] = agent.agent_dir
#         agent_dict['initial'][-1]['color'] = agent.agent_color
#         agent_dict['initial'][-1]['step_size'] = agent.step_size
#         agent_dict['initial'][-1]['forgetness'] = agent.forgetness
#         agent_dict['initial'][-1]['mission_preference_initial'] = agent.mission_preference
#         agent_dict['initial'][-1]['cur_mission'] = agent.cur_mission
#         agent_dict['initial'][-1]['cur_subgoal'] = agent.cur_subgoal

#         agent_dict['initial'][-1]['carrying'] = {}
#         carrying_set = agent.carrying
#         agent_dict['initial'][-1]['carrying']['num'] = len(carrying_set)
#         agent_dict['initial'][-1]['carrying']['initial'] = []
#         for obj in carrying_set:
#             agent_dict['initial'][-1]['carrying']['initial'].append({})
#             agent_dict['initial'][-1]['carrying']['initial'][-1]['type'] = obj.type
#             agent_dict['initial'][-1]['carrying']['initial'][-1]['name'] = obj.name
#             agent_dict['initial'][-1]['carrying']['initial'][-1]['id'] = obj.id
#             agent_dict['initial'][-1]['carrying']['initial'][-1]['pos'] = [int(obj.cur_pos[0]), int(obj.cur_pos[1])]
#             states = obj.get_ability_values(env)
#             agent_dict['initial'][-1]['carrying']['initial'][-1]['state'] = states

#     room_dict = {}
#     room_dict['num'] = len(env.room_instances)
#     room_dict['initial'] = []
#     for room in env.room_instances:
#         room_dict['initial'].append({})
#         room_dict['initial'][-1]['type'] = room.type
#         room_dict['initial'][-1]['name'] = room.name
#         room_dict['initial'][-1]['id'] = room.id
#         room_dict['initial'][-1]['top'] = room.top
#         room_dict['initial'][-1]['size'] = room.size
#         room_dict['initial'][-1]['furnitures'] = {}

#         room_dict['initial'][-1]['furnitures']['num'] = len(room.furnitures)
#         room_dict['initial'][-1]['furnitures']['initial'] = []
#         for fur in room.furnitures:
#             room_dict['initial'][-1]['furnitures']['initial'].append({})
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['type'] = fur.type
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['name'] = fur.name
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['id'] = fur.id
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['pos'] = [
#                 int(fur.cur_pos[0]), int(fur.cur_pos[1])]
#             states = fur.get_ability_values(env)
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['state'] = states
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['objs'] = {}

#             room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['num'] = len(
#                 fur.objects)
#             room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'] = []
#             for obj in fur.objects:
#                 room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'].append({
#                 })
#                 room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'][-1]['type'] = obj.type
#                 room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'][-1]['name'] = obj.name
#                 room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'][-1]['id'] = obj.id
#                 room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'][-1]['pos'] = [
#                     int(obj.cur_pos[0]), int(obj.cur_pos[1])]
#                 states = obj.get_ability_values(env)
#                 room_dict['initial'][-1]['furnitures']['initial'][-1]['objs']['initial'][-1]['state'] = states

#         room_dict['initial'][-1]['floor_objs'] = {}
#         room_dict['initial'][-1]['floor_objs'] = room_dict['initial'][-1]['floor_objs']
#         room_dict['initial'][-1]['floor_objs']['num'] = len(room.floor_objs)
#         room_dict['initial'][-1]['floor_objs']['initial'] = []
#         for obj in room.floor_objs:
#             room_dict['initial'][-1]['floor_objs']['initial'].append({})
#             room_dict['initial'][-1]['floor_objs']['initial'][-1]['type'] = obj.type
#             room_dict['initial'][-1]['floor_objs']['initial'][-1]['name'] = obj.name
#             room_dict['initial'][-1]['floor_objs']['initial'][-1]['id'] = obj.id
#             room_dict['initial'][-1]['floor_objs']['initial'][-1]['pos'] = [
#                 int(obj.cur_pos[0]), int(obj.cur_pos[1])]
#             states = obj.get_ability_values(env)
#             room_dict['initial'][-1]['floor_objs']['initial'][-1]['state'] = states

#     door_dict = {}
#     door_dict['num'] = len(env.objs['door'])
#     door_dict['initial'] = []
#     for door in env.objs['door']:
#         door_dict['initial'].append({})
#         door_dict['initial'][-1]['type'] = door.type
#         door_dict['initial'][-1]['name'] = door.name
#         door_dict['initial'][-1]['id'] = door.id
#         door_dict['initial'][-1]['dir'] = door.dir
#         door_dict['initial'][-1]['pos'] = [int(door.cur_pos[0]) , int(door.cur_pos[1])]
#         door_dict['initial'][-1]['state'] = 'open' if door.is_open else 'close'

#     env.state_dict['Grid']['agents'] = agent_dict
#     env.state_dict['Grid']['rooms'] = room_dict
#     env.state_dict['Grid']['doors'] = door_dict

#     return env.state_dict

# def get_cur_arrays(env):
#     """
#     return current (grid_w, grid_h, 8) array
#     """
#     width = env.grid.width
#     height = env.grid.height

#     array = np.zeros(shape=(height, width, 8), dtype=np.uint8)
#     # room tpye
#     # obj/fur type
#     # obj/fur states
#     # agent direction
#     # 1. Room type
#     room_type_ch = 0
#     for room in env.room_instances:
#         top = room.top
#         size = room.size
#         room_type_idx = ROOM_TYPE_TO_IDX[room.type]
#         for i in range(top[0], top[0] + size[0]):
#             for j in range(top[1], top[1] + size[1]):
#                 ymin = j
#                 ymax = (j+1)
#                 xmin = i
#                 xmax = (i+1)
#                 array[ymin:ymax, xmin:xmax, room_type_ch] = room_type_idx
    
#     # 2. Furniture type & states
#     fur_type_ch = 1
#     fur_state_ch = 2
#     for obj in list(env.obj_instances.values()):
#         if obj.is_furniture():
#             if obj.cur_pos is not None:
#                 try:
#                     i, j = obj.cur_pos
#                     ymin = j
#                     ymax = (j + obj.height)
#                     xmin = i
#                     xmax = (i + obj.width)
#                     # type
#                     fur_type_idx = OBJECT_TO_IDX[obj.type]
#                     array[ymin:ymax, xmin:xmax, fur_type_ch] = fur_type_idx
#                     # state
#                     state_values = obj.get_ability_values(env)
#                     fur_states_idx = states_to_idx(state_values)
#                     array[ymin:ymax, xmin:xmax, fur_state_ch] = fur_states_idx
#                 except:
#                     pass
    
#     # 3. Object type & states & ids
#     obj_type_ch = 3
#     obj_state_ch = 4
#     obj_id_ch = 5
#     check_obj_ids = set()
#     for obj in list(env.obj_instances.values()):
#         check_obj_ids.add(obj.id)
#         if not obj.is_furniture():
#             if obj.cur_pos is not None:
#                 try:
#                     i, j = obj.cur_pos
#                     ymin = j
#                     ymax = (j + obj.height)
#                     xmin = i
#                     xmax = (i + obj.width)
#                     # type
#                     obj_type_idx = OBJECT_TO_IDX[obj.type]
#                     array[ymin:ymax, xmin:xmax, obj_type_ch] = obj_type_idx
#                     # state
#                     state_values = obj.get_ability_values(env)
#                     obj_states_idx = states_to_idx(state_values)
#                     array[ymin:ymax, xmin:xmax, obj_state_ch] = obj_states_idx
#                     # id
#                     array[ymin:ymax, xmin:xmax, obj_id_ch] = obj.id
#                 except:
#                     pass
#     assert len(check_obj_ids) == len(env.obj_instances)

#     # 5. Agent id and dir
#     agent_id_ch = 6
#     agent_dir_ch = 7
#     for agent in env.agents:
#         i, j = agent.agent_pos
#         ymin = j
#         ymax = (j + 1)
#         xmin = i
#         xmax = (i + 1)
#         # id
#         array[ymin:ymax, xmin:xmax, agent_id_ch] = agent.id
#         # dir
#         array[ymin:ymax, xmin:xmax, agent_dir_ch] = agent.agent_dir + 1

#     return array


# def get_cur_step(env):
#     action = 'none' if env.last_action is None else env.last_action.name
#     state = env.get_state()
#     state_values = all_state_values(env)

# # save last action, all states, all obj pos, all door pos


# def get_step(env):
#     step_count = env.step_count
#     action = 'none' if env.last_action is None else env.last_action.name
#     state = env.get_state()
#     state_values = all_state_values(env)

#     # door_pos = []
#     # for door in env.doors:
#     #     door_pos.append(door.cur_pos)
#     step = {'action': action,
#             'predicates': state_values,
#             'agent_dirs': state['agent_dirs'],
#             'agent_poses': state['agent_poses']
#             # 'door_pos': door_pos
#             }

#     return step_count, step


# # save demo_16 as a pkl file
# def save_demo(all_steps, env_name, episode):
#     demo_dir = os.path.join('../../../demos', env_name)
#     if not os.path.isdir(demo_dir):
#         os.mkdir(demo_dir)
#     demo_dir = os.path.join(demo_dir, str(episode))
#     if not os.path.isdir(demo_dir):
#         os.mkdir(demo_dir)
#     all_files = os.listdir(demo_dir)
#     demo_num = len(all_files)
#     demo_file = os.path.join(demo_dir, '{}_{}'.format(env_name, demo_num))
#     assert not os.path.isfile(demo_file)

#     print('saving demo_16 to: {}'.format(demo_file))

#     with open(demo_file, 'wb') as f:
#         pkl.dump(all_steps, f)

#     print('saved')


# # save demo_16 as a pkl file
# def save_snapshots(env_steps, model_name='', date=''):
#     dir = '../snapshots'
#     if not os.path.isdir(dir):
#         os.mkdir(dir)
#     snapshot_file = os.path.join(dir, f'{model_name}_{date}')

#     print('saving snapshot to: {}'.format(snapshot_file))

#     # hf = h5py.File('{demo_file}.h5', 'w')

#     with open(snapshot_file, 'wb') as f:
#         pkl.dump(env_steps, f)

#     print('saved')


# def open_demo(demo_file):
#     assert os.path.isfile(demo_file)

#     with open(demo_file, 'rb') as f:
#         demo = pkl.load(f)
#         print('num_steps in demo_16: {}'.format(len(demo)))
#         return demo


# def get_step_num(step_num, demo_file):
#     # returns dict with keys: action, grid, agent_carrying, agent_pos
#     demo = open_demo(demo_file)
#     return demo[step_num]


# def get_action_num(step_num, demo_file):
#     demo = open_demo(demo_file)
#     action = demo[step_num]['action']
#     print('action at step {}: {}'.format(step_num, action.name))
#     return action


# def get_states(step_num, demo_file):
#     demo = open_demo(demo_file)
#     states = demo[step_num]['states']
#     return states


# def print_actions_states(demo_file):
#     demo = open_demo(demo_file)
#     for step_num in demo:
#         print('{}: {}'.format(step_num,  demo[step_num]['action']))
#         print('true predicates')
#         for state in demo[step_num]['predicates']:
#             if demo[step_num]['predicates'][state]:
#                 print(state)


# def print_actions(demo_file):
#     demo = open_demo(demo_file)
#     for step_num in demo:
#         print('{}: {}'.format(step_num,  demo[step_num]['action']))


# # demo_file = '/Users/emilyjin/Code/behavior/demos/MiniGrid-ThrowLeftoversFourRooms-8x8-N2-v1/2/MiniGrid-ThrowLeftoversFourRooms-8x8-N2-v1_10'
# # print_actions_states(demo_file)
