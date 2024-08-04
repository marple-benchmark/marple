import json
import os
import numpy as np

def load_json(filename):
    path = os.path.abspath(filename)
    with open(path, 'r') as f:
        return json.load(f)


# def load_npy(filepath, env):
#     """
#     return current (grid_w, grid_h, 8) array
#     """
#     init_npy = np.load(filepath)  
    
#     width = env.grid.width
#     height = env.grid.height

#     #

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
#                     # stateconda 
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
#         array[ymin:ymax, xmin:xmax, agent_dir_ch] = agent.agent_dir

#     return array