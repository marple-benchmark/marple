import os
import math
import names
import numpy as np
from marple_mini_behavior.mini_behavior.minibehavior import MiniBehaviorEnv
from marple_mini_behavior.mini_behavior.grid import BehaviorGrid
from marple_mini_behavior.mini_behavior.register import register
from marple_mini_behavior.mini_behavior.objects import *
from marple_mini_behavior.mini_behavior.agents import *
from marple_mini_behavior.mini_behavior.rooms import *
from marple_mini_behavior.bddl import *
from marple_mini_behavior.mini_behavior.utils.scene_to_grid import img_to_array
from .utils.globals import *
from marple_mini_behavior.mini_behavior.missions import *

# generate grid
class FloorPlanEnv(MiniBehaviorEnv):
    def __init__(self,
                 mode='human',
                 initial_dict=None,
                 scene_id='beechwood_0_int',
                 max_steps=1e5,
                 from_secne=False,
                 ):

        if from_secne:
            self.scene_id = scene_id
            img_path = get_floorplan(scene_id)
            self.img_path = img_path
            # assume img_path is to grid version of floorplan
            self.floor_plan = img_to_array(img_path)
            self.height, self.width = np.shape(self.floor_plan)
        else:
            self.initial_dict = initial_dict
            self.height = initial_dict['Grid']['height']
            self.width = initial_dict['Grid']['width']
        self.room_size = self.height * self.width
        self.num_cols = self.width
        self.num_rows = self.height

        super().__init__(mode=mode,
                         width=self.width,
                         height=self.height,
                         max_steps=max_steps)

    def add_walls(self, width, height):
        # Generate the surrounding walls
        self.objs = {}
        self.obj_instances = {}
        self.grid.wall_rect(0, 0, width, height)
        for dir, (x, y), length in self.floor_plan_walls:
            if dir == 'horz':
                self.grid.horz_wall(x, y, length)
            else:
                self.grid.vert_wall(x, y, length)

        obj_type = "door"
        if obj_type not in self.objs:
            self.objs[obj_type] = []
        if 'doors' in self.initial_dict['Grid']:
            num_door = self.initial_dict['Grid']['doors']['num']
            for idx in range(num_door):
                obj_name = self.initial_dict['Grid']['doors']['initial'][idx]['name']
                obj_id = self.initial_dict['Grid']['doors']['initial'][idx]['id']
                obj_dir = self.initial_dict['Grid']['doors']['initial'][idx]['dir']
                pos = self.initial_dict['Grid']['doors']['initial'][idx]['pos']
                state = self.initial_dict['Grid']['doors']['initial'][idx]['state']
                door = Door(obj_dir, width=1, height=1, is_open=state=='open')
                door.name = obj_name
                door.dir = obj_dir
                door.id = obj_id
                self.objs[obj_type].append(door)
                self.obj_instances[obj_name] = door
                # all_items = self.grid.get_all_items(*pos)
                # print(all_items)
                self.place_obj(door, (pos[0], pos[1]), (door.width, door.height))
        else: # auto generate doors
            door_count = 0
            for dir, (x, y), length in self.floor_plan_walls:
                if door_count >= 3:
                    break
                if dir == 'horz' and 0 < y < height - 1:
                    open_status = self._rand_bool()
                    # open_status = False
                    if obj_type not in self.objs:
                        self.objs[obj_type] = []
                    obj_name = f'{obj_type}_{len(self.objs[obj_type])}'
                    door = Door(dir='horz', width=1, height=1,
                                is_open=open_status)
                    self.objs[obj_type].append(door)
                    door.id = len(self.obj_instances) + 1
                    self.obj_instances[obj_name] = door
                    self.place_obj(door, (x, y), (length, 1))
                    door_count += 1
                elif dir == 'vert' and 0 < x < width - 1:
                    open_status = self._rand_bool()
                    # open_status = False
                    if obj_type not in self.objs:
                        self.objs[obj_type] = []
                    obj_name = f'{obj_type}_{len(self.objs[obj_type])}'
                    door = Door(dir='vert', width=1, height=1,
                                is_open=open_status)
                    self.objs[obj_type].append(door)
                    door.id = len(self.obj_instances) + 1
                    self.obj_instances[obj_name] = door
                    # print((x, y + 1), (1, height - 3 - y))
                    # try:
                    self.place_obj(door, (x, y + 1), (1, length))
                    # except:
                        # print('could not place door', x, y+1, 1, length)
                    # self.grid.set(x, doorIdx, door)
                    door_count += 1
                else:
                    continue

    def _gen_grid(self, width, height):
        self.grid = BehaviorGrid(width, height)
        self._gen_floorplan()
        self._gen_random_objs()
        self._gen_agents()
        assert self._init_conditions(), "Does not satisfy initial conditions"
        self.place_agent()
        # self.set_agent() # set agent to the one with mission TODO: debug

    def _gen_agents(self):
        self.agents = []
        if self.initial_dict['Grid']['agents']['num'] is not None:
            num_agent = self.initial_dict['Grid']['agents']['num']
        else:
            num_agent = self._rand_int(
                1, self.auto_max_num_agent)
        agent_initial = self.initial_dict['Grid']['agents']['initial']
        agent_colors = set()
        for num in range(num_agent):
            if num < len(agent_initial):
                if 'id' in agent_initial[num] and agent_initial[num]['id'] is not None:
                    agent_id = agent_initial[num]['id']
                else:
                    agent_id = num + 1
                if agent_initial[num]['gender'] is not None:
                    gender = agent_initial[num]['gender']
                else:
                    gender = self._rand_subset(GENDER, 1)[0]
                if 'name' in agent_initial[num] and agent_initial[num]['name'] is not None:
                    name = agent_initial[num]['name']
                else:
                    name = names.get_full_name(gender=gender)
                if agent_initial[num].get('color', None) is not None:
                    color = agent_initial[num]['color']
                else:
                    color = self._rand_subset(COLOR_TO_IDX.keys(), 1)[0]
                    while color in agent_colors or color == 'black':
                        color = self._rand_subset(COLOR_TO_IDX.keys(), 1)[0]
                    agent_colors.add(color)
                if agent_initial[num].get('step_size', None) is not None:
                    step_size = agent_initial[num]['step_size']
                else:
                    # step_size = self._rand_subset(STEP_SIZE, 1)[0]
                    step_size = 1
                if agent_initial[num].get('mission_preference_initial', None) is not None:
                    mission_preference = agent_initial[num]['mission_preference_initial']
                else:
                    # mission_preference = None
                    raise NotImplementedError("Didn't specify agent mission preference")
                if agent_initial[num].get('forgetness', None) is not None:
                    forgetness = agent_initial[num]['forgetness']
                else:
                    forgetness = random.random()
                if agent_initial[num].get('cur_mission', None) is not None:
                    cur_mission = agent_initial[num]['cur_mission']
                else:
                    cur_mission = None
                if agent_initial[num].get('cur_subgoal', None) is not None:
                    cur_subgoal = agent_initial[num]['cur_subgoal']
                else:
                    cur_subgoal = None
                if agent_initial[num]['carrying'] is not None:
                    carrying_initial = agent_initial[num]['carrying']['initial']
                    if agent_initial[num]['carrying']['num'] is not None:
                        num_carrying = agent_initial[num]['carrying']['num']
                    else:
                        num_carrying = 0
                else:
                    num_carrying = 0
                    carrying_initial = []
            else:
                gender = self._rand_subset(GENDER, 1)[0]
                # mission_preference = dict(zip(MISSION_TO_SUBGOALS.keys(), self._rand_distribution(
                #     len(MISSION_TO_SUBGOALS))))
                name = names.get_full_name(gender=gender)
                agent_id = num + 1
                step_size = self._rand_subset(STEP_SIZE, 1)[0]
                color = self._rand_subset(COLOR_TO_IDX.keys(), 1)[0]
                while color in agent_colors or color == 'black':
                    color = self._rand_subset(COLOR_TO_IDX.keys(), 1)[0]
                forgetness = random.random()
                cur_mission = None
                cur_subgoal = None
                num_carrying = 0
                carrying_initial = []

            agent = Agent(name)
            self.agents.append(agent)
            agent.set_step_size(step_size)
            agent.set_color(color)
            agent.set_gender(gender)
            agent.set_mission_preference(mission_preference)
            agent.set_forgetness(forgetness)
            agent.cur_mission = cur_mission
            agent.cur_subgoal = cur_subgoal
            agent.set_id(agent_id)

            # generate carrying objects
            for obj_id in range(num_carrying):
                if obj_id < len(carrying_initial) and carrying_initial[obj_id]['type'] is not None:
                    obj_type = carrying_initial[obj_id]['type']
                else:
                    object_types = set(OBJECT_TO_IDX.keys()) - set(OBJECT_CLASS.keys())
                    obj_type = self._rand_subset(object_types, 1)[0]

                if obj_id < len(carrying_initial) and carrying_initial[obj_id]['id'] is not None:
                    obj_exist_id = carrying_initial[obj_id]['id']
                else:
                    obj_exist_id = len(self.obj_instances) + 1

                if obj_id < len(carrying_initial) and carrying_initial[obj_id]['name'] is not None:
                    obj_name = carrying_initial[obj_id]['name']
                else:
                    obj_name = f'{obj_type}_{obj_exist_id}_carrying'

                if obj_type not in self.objs:
                    self.objs[obj_type] = []
            
                # generate obj instance
                # print("generate obj instance")
                if obj_type in OBJECT_CLASS.keys():
                    obj_instance = OBJECT_CLASS[obj_type](
                        name=obj_name)
                else:
                    obj_instance = WorldObj(
                        obj_type, None, obj_name)
                obj_instance.in_room = None
                obj_instance.on_fur = None

                self.objs[obj_type].append(obj_instance)
                obj_instance.id = obj_exist_id
                self.obj_instances[obj_name] = obj_instance
                # set obj pos
                obj_instance.update_pos((-1, -1))
                # set obj states
                if obj_id < len(carrying_initial) and carrying_initial[obj_id]['state'] is not None:
                    states = carrying_initial[obj_id]['state']
                    for state_name, flag in states.items():
                        if flag:
                            obj_instance.states[state_name].set_value(
                                True)
                else:
                    num_ability = self._rand_int(0, len(ABILITIES))
                    abilities = self._rand_subset(
                        ABILITIES, num_ability)
                    for ability in abilities:
                        if ability in list(obj_instance.states.keys()):
                            obj_instance.states[ability].set_value(
                                True)
                obj_instance.update(self)
                agent.carrying.add(obj_instance)

    def _gen_floorplan(self):
        # floor plan
        self.floor_plan_walls = []
        # rooms
        self.rooms = {}
        self.room_instances = []

        # num rooms
        if self.initial_dict['Grid']['rooms']['num'] is not None:
            num_room = self.initial_dict['Grid']['rooms']['num']
        else:
            num_room = self._rand_int(1, self.auto_max_num_room)

        room_initial = self.initial_dict['Grid']['rooms']['initial']
        # num_room = max(num_room, len(room_initial))

        # check if generate floor plan automatically
        auto_floor_plan = True
        if room_initial and room_initial[0]['top'] is not None:
            assert len(
                room_initial) == num_room, 'Please specify all room initials, tops and size'
            for room_init in room_initial:
                if room_init['top'] is None or room_init['size'] is None:
                    raise Exception(
                        'Please specify all room tops and sizes')
            auto_floor_plan = False

        if auto_floor_plan:
            tops, sizes = self._gen_random_floorplan(num_room)

        for num in range(num_room):
            if num < len(room_initial):
                if room_initial[num]['type'] is not None:
                    room_type = room_initial[num]['type']
                else:
                    room_type = self._rand_subset(ROOM_TYPE, 1)[0]
                if room_type not in self.rooms:
                    self.rooms[room_type] = []

                if 'id' in room_initial[num] and room_initial[num]['id'] is not None:
                    room_exist_id = room_initial[num]['id']
                else:
                    room_exist_id = num + 1

                if 'name' in room_initial[num] and room_initial[num]['name'] is not None:
                    room_name = room_initial[num]['name']
                else:
                    room_name = f"{room_type}_{room_exist_id}"

                if auto_floor_plan:
                    top = tops[num]
                    size = sizes[num]
                else:
                    top = room_initial[num]['top']
                    size = room_initial[num]['size']

                    # if top[0] - 1 not in [0, self.width - 1]:
                    #     self.floor_plan_walls.append(
                    #         ('vert', (top[0] - 1, top[1]), size[1] + 1))
                    # if top[0] + size[0] not in [0, self.width - 1]:
                    #     self.floor_plan_walls.append(
                    #         ('vert', (top[0] + size[0], top[1] - 1), size[1] + 1))
                    # if top[1] - 1 not in [0, self.height - 1]:
                    #     self.floor_plan_walls.append(
                    #         ('horz', (top[0] - 1, top[1] - 1), size[0] + 1))
                    # if top[1] + size[1] not in [0, self.height - 1]:
                    #     self.floor_plan_walls.append(
                    #         ('horz', (top[0], top[1] + size[1]), size[0] + 1))

# # """
# # TODO: ALSO NEW
# #                     if top[0] - 1 not in [0, self.width - 1]:
# #                         self.floor_plan_walls.append(
# #                             ('vert', (top[0] - 1, top[1]), size[1] + 1))
# #                     if top[0] + size[0] not in [0, self.width - 1]:
# #                         self.floor_plan_walls.append(
# #                             ('vert', (top[0] + size[0], top[1] - 1), size[1] + 1))
# #                     if top[1] - 1 not in [0, self.height - 1]:
# #                         self.floor_plan_walls.append(
# #                             ('horz', (top[0] - 1, top[1] - 1), size[0] + 1))
# #                     if top[1] + size[1] not in [0, self.height - 1]:
# #                         self.floor_plan_walls.append(
# #                             ('horz', (top[0], top[1] + size[1]), size[0] + 1))
# #             else:
# #                 room_exist_id = num + 1
# #                 room_name = f"room_{room_exist_id}"
# # """
#                     self.floor_plan_walls.append(
#                         ('vert', (top[0] - 1, top[1]), size[1] + 1))
#                     self.floor_plan_walls.append(
#                         ('vert', (top[0] + size[0], top[1] - 1), size[1] + 1))
#                     self.floor_plan_walls.append(
#                         ('horz', (top[0] - 1, top[1] - 1), size[0] + 1))
#                     self.floor_plan_walls.append(
#                         ('horz', (top[0], top[1] + size[1]), size[0] + 1))
            else:
                # create a random room
                room_exist_id = num + 1
                room_name = f"room_{room_exist_id}"
                room_type = self._rand_subset(ROOM_TYPE, 1)[0]
                top = tops[num]
                size = sizes[num]

            room = Room(room_exist_id, room_type, room_name, top, size)
            self.rooms[room_type].append(room)
            self.room_instances.append(room)
            
            # add walls if not auto generate floor plan, and only add walls for rooms except the last one
            if not auto_floor_plan and num != num_room - 1:
                self.floor_plan_walls.append(
                    ('vert', (top[0] - 1, top[1]), size[1] + 1))
                self.floor_plan_walls.append(
                    ('vert', (top[0] + size[0], top[1] - 1), size[1] + 1))
                self.floor_plan_walls.append(
                    ('horz', (top[0] - 1, top[1] - 1), size[0] + 1))
                self.floor_plan_walls.append(
                    ('horz', (top[0], top[1] + size[1]), size[0] + 1))

        # print("add last room", room_exist_id, room_type, room_name, top, size)

        # # add last room
        # num = num_room - 1
        # if num < len(room_initial):
        #     # print("add last room", num, room_initial[num])
        #     if room_initial[num]['type'] is not None:
        #         room_type = room_initial[num]['type']
        #     else:
        #         room_type = self._rand_subset(ROOM_TYPE, 1)[0]

        #     if room_type not in self.rooms:
        #         self.rooms[room_type] = []

        #     # print("get room type", room_type)
        #     if 'id' in room_initial[num] and room_initial[num]['id'] is not None:
        #         room_exist_id = room_initial[num]['id']
        #     else:
        #         room_exist_id = num + 1

        #     # print("get room id", room_exist_id)
        #     if 'name' in room_initial[num] and room_initial[num]['name'] is not None:
        #         room_name = room_initial[num]['name']
        #     else:
        #         room_name = f"{room_type}_{room_exist_id}"

        #     # print("get room name", room_name)

        #     if auto_floor_plan:
        #         top = tops[num]
        #         size = sizes[num]
        #     else:
        #         top = room_initial[num]['top']
        #         size = room_initial[num]['size']

        # else:
        #     room_exist_id = num + 1
        #     room_name = f"room_{room_exist_id}"
        #     room_type = self._rand_subset(ROOM_TYPE, 1)[0]
        #     top = tops[num]
        #     size = sizes[num]

        # # print("add last room", room_exist_id, room_type, room_name, top, size)

        # room = Room(room_exist_id, room_type, room_name, top, size)
        # self.rooms[room_type].append(room)
        # self.room_instances.append(room)

        # add last room
        # num += 1
        # if num < len(room_initial):
        #     # print("add last room", num, room_initial[num])
        #     if room_initial[num]['type'] is not None:
        #         room_type = room_initial[num]['type']
        #     else:
        #         room_type = self._rand_subset(ROOM_TYPE, 1)[0]

        #     if room_type not in self.rooms:
        #         self.rooms[room_type] = []

        #     # print("get room type", room_type)
        #     if 'id' in room_initial[num] and room_initial[num]['id'] is not None:
        #         room_exist_id = room_initial[num]['id']
        #     else:
        #         room_exist_id = num + 1

        #     # print("get room id", room_exist_id)
        #     if 'name' in room_initial[num] and room_initial[num]['name'] is not None:
        #         room_name = room_initial[num]['name']
        #     else:
        #         room_name = f"{room_type}_{room_exist_id}"

        #     # print("get room name", room_name)

        #     if auto_floor_plan:
        #         top = tops[num]
        #         size = sizes[num]
        #     else:
        #         top = room_initial[num]['top']
        #         size = room_initial[num]['size']

        # else:
        #     room_exist_id = num + 1
        #     room_name = f"room_{room_exist_id}"
        #     room_type = self._rand_subset(ROOM_TYPE, 1)[0]
        #     top = tops[num]
        #     size = sizes[num]

        # # print("add last room", room_exist_id, room_type, room_name, top, size)

        # room = Room(room_exist_id, room_type, room_name, top, size)
        # self.rooms[room_type].append(room)
        # self.room_instances.append(room)

    def _gen_random_objs(self):
        # objs
        self.objs = {}
        self.obj_instances = {}

        # TODO: debug, clean
        # print(self.floor_plan_walls)
        self.add_walls(self.width, self.height)

        for room_id, room in enumerate(self.room_instances):
            # Generate furs and objs for the room
            # print("Generate furs and objs for the room")
            if room_id < len(self.initial_dict['Grid']['rooms']['initial']) and self.initial_dict['Grid']['rooms']['initial'][room_id]['furnitures']['num'] is not None:
                num_furs = self.initial_dict['Grid']['rooms']['initial'][room_id]['furnitures']['num']
            else:
                num_furs = self._rand_int(
                    1, max(2, int(room.size[0]*room.size[1]/12)))

            if room_id < len(self.initial_dict['Grid']['rooms']['initial']) and self.initial_dict['Grid']['rooms']['initial'][room_id]['furnitures'] is not None:
                furniture_initial = self.initial_dict['Grid']['rooms']['initial'][room_id]['furnitures']['initial']
            else:
                furniture_initial = []

            num_furs = max(num_furs, len(furniture_initial))
            room.num_furs = num_furs
            room.self_id = room_id

            for fur_id in range(num_furs):
                if fur_id < len(furniture_initial) and furniture_initial[fur_id]['type'] is not None:
                    furniture_type = furniture_initial[fur_id]['type']
                else:
                    furniture_type = self._rand_subset(
                        ROOM_FURNITURE_MAP[room.type], 1)[0]
                if furniture_type not in self.objs:
                    self.objs[furniture_type] = []
                
                if fur_id < len(furniture_initial) and 'id' in furniture_initial[fur_id] and furniture_initial[fur_id]['id'] is not None:
                    fur_exist_id = furniture_initial[fur_id]['id']
                else:
                    fur_exist_id = len(self.obj_instances) + 1

                if fur_id < len(furniture_initial) and 'name' in furniture_initial[fur_id] and furniture_initial[fur_id]['name'] is not None:
                    fur_name = furniture_initial[fur_id]['name']
                else:
                    fur_name = f'{furniture_type}_{fur_exist_id}'
                

                # generate furniture instance
                if furniture_type in OBJECT_CLASS.keys():
                    fur_instance = OBJECT_CLASS[furniture_type](
                        name=fur_name)
                else:
                    fur_instance = WorldObj(furniture_type, None, fur_name)

                fur_instance.in_room = room
                self.objs[furniture_type].append(fur_instance)
                fur_instance.id = fur_exist_id
                self.obj_instances[fur_name] = fur_instance
                room.furnitures.append(fur_instance)

                # place furniture
                # print("place furniture", fur_instance.name)
                if fur_id < len(furniture_initial) and furniture_initial[fur_id]['pos'] is not None:
                    pos = self.place_obj_pos(
                        fur_instance, pos=furniture_initial[fur_id]['pos'], top=room.top, size=room.size)
                else:
                    # try:
                    pos = self.place_obj(fur_instance, room.top, room.size)
                    # except:
                    #     breakpoint()
                        # print('unable to place obj', fur_instance.name, room.top, room.size)

                # set furniture states
                # print('set furniture states')
                for ability in ABILITIES:
                    if ability in list(fur_instance.states.keys()):
                        if fur_id < len(furniture_initial) and furniture_initial[fur_id]['state'] is not None \
                                and ability in furniture_initial[fur_id]['state']:
                            if furniture_initial[fur_id]['state'][ability] == 1:
                                fur_instance.states[ability].set_value(True)
                            elif furniture_initial[fur_id]['state'][ability] == 0:
                                fur_instance.states[ability].set_value(False)
                        else:
                            state_value = self._rand_bool()
                            if state_value:
                                fur_instance.states[ability].set_value(True)
                            else:
                                fur_instance.states[ability].set_value(False)

                fur_instance.update(self)
                # for every furniture, generate objs on it
                # print("for every furniture, generate objs on it")
                if furniture_type not in FURNITURE_CANNOT_ON and fur_instance.width * fur_instance.height > 1:
                    if fur_id < len(furniture_initial) and furniture_initial[fur_id]['objs'] is not None:
                        if furniture_initial[fur_id]['objs']['num'] > 0:
                            obj_initial = furniture_initial[fur_id]['objs']['initial']
                            if furniture_initial[fur_id]['objs']['num'] is not None:
                                num_objs = furniture_initial[fur_id]['objs']['num']
                            else:
                                num_objs = max(len(obj_initial), self._rand_int(
                                    1, max(1, fur_instance.width * fur_instance.height)))
                        else:
                            num_objs = 0
                            obj_initial = []
                    else:
                        num_objs = self._rand_int(
                            1, max(1, fur_instance.width * fur_instance.height))
                        obj_initial = []
                    
                    num_objs = max(num_objs, len(obj_initial))

                    obj_poses = set()
                    for obj_id in range(num_objs):
                        if obj_id < len(obj_initial) and obj_initial[obj_id]['type'] is not None:
                            obj_type = obj_initial[obj_id]['type']
                        else:
                            obj_type = self._rand_subset(
                                ROOM_OBJ_MAP[room.type], 1)[0]
                        if obj_type not in self.objs:
                            self.objs[obj_type] = []

                        if obj_id < len(obj_initial) and 'id' in obj_initial[obj_id] and obj_initial[obj_id]['id'] is not None:
                            obj_exist_id = obj_initial[obj_id]['id']
                        else:
                            obj_exist_id = len(self.obj_instances) + 1

                        if obj_id < len(obj_initial) and 'name' in obj_initial[obj_id] and obj_initial[obj_id]['name'] is not None:
                            obj_name = obj_initial[obj_id]['name']
                        else:
                            obj_name = f'{obj_type}_{obj_exist_id}'

                        # generate obj instance
                        if obj_type in OBJECT_CLASS.keys():
                            obj_instance = OBJECT_CLASS[obj_type](
                                name=obj_name)
                        else:
                            obj_instance = WorldObj(
                                obj_type, None, obj_name)
                        obj_instance.in_room = room
                        obj_instance.on_fur = fur_instance

                        self.objs[obj_type].append(obj_instance)
                        obj_instance.id = obj_exist_id
                        self.obj_instances[obj_name] = obj_instance
                        fur_instance.objects.append(obj_instance)
                        room.objs.append(obj_instance)

                        # put obj instance
                        # print("put obj instance", obj_instance.name)
                        if obj_id < len(obj_initial) and obj_initial[obj_id]['pos'] is not None:
                            pos = obj_initial[obj_id]['pos']
                            self.put_obj(obj_instance, *pos)
                            obj_poses.add((pos[0], pos[1]))
                        else:
                            pos = self._rand_subset(
                                fur_instance.all_pos, 1)[0]
                            while pos in obj_poses:
                                pos = self._rand_subset(
                                    fur_instance.all_pos, 1)[0]
                            # print(f"obj instance {obj_instance}, pos {pos}")
                            self.put_obj(obj_instance, *pos)
                            obj_poses.add((pos[0], pos[1]))

                        # set obj states
                        for ability in ABILITIES:
                            if ability in list(obj_instance.states.keys()):
                                if obj_id < len(obj_initial) and obj_initial[obj_id]['state'] is not None \
                                        and ability in obj_initial[obj_id]['state']:
                                    if obj_initial[obj_id]['state'][ability] == 1:
                                        obj_instance.states[ability].set_value(
                                            True)
                                    elif obj_initial[obj_id]['state'][ability] == 0:
                                        obj_instance.states[ability].set_value(
                                            False)
                                else:
                                    state_value = self._rand_bool()
                                    if state_value:
                                        obj_instance.states[ability].set_value(
                                            True)
                                    else:
                                        obj_instance.states[ability].set_value(
                                            False)

                        obj_instance.update(self)

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """
        for idx, agent in enumerate(self.agents):
            agent.agent_pos = None
            if idx < len(self.initial_dict['Grid']['agents']['initial']):
                if self.initial_dict['Grid']['agents']['initial'][idx]['pos'] is None:
                    pos = self.place_obj(None, top, size, max_tries=max_tries)
                    agent.agent_pos = (pos[0], pos[1])
                else:
                    pos = self.initial_dict['Grid']['agents']['initial'][idx]['pos']
                    self.place_obj_pos(None, pos, top, size)
                    agent.agent_pos = (pos[0], pos[1])

                if idx >= len(self.initial_dict['Grid']['agents']['initial']) or \
                    idx < len(self.initial_dict['Grid']['agents']['initial']) and \
                        self.initial_dict['Grid']['agents']['initial'][idx]['dir'] is None:
                    agent.agent_dir = self._rand_int(0, 4)
                else:
                    agent.agent_dir = self.initial_dict['Grid']['agents']['initial'][idx]['dir']
            else:
                pos = self.place_obj(None, top, size, max_tries=max_tries)
                agent.agent_pos = (pos[0], pos[1])
                agent.agent_dir = self._rand_int(0, 4)

            self.cur_agent_id = idx
            self.agent_pos = agent.agent_pos
            self.agent_dir = agent.agent_dir
            self.agent_color = agent.agent_color

            # print("placed agent", self.agent_pos)

        return pos

    def set_agent(self):
        agent = [agent for agent in self.initial_dict['Grid']['agents']['initial'] if agent['cur_mission'] is not None][0]
        self.cur_agent_id = agent['id'] - 1
        self.agent_pos = agent['pos']
        self.agent_dir = agent['dir']
        self.agent_color = agent['color']

    def _gen_objs(self):
        goal = self.objs['goal'][0]
        self.target_pos = self.place_obj(goal)

    def _reward(self):
        if self._end_conditions():
            return 1
        else:
            return 0

    def room_from_pos(self, x, y):
        """Get the room a given position maps to"""

        assert x >= 0
        assert y >= 0

        i = x // (self.room_size-1)
        j = y // (self.room_size-1)

        assert i < self.num_cols
        assert j < self.num_rows

        return self.room_grid[j][i]

    def _end_conditions(self):
        pass
        # if np.all(self.agent_pos == self.target_pos):
        #     return True
        # else:
        #     return False
