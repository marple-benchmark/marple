# MODIFIED FROM MINIGRID REPO

import os
import pickle as pkl
from enum import IntEnum
from xml.dom import ValidationErr
from gym import spaces
from gym_minigrid.minigrid import MiniGridEnv

import sys
from marple_mini_behavior.bddl.actions import ACTION_FUNC_MAPPING
from .objects import *
from .agents import *
from .audios import *
from .grid import BehaviorGrid, GridDimension, is_obj
from marple_mini_behavior.mini_behavior.window import Window
# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32


class MiniBehaviorEnv(MiniGridEnv):
    """
    2D grid world game environment
    """
    metadata = {
        # Deprecated: use 'render_modes' instead
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,  # Deprecated: use 'render_fps' instead
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 10,
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        pickup = 3
        drop = 4
        toggle = 6
        open = 7
        close = 8
        slice = 9
        cook = 10

    def __init__(
        self,
        mode='not_human',
        grid_size=None,
        width=None,
        height=None,
        max_steps=1e5,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7,
        highlight=True,
        tile_size=TILE_PIXELS,
    ):
        self.episode = 0
        self.mode = mode
        self.last_action = None
        self.action_done = None

        self.render_dim = None

        self.highlight = highlight
        self.tile_size = tile_size

        # Initialize the RNG
        # self.seed(seed=seed)
        self.furniture_view = None

        super().__init__(grid_size=grid_size,
                         width=width,
                         height=height,
                         max_steps=max_steps,
                         see_through_walls=see_through_walls,
                         agent_view_size=agent_view_size,
                         )

        self.grid = BehaviorGrid(width, height)

        # Action enumeration for this environment, actions are discrete int
        self.actions = MiniBehaviorEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.carrying = set()

    def switch_agent(self, agent_id=None):
        if agent_id is not None:
            self.cur_agent_id = agent_id
        else:
            cur_agent = self.agents[self.cur_agent_id]
            cur_agent.agent_dir = self.agent_dir
            cur_agent.agent_pos = self.agent_pos
            cur_agent.carrying = self.carrying
            self.cur_agent_id = (self.cur_agent_id + 1) % len(self.agents)
        new_agent = self.agents[self.cur_agent_id]
        self.agent_dir = new_agent.agent_dir
        self.agent_pos = new_agent.agent_pos
        self.agent_color = new_agent.agent_color
        self.carrying = new_agent.carrying

    def set_agent(self, i):
        if i < len(self.agents):
            self.cur_agent_id = i
            new_agent = self.agents[self.cur_agent_id]
            self.agent_dir = new_agent.agent_dir
            self.agent_pos = new_agent.agent_pos
            self.agent_color = new_agent.agent_color
            self.carrying = new_agent.carrying

    def copy_objs(self):
        from copy import deepcopy
        return deepcopy(self.objs), deepcopy(self.obj_instances)

    def copy_rooms(self):
        from copy import deepcopy
        return deepcopy(self.rooms), deepcopy(self.room_instances)

    # TODO: check this works
    def load_objs(self, state):
        obj_instances = state['obj_instances']
        grid = state['grid']
        for obj in self.obj_instances.values():
            if type(obj) != Wall and type(obj) != Door:
                load_obj = obj_instances[obj.name]
                obj.load(load_obj, grid, self)

        for obj in self.obj_instances.values():
            obj.contains = []
            for other_obj in self.obj_instances.values():
                if other_obj.check_rel_state(self, obj, 'inside'):
                    obj.contains.append(other_obj)

    # TODO: check this works
    def get_state(self):
        grid = self.grid.copy()
        agent_poses = []
        agent_dirs = []
        agent_colors = []
        for agent in self.agents:
            agent_poses.append(agent.agent_pos)
            agent_dirs.append(agent.agent_dir)
            agent_colors.append(agent.agent_color)
        rooms, room_instances = self.copy_rooms()
        objs, obj_instances = self.copy_objs()
        state = {'grid': grid,
                 'agent_poses': agent_poses,
                 'agent_dirs': agent_dirs,
                 'agent_colors': agent_colors,
                 'objs': objs,
                 'obj_instances': obj_instances,
                 'rooms': rooms,
                 'room_instances': room_instances,
                 }
        return state

    def save_state(self, out_file='cur_state.pkl'):
        state = self.get_state()
        with open(out_file, 'wb') as f:
            pkl.dump(state, f)
            print(f'saved to: {out_file}')

    # TODO: check this works
    def load_state(self, load_file):
        assert os.path.isfile(load_file)
        with open(load_file, 'rb') as f:
            state = pkl.load(f)
            self.load_objs(state)
            self.grid.load(state['grid'], self)
            self.agent_pos = state['agent_pos']
            self.agent_dir = state['agent_dir']
            self.agent_color = state['agent_color']
        return self.grid

    def reset(self):
        if self.load_init:
            self.step_count = 0
            self.episode = 0

            # Return first observation
            # self.agent_dir = self.initial_dict['Grid']['agents']['initial'][0]['dir']
            # self.agent_pos = self.initial_dict['Grid']['agents']['initial'][0]['pos']
            obs = self.gen_obs()
            return obs

        else:
            # Reinitialize episode-specific variables
            self.agent_pos = (-1, -1)
            self.agent_dir = -1

            # for agent in self.agents:
            #     agent.agent_pos = (-1, -1)
            #     agent.agent_dir = -1

            self.carrying = set()
            if self.obj_instances:
                for obj in self.obj_instances.values():
                    obj.reset()

            self.reward = 0

            # Generate a new random grid at the start of each episode
            # To keep the same grid for each episode, call env.seed() with
            # the same seed before calling env.reset()
            self._gen_grid(self.width, self.height)

            # generate furniture view
            self.furniture_view = self.grid.render_furniture(
                tile_size=TILE_PIXELS, obj_instances=self.obj_instances)

            # These fields should be defined by _gen_grid
            assert self.agent_pos is not None
            assert self.agent_dir is not None

            # Check that the agent doesn't overlap with an object
            assert self.grid.is_empty(*self.agent_pos)

#         else:
#             # Reinitialize episode-specific variables
#             self.agent_pos = (-1, -1)
#             self.agent_dir = -1

#             self.carrying = set()
#             if self.obj_instances:
#                 for obj in self.obj_instances.values():
#                     obj.reset()

#             self.reward = 0

#             # Generate a new random grid at the start of each episode
#             # To keep the same grid for each episode, call env.seed() with
#             # the same seed before calling env.reset()
#             self._gen_grid(self.width, self.height)

#             # generate furniture view
#             self.furniture_view = self.grid.render_furniture(
#                 tile_size=TILE_PIXELS, obj_instances=self.obj_instances)

#             # These fields should be defined by _gen_grid
#             assert self.agent_pos is not None
#             assert self.agent_dir is not None

#             # Check that the agent doesn't overlap with an object
#             assert self.grid.is_empty(*self.agent_pos)

            # Step count since episode start
            self.step_count = 0
            self.episode += 1

            # Return first observation
            obs = self.gen_obs()
            return obs

    def _gen_grid(self, width, height):
        self._gen_objs()
        # assert self._init_conditions(), "Does not satisfy initial conditions"
        self.place_agent()

    def _gen_objs(self):
        assert False, "_gen_objs needs to be implemented by each environment"

    def _init_conditions(self):
        # print('no init conditions')
        return True

    def _end_conditions(self):
        # print('no end conditions')
        return False

    def place_obj_pos(self,
                      obj,
                      pos,
                      top=None,
                      size=None,
                      reject_fn=None
                      ):
        """
        Place an object at a specific position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param obj: the object to place
        :param pos: the top left of the pos we want the object to be placed
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        width = 1 if obj is None else obj.width
        height = 1 if obj is None else obj.height

        valid = True

        if pos[0] < top[0] or pos[0] > min(top[0] + size[0], self.grid.width - width + 1)\
                or pos[1] < top[1] or pos[1] > min(top[1] + size[1], self.grid.height - height + 1):
            raise ValidationErr(f'position {pos} not in grid')

        for dx in range(width):
            for dy in range(height):
                x = pos[0] + dx
                y = pos[1] + dy

                # Don't place the object on top of another object
                if not self.grid.is_empty(x, y):
                    print('grid is not empty', self.grid.get(x,y))
                    break

                # Don't place the object where the agent is
                if np.array_equal((x, y), self.agent_pos):
                    valid = False
                    print('agent is there')
                    break

                # Check if there is a filtering criterion
                if reject_fn and reject_fn(self, (x, y)):
                    valid = False
                    print('failed filtering criterion')
                    break

        if not valid:
            raise ValidationErr(f'failed in place_obj at {pos}')

        self.grid.set(*pos, obj)

        if obj:
            self.put_obj(obj, *pos)

        return pos

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=100
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError(
                    f'rejection sampling failed in place_obj, top: {top}, size: {size}, \
                        width: {width}, height: {height}')

            num_tries += 1

            width = 1 if obj is None else obj.width
            height = 1 if obj is None else obj.height

            # TODO: edited
            pos = np.array((
                self._rand_int(top[0], max(top[0] + 1, min(top[0] + size[0],
                               self.grid.width - width + 1))),
                self._rand_int(top[1], max(top[1] + 1, min(top[1] + size[1],
                               self.grid.height - height + 1)))
            ))

            valid = True 
            for dx in range(width):
                for dy in range(height):
                    x = pos[0] + dx
                    y = pos[1] + dy
                    
                    # Don't place the object on top of another object
                    if obj is not None and obj.type == "door":
                        left, right, up, down = 0, 0, 0, 0
                        if (x < self.grid.width-1 and not self.grid.is_empty(x+1, y)) or x == self.grid.width - 1:
                            right = 1
                        if (x > 1 and not self.grid.is_empty(x-1, y)) or x == 0:
                            left = 1
                        if (y < self.grid.height-1 and not self.grid.is_empty(x, y+1)) or y == self.grid.height - 1:
                            down = 1
                        if (y > 1 and not self.grid.is_empty(x, y-1)) or y == 0:
                            up = 1
                        if obj.dir=='horz' and (up or down):
                            valid = False
                            # print(
                            #      'placing door on a position of a horizonal wall but there is blocked up or down')
                            # print(left, right, up, down)
                            break
                        if obj.dir=='vert' and (left or right):
                            valid = False
                            # print(
                            #      'placing door on a position of a vertical wall but there is blocked left or right'
                            # )
                            # print(left, right, up, down)
                            break
                    else:
                        if not self.grid.is_empty(x, y):
                            valid = False
                            # print(
                            #     'Placing a non-door object in a position of a non-empty cell')
                            break

                    # Don't place the object next to door

                    if obj is not None and obj.type != "door":
                        if x < self.grid.width-1:
                            fur = self.grid.get_furniture(x+1, y)
                            # print("right", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                # print('try to place object next to door')
                                break
                        if x > 1:
                            fur = self.grid.get_furniture(x-1, y)
                            # print("left", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                # print('try to place object next to door')
                                break
                        if y < self.grid.height-1:
                            fur = self.grid.get_furniture(x, y+1)
                            # print("down", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                # print('try to place object next to door')
                                break
                        if y > 1:
                            fur = self.grid.get_furniture(x, y-1)
                            # print("top", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                # print('try to place object next to door')
                                break

                    # Don't place the object where the agent is
                    # TODO: fixed for single agent
                    # for agent in self.agents:
                    #     if np.array_equal((x, y), agent.agent_pos):
                    #         valid = False
                    #         print('try to place object on agent')
                    #         break
                    if np.array_equal((x, y), self.agent_pos):
                        valid = False
                        # print('try to place object on agent')
                        break

                    # Check if there is a filtering criterion
                    if reject_fn and reject_fn(self, (x, y)):
                        valid = False
                        # print('try to place object but got rejected by reject_fn')
                        break

            if not valid:
                continue

            break

        self.grid.set(*pos, obj)

        if obj:
            self.put_obj(obj, *pos)

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.update_pos((i, j))

        if obj.is_furniture():
            for pos in obj.all_pos:
                self.grid.set(*pos, obj)

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = BehaviorGrid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        if obs_grid.is_empty(vx, vy):
            return False

        for i in range(3):
            if [obj.type for obj in obs_cell[i]] != [obj.type for obj in world_cell[i]]:
                return False

        return True

    def step(self, action):
        # keep track of last action
        if self.mode == 'human':
            self.last_action = action
        else:
            self.last_action = self.actions(action)

        self.step_count += 1
        self.action_done = True

        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            self.agents[self.cur_agent_id].agent_dir = self.agent_dir

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            self.agents[self.cur_agent_id].agent_dir = self.agent_dir

        # Move forward
        elif action == self.actions.forward:
            can_overlap = True
            for _ in range(self.agents[self.cur_agent_id].step_size):
                for obj in fwd_cell:
                    if is_obj(obj) and not obj.can_overlap:
                        can_overlap = False
                        break
                if can_overlap:
                    self.agent_pos = fwd_pos
                    self.agents[self.cur_agent_id].agent_pos = self.agent_pos
                else:
                    self.action_done = False
                fwd_pos = self.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

        else:
            if self.mode == 'human':
                self.last_action = None
                if action == 'choose':
                    choices = self.all_reachable()
                    if not choices:
                        print("No reachable objects")
                    else:
                        # get all reachable objects
                        text = ''.join('{}) {} \n'.format(
                            i, choices[i].name) for i in range(len(choices)))
                        obj = input(
                            "Choose one of the following reachable objects: \n{}".format(text))
                        while not obj.isdigit():
                            obj = input(
                                "Please select digit! Choose one of the following reachable objects: \n{}".format(text))
                        obj = choices[int(obj)]
                        print("chosen object: ", obj)
                        assert obj is not None, "No object chosen"

                        actions = []
                        for action in self.actions:
                            action_class = ACTION_FUNC_MAPPING.get(
                                action.name, None)
                            if action_class and action_class(self).can(obj):
                                actions.append(action.name)

                        if len(actions) == 0:
                            print("No actions available")
                        else:
                            text = ''.join('{}) {} \n'.format(
                                i, actions[i]) for i in range(len(actions)))
                            action = input(
                                f"Choose one of the following actions: {text}, or quit")
                            while not action.isdigit():
                                action = input(
                                    f"Please select digit! Choose one of the following actions: {text}, or quit")
                            action = actions[int(action)]  # action name
                            ACTION_FUNC_MAPPING[action](self).do(obj)
                            self.last_action = self.actions[action]

                # Done action (not used by default)
                else:
                    assert False, "unknown action {}".format(action)
            else:
                # TODO: with agent centric, how does agent choose which obj to do the action on
                obj_action = self.actions(action).name.split(
                    '/')  # list: [obj, action]

                # try to perform action
                obj = self.obj_instances[obj_action[0]]
                action_class = ACTION_FUNC_MAPPING[obj_action[1]]
                if action_class(self).can(obj):
                    action_class(self).do(obj)
                else:
                    self.action_done = False

        self.update_states()
        reward = self._reward()
        done = self._end_conditions() or self.step_count >= self.max_steps
        obs = self.gen_obs()

        return obs, reward, done, {}

    def all_reachable(self):
        return [obj for obj in self.obj_instances.values() if obj.check_abs_state(self, 'inreachofrobot')]

    def update_states(self):
        for obj in self.obj_instances.values():
            for name, state in obj.states.items():
                if state.type == 'absolute':
                    state._update(self)

        self.grid.state_values = {obj: obj.get_ability_values(
            self) for obj in self.obj_instances.values()}

    def render(self, mode='human', highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        if mode == "human" and not self.window:
            self.window = Window("mini_behavior")
            self.window.show(block=False)

        # img = super().render(mode='rgb_array', highlight=highlight, tile_size=tile_size)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * \
            (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        agent_poses = [agent.agent_pos for agent in self.agents]
        agent_dirs = [agent.agent_dir for agent in self.agents]
        agent_colors = [agent.agent_color for agent in self.agents]
        # render agent poses is only the current agent pos
        agent_poses = [self.agent_pos]
        agent_dirs = [self.agent_dir]
        agent_colors = [self.agent_color]
        img = self.grid.render(
            tile_size,
            agent_poses,
            agent_dirs,
            agent_colors,
            highlight_mask=highlight_mask if highlight else None
        )

        if self.render_dim is None:
            img = self.render_furniture_states(img)
        else:
            img = self.render_furniture_states(img, dim=self.render_dim)

        if self.window:
            self.window.set_inventory(self)

        img = self.render_furniture_states(img)

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def render_states(self, tile_size=TILE_PIXELS):
        pos = self.front_pos
        imgs = []
        furniture = self.grid.get_furniture(*pos)
        img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
        if furniture:
            furniture.render(img)
            state_values = furniture.get_ability_values(self)
            GridDimension.render_furniture_states(img, state_values)
            imgs.append(img)
            # render for state
            # TODO: Modify as render object to be state size extenable
            state_img = np.zeros(
                shape=(tile_size, tile_size, 3), dtype=np.uint8)
            tile_size_half = int(tile_size/2)
            furniture.render_state(
                state_img[:tile_size_half, :tile_size_half, :], 'dustyable'
                if state_values.get('dustyable', False) else None)
            furniture.render_state(
                state_img[tile_size_half:, :tile_size_half, :], 'openable'
                if state_values.get('openable', False) else None)
            furniture.render_state(
                state_img[:tile_size_half, tile_size_half:, :], 'stainable'
                if state_values.get('stainable', False) else None)
            furniture.render_state(
                state_img[tile_size_half:, tile_size_half:, :], 'toggleable'
                if state_values.get('toggleable', False) else None)
            imgs.append(state_img)
        else:
            imgs.append(img)
            imgs.append(img)

        furniture, obj = self.grid.get(*pos)
        state_values = obj.get_ability_values(self) if obj else None

        # render object
        img = GridDimension.render_tile_with_state(furniture, obj)
        # render object state
        imgs.append(img)
        img = GridDimension.render_obj_states_icon(state_values)
        imgs.append(img)

        return imgs

    def render_furniture_states(self, img, tile_size=TILE_PIXELS, dim=None):
        for obj in self.obj_instances.values():
            if obj.is_furniture():
                if dim is None or dim in obj.dims:
                    if obj.cur_pos is not None:
                        i, j = obj.cur_pos
                        ymin = j * tile_size
                        ymax = (j + obj.height) * tile_size
                        xmin = i * tile_size
                        xmax = (i + obj.width) * tile_size
                        sub_img = img[ymin:ymax, xmin:xmax, :]
                        state_values = obj.get_ability_values(self)
                        GridDimension.render_furniture_states(
                            sub_img, state_values, tile_size)
        return img

    def render_objects_states(self, img, tile_size=TILE_PIXELS, dim=None):
        for obj in self.obj_instances.values():
            if not obj.is_furniture():
                if dim is None or dim in obj.dims:
                    if obj.cur_pos is not None:
                        i, j = obj.cur_pos
                        ymin = j * tile_size
                        ymax = (j + obj.height) * tile_size
                        xmin = i * tile_size
                        xmax = (i + obj.width) * tile_size
                        sub_img = img[ymin:ymax, xmin:xmax, :]
                        state_values = obj.get_ability_values(self)
                        GridDimension.render_objects_states(
                            sub_img, state_values)
        return img

    def switch_dim(self, dim):
        self.render_dim = dim
        self.grid.render_dim = dim
