import random
import time
import numpy as np

from src.mini_behavior.mini_behavior.nav import MazeSolver
from src.mini_behavior.mini_behavior.objects import *

GENDER = ['female', 'male']
STEP_SIZE = range(1, 4)
AGENT_NAMES = []
MAX_NUM_ACTIONS = 20


class Agent:
    def __init__(self, name):
        self.name = name
        self.id = None
        self.init_pos = None
        self.agent_pos = None
        self.agent_dir = None
        # 0 right, 1 down, 2 left, 3 up
        self.agent_color = None
        self.step_size = 1
        self.gender = None
        self.mission_preference = None
        self.forgetness = None
        self.cur_mission = None
        self.cur_subgoal = None
        self.carrying = set()

    def set_mission_preference(self, mission_preference: dict):
        self.mission_preference = mission_preference

    def set_forgetness(self, forgetness: float):
        self.forgetness = forgetness

    def set_id(self, id:int):
        self.id = id

    def navigate(self, env, startpos, startdir: int, tgtpos, exact=False):
        '''
        navigate from startpos to tgtpos
        if exact is True, then navigate to the exact position of tgtpos
        if exact is False, then navigate to the position next to tgtpos
        '''
        # src and tgt are positions
        # cur_pos = self.agent_pos
        # print("navigate start pos type", type(startpos), startpos)
        cur_pos = startpos
        cur_dir = startdir
        if not exact:
            # steps = astar(env, cur_pos, tgtpos, self.step_size)
            base_action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dir = random.sample(base_action_space, 1)[0]
            end = (tgtpos[0] + dir[0], tgtpos[1] + dir[1])
            end_found = False
            for dir in base_action_space:
                end = (tgtpos[0] + dir[0], tgtpos[1] + dir[1])
                if end[0] >= 0 and end[0] < env.grid.width \
                        and end[1] >= 0 and end[1] < env.grid.height and env.grid.is_empty(*end):
                    end_found = True
                    break
            # print(f"navgivate with step size {self.step_size}")
            if not end_found:
                return None, cur_pos, startdir
        else:
            end = tgtpos
        nav_res = MazeSolver(env, self.step_size).astar(cur_pos, cur_dir, end)
        if nav_res:
            steps = list(nav_res)
        else:
            return None, cur_pos, startdir
            

        # cur_dir = self.agent_dir
        cur_dir = startdir
        # print(
        #     f"start navigate from {cur_pos}, {cur_dir}, to {end} with {steps}.")
        actions = []
        for step in steps[1:]:
            nx, ny = step[0], step[1]
            cx, cy = cur_pos[0], cur_pos[1]

            # close door if door is behind the agent and if not forget
            # get agent back position
            # if cur_dir == 0:
            #     back_pos = cx - 1, cy   
            # elif cur_dir == 1:
            #     back_pos = cx, cy - 1
            # elif cur_dir == 2:
            #     back_pos = cx + 1, cy
            # else:
            #     back_pos = cx, cy + 1
            
            # check if there is a door behind the agent
            # furniture, obj = env.grid.get(*back_pos)
            # if furniture is not None and furniture.type == 'door' and furniture.is_open:
            #     close_score = random.random()
            #     if close_score > self.forgetness:
            #         actions += [[env.actions.left], [env.actions.left]]
            #         actions.append(['close', furniture])
            #         actions += [[env.actions.right], [env.actions.right]]
            # if len(actions) >= 3 and actions[-3][0] == 'open':
            #     if isinstance(actions[-3][1], Door):
            #         door = actions[-3][1]
            #         close_score = random.random()
            #         if close_score > self.forgetness:
            #             actions += [[env.actions.left], [env.actions.left]]
            #             actions.append(['close', door])
            #             actions += [[env.actions.right], [env.actions.right]]   

            # align direction
            deltax, deltay = nx - cx, ny - cy
            if deltax == 0 and deltay > 0:
                move_dir = 1
            elif deltax == 0 and deltay < 0:
                move_dir = 3
            elif deltax > 0 and deltay == 0:
                move_dir = 0
            elif deltax < 0 and deltay == 0:
                move_dir = 2
            else:
                raise Exception('not valid move')
            action_list, new_dir = self.face_towards(env, move_dir, cur_dir)
            actions += action_list
            cur_dir = new_dir

            # move forward
            if nx == cx:
                rest_dist = max(ny, cy) - min(ny, cy)
                for dy in range(min(ny, cy), max(ny, cy) + 1):
                    if dy == cy:
                        continue
                    # look at any where between the start pos and end pos of the action
                    # if there is a door in the way, open it
                    furniture, obj = env.grid.get(nx, dy)
                    if furniture is not None and furniture.type == 'door' and not furniture.is_open:
                        # if encountered door, add open action
                        dist = dy - min(ny, cy)
                        if dist > 1:
                            actions.append(
                                [env.actions.forward, dist - 1])
                            rest_dist = rest_dist - dist + 1
                        actions.append(['open', furniture])
                        break
            else:
                rest_dist = max(nx, cx) - min(nx, cx)
                for dx in range(min(nx, cx), max(nx, cx) + 1):
                    if dx == cx:
                        continue
                    furniture, obj = env.grid.get(dx, ny)
                    if furniture is not None and furniture.type == 'door' and not furniture.is_open:
                        # distance larger than 1
                        dist = dx - min(nx, cx)
                        if dist > 1:
                            actions.append(
                                [env.actions.forward, dist - 1])
                            rest_dist = rest_dist - dist + 1
                        actions.append(['open', furniture])
                        break
            assert rest_dist >= 0, "rest dist smaller than 0"
            actions.append(
                [env.actions.forward, rest_dist])
            cur_pos = step
            cur_dir = move_dir

        return actions, cur_pos, cur_dir

    def face_towards(self, env, move_dir, cur_dir=None):
        # 0 right, 1 down, 2 left 3 up
        if cur_dir is None:
            cur_dir = self.agent_dir

        # Calculate the difference in direction
        dir_diff = (move_dir - cur_dir) % 4

        # Determine the shortest turn direction
        if dir_diff == 3:
            actions = [[env.actions.left]]
        elif dir_diff == 1:
            actions = [[env.actions.right]]
        else:
            # For dir_diff == 2 or 0, it doesn't matter if we turn left or right
            actions = [[env.actions.right if dir_diff ==
                        0 else env.actions.left]] * abs(dir_diff)

        return actions, move_dir

    def face_to(self, env, tgt, cur_pos=None, cur_dir=None):
        if isinstance(tgt, (tuple, np.ndarray)):
            tgt_pos = tgt[:2]
        else:
            tgt_pos = tgt.cur_pos[:2]

        if cur_pos is None:
            cur_pos = self.agent_pos
        if cur_dir is None:
            cur_dir = self.agent_dir

        assert abs(cur_pos[0] - tgt_pos[0]) + \
            abs(cur_pos[1] - tgt_pos[1]) == 1, 'not arrived'

        direction_map = {(1, 0): 0, (-1, 0): 2, (0, 1): 1, (0, -1): 3}
        move_dir = (tgt_pos[0] - cur_pos[0], tgt_pos[1] - cur_pos[1])
        move_dir = direction_map.get(move_dir)

        actions, new_dir = self.face_towards(env, move_dir, cur_dir)

        return actions, cur_pos, new_dir

    def check_forward(self, env, cur_pos, move_dir, forward_size=1):
        assert move_dir in [0, 1, 2, 3], 'direction not exist'

        walkable = True

        for size in range(1, forward_size+1):
            if move_dir == 0:
                next_pos = cur_pos[0] + size, cur_pos[1]
            elif move_dir == 1:
                next_pos = cur_pos[0], cur_pos[1] + forward_size
            elif move_dir == 2:
                next_pos = cur_pos[0] - size, cur_pos[1]
            else:
                next_pos = cur_pos[0] - size, cur_pos[1]

            if 0 <= next_pos[0] < env.grid.width and \
                    0 <= next_pos[1] < env.grid.height:
                furniture, obj = env.grid.get(*next_pos)
                if furniture is None and obj is None:
                    continue
                else:
                    walkable = False
                    return walkable, None
            else:
                walkable = False
                return walkable, None

        return walkable, next_pos

    def random_walk(self, env):
        steps = random.randint(1, MAX_NUM_ACTIONS)
        actions = []
        cur_pos = self.agent_pos
        cur_dir = self.agent_dir
        step_size = self.step_size
        for step in range(steps):
            move_dir = random.randint(0, 3)
            action_list, new_dir = self.face_towards(env, move_dir, cur_dir)
            actions += action_list
            cur_dir = new_dir
            cur_step_size = random.randint(1, step_size)
            walkable, next_pos = self.check_forward(
                env, cur_pos, cur_dir, cur_step_size)
            if walkable:
                actions.append([env.actions.forward, cur_step_size])
                cur_pos = next_pos

        return actions, cur_pos, cur_dir

    def policy(self):
        raise NotImplementedError

    def reward(self):
        raise NotImplementedError

    def habits(self, last_action):
        next_action_map = {
        }
        raise NotImplementedError

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_color(self, color):
        self.agent_color = color

    def set_gender(self, gender):
        self.gender = gender
