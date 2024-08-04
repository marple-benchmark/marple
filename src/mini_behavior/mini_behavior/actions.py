import numpy as np
from marple_mini_behavior.bddl import *

def find_tool(env, possible_tool_types):
    # returns whether agent is carrying a obj of possible_tool_types, and the obj_instance
    for tool_type in possible_tool_types:
        tools = env.objs.get(tool_type, []) # objs of type tool in the env
        for tool in tools:
            if tool.check_abs_state(env, 'inhandofrobot'):
                return True
    return False


class BaseAction:
    def __init__(self, env):
        """
        initialize action
        """
        super(BaseAction, self).__init__()
        self.env = env
        self.key = None

    def can(self, obj):
        """
        check if possible to do action
        """

        # check if possible to do the action on the object
        if not obj.possible_action(self.key):
            print("not possible action", obj.type)
            return False

        iotal = 0
        # check if the object is in reach of the agent
        if not obj.check_abs_state(self.env, 'inreachofrobot'):
            print(f"not reachable, {obj.cur_pos}, {self.env.front_pos}, {self.env.agent_pos}")
            return False

        return True

    def pre_can(self, obj):
        """
        check if possible to do action pre check
        """
        # check if possible to do the action on the object
        if obj is None or obj == 'null':
            print('not an obj')
            return False   
                
        # if type(obj).__name__ != 'WorldObj' or type(obj).__bases__[0].__name__ != 'FurnitureObj':
        #     # print(type(obj).__name__)
        #     print('not an obj')
        #     return False
        if not obj.possible_action(self.key):
            print("not possible action")
            return False
        
        return True

    def do(self, obj):
        """
        do action
        """
        assert self.can(obj), f'Cannot perform action {self.key}'

    def update_state_dict(self, agent=None, obj=None):
        """
        after do action
        update obj state and position
        """
        return

class Close(BaseAction):
    def __init__(self, env):
        super(Close, self).__init__(env)
        self.key = 'close'

    def can(self, obj):
        """
        can perform action if:
        - obj is openable
        - obj is open
        """
        if not super().can(obj):
            return False
        
        if not obj.check_abs_state(self.env, 'openable'):
            return False

        return True

    def pre_can(self, obj):
        """
        can perform action if:
        - obj is openable
        - obj is open
        """
        if not super().pre_can(obj):
            return False
        
        if not obj.check_abs_state(self.env, 'openable'):
            return False

        return True
    
    def do(self, obj):
        super().do(obj)
        obj.states['openable'].set_value(False)
        obj.update(self.env)


class Cook(BaseAction):
    def __init__(self, env):
        super(Cook, self).__init__(env)
        self.key = 'cook'
        self.tools = ['pan']
        self.heat_sources = ['stove']

    def can(self, obj):
        """
        can perform action if:
        - obj is cookable
        - agent is carrying a cooking tool
        - agent is infront of a heat source
        - the heat source is toggled on
        """
        if not super().can(obj):
            return False

        if find_tool(self.env, self.tools):
            front_cell = self.env.grid.get_all_items(*self.env.agent_pos)
            for obj2 in front_cell:
                if obj2 is not None and obj2.type in self.heat_sources:
                    return obj2.check_abs_state(self.env, 'toggleable')
        return False

    def do(self, obj):
        super().do(obj)
        obj.states['cookable'].set_value(True)

class Idle(BaseAction):
    def __init__(self, env):
        super(Idle, self).__init__(env)
        self.key = 'idle'

    def can(self, obj):
        # if not super().pre_can(obj):
        #     return False

        # fwd_pos = self.env.front_pos
        # all_items = self.env.grid.get_all_items(*fwd_pos)
        # fur = all_items[0]
        # obj = all_items[1]

        # if 'toggleable' in fur.states.keys():
        #     if not fur.check_abs_state(self.env, 'toggleable'):
        #         return False
        
        return True
    
    def pre_can(self, obj, fur):
        # if not super().pre_can(obj):
        #     return False

        # if 'toggleable' in fur.states.keys():
        #     if not fur.check_abs_state(self.env, 'toggleable'):
        #         return False
        
        return True

    def do(self, obj):
        pass


class Drop(BaseAction):
    def __init__(self, env):
        super(Drop, self).__init__(env)
        self.key = 'drop'

    def can_drop(self, pos):
        flag = False

        all_items = self.env.grid.get_all_items(*pos)
        obj = all_items[1]

        if obj is None:
            flag = True 
        return flag

    def can(self, obj):
        """
        can drop obj if:
        - agent is carrying obj
        - there is no obj in base of forward cell
        """
        if not super().can(obj):
            print("super can't drop")
            return False

        if not obj.check_abs_state(self.env, 'inhandofrobot'):
            print("cannot drop, obj not in hand of robot")
            return False

        fwd_pos = self.env.front_pos
        all_items = self.env.grid.get_all_items(*fwd_pos)
        fur = all_items[0]
        obj = all_items[1]

        if obj is None:
            return True

        if fur is not None and 'openable' in fur.states.keys():
            if not fur.check_abs_state(self.env, 'openable'):
                return False
            
        final_flag = False
        for sampled_pos in fur.all_pos:
            flag = self.can_drop(sampled_pos)
            if flag is True:
                final_flag = True

        return final_flag
    
    def pre_can(self, obj, fur):
        """
        can drop obj if:
        - agent is carrying obj
        - there is no obj in base of forward cell
        """
        if not super().pre_can(obj):
            print("super can't drop")
            return False

        if not obj.check_abs_state(self.env, 'inhandofrobot'):
            print("not in hand of robot")
            return False

        # check_pos = obj.cur_pos
        # print("check action, obj's position", obj, obj.name, check_pos)
        # all_items = self.env.grid.get_all_items(*check_pos)
        # fur = all_items[0]
        # obj = all_items[1]

        if 'openable' in fur.states.keys():
            if not fur.check_abs_state(self.env, 'openable'):
                return False
            
        final_flag = False
        for sampled_pos in fur.all_pos:
            flag = self.can_drop(sampled_pos)
            if flag is True:
                final_flag = True
                break

        return final_flag

    def do(self, obj):
        super().do(obj)

        self.env.carrying.discard(obj)
        self.env.agents[self.env.cur_agent_id].carrying.discard(obj)

        fwd_pos = self.env.front_pos
        all_items = self.env.grid.get_all_items(*fwd_pos)
        fur = all_items[0]
        drop_pos = None
        for sampled_pos in fur.all_pos:
            flag = self.can_drop(sampled_pos)
            if flag is True:
                drop_pos = sampled_pos
                break

        # change object properties
        obj.cur_pos = drop_pos
        # change agent / grid
        self.env.grid.set(*drop_pos, obj)
        self.update_state_dict(obj)

        # print(f'dropped {obj.type}')

    def update_state_dict(self, obj):

        # check where is the obj's new position
        for id, room in enumerate(self.env.room_instances):
            room_top = room.top
            room_size = room.size
            if room_top[0] <= obj.cur_pos[0] <= room_top[0] + room_size[0] - 1 and \
                    room_top[1] <= obj.cur_pos[1] <= room_top[1] + room_size[1] - 1:
                # in this room
                obj.in_room = room
                if obj not in room.objs:
                    room.objs.append(obj)
                # check on which furniture
                on_furniture = False
                for fur in room.furnitures:
                    # get furniture instance
                    # on one furniture
                    obj_cur_pos_tuple = (obj.cur_pos[0], obj.cur_pos[1])
                    if obj_cur_pos_tuple in fur.all_pos:
                        obj.on_fur = fur
                        if obj not in fur.objects:
                            obj.self_id = len(fur.objects)
                            fur.objects.append(obj)
                        on_furniture = True
                        break
                # on room's floor
                if not on_furniture:
                    obj.on_fur = None
                    if obj not in room.floor_objs:
                        obj.self_id = len(room.floor_objs)
                        room.floor_objs.append(obj)
                break



class DropIn(BaseAction):
    def __init__(self, env):
        super(DropIn, self).__init__(env)
        self.key = 'drop_in'

    def drop_dims(self, pos):
        dims = []

        all_items = self.env.grid.get_all_items(*pos)
        last_furniture, last_obj = 'floor', 'floor'
        for i in range(3):
            furniture = all_items[2*i]
            obj = all_items[2*i + 1]

            if obj is None and furniture is not None and furniture.can_contain and i in furniture.can_contain:
                if 'openable' not in furniture.states or furniture.check_abs_state(self.env, 'openable'):
                    if last_obj is not None:
                        dims.append(i)

            last_furniture = furniture
            last_obj = obj
        return dims

    def can(self, obj):
        """
        can drop obj under if:
        - agent is carrying obj
        - middle of forward cell is open
        - obj does not contain another obj
        """
        if not super().can(obj):
            return False

        if not obj.check_abs_state(self.env, 'inhandofrobot'):
            return False

        fwd_pos = self.env.front_pos
        dims = self.drop_dims(fwd_pos)
        return dims != []

    def do(self, obj, dim):
        # drop
        super().do(obj)
        self.env.carrying.discard(obj)
        self.env.agents[self.env.cur_agent_id].carrying.discard(obj)

        fwd_pos = self.env.front_pos
        obj.cur_pos = fwd_pos
        self.env.grid.set(*fwd_pos, obj, dim)

        # drop in and update
        furniture = self.env.grid.get_furniture(*fwd_pos, dim)
        obj.states['inside'].set_value(furniture, True)


class Open(BaseAction):
    def __init__(self, env):
        super(Open, self).__init__(env)
        self.key = 'open'
    
    def can(self, obj):
        """
        can perform action if:
        - obj is openable
        - obj is open
        """
        if not super().can(obj):
            return False
        
        if obj.check_abs_state(self.env, 'openable'):
            return False

        return True

    def pre_can(self, obj):
        """
        can perform action if:
        - obj is openable
        - obj is close
        """
        if not super().pre_can(obj):
            return False
        
        if obj.check_abs_state(self.env, 'openable'):
            return False
        
        return True

    def do(self, obj):
        super().do(obj)
        obj.states['openable'].set_value(True)
        obj.update(self.env)


class Clean(BaseAction):
    def __init__(self, env):
        super(Clean, self).__init__(env)
        self.key = 'clean'

    def do(self, obj):
        super().do(obj)
        obj.states['dustyable'].set_value(False)
        obj.update(self.env)

class Pickup(BaseAction):
    def __init__(self, env):
        super(Pickup, self).__init__(env)
        self.key = 'pickup'

    def can(self, obj):
        if not super().can(obj):
            print('cannot do basic action')
            return False
        
        if obj.check_abs_state(self.env, 'inhandofrobot'):
            print('cannot pickup bc already carrying')
            return False
        
        fwd_pos = self.env.front_pos
        all_items = self.env.grid.get_all_items(*fwd_pos)
        fur = all_items[0]
        obj = all_items[1]
        
        if 'openable' in fur.states.keys():
            if not fur.check_abs_state(self.env, 'openable'):
                print(f'cannot pickup bc fur {fur.type} closed')
                return False

        return True

    def pre_can(self, obj, fur):
        if not super().pre_can(obj):
            return False
        
        # cannot pickup if carrying
        if obj.check_abs_state(self.env, 'inhandofrobot'):
            return False
        
        # check_pos = obj.cur_pos
        # all_items = self.env.grid.get_all_items(*check_pos)
        # fur = all_items[0]
        # obj = all_items[1]

        if 'openable' in fur.states.keys():
            if not fur.check_abs_state(self.env, 'openable'):
                return False

        return True

    def do(self, obj):
        super().do(obj)
        self.env.carrying.add(obj)
        self.env.agents[self.env.cur_agent_id].carrying.add(obj)

        obj = self.env.grid.get_obj(*obj.cur_pos)

        # remove obj from the grid and shift remaining objs
        self.env.grid.remove(*obj.cur_pos, obj)

        new_obj = None
        self.env.grid.set(*obj.cur_pos, new_obj)

        # update cur_pos of obj
        obj.update_pos(np.array([-1, -1]))

        # check dependencies
        assert obj.check_abs_state(self.env, 'inhandofrobot')
        assert not obj.check_abs_state(self.env, 'onfloor')

        self.update_state_dict(obj)

    def update_state_dict(self, obj):
        obj.in_room.objs.remove(obj)
        obj.on_fur.objects.remove(obj)
        obj.in_room = None
        obj.on_fur = None


class Slice(BaseAction):
    def __init__(self, env):
        super(Slice, self).__init__(env)
        self.key = 'slice'
        self.slicers = ['carving_knife', 'knife']

    def can(self, obj):
        """
        can perform action if:
        - action is sliceable
        - agent is holding a slicer
        """
        if not super().can(obj):
            return False
        return find_tool(self.env, self.slicers)

    def do(self, obj):
        super().do(obj)
        obj.states['sliceable'].set_value()


class Toggle(BaseAction):
    def __init__(self, env):
        super(Toggle, self).__init__(env)
        self.key = 'toggle'

    def do(self, obj):
        """
        toggle from on to off, or off to on
        """
        super().do(obj)
        cur = obj.check_abs_state(self.env, 'toggleable')
        obj.states['toggleable'].set_value(not cur)
        obj.update(self.env)
        # TODO: debug
        # print('state after toggle', obj.states['toggleable'].get_value(self.env))

# ACTION_TO_IDX = {
#     'idle': 1,
#     'forward': 2,
#     'left': 3,
#     'right': 4,
#     'toggle': 5,
#     'open': 6,
#     'close': 7,
#     'pickup': 8,
#     'drop': 9,
#     'clean': 10,
#     'random_walk': 11
# }

# IDX_TO_ACTION = {
#     1: 'idle', 
#     2: 'forward', 
#     3: 'left',
#     4: 'right',
#     5: 'toggle', 
#     6: 'open',
#     7: 'close',
#     8: 'pickup',
#     9: 'drop',
#     10: 'clean',
#     11: 'random_walk'
# }
