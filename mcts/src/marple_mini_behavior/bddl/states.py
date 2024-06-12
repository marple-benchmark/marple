from marple_mini_behavior.bddl.objs import *
from marple_mini_behavior.mini_behavior.states import *

ALL_STATES = [
    'atsamelocation',
    'cleaningTool',
    'coldSource',
    'cookable',
    'dustyable',
    'freezable',
    'heatSource',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'insameroomasrobot',
    'inside',
    'nextto',
    'onfloor',
    'onTop',
    'openable',
    'sliceable',
    'slicer',
    'soakable',
    'stainable',
    'toggleable',
    'under'
    'waterSource'
    # 'touching', TODO: uncomment once implemented
]

# Touching
# ObjectsInFOVOfRobot,

DEFAULT_STATES = [
    'atsamelocation',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'insameroomasrobot',
    'inside',
    'nextto',
    'onfloor',
    'onTop',
    'under'
]


ABILITIES = [
    'cookable',
    'dustyable',
    'freezable',
    'openable',
    'sliceable',
    'soakable',
    'stainable',
    'toggleable',
]


FURNITURE_STATES = [
    'dustyable',
    'openable',
    'stainable',
    'toggleable',
]


# state (str) to state (function) mapping
STATE_FUNC_MAPPING = {
    'atsamelocation': AtSameLocation,
    'cleaningTool': CleaningTool,
    'coldSource': HeatSourceOrSink,
    'cookable': Cooked,
    'dustyable': Dusty,
    'freezable': Frozen,
    'heatSource': HeatSourceOrSink,
    'infovofrobot': InFOVOfRobot,
    'inhandofrobot': InHandOfRobot,
    'inreachofrobot': InReachOfRobot,
    'insameroomasrobot': InSameRoomAsRobot,
    'inside': Inside,
    'nextto': NextTo,
    'onfloor': OnFloor,
    'onTop': OnTop,
    'openable': Opened,
    'sliceable': Sliced,
    'slicer': Slicer,
    'soakable': Soaked,
    'stainable': Stained,
    'toggleable': ToggledOn,
    'under': Under,
    'waterSource': WaterSource
    # 'touching', TODO: uncomment once implemented
}

# state (str) to state (function) mapping
def states_to_idx(state_values):
    bi_str = ''
    for i, ability in enumerate(ABILITIES):
        if state_values.get(ability, False):
            bi_str += '1'
        else:
            bi_str += '0'
    idx = 1 + int(bi_str, 2) 
    # add 1 to differentiate empty cell and object with no states

    return idx


########################################################################################################################

# FROM BDDL

# TEXTURE_CHANGE_PRIORITY = {
#     Frozen: 4,
#     Burnt: seed 10_3,
#     Cooked: seed 0_2,
#     Soaked: seed 0_2,
#     ToggledOn: 0,
# }

