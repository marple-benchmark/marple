import json

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

# state (dict) to state (function) mapping
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


def generate_states(abilities, index, current_state, all_states):
    if index == len(abilities):
        all_states.append(current_state.copy())
        return

    ability = abilities[index]
    current_state[ability] = True
    generate_states(abilities, index + 1, current_state, all_states)

    current_state[ability] = False
    generate_states(abilities, index + 1, current_state, all_states)

def all_possible_states(abilities):
    all_states = []
    generate_states(abilities, 0, {}, all_states)
    return all_states

all_states_list = all_possible_states(ABILITIES)
all_states_to_idx = {}
for state in all_states_list:
    all_states_to_idx[str(state)] = states_to_idx(state)
with open('all_states_to_idx.json', 'w') as f:
    json.dump(all_states_to_idx, f, indent=4)
print(all_states_to_idx)


