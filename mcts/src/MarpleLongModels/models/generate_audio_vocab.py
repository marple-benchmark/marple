import json
import vocab

action_to_audio_idx = {}

with open('../../marple_mini_behavior/mini_behavior/utils/object_actions.json', 'r') as f:
    object_actions = json.load(f)

def generate_action_to_audio_idx():
    action_to_audio_idx['empty'] = 0
    action_to_audio_idx['forward_a'] = 1
    action_to_audio_idx['forward_b'] = 2
    action_to_audio_idx['forward_c'] = 3
    action_to_audio_idx['forward_d'] = 4

    for action_name in vocab.ACTION_TO_IDX:
        if action_name not in ['null', 'forward', 'left', 'right']:
            for object_name in list(vocab.OBJECT_TO_IDX):
                if object_name not in ['null', 'empty']:
                    if action_name in object_actions[object_name]:
                        action_to_audio_idx[f'{action_name}_{object_name}'] = len(action_to_audio_idx)

generate_action_to_audio_idx()

with open('action_to_audio_idx.json', 'w') as f:
    json.dump(action_to_audio_idx, f, indent=4)