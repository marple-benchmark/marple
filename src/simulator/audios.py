import os
from email.mime import audio
from pydub import AudioSegment

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))#, ".."))
audio_dir = os.path.join(root_dir, 'simulator/audios')

def get_audio_file_idx(action=None, object=None, state=None, empty=False):
    empty_audio_file_name = 'empty.wav'
    empty_audio_file_path = os.path.join(audio_dir, 'empty.wav')
    empty_audio_file_idx = os.listdir(audio_dir).index('empty.wav')
    if empty:
        # print(f"audio check {'*' * 100} {empty_audio_file_name, empty_audio_file_path, empty_audio_file_idx}")
        return empty_audio_file_path, empty_audio_file_idx
    else:
        if action == 'toggle':
            audio_file_name = f'{action}_{object}_{state}.wav'
            # print(f"audio check new {audio_file_name}")
            audio_file_path = os.path.join(audio_dir, audio_file_name)
        else:
            audio_file_name = f'{action}_{object}.wav'
            # print(f"audio check new {audio_file_name}")
            audio_file_path = os.path.join(audio_dir, f'{action}_{object}.wav')
        if audio_file_name in os.listdir(audio_dir):
            audio_file_idx = os.listdir(audio_dir).index(audio_file_name)
            # print(f"audio check {'*' * 100} {audio_file_name, audio_file_path, audio_file_idx}")
            return audio_file_path, audio_file_idx
        else:
            # print(f"audio check {'*' * 100} {empty_audio_file_name, empty_audio_file_path, empty_audio_file_idx}")
            return empty_audio_file_path, empty_audio_file_idx


AUDIO_MAP = {
    'empty': os.path.join(audio_dir, 'empty.wav'),
    'forward_one_step': os.path.join(audio_dir, 'forward_one_step.wav'),
    'forward_two_step': os.path.join(audio_dir, 'forward_two_step.wav'),
    'forward_three_step': os.path.join(audio_dir, 'forward_three_step.wav')
    
}

STEP_SIZE_AUDIO_MAP = {
    1: 'forward_one_step',
    2: 'forward_two_step',
    3: 'forward_three_step'
}

OPEN_CLOSE_OBJECT_AUDIO_MAP = {
    'door': {
        'open': os.path.join(audio_dir, 'toggle_door_to_open.wav'),
        'close': os.path.join(audio_dir, 'toggle_door_to_close.wav')
    },
    'light': {
        'open': os.path.join(audio_dir, 'toggle_light_to_on.wav'),
        'close': os.path.join(audio_dir, 'toggle_light_to_off.wav'),
    },
    'electric_refrigerator': {
        'open': os.path.join(audio_dir, 'open_fridge.wav'),
        'close': os.path.join(audio_dir, 'close_fridge.wav')
    },
    'television': {
        'open': os.path.join(audio_dir, 'remote_use.wav'),
        'close': os.path.join(audio_dir, 'remote_use.wav')
    },
    'shower': {
        'open': os.path.join(audio_dir, 'shower_toggle_to_on.wav'),
        'close': os.path.join(audio_dir, 'shower_toggle_to_off.wav'),
        'idle_on': os.path.join(audio_dir, 'shower_use.wav')
    },
    'laundry': {
        'open': os.path.join(audio_dir, 'laundry_toggle_to_open.wav'),
        'close': os.path.join(audio_dir, 'laundry_toggle_to_close.wav'),
        'idle_on': os.path.join(audio_dir, 'laundry_use.wav')
    },
    'closet': {
        'open': os.path.join(audio_dir, 'closet_door_open.wav'),
        'close': os.path.join(audio_dir, 'closet_door_close.wav')
    }
}

PICK_UP_DROP_OBJECT_AUDIO_MAP = {
    'sandwich': {
        'pickup': os.path.join(audio_dir, 'snack_pickup.wav'),
        'drop': os.path.join(audio_dir, 'snack_drop.wav')
    },
    'remote': {
        'pickup': os.path.join(audio_dir, 'remote_pickup.wav'),
        'drop': os.path.join(audio_dir, 'remote_drop.wav')
    },
    'pot_plant': {
        'pickup': os.path.join(audio_dir, 'vase_pickup.wav'),
        'drop': os.path.join(audio_dir, 'vase_drop.wav')
    },
    'clothes': {
        'pickup': os.path.join(audio_dir, 'clothes_pickup.wav'),
        'drop': os.path.join(audio_dir, 'clothes_drop.wav')
    }
}


# audios = []
# sound1 = AudioSegment.from_wav(AUDIO_MAP['empty'])
# sound2 = AudioSegment.from_wav(AUDIO_MAP['forward_one_step'])
# audios.append(sound1)
# audios.append(sound2)
# combined_sounds = sum(audios)
# combined_sounds.export("output.wav", format="wav")
