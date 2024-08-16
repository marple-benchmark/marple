import random
import os
import json
import os


# subgoals
# 1
# (room, fur, fur_states, max_obj, req_obj)
# to gen config given a list of missions, get lsit of all req obj
# check that it is less than min(max_obj)

get_night_snack = [
    ('Kitchen', 'light', {'toggleable': False}, 0, []),
    ('Kitchen', 'electric_refrigerator', {'openable': False}, 4, ['sandwich']),
    ('Bedroom', 'table', None, 4, [])
]

get_snack = [
    ('Kitchen', 'electric_refrigerator', {'openable': False}, 4, ['sandwich']),
    ('Bedroom', 'table', None, 4, [])
]


################################################################################################

watch_news_on_tv = [
    ('LivingRoom', 'sofa', None, 4, ['remote']),
    ('LivingRoom', 'table', None, 4, []),
    ('LivingRoom', 'television', {'toggleable': False}, 0, [])
]  

watch_movie_cozily = [
    ('Bedroom', 'bed', {}, 4, ['pillow']), 
    ('LivingRoom', 'sofa', None, 4, ['remote']),
    ('LivingRoom', 'table', None, 4, []),
    ('LivingRoom', 'television', {'toggleable': False}, 0, [])
]

################################################################################################

do_laundry = [
    ('Bedroom', 'closet', {'openable': False}, 4, ['clothes']), 
    ('Bedroom', 'bed', {}, 4, ['clothes']), 
    ('Bathroom', 'laundry', {'openable': False, 'toggleable': False}, 1, [])
]

feed_dog = [
    ('Kitchen', 'table', None, 4, ['dogfood']),
    ('Bedroom', 'dog', None, 0, []),
    ('Kitchen', 'closet', {'openable': False}, 4, [])
]

################################################################################################

change_outfit = [
    ('Bedroom', 'closet', {'openable': False}, 4, ['clothes']), 
    ('Bathroom', 'laundry', {'openable': False, 'toggleable': False}, 1, []) 
]


move_plant_at_night = [
    ('Kitchen', 'light', {'toggleable': False}, 0, []),
    ('LivingRoom', 'table', None, 4, ['pot_plant']),
    ('Kitchen', 'table', None, 4, [])
]

################################################################################################

clean_living_room_table = [
    ('Kitchen', 'closet', {'openable': False}, 4, ['towel']),
    ('LivingRoom', 'table', {'dustyable': True}, 4, [])
]

take_shower = [
    ('Bathroom', 'shower', {'toggleable': False}, 0, []),
    ('Bedroom', 'closet', {'openable': False}, 4, ['clothes'])
]

MISSION_TO_INIT_CONDITIONS = {
    'get_night_snack': get_night_snack,
    'get_snack': get_snack,
    'feed_dog': feed_dog,
    'watch_news_on_tv': watch_news_on_tv,
    'move_plant_at_night': move_plant_at_night,
    'take_shower': take_shower,
    'do_laundry': do_laundry,
    'watch_movie_cozily': watch_movie_cozily,
    'change_outfit': change_outfit,
    'clean_living_room_table': clean_living_room_table
}

