
get_night_snack = [
    "toogle-on-*-light-Kitchen", {
        "obj": None,
        "fur": "light",
        "room": "Kitchen",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }
]
# 2
get_snack = [
    "close-*-*-electric_refrigerator-Kitchen", 
    {
        "obj": None,
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": False
    }
]

# 3
do_laundry = [   
    "toggle-on-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }
]

feed_dog = [
    "drop-*-dogfood-closet-Kitchen", 
    {
        "obj": "dogfood",
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False,
    }
]

# 4
watch_news_on_tv = [ 
    "idle", 
    {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "idle",
        "state": None,
        "can_skip": False,
        "end_state": False
    }
]

# 5
move_plant_at_night = [ 
    "pickup-*-pot_plant-table-LivingRoom", {
        "obj": "pot_plant",
        "fur": "table",
        "room": "LivingRoom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }
]

# 6
take_shower = [
    "toogle-on-*-shower-Bathroom", {
        "obj": None,
        "fur": "shower",
        "room": "Bathroom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }
]

# 8
watch_movie_cozily = [ 
    # "idle", 
    # {
    #     "obj": None,
    #     "fur": "television",
    #     "room": "LivingRoom",
    #     "pos": None,
    #     "action": "idle",
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": False
    # }
    "pickup-*-pillow-bed-Bedroom", {
        "obj": "pillow",
        "fur": "bed",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }
]

# 9
change_outfit = [ 
    "pickup-*-clothes-closet-Bedroom", {
        "obj": "clothes",
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }
]

# 10
clean_living_room_table = [
    "clean-*-*-table-LivingRoom", 
    {
        "obj": None,
        "fur": "table",
        "room": "LivingRoom",
        "pos": None,
        "action": "clean",
        "state": ["dustyable", 0],
        "can_skip": False,
        "end_state": False
    }
]

MISSION_TO_INFERENCE_STATE = {
    'get_night_snack': get_night_snack,
    'get_snack': get_snack,
    'feed_dog': feed_dog,
    'watch_news_on_tv': watch_news_on_tv,
    'move_plant_at_night': move_plant_at_night,
    'take_shower': take_shower,
    'do_laundry': do_laundry,
    'watch_movie_cozily': watch_movie_cozily,
    'change_outfit': change_outfit,
    'clean_living_room_table': clean_living_room_table,
}
 
