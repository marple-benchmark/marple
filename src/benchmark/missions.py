Get_night_snack_subgoals_order = [
    ("toogle-on-*-light-Kitchen", {
        "obj": None,
        "fur": "light",
        "room": "Kitchen",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("open-*-*-electric_refrigerator-Kitchen", {
        "obj": None,
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-sandwich-electric_refrigerator-Kitchen", {
        "obj": "sandwich",
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-electric_refrigerator-Kitchen", {
        "obj": None,
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("toogle-off-*-light-Kitchen", {
        "obj": None,
        "fur": "light",
        "room": "Kitchen",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("drop-*-sandwich-table-Bedroom", {
        "obj": "sandwich",
        "fur": "table",
        "room": "Bedroom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": True
    })
]

# 2
Get_snack_subgoals_order = [
    ("open-*-*-electric_refrigerator-Kitchen", {
        "obj": None,
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-sandwich-electric_refrigerator-Kitchen", {
        "obj": "sandwich",
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-electric_refrigerator-Kitchen", {
        "obj": None,
        "fur": "electric_refrigerator",
        "room": "Kitchen",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("drop-*-sandwich-table-Bedroom", {
        "obj": "sandwich",
        "fur": "table",
        "room": "Bedroom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": True
    })
]

# 3
Feed_dog_subgoals_order = [
    ("pickup-*-dogfood-table-Kitchen", {
        "obj": "dogfood",
        "fur": "table",
        "room": "Kitchen",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("drop-*-dogfood-dog-Bedroom", {
        "obj": "dogfood",
        "fur": "dog",
        "room": "Bedroom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-dogfood-dog-Bedroom", {
        "obj": "dogfood",
        "fur": "dog",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": True,
        "end_state": False
    }),
    ("open-*-*-closet-Kitchen", {
        "obj": None,
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False,
    }),
    ("drop-*-dogfood-closet-Kitchen", {
        "obj": "dogfood",
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False,
    }),
    ("close-*-*-closet-Kitchen", {
        "obj": None,
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
]

# 4
Watch_news_on_tv_subgoals_order = [
    ("pickup-*-remote-sofa-LivingRoom", {
        "obj": "remote",
        "fur": "sofa",
        "room": "LivingRoom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("toggle-on-*-television-LivingRoom", {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("idle", {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "idle",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("drop-*-remote-table-LivingRoom", {
        "obj": "remote",
        "fur": "table",
        "room": "LivingRoom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": True,
        "end_state": False
    }),
    ("toogle-off-*-television-LivingRoom", {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 0],
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
]

# 5
Move_plant_at_night_subgoals_order = [
    ("toogle-on-*-light-Kitchen", {
        "obj": None,
        "fur": "light",
        "room": "Kitchen",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-pot_plant-table-LivingRoom", {
        "obj": "pot_plant",
        "fur": "table",
        "room": "LivingRoom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("drop-*-pot_plant-table-Kitchen", {
        "obj": "pot_plant",
        "fur": "table",
        "room": "Kitchen",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False,
    }),
    ("toogle-off-*-light-Kitchen", {
        "obj": None,
        "fur": "light",
        "room": "Kitchen",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 0],
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
    # add final navigation action
    # action go to random room
]

# 6
Take_shower_subgoals_order = [
    ("toogle-on-*-shower-Bathroom", {
        "obj": None,
        "fur": "shower",
        "room": "Bathroom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("idle", {
        "obj": None,
        "fur": "shower",
        "room": "Bathroom",
        "pos": None,
        "action": "idle",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("toogle-off-*-shower-Bathroom", {
        "obj": None,
        "fur": "shower",
        "room": "Bathroom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("open-*-*-closet-Bedroom", {
        "obj": None,
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False,
    }),
    ("pickup-*-clothes-closet-Bedroom", {
        "obj": "clothes",
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False,
    }),
    ("close-*-*-closet-Bedroom", {
        "obj": None,
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
]

# 7
Do_laundry_subgoals_order = [
    ("pickup-*-clothes-bed-Bedroom", {
        "obj": "clothes",
        "fur": "bed",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("open-*-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("drop-*-clothes-laundry-Bathroom", {
        "obj": "clothes",
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": False,
        "end_state": False
    }),
    ("toggle-on-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("idle", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "idle",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("toggle-off-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("open-*-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-clothes-laundry-Bathroom", {
        "obj": "clothes",
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("open-*-*-closet-Bedroom", {
        "obj": None,
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False,
    }),
    ("drop-*-clothes-closet-Bedroom", {
        "obj": "clothes",
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False,
    }),
    ("close-*-*-closet-Bedroom", {
        "obj": None,
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
]

# 8
Watch_movie_cozily_subgoals_order = [
    ("pickup-*-pillow-bed-Bedroom", {
        "obj": "pillow",
        "fur": "bed",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-remote-sofa-LivingRoom", {
        "obj": "remote",
        "fur": "sofa",
        "room": "LivingRoom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("toogle-on-*-television-LivingRoom", {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("drop-*-pillow-sofa-LivingRoom", {
        "obj": "pillow",
        "fur": "sofa",
        "room": "LivingRoom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("idle", {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "idle",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("toogle-off-*-television-LivingRoom", {
        "obj": None,
        "fur": "television",
        "room": "LivingRoom",
        "pos": None,
        "action": "toggle",
        "state": ["toggleable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("drop-*-remote-sofa-LivingRoom", {
        "obj": "remote",
        "fur": "sofa",
        "room": "LivingRoom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
]

# 9
Change_outfit_subgoals_order = [
    ("open-*-*-closet-Bedroom", {
        "obj": None,
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("pickup-*-clothes-closet-Bedroom", {
        "obj": "clothes",
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-closet-Bedroom", {
        "obj": None,
        "fur": "closet",
        "room": "Bedroom",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": False
    }),
    ("open-*-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False
    }),
    ("drop-*-clothes-laundry-Bathroom", {
        "obj": "clothes",
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-laundry-Bathroom", {
        "obj": None,
        "fur": "laundry",
        "room": "Bathroom",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": True
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
]

# 10
Clean_LivingRoom_table_subgoals_order = [
    ("open-*-*-closet-Kitchen", {
        "obj": None,
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False,
    }),
    ("pickup-*-towel-closet-Kitchen", {
        "obj": "towel",
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "pickup",
        "state": None,
        "can_skip": False,
        "end_state": False
    }),
    ("close-*-*-closet-Kitchen", {
        "obj": None,
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": False,
        "end_state": False,
    }),
    ("clean-*-*-table-LivingRoom", {
        "obj": None,
        "fur": "table",
        "room": "LivingRoom",
        "pos": None,
        "action": "clean",
        "state": ["dustyable", 0],
        "can_skip": False,
        "end_state": False
    }),
    ("open-*-*-closet-Kitchen", {
        "obj": None,
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "open",
        "state": ["openable", 1],
        "can_skip": False,
        "end_state": False,
    }),
    ("drop-*-towel-closet-Kitchen", {
        "obj": "towel",
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "drop",
        "state": None,
        "can_skip": True,
        "end_state": False
    }),
    ("close-*-*-closet-Kitchen", {
        "obj": None,
        "fur": "closet",
        "room": "Kitchen",
        "pos": None,
        "action": "close",
        "state": ["openable", 0],
        "can_skip": True,
        "end_state": True,
    }),
    # ("random_walk", {
    #     "obj": None,
    #     "fur": None,
    #     "room": None,
    #     "pos": None,
    #     "action": None,
    #     "state": None,
    #     "can_skip": False,
    #     "end_state": True
    # })
    # add final navigation action
    # action go to random rooms
]

MISSION_TO_SUBGOALS = {
    'get_night_snack': Get_night_snack_subgoals_order,
    'get_snack': Get_snack_subgoals_order,
    'feed_dog': Feed_dog_subgoals_order,
    'watch_news_on_tv': Watch_news_on_tv_subgoals_order,
    'move_plant_at_night': Move_plant_at_night_subgoals_order,
    'take_shower': Take_shower_subgoals_order,
    'do_laundry': Do_laundry_subgoals_order,
    'watch_movie_cozily': Watch_movie_cozily_subgoals_order,
    'change_outfit': Change_outfit_subgoals_order,
    'clean_living_room_table': Clean_LivingRoom_table_subgoals_order,
    # 'random_walk': None
}

MISSION_TO_INFERENCE = {
    mission: [subgoal[0] for subgoal in MISSION_TO_SUBGOALS[mission]] for mission in MISSION_TO_SUBGOALS.keys()
}
