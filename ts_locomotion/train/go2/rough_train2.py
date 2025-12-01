"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from go2_base2 import train_main

# ---- patch only the bits that change --------------------------------------
env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    "base_init_pos": [0.0, 0.0, 0.45],
    'max_push_vel_xy': 1.0,
    "episode_length_s": 30.0,
    "resampling_time_s": 10.0,
    "termination_if_roll_greater_than": 60,  # degree.
    "termination_if_pitch_greater_than": 60,    
    "termination_if_relative_height_lower_than": 0.15,
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "tracking_lin_vel": 8. , # 5.0, # 1.5,
        "tracking_ang_vel": 4. , # 2.5, # 0.75,
        "lin_vel_z": -5.,
        "relative_base_height": -30.0,
        "orientation": -10., #-0.01,
        "ang_vel_xy": -0.1,
        "collision": -10.,
        "front_feet_clearance": 30.0,
        "rear_feet_clearance": 30.0,
        # "foot_clearance": -0.5,
        "action_rate": -0.05,
        "action_smoothness": -0.05,
        "dof_acc": -2.5e-7,
        "dof_pos_limits": -10.0,
        "powers": -2e-5,
        "termination": -30.0,
        "contact_no_vel": -0.05,
        "stand_pos": -.5,
        "feet_contact_forces": -1e-4,
        "contact_no_commands": -2.0, # -1.0, # -0.5,
        "contact_stair": -0.,
        "both_front_feet_airborne": -1.,
        "both_rear_feet_airborne": -1.,
        # "stand_still": -0.5,
        "swing_stuck": -10.0,
        "front_step_distance_commmands":5.0,
        "rear_step_distance_commmands":4.0, # 2.5,
    },
}


terrain_cfg_patch = {
    # "terrain_type": "trimesh", #plane
    "terrain_type": "variable_terrain3", #plane
    "subterrain_size": 5.,
    "horizontal_scale": .0025,
    "vertical_scale": 0.02,
    "cols": 7,  #should be more than 5
    "rows": 3,   #should be more than 5
    "selected_terrains":{
        # "flat_terrain" : {"probability": 0.3},
        # "stamble_terrain" : {"probability": 0.1},
        # "pyramid_sloped_terrain" : {"probability": 0.1},
        # "discrete_obstacles_terrain" : {"probability": 0.1},
        "pyramid_down_stairs_terrain" : {"probability": 0.2},
        # "blocky_terrain": {"probability": 0.1},
        "pyramid_steep_down_stairs_terrain" : {"probability": 0.2},
    }
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": True,
    "curriculum_tasks": {
        0: {
            # "stand_plane": "plane_center",
            # "stand_rough": "rough_center",
            "walk_linear_plane": "plane",
            "walk_angular_plane": "plane_center",
            },
        1: {
            # "walk_linear_rough": "rough",
            # "walk_angular_rough": "rough_center",
            # "stand_stair_5cm": "stair_5cm_center",
            "walk_up_slope_5cm": "slope_0-5cm",
            "walk_up_stair_5cm": "stair_0-5cm",
            "walk_down_slope_5cm": "slope_5-10cm",
            "walk_down_stair_5cm": "stair_5-10cm",
            },
        2: {
            # "walk_angular_rough": "rough_center",
            # "stand_stair_10cm": "stair_10cm_center",
            "walk_up_slope_10cm": "slope_5-10cm",
            "walk_up_stair_10cm": "stair_5-10cm",
            "walk_down_slope_10cm": "slope_10-15cm",
            "walk_down_stair_10cm": "stair_10-15cm",
            },
        3: {
            # "stand_stair_15cm": "stair_15cm_center",
            "walk_up_slope_15cm": "slope_10-15cm",
            "walk_up_stair_15cm": "stair_10-15cm",
            "walk_down_slope_15cm": "slope_15-cm",
            "walk_down_stair_15cm": "stair_15-cm",
            },
        # 4: {},
    },
    "start_curriculum": 0, # 0,
    "curriculum_iteration_threshold": 3000, #1 calculated 1 iteration is 1 seocnd 2000 = 
    "lin_vel_x_range": [-1.0, 1.0],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-1.0, 1.0],
}


# leave other five cfgs untouched
CFG_PATCHES = (
    env_cfg_patch,  # env_cfg
    {},  # obs_cfg
    {},  # noise_cfg
    reward_cfg_patch,  # reward_cfg
    command_cfg_patch,  # command_cfg
    terrain_cfg_patch,
)

if __name__ == "__main__":
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="go2_walking")
