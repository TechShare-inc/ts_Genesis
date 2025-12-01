"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from Zxxx_base import train_main


# ---- patch only the bits that change --------------------------------------

env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    "base_init_pos": [0.0, 0.0, 0.45],
    "termination_if_roll_greater_than": 100,
    "angle_termination_duration": 5.0, #seconds
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 0.9,
    "reward_scales": {
        "tracking_lin_vel": 8., # 5.0, # 1.5,
        "tracking_ang_vel": 4., # 2.5, # 0.75,
        "lin_vel_z": -5.0, # -5.0,
        "relative_base_height": -30.0,
        "orientation": -30.0, #fixed!
        "orientation_x": -1.0, #fixed!
        "ang_vel_xy": -.3, #-0.5, # -0.05, #fixed!
        "ang_vel_x": -.5, # -0.05, #fixed!
        "collision": -10.0, #fixed!
        "hip_pos": -1.0,
        "front_feet_clearance": 15.0,
        "rear_feet_clearance": 15.0,
        "action_rate": -0.05,
        "action_smoothness": -0.05,
        # "dof_vel": -2.5e-4,
        "hip_dof_vel": -2.5e-2,
        "thigh_dof_vel": -5.0e-4, # -2.5e-3,
        "dof_acc": -1e-6, # -2.5e-7,
        "dof_pos_limits": -10.0, #fixed!
        "powers": -2e-5, # -1e-5,
        "termination": -30.0,
        "contact_no_vel": -0.5, # -0.2,
        "feet_contact_forces": -0.05,
        # "stand_still": -.3, # -0.05, # -0.5,
        "foot_vel": -1., # -.3,
        "contact_no_commands": -2.0, # -1.0, # -0.5,
        "both_front_feet_airborne": -1.0,
        "both_rear_feet_airborne": -1.0,
        # "hip_thigh_vel": -2.0e-1, # -1.0e-3, # 1.0e-2
        # "step_distance": 15.0,
        # "front_step_distance": 10.0,
        # "rear_step_distance": 5.0,
        "front_step_distance_commmands":10.0,
        "rear_step_distance_commmands":8.0, # 2.5,
        # "step_distance_penalty": -5.0,
        # "step_timing": -5.0,
        "front_step_timing": -10.0,
        "rear_step_timing": -8.0,
        "swing_foot": -0.5,
    },
}

terrain_cfg_patch = {
    "terrain_type": "plane", #plane
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": True,
    "curriculum_iteration_threshold": 2000,
    "mean_reward_threshold": 60,
    "lin_vel_x_range": [-0.5, 0.5],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-.5, .5],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="Z_walking")
