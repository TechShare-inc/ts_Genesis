def mirror_linear(linear):
    linear_mirrored = linear.clone()
    linear_mirrored[..., 1] *= -1 # y
    return linear_mirrored

def mirror_angle(angle):
    angle_mirrored = angle.clone()
    angle_mirrored[..., 0] *= -1 # roll
    angle_mirrored[..., 2] *= -1 # yaw
    return angle_mirrored

def mirror_commands(commands):
    commands_mirrored = commands.clone()
    commands_mirrored[..., 1] *= -1 # y
    commands_mirrored[..., 2] *= -1 # yaw
    return commands_mirrored

def mirror_go_joint(joint):
    joint_mirrored = joint.clone()
    joint_mirrored[..., [0, 1, 2]] = joint[..., [3, 4, 5]]
    joint_mirrored[..., [3, 4, 5]] = joint[..., [0, 1, 2]]
    joint_mirrored[..., [6, 7, 8]] = joint[..., [9, 10, 11]]
    joint_mirrored[..., [9, 10, 11]] = joint[..., [6, 7, 8]]
    joint_mirrored[..., 0::3] = -joint_mirrored[..., 0::3] # hip joint
    return joint_mirrored

def mirror_go_foot(foot):
    foot_mirrored = foot.clone()
    foot_mirrored[..., 0] = foot[..., 1]
    foot_mirrored[..., 1] = foot[..., 0]
    foot_mirrored[..., 2] = foot[..., 3]
    foot_mirrored[..., 3] = foot[..., 2]
    return foot_mirrored
