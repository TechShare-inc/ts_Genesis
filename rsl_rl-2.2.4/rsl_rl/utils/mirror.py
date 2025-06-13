# RNNには現時点で未対応.

import torch

def mirror_data_augmentation(obs=None, actions=None, env=None, is_critic=False):
    """
    Applies mirror augmentation to observations and/or actions for data augmentation.
    
    Args:
        obs (torch.Tensor or None): Observations of shape [batch_size, obs_dim]
        actions (torch.Tensor or None): Actions of shape [batch_size, act_dim]
        env: The environment with `mirror_observation()` and `mirror_func_dict`
        is_critic (bool): Whether this is for critic (True disables action augmentation)

    Returns:
        Tuple[obs_aug, actions_aug]: Each is a tensor of shape [2 * batch_size, ...]
    """
    assert obs is not None or actions is not None, "Either obs or actions must be provided."

    obs_aug, actions_aug = None, None

    if obs is not None:
        obs_components = (
            env.privileged_obs_components if is_critic and env.privileged_obs_components is not None
            else env.obs_components
        )
        if obs.ndim == 3:
            # RNN: obs.shape = [B, T, D]
            B, T, D = obs.shape
            mirrored_obs = torch.stack([
                env.mirror_observation(obs[i], obs_components)
                for i in range(B)
            ])  # shape: [B, T, D]
        else:
            # default: obs.shape = [B, D]
            mirrored_obs = env.mirror_observation(obs, obs_components)

        obs_aug = torch.cat([obs, mirrored_obs], dim=0) # shape: [2*B, T, D] or [2*B, D]

        # if is_critic and env.privileged_obs_components is not None:
        #     mirrored_obs = env.mirror_observation(obs, env.privileged_obs_components)
        # else:
        #     mirrored_obs = env.mirror_observation(obs, env.obs_components)
        # obs_aug = torch.cat([obs, mirrored_obs], dim=0)  # shape: [2 * batch_size, obs_dim]

    if not is_critic and actions is not None:
        mirror_action_fn = env.mirror_func_dict["actions"]

        if actions.ndim == 3:
            # RNN: actions.shape = [B, T, A]
            B, T, A = actions.shape
            mirrored_actions = torch.stack([
                mirror_action_fn(actions[i]) for i in range(B)
            ])  # shape: [B, T, A]
        else:
            # default: [B, A]
            mirrored_actions = mirror_action_fn(actions)

        actions_aug = torch.cat([actions, mirrored_actions], dim=0)  # [2*B, T, A] or [2*B, A]

    # if not is_critic and actions is not None:
    #     mirror_action_fn = env.mirror_func_dict["actions"]
    #     mirrored_actions = mirror_action_fn(actions)
    #     actions_aug = torch.cat([actions, mirrored_actions], dim=0)  # shape: [2 * batch_size, act_dim]

    return obs_aug, actions_aug