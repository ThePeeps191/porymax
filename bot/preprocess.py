# Battle -> tensor conversion plus 13-action illegal-action mask for Kakuna.

import numpy as np
import torch

from metamon.interface import UniversalState, UniversalAction


def battle_to_obs(battle, obs_space, act_space):
    state = UniversalState.from_Battle(battle)
    obs = obs_space.state_to_obs(state)
    obs["illegal_actions"] = _build_illegal_mask(state, battle, act_space)
    return obs


def _build_illegal_mask(state, battle, act_space):
    legal = UniversalAction.definitely_valid_actions(state=state, battle=battle)
    legal_idxs = [
        act_space.action_to_agent_output(state=state, action=a) for a in legal
    ]
    mask = np.ones(act_space.gym_space.n, dtype=bool)
    for idx in legal_idxs:
        mask[idx] = False
    return mask


def obs_to_tensors(obs, rl2s, time_idx, device):
    obs_t = {
        k: torch.from_numpy(np.asarray(v)).to(device).reshape(1, 1, *v.shape)
        for k, v in obs.items()
    }
    rl2s_t = torch.from_numpy(np.asarray(rl2s)).to(device).reshape(1, 1, -1)
    time_t = torch.tensor([time_idx], dtype=torch.long, device=device).reshape(1, 1, 1)
    return obs_t, rl2s_t, time_t


def init_rl2s(action_size):
    return np.zeros(action_size + 1, dtype=np.float32)


def next_rl2s(reward, action_idx, action_size):
    rl2s = np.zeros(action_size + 1, dtype=np.float32)
    rl2s[0] = float(reward)
    rl2s[1 + int(action_idx)] = 1.0
    return rl2s


def action_idx_to_order(action_idx, battle, act_space):
    state = UniversalState.from_Battle(battle)
    action = act_space.agent_output_to_action(state=state, agent_output=int(action_idx))
    return action.to_BattleOrder(battle)
