# Batched MCTS: critic-guided UCB1 search over legal actions using Kakuna's value head.

import math
import random

import numpy as np
import torch


def run_mcts(policy, obs_t, rl2s_t, time_t, battle, n_sims=50, c_puct=1.0):
    """Run MCTS and return the best action index.

    Depth-1 MCTS with batched critic evaluation. Compares 13 legal actions using
    UCB1, where each child's Q-value comes from Kakuna's NCriticsTwoHot value head
    (which already accounts for expected opponent response from self-play training).
    """
    device = obs_t["numbers"].device
    illegal_mask = obs_t["illegal_actions"]  # (1, 1, 13)
    legal = torch.where(~illegal_mask[0, 0])[0].tolist()
    if not legal:
        return 0

    with torch.no_grad():
        tstep_emb = policy.tstep_encoder(obs=obs_t, rl2s=rl2s_t)
        traj_emb, _ = policy.traj_encoder(tstep_emb, time_idxs=time_t)
        action_dist = policy.actor(
            traj_emb,
            straight_from_obs={"illegal_actions": obs_t["illegal_actions"]},
        )
        priors = action_dist.probs[0, 0, -1, :].cpu().numpy()

    B = len(legal)
    num_gammas = policy.actor.num_gammas
    action_onehot = torch.zeros(1, B, 1, num_gammas, 13, device=device)
    for i, a in enumerate(legal):
        action_onehot[0, i, 0, :, a] = 1.0

    traj_emb_batch = traj_emb.repeat(B, 1, 1)
    q_dist = policy.critics(state=traj_emb_batch, action=action_onehot)
    q_scalar = policy.critics.bin_dist_to_raw_vals(q_dist)
    q_vals = q_scalar[0, :, 0, :, -1, 0].mean(dim=-1).detach().cpu().numpy()

    q_min = q_vals.min()
    q_max = q_vals.max()
    q_range = max(q_max - q_min, 1.0)
    q_norm = (q_vals - q_min) / q_range

    root = _Node(action_idx=None, parent=None)
    for i, a in enumerate(legal):
        root.children[a] = _Node(
            action_idx=a,
            parent=root,
            prior=float(priors[a]),
            q_norm=float(q_norm[i]),
        )

    for _ in range(n_sims):
        best = _select_ucb1(root, legal, c_puct)
        root.children[best].visits += 1
        root.visits += 1

    return max(legal, key=lambda a: root.children[a].visits)


class _Node:
    __slots__ = ("action_idx", "parent", "children", "visits", "prior", "q_norm")

    def __init__(self, action_idx, parent, prior=0.0, q_norm=0.0):
        self.action_idx = action_idx
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.prior = prior
        self.q_norm = q_norm

    @property
    def avg_value(self):
        if self.visits == 0:
            return self.q_norm
        return self.q_norm


def _select_ucb1(root, legal, c_puct):
    best = None
    best_score = -float("inf")
    sqrt_parent = math.sqrt(max(root.visits, 1))
    for a in legal:
        child = root.children[a]
        q = child.q_norm
        prior = child.prior
        u = c_puct * prior * sqrt_parent / (1.0 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best = a
    return best
