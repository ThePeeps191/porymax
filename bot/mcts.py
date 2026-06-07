# Batched MCTS: depth-1 UCB1 + depth-2 opponent modeling via policy self-play.

import math
import random

import numpy as np
import torch


def run_mcts(policy, obs_t, rl2s_t, time_t, battle, hidden_state,
             obs_space=None, act_space=None,
             n_sims=50, c_puct=1.0, depth=2):
    """Run MCTS and return (best_action_idx, new_hidden_state).

    depth=1: UCB1 over 13 immediate actions with critic Q-values.
    depth=2: Also models opponent's response by running the policy on the
             current observation (self-play approximation), then evaluates
             grandchildren with the critic.
    """
    device = obs_t["numbers"].device
    illegal_mask = obs_t["illegal_actions"]
    legal = torch.where(~illegal_mask[0, 0])[0].tolist()
    if not legal:
        return 0, hidden_state

    with torch.no_grad():
        tstep_emb = policy.tstep_encoder(obs=obs_t, rl2s=rl2s_t)
        traj_emb, new_hidden = policy.traj_encoder(
            tstep_emb, time_idxs=time_t, hidden_state=hidden_state,
        )
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
    q_range = max(q_vals.max() - q_min, 1.0)
    q_norm = (q_vals - q_min) / q_range

    root = _Node(action_idx=None, parent=None)
    for i, a in enumerate(legal):
        root.children[a] = _Node(
            action_idx=a,
            parent=root,
            prior=float(priors[a]),
            q_norm=float(q_norm[i]),
        )

    if depth >= 2:
        _build_depth2(root, legal, priors, policy, obs_t, rl2s_t, time_t,
                      device, num_gammas)

    for _ in range(n_sims):
        node = root
        best = _select_ucb1(root, legal, c_puct)
        child = root.children[best]
        child.visits += 1
        root.visits += 1

        if child.children:
            opp_legal = list(child.children.keys())
            opp_best = _select_ucb1(child, opp_legal, c_puct)
            child.children[opp_best].visits += 1

    return max(legal, key=lambda a: root.children[a].visits), new_hidden


def _build_depth2(root, legal, our_priors, policy, obs_t, rl2s_t, time_t,
                  device, num_gammas):
    """Pre-build depth-2 grandchildren by modeling opponent's likely actions.

    Uses the policy's own output as an opponent model (self-play approximation).
    The policy distribution over our 13 actions is treated as an estimate of what
    the opponent is likely to do, since both sides use the same policy in self-play.
    """
    top_n = min(5, len(legal))
    sorted_legal = sorted(legal, key=lambda a: our_priors[a], reverse=True)
    top_opp = sorted_legal[:top_n]

    for child in root.children.values():
        for a_opp in top_opp:
            gc = _Node(
                action_idx=a_opp,
                parent=child,
                prior=float(our_priors[a_opp]),
            )
            child.children[a_opp] = gc

    B = len(top_opp)
    opp_onehot = torch.zeros(1, B, 1, num_gammas, 13, device=device)
    for i, a in enumerate(top_opp):
        opp_onehot[0, i, 0, :, a] = 1.0

    with torch.no_grad():
        tstep = policy.tstep_encoder(obs=obs_t, rl2s=rl2s_t)
        traj_emb, _ = policy.traj_encoder(tstep, time_idxs=time_t)

    traj_batch = traj_emb.repeat(B, 1, 1)
    with torch.no_grad():
        q_dist = policy.critics(state=traj_batch, action=opp_onehot)
        q_scalar = policy.critics.bin_dist_to_raw_vals(q_dist)
        q_vals = q_scalar[0, :, 0, :, -1, 0].mean(dim=-1).detach().cpu().numpy()

    q_m = q_vals.min()
    q_r = max(q_vals.max() - q_m, 1.0)
    for i, a_opp in enumerate(top_opp):
        qn = float((q_vals[i] - q_m) / q_r)
        for child in root.children.values():
            if a_opp in child.children:
                child.children[a_opp].q_norm = qn


class _Node:
    __slots__ = ("action_idx", "parent", "children", "visits", "prior", "q_norm")

    def __init__(self, action_idx, parent, prior=0.0, q_norm=0.0):
        self.action_idx = action_idx
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.prior = prior
        self.q_norm = q_norm


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
