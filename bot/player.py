# poke_env.Player subclass: runs Kakuna inference (optionally MCTS) in choose_move().

import random

from poke_env.player import BattleOrder, Player

from bot.model import PorymaxModel
from bot.preprocess import (
    action_idx_to_order,
    battle_to_obs,
    init_rl2s,
    next_rl2s,
    obs_to_tensors,
)
from bot.team_guide import (
    get_forced_switch_actions,
    get_hazard_malus,
    get_lead_species,
    get_move_failure_actions,
    get_preferred_actions,
    is_guide_active,
)


class PorymaxPlayer(Player):

    def __init__(
        self,
        *,
        server_configuration=None,
        username=None,
        team=None,
        battle_format="gen9ou",
        temperature=1.0,
        mcts_enabled=False,
        team_file=None,
        **kwargs,
    ):
        player_kwargs = {"battle_format": battle_format, **kwargs}
        if server_configuration is not None:
            player_kwargs["server_configuration"] = server_configuration
        if username is not None and "account_configuration" not in player_kwargs:
            from poke_env import AccountConfiguration
            player_kwargs["account_configuration"] = AccountConfiguration(username, None)
        if team is not None:
            player_kwargs["team"] = team
        super().__init__(**player_kwargs)

        self._temperature = temperature
        self._mcts_enabled = mcts_enabled
        self._team_file = team_file
        self._use_guide = is_guide_active(team_file)
        self._battle_states = {}

    def teampreview(self, battle):
        if self._use_guide:
            lead = get_lead_species(battle)
            team_list = list(battle.team.values())
            lead_pos = None
            for i, p in enumerate(team_list):
                if p.species.lower() == lead:
                    lead_pos = i + 1
                    break
            if lead_pos is not None:
                members = [lead_pos] + [j + 1 for j in range(len(team_list)) if j + 1 != lead_pos]
                return "/team " + "".join(str(c) for c in members)
        return self.random_teampreview(battle)

    def choose_move(self, battle):
        if self._mcts_enabled:
            return self._mcts_choose(battle)
        return self._policy_choose(battle)

    def _policy_choose(self, battle):
        tag = battle.battle_tag
        if tag not in self._battle_states:
            self._battle_states[tag] = _fresh_battle_state()
        bs = self._battle_states[tag]

        try:
            obs = battle_to_obs(
                battle, PorymaxModel.obs_space(), PorymaxModel.act_space()
            )
            obs_t, rl2s_t, time_t = obs_to_tensors(
                obs, bs["rl2s"], bs["time_idx"], PorymaxModel.device()
            )

            with _nograd():
                actions, new_hidden = PorymaxModel.policy().get_actions(
                    obs=obs_t,
                    rl2s=rl2s_t,
                    time_idxs=time_t,
                    hidden_state=bs["hidden_state"],
                    sample=self._temperature > 0.0,
                )

            action_idx = int(actions.squeeze().cpu().numpy())

            if self._use_guide:
                action_idx = _apply_guide(battle, action_idx, bs)

            order = action_idx_to_order(
                action_idx, battle, PorymaxModel.act_space()
            )

            bs["hidden_state"] = new_hidden
            bs["rl2s"] = next_rl2s(0.0, action_idx, _ACTION_SIZE)
            bs["time_idx"] += 1
            bs["last_action_idx"] = action_idx
            bs["last_opp_hp_pct"] = _opp_hp(battle)
            bs["last_opp_species"] = _opp_species(battle)

            if order is None:
                return self.choose_random_move(battle)
            return order

        except Exception:
            return self.choose_random_move(battle)

    def _mcts_choose(self, battle):
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self._battle_states.pop(battle.battle_tag, None)


_ACTION_SIZE = 13


def _apply_guide(battle, action_idx, bs):
    return _apply_forced_switch(battle, action_idx, bs)


def _apply_forced_switch(battle, action_idx, bs):
    forced = get_forced_switch_actions(battle)
    if not forced:
        return _apply_hazard_check(battle, action_idx, bs)

    if random.random() < 0.50:
        forced_list = sorted(forced)
        return random.choice(forced_list)

    return _apply_hazard_check(battle, action_idx, bs)


def _apply_hazard_check(battle, action_idx, bs):
    malus = get_hazard_malus(battle)
    is_switch = 4 <= action_idx <= 8
    is_tera_move = action_idx >= 9

    if is_switch and malus >= 3:
        switches = battle.available_switches
        switch_idx = action_idx - 4
        if 0 <= switch_idx < len(switches):
            target = switches[switch_idx]
            hp = target.current_hp_fraction if hasattr(target, "current_hp_fraction") else None
            if hp is not None and hp < 0.5:
                return _resample_move(battle, action_idx, bs)

    if is_switch and malus >= 4:
        return _resample_move(battle, action_idx, bs)

    if is_tera_move and malus >= 4:
        return _resample_move(battle, action_idx, bs)

    return _apply_move_failure(battle, action_idx, bs)


def _apply_move_failure(battle, action_idx, bs):
    last = bs.get("last_action_idx")
    prev_hp = bs.get("last_opp_hp_pct")
    prev_species = bs.get("last_opp_species")

    failed_action = get_move_failure_actions(battle, last, prev_hp, prev_species)
    if failed_action is not None and action_idx == failed_action:
        return _resample_any(battle)

    return _apply_team_hints(battle, action_idx)


def _apply_team_hints(battle, action_idx):
    preferred = get_preferred_actions(battle)
    if not preferred:
        return action_idx
    if action_idx in preferred:
        return action_idx
    if random.random() < 0.20:
        return random.choice(sorted(preferred))
    return action_idx


def _resample_move(battle, action_idx, bs):
    if random.random() < 0.70:
        return _resample_any(battle)
    return _apply_team_hints(battle, action_idx)


def _resample_any(battle):
    import random as _random
    moves = list(range(len(battle.available_moves)))
    switches = [4 + i for i in range(len(battle.available_switches))]
    all_legal = moves + switches
    if battle.can_tera:
        all_legal = all_legal + [9 + i for i in range(len(battle.available_moves))]
    return _random.choice(all_legal) if all_legal else 0


def _opp_hp(battle):
    opp = battle.opponent_active_pokemon
    if opp is None:
        return None
    return getattr(opp, "current_hp_fraction", None)


def _opp_species(battle):
    opp = battle.opponent_active_pokemon
    if opp is None:
        return None
    return opp.species.lower()


def _fresh_battle_state():
    return {
        "hidden_state": None,
        "rl2s": init_rl2s(_ACTION_SIZE),
        "time_idx": 0,
        "last_action_idx": None,
        "last_opp_hp_pct": None,
        "last_opp_species": None,
    }


def _nograd():
    import torch
    return torch.no_grad()
