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
    is_action_immune,
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
        self._game_counter = 0

    def teampreview(self, battle):
        if self._use_guide:
            lead = get_lead_species(battle)
            team_list = list(battle.team.values()) or list(getattr(battle, "teampreview_team", []))
            if not team_list:
                return self.random_teampreview(battle)
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
        if battle.battle_tag not in self._battle_states:
            self._on_battle_start(battle)
        if self._mcts_enabled:
            return self._mcts_choose(battle)
        return self._policy_choose(battle)

    def _on_battle_start(self, battle):
        self._game_counter += 1
        tag = battle.battle_tag
        our_names = [p.species for p in battle.team.values() if p]
        opp_names = [p.species for p in battle.teampreview_opponent_team if p]
        print(f"\n  Game #{self._game_counter}")
        print(f"  BATTLE START: {tag}")
        print(f"  Us: {', '.join(our_names)}")
        print(f"  Them: {', '.join(opp_names)}")
        print(f"  Spectate: https://play.pokemonshowdown.com/{tag}\n")

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
            source = "POLICY"

            if self._use_guide:
                action_idx, trace = _apply_guide(battle, action_idx, bs)
                _print_turn_trace(battle, source, bs["time_idx"], action_idx, trace)

            order = action_idx_to_order(
                action_idx, battle, PorymaxModel.act_space()
            )

            bs["hidden_state"] = new_hidden
            bs["rl2s"] = next_rl2s(0.0, action_idx, _ACTION_SIZE)
            bs["time_idx"] += 1
            bs["last_action_idx"] = action_idx
            bs["last_opp_hp_pct"] = _opp_hp(battle)
            bs["last_opp_species"] = _opp_species(battle)
            if _is_iron_treads(battle) and _is_move(action_idx, "steel beam", battle):
                bs["steel_beam_used"] = True
            elif not _is_iron_treads(battle):
                bs["steel_beam_used"] = False

            if order is None:
                return self.choose_random_move(battle)
            return order

        except Exception:
            return self.choose_random_move(battle)

    def _mcts_choose(self, battle):
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

            from bot.mcts import run_mcts
            action_idx, new_hidden = run_mcts(
                PorymaxModel.policy(), obs_t, rl2s_t, time_t, battle,
                hidden_state=bs["hidden_state"],
                obs_space=PorymaxModel.obs_space(),
                act_space=PorymaxModel.act_space(),
                n_sims=50, c_puct=1.0, depth=2,
            )
            source = "MCTS"

            if self._use_guide:
                action_idx, trace = _apply_guide(battle, action_idx, bs)
                _print_turn_trace(battle, source, bs["time_idx"], action_idx, trace)

            order = action_idx_to_order(
                action_idx, battle, PorymaxModel.act_space()
            )

            bs["hidden_state"] = new_hidden
            bs["rl2s"] = next_rl2s(0.0, action_idx, _ACTION_SIZE)
            bs["time_idx"] += 1
            bs["last_action_idx"] = action_idx
            bs["last_opp_hp_pct"] = _opp_hp(battle)
            bs["last_opp_species"] = _opp_species(battle)
            if _is_iron_treads(battle) and _is_move(action_idx, "steel beam", battle):
                bs["steel_beam_used"] = True
            elif not _is_iron_treads(battle):
                bs["steel_beam_used"] = False

            if order is None:
                return self.choose_random_move(battle)
            return order

        except Exception:
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        tag = battle.battle_tag
        won = getattr(battle, "won", None)
        result = "WON" if won else ("LOST" if won is False else "TIED")
        rating = getattr(battle, "rating", None)
        rating_str = f" (rating: {rating})" if rating is not None else ""
        print(f"  BATTLE END: {tag} — {result}{rating_str}\n")
        self._battle_states.pop(tag, None)


_ACTION_SIZE = 13


def _print_turn_trace(battle, source, turn, action_idx, trace):
    active = battle.active_pokemon
    mon = active.species if active else "???"
    final = _action_name(action_idx, battle)
    if not trace:
        print(f"  {mon} T{turn}: {source} → {final}")
        return
    parts = " → ".join(t for t in trace if t)
    print(f"  {mon} T{turn}: {source} → {final}  [{parts}]")


def _action_name(action_idx, battle):
    if 0 <= action_idx <= 3 and action_idx < len(battle.available_moves):
        return battle.available_moves[action_idx].id.upper()
    if 4 <= action_idx <= 8:
        idx = action_idx - 4
        if idx < len(battle.available_switches):
            return f"switch {battle.available_switches[idx].species.upper()}"
    if action_idx >= 9 and battle.can_tera:
        idx = action_idx - 9
        if idx < len(battle.available_moves):
            return f"tera {battle.available_moves[idx].id.upper()}"
    return f"idx={action_idx}"


def _apply_guide(battle, action_idx, bs):
    trace = []
    action_idx = _apply_immunity(battle, action_idx, trace)
    action_idx = _apply_steel_beam_limit(battle, action_idx, bs, trace)
    action_idx = _apply_forced_switch(battle, action_idx, bs, trace)
    action_idx = _apply_hazard_check(battle, action_idx, bs, trace)
    action_idx = _apply_move_failure(battle, action_idx, bs, trace)
    action_idx = _apply_team_hints(battle, action_idx, trace)
    return action_idx, trace


def _apply_immunity(battle, action_idx, trace):
    for _ in range(50):
        if not is_action_immune(action_idx, battle):
            return action_idx
        old = _action_name(action_idx, battle)
        action_idx = _resample_any(battle)
        new = _action_name(action_idx, battle)
        if old != new:
            trace.append(f"IMMUNE: {old} → {new}")
    return action_idx


def _apply_steel_beam_limit(battle, action_idx, bs, trace):
    used = bs.get("steel_beam_used", False)
    if used and _is_iron_treads(battle) and _is_move(action_idx, "steel beam", battle):
        old = _action_name(action_idx, battle)
        action_idx = _resample_any(battle)
        new = _action_name(action_idx, battle)
        trace.append(f"STEEL BEAM LIMIT: {old} → {new}")
    return action_idx


def _apply_forced_switch(battle, action_idx, bs, trace):
    forced = get_forced_switch_actions(battle)
    if not forced:
        return _apply_hazard_check(battle, action_idx, bs, trace)

    if random.random() < 0.15:
        forced_list = sorted(forced)
        old = _action_name(action_idx, battle)
        action_idx = random.choice(forced_list)
        new = _action_name(action_idx, battle)
        trace.append(f"FORCED SWITCH: {old} → {new}")
        return action_idx

    return _apply_hazard_check(battle, action_idx, bs, trace)


def _apply_hazard_check(battle, action_idx, bs, trace):
    malus = get_hazard_malus(battle)
    is_switch = 4 <= action_idx <= 8
    is_tera_move = action_idx >= 9

    if is_switch and malus >= 1 and _spinner_is_dead(battle):
        switches = battle.available_switches
        switch_idx = action_idx - 4
        if 0 <= switch_idx < len(switches):
            target = switches[switch_idx]
            hp = target.current_hp_fraction if hasattr(target, "current_hp_fraction") else None
            if hp is not None and hp < 0.4:
                old = _action_name(action_idx, battle)
                action_idx = _resample_move(battle, action_idx, bs)
                new = _action_name(action_idx, battle)
                trace.append(f"HAZARD BLOCK (spinner dead, {old} hp<40%): → {new}")
                return action_idx

    if is_switch and malus >= 3:
        switches = battle.available_switches
        switch_idx = action_idx - 4
        if 0 <= switch_idx < len(switches):
            target = switches[switch_idx]
            hp = target.current_hp_fraction if hasattr(target, "current_hp_fraction") else None
            if hp is not None and hp < 0.5:
                old = _action_name(action_idx, battle)
                action_idx = _resample_move(battle, action_idx, bs)
                new = _action_name(action_idx, battle)
                trace.append(f"HAZARD WARN: {old} hp<50% → {new}")
                return action_idx

    if is_switch and malus >= 4:
        old = _action_name(action_idx, battle)
        action_idx = _resample_move(battle, action_idx, bs)
        new = _action_name(action_idx, battle)
        trace.append(f"HAZARD BLOCK (heavy hazards): {old} → {new}")
        return action_idx

    if is_tera_move and malus >= 4:
        old = _action_name(action_idx, battle)
        action_idx = _resample_move(battle, action_idx, bs)
        new = _action_name(action_idx, battle)
        trace.append(f"HAZARD BLOCK: {old} → {new}")
        return action_idx

    return _apply_move_failure(battle, action_idx, bs, trace)


def _apply_move_failure(battle, action_idx, bs, trace):
    last = bs.get("last_action_idx")
    prev_hp = bs.get("last_opp_hp_pct")
    prev_species = bs.get("last_opp_species")

    failed_action = get_move_failure_actions(battle, last, prev_hp, prev_species)
    if failed_action is not None and action_idx == failed_action:
        old = _action_name(action_idx, battle)
        action_idx = _resample_any(battle)
        new = _action_name(action_idx, battle)
        trace.append(f"FAILED LAST TURN: {old} → {new}")
        return action_idx

    return _apply_team_hints(battle, action_idx, trace)


def _apply_team_hints(battle, action_idx, trace):
    preferred = get_preferred_actions(battle)
    if not preferred:
        return action_idx
    if action_idx in preferred:
        return action_idx
    if random.random() < 0.10:
        old = _action_name(action_idx, battle)
        preferred_list = sorted(preferred)
        if preferred_list:
            action_idx = random.choice(preferred_list)
            new = _action_name(action_idx, battle)
            trace.append(f"TEAM HINT: {old} → {new}")
    return action_idx


def _resample_move(battle, action_idx, bs):
    if random.random() < 0.30:
        return _resample_any(battle)
    return _apply_team_hints(battle, action_idx, [])


def _resample_any(battle):
    moves = list(range(len(battle.available_moves)))
    switches = [4 + i for i in range(len(battle.available_switches))]
    all_legal = moves + switches
    if battle.can_tera:
        all_legal = all_legal + [9 + i for i in range(len(battle.available_moves))]
    return random.choice(all_legal) if all_legal else 0


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


def _spinner_is_dead(battle):
    for p in battle.team.values():
        if p and p.species and p.species.lower() == "iron treads":
            return getattr(p, "fainted", False)
    return False


def _is_iron_treads(battle):
    active = battle.active_pokemon
    return active is not None and active.species and active.species.lower() == "iron treads"


def _is_move(action_idx, move_id, battle):
    if 0 <= action_idx <= 3 and action_idx < len(battle.available_moves):
        return battle.available_moves[action_idx].id.lower() == move_id
    if action_idx >= 9 and battle.can_tera:
        idx = action_idx - 9
        if idx < len(battle.available_moves):
            return battle.available_moves[idx].id.lower() == move_id
    return False


def _fresh_battle_state():
    return {
        "hidden_state": None,
        "rl2s": init_rl2s(_ACTION_SIZE),
        "time_idx": 0,
        "last_action_idx": None,
        "last_opp_hp_pct": None,
        "last_opp_species": None,
        "steel_beam_used": False,
    }


def _nograd():
    import torch
    return torch.no_grad()
