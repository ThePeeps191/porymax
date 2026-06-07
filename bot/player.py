# poke_env.Player subclass: runs Kakuna inference (optionally MCTS) in choose_move().

from poke_env.player import BattleOrder, Player

from bot.model import PorymaxModel
from bot.preprocess import (
    action_idx_to_order,
    battle_to_obs,
    init_rl2s,
    next_rl2s,
    obs_to_tensors,
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
        self._battle_states = {}

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
            order = action_idx_to_order(
                action_idx, battle, PorymaxModel.act_space()
            )

            bs["hidden_state"] = new_hidden
            bs["rl2s"] = next_rl2s(0.0, action_idx, _ACTION_SIZE)
            bs["time_idx"] += 1
            bs["last_action"] = action_idx

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


def _fresh_battle_state():
    return {
        "hidden_state": None,
        "rl2s": init_rl2s(_ACTION_SIZE),
        "time_idx": 0,
        "last_action": None,
    }


def _nograd():
    import torch
    return torch.no_grad()
