import functools
import json
import logging
import os
import sys
import tempfile
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym
import torch

if sys.platform == "win32":
    torch._dynamo.config.disable = True
    torch.set_default_dtype(torch.float32)

    def _identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    torch.compile = _identity_compile

warnings.filterwarnings("ignore")

import amago
import amago.cli_utils
import amago.nets.transformer
import gin

import metamon
from metamon.baselines import get_baseline
from metamon.env import BattleAgainstBaseline
from metamon.rl.metamon_to_amago import (
    make_placeholder_experiment,
    MetamonAMAGOWrapper,
)
from metamon.rl.pretrained import PretrainedModel
from metamon.interface import (
    get_observation_space,
    get_action_space,
    get_reward_function,
)
from metamon.tokenizer import get_tokenizer


HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.normpath(os.path.join(HERE, "..", "weights"))
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "kakuna.pt")
CONFIG_PATH = os.path.join(WEIGHTS_DIR, "config.txt")

GENS = [9]
FORMATS = ["ou"]
TOTAL_BATTLES = 1
BATTLE_BACKEND = "metamon"
TEAM_SET = "gl_05_26"
OPPONENT = "RandomBaseline"
TEMPERATURE = 1.0
ASYNC_MP_CONTEXT = "spawn" if sys.platform == "win32" else "forkserver"
TOKENIZER = get_tokenizer("DefaultObservationSpace-v1")


def _select_attention_class():
    if not torch.cuda.is_available():
        return (
            amago.nets.transformer.VanillaAttention,
            "VanillaAttention (no CUDA detected)",
        )
    try:
        import flash_attn  # noqa: F401
        return (
            amago.nets.transformer.FlashAttention,
            "FlashAttention (CUDA + flash_attn available)",
        )
    except ImportError:
        return (
            amago.nets.transformer.VanillaAttention,
            "VanillaAttention (flash_attn not installed)",
        )


class LocalKakuna(PretrainedModel):
    def __init__(self, checkpoint_path, attn_class):
        super().__init__(
            model_name="kakuna",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna.gin",
            default_checkpoint=0,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=TOKENIZER,
            battle_backend=BATTLE_BACKEND,
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": TOKENIZER,
            },
        )
        self._checkpoint_path = checkpoint_path
        self._attn_class = attn_class

    def get_path_to_checkpoint(self, checkpoint):
        return self._checkpoint_path

    def initialize_agent(self, checkpoint=None, log=False, action_temperature=1.0):
        amago.cli_utils.use_config(
            self.base_config | {"MetamonDiscrete.temperature": action_temperature},
            [self.model_gin_config_path, self.train_gin_config_path],
            finalize=False,
        )
        gin.bind_parameter(
            "amago.nets.traj_encoders.TformerTrajEncoder.attention_type",
            self._attn_class,
        )
        gin.bind_parameter(
            "MetamonPerceiverTstepEncoder.tokenizer",
            TOKENIZER,
        )
        ckpt = checkpoint if checkpoint is not None else self.default_checkpoint
        ckpt_path = self.get_path_to_checkpoint(ckpt)
        ckpt_base_dir = tempfile.mkdtemp(prefix="metamon_eval_")
        experiment = make_placeholder_experiment(
            ckpt_base_dir=ckpt_base_dir,
            run_name=self.model_name,
            log=log,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        experiment.start()
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        model_state = experiment.policy.state_dict()
        self._validate_checkpoint(ckpt_state, model_state)
        experiment.policy.load_state_dict(ckpt_state, strict=True)
        experiment.policy.on_checkpoint_loaded(is_resume=False)
        return experiment


_STATUS_NAMES = {
    "SLP": "asleep",
    "PAR": "paralyzed",
    "BRN": "burned",
    "FRZ": "frozen",
    "PSN": "poisoned",
    "TOX": "badly poisoned",
    "FNT": "fainted",
}


class BattleCommentaryWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._seen_turn = -1
        self._prev_our = None
        self._prev_opp = None
        self._printed_preview = False

    def reset(self, *args, **kwargs):
        self._seen_turn = -1
        self._prev_our = None
        self._prev_opp = None
        obs, info = self.env.reset(*args, **kwargs)
        battle = self._get_battle()
        if battle and not self._printed_preview:
            self._print_team_preview(battle)
            self._printed_preview = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        battle = self._get_battle()
        if battle and len(battle._mm_battle.turnlist) >= 2:
            completed = battle._mm_battle.turnlist[-2]
            if completed.turn_number > self._seen_turn:
                self._seen_turn = completed.turn_number
                if completed.turn_number > 0:
                    self._print_turn(battle, completed)
        if terminated or truncated:
            self._print_outcome(info)
        return obs, reward, terminated, truncated, info

    def _get_battle(self):
        env = self.env
        for _ in range(10):
            if hasattr(env, "current_battle") and env.current_battle is not None:
                return env.current_battle
            if not hasattr(env, "env"):
                break
            env = env.env
        return None

    def _print_team_preview(self, battle):
        p1 = battle.player_role == "p1"
        turn = battle._current_turn
        our_preview = [p for p in turn.get_teampreview(p1) if p]
        opp_preview = [p for p in turn.get_teampreview(not p1) if p]
        our_names = [p.name for p in our_preview]
        opp_names = [p.name for p in opp_preview]
        print()
        print("=" * 50)
        print("  TEAM PREVIEW")
        print("=" * 50)
        print(f"  Your team:       {', '.join(our_names)}")
        print(f"  Opponent's team: {', '.join(opp_names)}")
        our_lead = turn.get_active_pokemon(p1)[0]
        opp_lead = turn.get_active_pokemon(not p1)[0]
        if our_lead and opp_lead:
            print(f"  Lead: {our_lead.name} vs {opp_lead.name}")
        print("=" * 50)
        print()
        if our_lead:
            self._prev_our = (our_lead.name, our_lead.current_hp, our_lead.max_hp)
        if opp_lead:
            self._prev_opp = (opp_lead.name, opp_lead.current_hp, opp_lead.max_hp)

    def _print_turn(self, battle, turn):
        p1 = battle.player_role == "p1"
        print(f"--- Turn {turn.turn_number} ---")

        our_actions = turn.moves_1 if p1 else turn.moves_2
        opp_actions = turn.moves_2 if p1 else turn.moves_1
        our_active = turn.get_active_pokemon(p1)[0]
        opp_active = turn.get_active_pokemon(not p1)[0]

        for actions, label in [
            (our_actions, ""),
            (opp_actions, "Opponent's "),
        ]:
            for action in actions:
                if action is None or action.is_noop:
                    continue
                user = action.user.name if action.user else "???"
                if action.is_switch:
                    target = action.target.name if action.target else "???"
                    print(f"  {label}{user}, come back!")
                    print(f"  {label}Go, {target}!")
                else:
                    print(f"  {label}{user} used {action.name}!")

        if our_active:
            print(f"  {self._hp_line('Your', our_active, self._prev_our)}")
        if opp_active:
            print(f"  {self._hp_line('Opp', opp_active, self._prev_opp)}")

        weather = turn.weather
        if weather is not None:
            wname = getattr(weather, "name", str(weather))
            if wname != "NO_WEATHER":
                print(f"  Weather: {self._weather_label(weather)}")

        self._prev_our = (our_active.name, our_active.current_hp, our_active.max_hp) if our_active and our_active.current_hp > 0 else None
        self._prev_opp = (opp_active.name, opp_active.current_hp, opp_active.max_hp) if opp_active and opp_active.current_hp > 0 else None
        print()

    @staticmethod
    def _hp_line(label, poke, prev):
        if poke.current_hp <= 0:
            return f"{label} {poke.name}: FAINTED"
        pct = poke.current_hp / max(poke.max_hp, 1) * 100
        delta = ""
        if prev is not None:
            prev_name, prev_hp, prev_max = prev
            if prev_name == poke.name and prev_max > 0:
                d = prev_hp - poke.current_hp
                if d > 0:
                    delta = f" [-{d} HP]"
                elif d < 0:
                    delta = f" [+{-d} HP]"
        st = BattleCommentaryWrapper._status_label_static(poke.status)
        return f"{label} {poke.name}: {poke.current_hp}/{poke.max_hp} HP ({pct:.0f}%){delta}{st}"

    @staticmethod
    def _status_label_static(status):
        if status is None:
            return ""
        s = getattr(status, "name", str(status))
        name = _STATUS_NAMES.get(s)
        return f" [{name}]" if name else ""

    def _weather_label(self, weather):
        s = getattr(weather, "name", str(weather)).lower()
        mapping = {
            "raindance": "Rain",
            "sunnydance": "Sun",
            "sandstorm": "Sandstorm",
            "hail": "Hail",
            "snow": "Snow",
            "deltastream": "Strong Winds",
            "primordialsea": "Heavy Rain",
            "desolateland": "Extremely Harsh Sun",
        }
        return mapping.get(s, s.replace("_", " ").title())

    def _print_outcome(self, info):
        won = info.get("won", None)
        if won == 1:
            print("*** YOU WIN! ***")
        elif won == 0:
            print("*** YOU LOSE! ***")
        else:
            print("*** BATTLE ENDED ***")


def _make_commentary_env(battle_format, obs_space, act_space, reward_fn,
                          battle_backend, team_set, opponent):
    logging.getLogger("Metamon.MetamonBackendBattle").setLevel(logging.ERROR)
    menv = BattleAgainstBaseline(
        battle_format=battle_format,
        observation_space=obs_space,
        action_space=act_space,
        reward_function=reward_fn,
        battle_backend=battle_backend,
        team_set=team_set,
        opponent_type=get_baseline(opponent),
    )
    print("Made Baseline Env")
    amago_env = MetamonAMAGOWrapper(menv)
    return BattleCommentaryWrapper(amago_env)


def _verify_prerequisites():
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"ERROR: checkpoint not found: {CHECKPOINT_PATH}", file=sys.stderr)
        return False
    if not os.path.isfile(CONFIG_PATH):
        print(f"ERROR: config.txt not found: {CONFIG_PATH}", file=sys.stderr)
        return False
    cache_dir = os.environ.get("METAMON_CACHE_DIR", "")
    if not cache_dir or not os.path.isdir(cache_dir):
        print(
            "ERROR: METAMON_CACHE_DIR is not set or invalid. Set it to a "
            "directory with plenty of free space before running this script.",
            file=sys.stderr,
        )
        return False
    return True


def main():
    if not _verify_prerequisites():
        return 1

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_text = f.read()
    print(f"Loaded reference config from {CONFIG_PATH} ({len(config_text)} chars)")
    print(
        f"Checkpoint: {CHECKPOINT_PATH} "
        f"({os.path.getsize(CHECKPOINT_PATH):,} bytes)"
    )
    attn_class, attn_label = _select_attention_class()
    print(f"Attention: {attn_label}")
    print(f"Async mp_context: {ASYNC_MP_CONTEXT}")
    print(
        f"Battle backend: {BATTLE_BACKEND} | team set: {TEAM_SET} | "
        f"opponent: {OPPONENT} | T={TEMPERATURE}"
    )

    model = LocalKakuna(CHECKPOINT_PATH, attn_class)

    agent = model.initialize_agent(
        checkpoint=None, log=False, action_temperature=TEMPERATURE
    )
    agent.async_env_mp_context = ASYNC_MP_CONTEXT

    for gen in GENS:
        for fmt in FORMATS:
            battle_format = f"gen{gen}{fmt.lower()}"
            team_set = metamon.env.get_metamon_teams(battle_format, TEAM_SET)
            print(
                f"\n=== {model.model_name} vs {OPPONENT} on {battle_format} "
                f"(1 battle, both sides draw random teams from {TEAM_SET}) ==="
            )
            make_env = functools.partial(
                _make_commentary_env,
                battle_format=battle_format,
                obs_space=model.observation_space,
                act_space=model.action_space,
                reward_fn=model.reward_function,
                battle_backend=BATTLE_BACKEND,
                team_set=team_set,
                opponent=OPPONENT,
            )
            agent.parallel_actors = 1
            results = agent.evaluate_test(
                [make_env],
                timesteps=TOTAL_BATTLES * 250,
                episodes=TOTAL_BATTLES,
            )
            print(json.dumps(results, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
