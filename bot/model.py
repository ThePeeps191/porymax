# Kakuna model singleton: loads checkpoint once, exposes policy, tokenizer, and spaces.

import logging
import sys
import tempfile
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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

from metamon.interface import (
    get_observation_space,
    get_action_space,
    get_reward_function,
)
from metamon.rl.metamon_to_amago import make_placeholder_experiment
from metamon.rl.pretrained import PretrainedModel
from metamon.tokenizer import get_tokenizer

from bot.config import PorymaxSettings


def _select_attention_class():
    if not torch.cuda.is_available():
        return amago.nets.transformer.VanillaAttention
    try:
        import flash_attn  # noqa: F401
        return amago.nets.transformer.FlashAttention
    except ImportError:
        return amago.nets.transformer.VanillaAttention


class _LocalKakuna(PretrainedModel):
    def __init__(self, checkpoint_path, attn_class, tokenizer):
        super().__init__(
            model_name="kakuna",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna.gin",
            default_checkpoint=0,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=tokenizer,
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": tokenizer,
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
            self.tokenizer,
        )
        ckpt = checkpoint if checkpoint is not None else self.default_checkpoint
        ckpt_path = self.get_path_to_checkpoint(ckpt)
        ckpt_base_dir = tempfile.mkdtemp(prefix="porymax_")
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


class PorymaxModel:
    _experiment = None
    _model = None
    _tokenizer = None
    _obs_space = None
    _act_space = None
    _loaded = False

    @classmethod
    def load(cls, settings: PorymaxSettings = None):
        if cls._loaded:
            return
        if settings is None:
            settings = PorymaxSettings()
        logging.getLogger("Metamon.MetamonBackendBattle").setLevel(logging.ERROR)

        cls._tokenizer = get_tokenizer("DefaultObservationSpace-v1")
        attn_class = _select_attention_class()

        cls._model = _LocalKakuna(
            checkpoint_path=str(settings.checkpoint_path),
            attn_class=attn_class,
            tokenizer=cls._tokenizer,
        )
        cls._experiment = cls._model.initialize_agent(
            checkpoint=None,
            log=False,
            action_temperature=settings.temperature,
        )
        cls._experiment.policy.eval()
        cls._obs_space = cls._model.observation_space
        cls._act_space = cls._model.action_space
        cls._loaded = True

    @classmethod
    def experiment(cls):
        cls._ensure_loaded()
        return cls._experiment

    @classmethod
    def policy(cls):
        return cls.experiment().policy

    @classmethod
    def device(cls):
        return cls.experiment().DEVICE

    @classmethod
    def tokenizer(cls):
        cls._ensure_loaded()
        return cls._tokenizer

    @classmethod
    def obs_space(cls):
        cls._ensure_loaded()
        return cls._obs_space

    @classmethod
    def act_space(cls):
        cls._ensure_loaded()
        return cls._act_space

    @classmethod
    def _ensure_loaded(cls):
        if not cls._loaded:
            raise RuntimeError("Model not loaded. Call PorymaxModel.load() first.")
