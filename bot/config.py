# PorymaxSettings dataclass loaded from .env, CLI args, and defaults

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

@dataclass
class PorymaxSettings:
    # Model
    checkpoint_path: Optional[Path] = None
    config_path: Optional[Path] = None
    device: str = "cpu"

    # Pokemon Showdown Connection
    username: str = "PorymaxBot"
    password: Optional[str] = None
    server_url: str = "localhost:8000"
    battle_format: str = "gen9ou"
    max_concurrent_battles: int = 1

    # Runtime Behavior
    temperature: float = 1.0
    mcts_enabled: bool = False
    mcts_simulations: int = 50
    mcts_max_depth: int = 2

    # Team
    team_set_name: str = "gl_05_26"

    # Logging
    log_level: int = 20

    # Metamon
    async_mp_context: str = "spawn" if sys.platform == "win32" else "forkserver"

    def __post_init__(self):
        if self.device == "cuda" and not _cuda_available():
            self.device = "cpu"
        if self.checkpoint_path is None:
            self.checkpoint_path = _project_root() / "weights" / "kakuna.pt"
        if self.config_path is None:
            self.config_path = _project_root() / "weights" / "config.txt"

    @classmethod
    def from_env(cls) -> "PorymaxSettings":
        s = cls()
        _str(s, "checkpoint_path", "PORYMAX_CKPT")
        _str(s, "config_path", "PORYMAX_CONFIG")
        _str(s, "device", "PORYMAX_DEVICE")
        _str(s, "username", "PORYMAX_USERNAME")
        _str(s, "password", "PORYMAX_PASSWORD")
        _str(s, "server_url", "PORYMAX_SERVER")
        _str(s, "battle_format", "PORYMAX_FORMAT")
        _int(s, "max_concurrent_battles", "PORYMAX_MAX_BATTLES")
        _float(s, "temperature", "PORYMAX_TEMPERATURE")
        _bool(s, "mcts_enabled", "PORYMAX_MCTS_ENABLED")
        _int(s, "mcts_simulations", "PORYMAX_MCTS_SIMS")
        _int(s, "mcts_max_depth", "PORYMAX_MCTS_DEPTH")
        _str(s, "team_set_name", "PORYMAX_TEAM_SET")
        _int(s, "log_level", "PORYMAX_LOG_LEVEL")
        if "METAMON_CACHE_DIR" not in os.environ:
            raise EnvironmentError(
                "METAMON_CACHE_DIR must be set in the environment or .env file"
            )
        if s.device == "cuda" and not _cuda_available():
            s.device = "cpu"
        if isinstance(s.checkpoint_path, str):
            s.checkpoint_path = Path(s.checkpoint_path)
        if isinstance(s.config_path, str):
            s.config_path = Path(s.config_path)
        return s

    @classmethod
    def from_cli(cls, args: "argparse.Namespace") -> "PorymaxSettings":
        s = cls.from_env()
        for field_name in [
            "checkpoint_path", "config_path", "device",
            "username", "password", "server_url", "battle_format",
            "temperature", "mcts_enabled", "mcts_simulations",
            "mcts_max_depth", "team_set_name", "log_level",
        ]:
            val = getattr(args, field_name, None)
            if val is not None:
                if field_name in ("checkpoint_path", "config_path"):
                    val = Path(val)
                setattr(s, field_name, val)
        return s

# env processing functions

def _str(settings: PorymaxSettings, field: str, env_var: str):
    val = os.environ.get(env_var)
    if val is not None:
        setattr(settings, field, val)

def _int(settings: PorymaxSettings, field: str, env_var: str):
    val = os.environ.get(env_var)
    if val is not None:
        setattr(settings, field, int(val))

def _float(settings: PorymaxSettings, field: str, env_var: str):
    val = os.environ.get(env_var)
    if val is not None:
        setattr(settings, field, float(val))

def _bool(settings: PorymaxSettings, field: str, env_var: str):
    val = os.environ.get(env_var)
    if val is not None:
        setattr(settings, field, val.strip().lower() in ("1", "true", "yes"))

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
