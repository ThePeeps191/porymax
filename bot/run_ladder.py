# CLI launcher: loads model, creates PorymaxPlayer, connects to Showdown.

import argparse
import asyncio
import sys
import time

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from bot.config import PorymaxSettings
from bot.model import PorymaxModel
from bot.player import PorymaxPlayer
from bot.team import constant_teambuilder, load_team_file


def parse_args():
    p = argparse.ArgumentParser(
        description="Porymax Bot - Pokemon Showdown ladder agent"
    )

    p.add_argument("--username", type=str, default="PorymaxBot",
                   help="Showdown username (default: PorymaxBot)")
    p.add_argument("--password", type=str, default=None,
                   help="Showdown password (required for public ladder)")

    p.add_argument("--local", action="store_true", default=True,
                   help="Use local Showdown server at localhost:8000 (default)")
    p.add_argument("--public", action="store_true", default=False,
                   help="Connect to public Showdown server (wss://sim3.psim.us)")
    p.add_argument("--server", type=str, default=None,
                   help="Custom server WebSocket URL (overrides --local and --public)")

    p.add_argument("--format", type=str, default="gen9ou",
                   help="Battle format (default: gen9ou)")
    p.add_argument("--team-file", type=str, default=None,
                   help="Path to a Showdown team export file (overrides --team-set)")
    p.add_argument("--team-set", type=str, default="gl_05_26",
                   help="Metamon team set name (default: gl_05_26)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Action sampling temperature (0 = greedy, default: 1.0)")
    p.add_argument("--battles", type=int, default=0,
                   help="Max battles to play (0 = unlimited, default: 0)")

    p.add_argument("--mcts", action="store_true", default=False,
                   help="Enable Monte Carlo Tree Search lookahead")
    p.add_argument("--ladder", action="store_true", default=False,
                   help="Enter Showdown ladder matchmaking instead of waiting for challenges")

    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override checkpoint path (default: weights/kakuna.pt)")
    p.add_argument("--device", type=str, default=None,
                   help="Inference device (default: auto-detect)")

    return p.parse_args()


def main():

    args = parse_args()

    settings = PorymaxSettings()
    if args.checkpoint:
        settings.checkpoint_path = args.checkpoint
    if args.device:
        settings.device = args.device
    settings.temperature = args.temperature
    settings.battle_format = args.format
    settings.team_set_name = args.team_set
    settings.username = args.username

    if args.server:
        from poke_env import ServerConfiguration
        server_config = ServerConfiguration(args.server,
                                            "https://play.pokemonshowdown.com/action.php?")
    elif args.public:
        server_config = ShowdownServerConfiguration
    else:
        server_config = LocalhostServerConfiguration

    print(f"Loading Kakuna model (device: {settings.device})...")
    PorymaxModel.load(settings)
    print("Model ready.")

    if args.team_file:
        print(f"Loading custom team from {args.team_file}...")
        team_str = load_team_file(args.team_file)
        team_set = constant_teambuilder(team_str)
        print("Team loaded.")
    else:
        import metamon
        from metamon.env import get_metamon_teams
        print(f"Loading team set '{args.team_set}' for {args.format}...")
        team_set = get_metamon_teams(args.format, args.team_set)

    account = AccountConfiguration(args.username, args.password)
    replay_kwargs = {}
    if args.public:
        replay_kwargs["save_replays"] = "replays"
    player = PorymaxPlayer(
        account_configuration=account,
        server_configuration=server_config,
        team=team_set,
        battle_format=args.format,
        temperature=args.temperature,
        mcts_enabled=args.mcts,
        team_file=args.team_file,
        **replay_kwargs,
    )

    n_battles = args.battles if args.battles > 0 else 999_999
    label = "public server" if args.public else "local server"
    mcts_label = " + MCTS" if args.mcts else ""
    guide_label = " (team guide active)" if player._use_guide else ""
    print(
        f"Bot '{args.username}' running on {label}{mcts_label}{guide_label} "
        f"(format: {args.format}, battles: {n_battles if args.battles > 0 else 'unlimited'})"
    )

    if args.ladder:
        print("Entering ladder matchmaking...")
        try:
            asyncio.run(player.ladder(n_battles))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(f"Ladder error: {e}")
        print("Ladder session finished.")
    else:
        print("Accepting challenges. Press Ctrl+C to stop.")
        try:
            asyncio.run(player.accept_challenges(None, n_battles))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Accept error: {e}")

    try:
        player.close()
    except AttributeError:
        pass
    print("Shut down.")


if __name__ == "__main__":
    for attempt in range(5):
        try:
            main()
            break
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            msg = str(e)
            if "JSONDecodeError" in type(e).__name__ or "Expecting value" in msg:
                wait = 5 * (attempt + 1)
                print(f"Login failed (server auth error). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
