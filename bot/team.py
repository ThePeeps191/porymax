# Loads a Showdown team export file and wraps it in a Teambuilder that always returns that team.

from poke_env.teambuilder import Teambuilder


def load_team_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def constant_teambuilder(team_str):
    return _ConstantTeambuilder(team_str)


class _ConstantTeambuilder(Teambuilder):

    def __init__(self, team_str):
        super().__init__()
        self._packed = self.join_team(self.parse_showdown_team(team_str))

    def yield_team(self):
        return self._packed
