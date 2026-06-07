# Team-specific heuristic guidance extracted from the "Soul of Rain" RMT strategy.
# Activated only when the bot is using the main_team.txt rain team.
# Provides preferred-action hints without overriding the model entirely.

MAIN_TEAM_FILE = "bot/teams/main_team.txt"

# Pokemon species names as they appear in Showdown protocol messages.
# The battle state exposes species via battle.active_pokemon.species.
RAIN_LEAD_TARGETS = {"kyurem", "raging bolt", "ninetales", "tyranitar"}
ZAMAZENTA = "zamazenta"
DRAGONITE = "dragonite"
GLOWKING = "slowking-galar"
PECHARUNT = "pecharunt"
HYDRAPPLE = "hydrapple"
GREAT_TUSK = "great tusk"
KINGAMBIT = "kingambit"
RILLABOOM = "rillaboom"
OGERPON_WELLSPRING = "ogerpon-wellspring"


def is_guide_active(team_file):
    if team_file is None:
        return False
    return team_file.replace("\\", "/").rstrip("/") == MAIN_TEAM_FILE


def get_preferred_actions(battle):
    pref = set()
    active = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    if active is None or opp is None:
        return pref

    species = active.species.lower() if active.species else ""
    opp_species = opp.species.lower() if opp.species else ""
    opp_team_species = {p.species.lower() for p in battle.opponent_team.values() if p}
    rain_active = _rain_is_active(battle)
    turn = getattr(battle, "turn", 0)
    rocks_up = _rocks_are_up(battle)

    if species == "iron treads":
        pref.update(_iron_treads_guide(battle, turn, opp_species, opp_team_species, rocks_up))
    elif species == "pelipper":
        pref.update(_pelipper_guide(battle, opp_species, turn))
    elif species == "barraskewda":
        pref.update(_barraskewda_guide(battle, opp_species, rain_active))
    elif species == "overqwil":
        pref.update(_overqwil_guide(battle, opp_species, opp_team_species))
    elif species == "raging bolt":
        pref.update(_raging_bolt_guide(battle, opp_species))
    elif species == "kingambit":
        pref.update(_kingambit_guide(battle, turn, opp_species, opp_team_species))

    return pref


def _rain_is_active(battle):
    from poke_env.environment import Weather
    weather = battle.weather
    return bool(weather and Weather.RAINDANCE in weather)


def _rocks_are_up(battle):
    from poke_env.environment import SideCondition
    return SideCondition.STEALTH_ROCK in battle.opponent_side_conditions


def _move_indices(battle, move_names):
    indices = set()
    for i, move in enumerate(battle.available_moves):
        if move.id.lower() in move_names:
            indices.add(i)
    return indices


def _switch_indices(battle, target_species):
    indices = set()
    for i, switch in enumerate(battle.available_switches):
        if switch.species.lower() in target_species:
            indices.add(4 + i)
    return indices


def _tera_move_indices(battle, move_names):
    if not battle.can_tera:
        return set()
    indices = set()
    for i, move in enumerate(battle.available_moves):
        if move.id.lower() in move_names:
            indices.add(9 + i)
    return indices


def _iron_treads_guide(battle, turn, opp_species, opp_team_species, rocks_up):
    pref = set()

    # Lead vs Kyurem: Steel Beam into the Kyurem lead
    if turn <= 1:
        if opp_species in RAIN_LEAD_TARGETS or any(
            t in RAIN_LEAD_TARGETS for t in opp_team_species
        ):
            pref.update(_move_indices(battle, {"steel beam"}))
            return pref

    # If rocks are already up and Treads is low: Steel Beam to preserve momentum
    if rocks_up:
        hp = battle.active_pokemon.current_hp_fraction
        if hp is not None and hp < 0.5:
            pref.update(_move_indices(battle, {"steel beam"}))
            pref.update(_move_indices(battle, {"earth power"}))
            return pref

    # Otherwise: prioritize getting rocks up
    pref.update(_move_indices(battle, {"stealth rock"}))
    return pref


def _pelipper_guide(battle, opp_species, turn):
    pref = set()

    # Primarily U-turn for momentum
    pref.update(_move_indices(battle, {"u-turn", "uturn"}))

    # Hurricane vs Zamazenta
    if ZAMAZENTA in opp_species:
        pref.update(_move_indices(battle, {"hurricane"}))

    # Roost to 1v1 threats
    if opp_species in (DRAGONITE, ZAMAZENTA, GREAT_TUSK):
        pref.update(_move_indices(battle, {"roost"}))

    # Weather Ball in rain is strong
    if _rain_is_active(battle):
        pref.update(_move_indices(battle, {"weather ball"}))

    return pref


def _barraskewda_guide(battle, opp_species, rain_active):
    pref = set()

    if rain_active:
        # Tera Water Liquidation is primary wincon - prioritize tera moves
        pref.update(_tera_move_indices(battle, {"liquidation"}))

        # Liquidation spam in rain
        pref.update(_move_indices(battle, {"liquidation"}))

        # Flip Turn on predicted switches to bring in counters
        if opp_species in (HYDRAPPLE, KINGAMBIT, "dondozo", "alomomola"):
            pref.update(_move_indices(battle, {"flip turn", "flipturn"}))

    # Flip Turn is always good for momentum
    pref.update(_move_indices(battle, {"flip turn", "flipturn"}))

    return pref


def _overqwil_guide(battle, opp_species, opp_team_species):
    pref = set()

    # Use Glowking and Pecharunt as setup fodder
    if opp_species in (GLOWKING, PECHARUNT):
        pref.update(_move_indices(battle, {"swords dance"}))
        return pref

    # Gunk Shot vs Ogerpon-Wellspring (OHKO)
    if OGERPON_WELLSPRING in opp_species:
        pref.update(_move_indices(battle, {"gunk shot"}))

    # Crunch OHKOs Glowking and Gholdengo
    if opp_species in (GLOWKING, "gholdengo"):
        pref.update(_move_indices(battle, {"crunch"}))

    # Liquidation in rain = strong
    if _rain_is_active(battle):
        pref.update(_move_indices(battle, {"liquidation"}))
        pref.update(_tera_move_indices(battle, {"liquidation"}))

    return pref


def _raging_bolt_guide(battle, opp_species):
    pref = set()

    # Thunder as primary attack in rain
    pref.update(_move_indices(battle, {"thunder"}))

    # Thunderclap for priority
    pref.update(_move_indices(battle, {"thunderclap"}))

    # Draco Meteor vs Kyurem and fat targets
    if opp_species in ("kyurem", GLOWKING, ZAMAZENTA):
        pref.update(_move_indices(battle, {"draco meteor"}))

    return pref


def _kingambit_guide(battle, turn, opp_species, opp_team_species):
    pref = set()

    # Primary attacks
    pref.update(_move_indices(battle, {"kowtow cleave", "iron head"}))

    # Sucker Punch for priority
    pref.update(_move_indices(battle, {"sucker punch"}))

    # Early-game: Gambit is a trade piece, not just a late-game sweeper
    # SD early to pressure Dragonite / Hydrapple
    if turn <= 8 and any(
        t in opp_team_species
        for t in (DRAGONITE, HYDRAPPLE, "corviknight")
    ):
        pref.update(_move_indices(battle, {"swords dance"}))

    return pref
