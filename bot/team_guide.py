# Team-specific heuristic guidance extracted from the "Soul of Rain" RMT strategy.
# Activated only when the bot is using the main_team.txt rain team.
# Provides preferred actions, lead preference, hazard-awareness, and move-failure avoidance.

MAIN_TEAM_FILE = "bot/teams/main_team.txt"

RAIN_LEAD_TARGETS = {"kyurem", "kyurem-black", "raging bolt", "ninetales", "tyranitar"}
ZAMAZENTA = "zamazenta"
DRAGONITE = "dragonite"
GLOWKING = "slowking-galar"
PECHARUNT = "pecharunt"
HYDRAPPLE = "hydrapple"
GREAT_TUSK = "great tusk"
KINGAMBIT = "kingambit"
RILLABOOM = "rillaboom"
OGERPON_WELLSPRING = "ogerpon-wellspring"

TEAM_LEAD = "pelipper"
LEAD_VS_KYUREM = "iron treads"

GROUND_TYPES = {
    "great tusk", "landorus-therian", "gliscor", "clodsire", "garchomp",
    "iron treads", "ting-lu", "hippowdon", "gastrodon", "excadrill",
    "ursaluna", "ursaluna-bloodmoon", "sandaconda", "mamoswine", "quagsire",
}
WATER_ABSORB = {
    "clodsire", "gastrodon", "quagsire", "vaporeon", "alomomola", "dondozo",
    "toxapex", "ogerpon-wellspring",
}


def is_guide_active(team_file):
    if team_file is None:
        return False
    return team_file.replace("\\", "/").rstrip("/") == MAIN_TEAM_FILE


def get_lead_species(battle):
    opp_team_names = {p.species.lower() for p in battle.opponent_team.values() if p}
    for t in RAIN_LEAD_TARGETS:
        if t in opp_team_names or any(t in n for n in opp_team_names):
            return LEAD_VS_KYUREM
    return TEAM_LEAD


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
        pref.update(_pelipper_guide(battle, opp_species))
    elif species == "barraskewda":
        pref.update(_barraskewda_guide(battle, opp_species, rain_active))
    elif species == "overqwil":
        pref.update(_overqwil_guide(battle, opp_species, opp_team_species))
    elif species == "raging bolt":
        pref.update(_raging_bolt_guide(battle, opp_species))
    elif species == "kingambit":
        pref.update(_kingambit_guide(battle, turn, opp_species, opp_team_species))

    return pref


def get_forced_switch_actions(battle):
    """Return switch action indices to strong targets, used when the active mon must switch out."""
    from poke_env.environment import SideCondition

    active = battle.active_pokemon
    if active is None:
        return set()

    species = active.species.lower() if active.species else ""
    switches = battle.available_switches

    forced = set()

    # Pelipper: if facing electric/rock and has switched in already (rain is up), U-turn or switch
    if species == "pelipper":
        opp = battle.opponent_active_pokemon
        opp_species = opp.species.lower() if opp else ""
        if any(t in opp_species for t in ("electric", "raging bolt", "zapdos", "iron hands")):
            for i, s in enumerate(switches):
                if s.species.lower() in ("iron treads", "kingambit"):
                    forced.add(4 + i)
            forced.update(_move_indices(battle, {"u-turn", "uturn"}))

    # Raging Bolt vs Ground-type: switch out aggressively
    if species == "raging bolt":
        opp = battle.opponent_active_pokemon
        opp_species = opp.species.lower() if opp else ""
        if opp_species in GROUND_TYPES or "tusk" in opp_species:
            for i, s in enumerate(switches):
                if s.species.lower() in ("barraskewda", "overqwil", "pelipper"):
                    forced.add(4 + i)

    # Iron Treads after rocks are up: stay in, don't switch — sac it
    if species == "iron treads":
        if SideCondition.STEALTH_ROCK in battle.opponent_side_conditions:
            hp = active.current_hp_fraction
            if hp is not None and hp < 0.6:
                forced.update(_move_indices(battle, {"steel beam", "earth power", "rapid spin"}))

    return forced


def get_hazard_malus(battle):
    """Penalty score for switching (0 = safe, 3+ = very dangerous)."""
    from poke_env.environment import SideCondition
    score = 0
    if SideCondition.STEALTH_ROCK in battle.side_conditions:
        score += 2
    spikes = battle.side_conditions.get(SideCondition.SPIKES, 0)
    score += spikes
    tspikes = battle.side_conditions.get(SideCondition.TOXIC_SPIKES, 0)
    score += min(tspikes, 2)
    return score


def get_move_failure_actions(battle, last_action_idx, last_opp_hp_pct, last_opp_species):
    """Return the previous action if it likely failed, for re-sample avoidance."""
    if last_action_idx is None or last_opp_hp_pct is None:
        return None
    opp = battle.opponent_active_pokemon
    if opp is None:
        return None
    curr_hp = opp.current_hp_fraction
    curr_species = opp.species.lower() if opp.species else ""
    if curr_species == last_opp_species and curr_hp is not None and last_opp_hp_pct is not None:
        if abs(curr_hp - last_opp_hp_pct) < 0.005:
            return last_action_idx
    return None


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

    if turn <= 1:
        if opp_species in RAIN_LEAD_TARGETS or any(
            t in RAIN_LEAD_TARGETS for t in opp_team_species
        ):
            pref.update(_move_indices(battle, {"steel beam"}))
            return pref

    if rocks_up:
        hp = battle.active_pokemon.current_hp_fraction
        if hp is not None and hp < 0.6:
            pref.update(_move_indices(battle, {"steel beam", "earth power", "rapid spin"}))
            return pref

    pref.update(_move_indices(battle, {"stealth rock", "rapid spin"}))
    return pref


def _pelipper_guide(battle, opp_species):
    pref = set()

    pref.update(_move_indices(battle, {"u-turn", "uturn"}))

    if ZAMAZENTA in opp_species:
        pref.update(_move_indices(battle, {"hurricane"}))

    if opp_species in (DRAGONITE, ZAMAZENTA, GREAT_TUSK):
        pref.update(_move_indices(battle, {"roost"}))

    if _rain_is_active(battle):
        pref.update(_move_indices(battle, {"weather ball"}))

    return pref


def _barraskewda_guide(battle, opp_species, rain_active):
    pref = set()

    if rain_active:
        # RMT strongly asserts "keep Tera for Barraskewda"
        pref.update(_tera_move_indices(battle, {"liquidation"}))
        pref.update(_move_indices(battle, {"liquidation"}))

        if opp_species in (HYDRAPPLE, KINGAMBIT, "dondozo", "alomomola"):
            pref.update(_move_indices(battle, {"flip turn", "flipturn"}))

    pref.update(_move_indices(battle, {"flip turn", "flipturn"}))

    return pref


def _overqwil_guide(battle, opp_species, opp_team_species):
    pref = set()

    if opp_species in (GLOWKING, PECHARUNT):
        pref.update(_move_indices(battle, {"swords dance"}))
        return pref

    if OGERPON_WELLSPRING in opp_species:
        pref.update(_move_indices(battle, {"gunk shot"}))

    if opp_species in (GLOWKING, "gholdengo"):
        pref.update(_move_indices(battle, {"crunch"}))

    if _rain_is_active(battle):
        pref.update(_move_indices(battle, {"liquidation"}))
        pref.update(_tera_move_indices(battle, {"liquidation"}))

    return pref


def _raging_bolt_guide(battle, opp_species):
    pref = set()

    # vs Ground types: DON'T Thunder — switch is preferred (handled by get_forced_switch_actions)
    is_ground = opp_species in GROUND_TYPES or "tusk" in opp_species
    if not is_ground:
        pref.update(_move_indices(battle, {"thunder"}))

    # Thunderclap only against predicted attacking mons — skip if opponent just used non-attack
    pref.update(_move_indices(battle, {"thunderclap"}))

    if opp_species in ("kyurem", GLOWKING, ZAMAZENTA):
        pref.update(_move_indices(battle, {"draco meteor"}))

    if is_ground:
        pref.update(_move_indices(battle, {"weather ball"}))

    return pref


def _kingambit_guide(battle, turn, opp_species, opp_team_species):
    pref = set()

    pref.update(_move_indices(battle, {"kowtow cleave", "iron head"}))
    pref.update(_move_indices(battle, {"sucker punch"}))

    if turn <= 8 and any(
        t in opp_team_species
        for t in (DRAGONITE, HYDRAPPLE, "corviknight")
    ):
        pref.update(_move_indices(battle, {"swords dance"}))

    return pref
