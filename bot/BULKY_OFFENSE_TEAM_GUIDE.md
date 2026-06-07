# Team Guide — Bulky Offense (Peaked 1967 ELO)

This team by noob1877 is a physical-heavy bulky offense built around Dragonite's Tera Blast Flying sweep and Choice Scarf Zapdos speed control. Unlike the rain team, this team has **no weather management, no pivot chains, and no complex sequencing requirements** — the model plays it raw (no heuristic guide activated).

---

## Team Composition

| Pokemon | Role | Key Moves |
|---------|------|-----------|
| **Dragonite** | Late-game sweeper | Dragon Dance, Tera Blast (Flying), Earthquake, Substitute |
| **Ogerpon-Cornerstone** | Wallbreaker / anti-offense | Swords Dance, Ivy Cudgel, Power Whip, Low Kick |
| **Hatterene** | Hazard deterrent / Zama check | Psychic Noise, Draining Kiss, Pain Split, Nuzzle (Magic Bounce) |
| **Iron Treads** | Hazard control | Stealth Rock, Earthquake, Knock Off, Rapid Spin (Air Balloon) |
| **Kingambit** | Late-game cleaner / pivot | Swords Dance, Sucker Punch, Low Kick, Iron Head |
| **Zapdos** | Speed control / special attacker | Thunderbolt, Hurricane, Volt Switch, Heat Wave (Choice Scarf) |

---

## Why No Guide Is Needed

The team's design naturally aligns with Kakuna's strengths:

- **Zapdos is Choice Scarf** — locked into one move after the first click. Cannot misclick.
- **Dragonite clicks DD → sweeps** — one setup move, then attack spam.
- **Ogerpon clicks SD → Ivy Cudgel** — same pattern.
- **Hatterene's Magic Bounce** auto-punishes hazard setters without any decision required.
- **Iron Treads clicks Rapid Spin after hazards** — natural reactive play.
- **Kingambit clicks Sucker Punch** — priority cleans up endgames.

No weather timers, no "preserve this mon for later," no "click U-turn exactly when rain has 2 turns left." The model + MCTS handle this team without guidance.

---

## Strategy Notes (for human reference)

- **Lead**: Zapdos or Iron Treads (Zapdos punishes Zama/Ogerpon leads with Hurricane, Treads sets rocks safely)
- **Win condition**: Chip the opponent's physical walls, then sweep with Dragonite or Kingambit
- **Tera**: Save for Dragonite (Tera Blast Flying OHKOs most unresisted targets at +1). Kingambit Tera Ghost as backup.
- **Dangerous matchups**: Tera Electric Zama, Darkrai, Kyurem, Ceruledge, hazard stack
- **Magic Bounce**: Hatterene switches into predicted Stealth Rock/Spikes/Sticky Web turns

---

## Activation

This team does NOT trigger any heuristic guide. When `--team-file bot/teams/bulky_offense_team.txt` is used, the bot plays with raw Kakuna policy + MCTS only. The rain-specific guide activates only when using the rain team.
