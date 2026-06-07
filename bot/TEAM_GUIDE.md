# Team Guide — Soul of Rain Heuristic Rules

These rules are extracted from the [Soul of Rain RMT](https://www.smogon.com/forums/threads/sv-ou-soul-of-rain-peaked-3-2028-elo.3759651/) by Delibird Heart (Peaked #3, 2028 ELO). They provide soft hints to the model — the AI still makes the final decision, but the guide nudges it toward strategies the team's creator validated across hundreds of ladder games.

Only active when using the rain team via `--team-file bot/teams/main_team.txt`.

---

## How It Works

After Kakuna picks an action, the guide checks if the current Pokemon has a preferred move for this situation. If the model's choice isn't in the preferred set, there's a **20% chance** of randomly selecting one of the guide's recommended actions instead. The model wins 80% of the time — the guide is a nudge, not an override.

---

## Per-Pokemon Rules

### Pelipper (Rain Setter)

| Priority | Action | When |
|----------|--------|------|
| Highest | **U-turn** | Always — slow pivot brings in sweepers safely |
| High | **Hurricane** | Opponent has Zamazenta |
| High | **Roost** | Opponent has Dragonite, Zamazenta, or Great Tusk (1v1 them) |
| Medium | **Weather Ball** | Rain is active |

**RMT logic:** *"U-Turn is an invaluable tool to grab momentum. Pelipper can always 1v1 Zamazenta thanks to Roost. Uninvested Weather Ball and Hurricane are pretty potent."*

---

### Barraskewda (Primary Wincon)

| Priority | Action | When |
|----------|--------|------|
| Highest | **Tera Water Liquidation** | Rain is up — nothing without Water Absorb switches in safely |
| High | **Liquidation** | Rain is active |
| High | **Flip Turn** | Opponent has Hydrapple, Kingambit, Dondozo, or Alomomola — pivot to Overqwil/Raging Bolt counters |

**RMT logic:** *"Most games you should keep your Tera for Barraskewda. Nothing that doesn't have water absorb or a 4x resist can switch into Tera Water Liquidation. Flip Turn forces switches that your other offensive monsters abuse."*

---

### Overqwil (Breaker / Setup Sweeper)

| Priority | Action | When |
|----------|--------|------|
| Highest | **Swords Dance** | Opponent has Slowking-Galar or Pecharunt — they're setup fodder |
| High | **Gunk Shot** | Opponent has Ogerpon-Wellspring (OHKO) |
| High | **Crunch** | Opponent has Glowking or Gholdengo (OHKO) |
| Medium | **Tera Water Liquidation** | Rain is active — 3 STABs worth of damage |
| Medium | **Liquidation** | Rain is active |

**RMT logic:** *"Overqwil completely counters Pecharunt and Glowking and can use them as setup fodder. Gunk Shot OHKOs Ogerpon-Wellspring. Having Liquidation in rain is like having 3 stabs."*

---

### Raging Bolt (Special Breaker / Revenge Killer)

| Priority | Action | When |
|----------|--------|------|
| Highest | **Thunder** | Primary attack — nearly as strong as +1 Thunderbolt in rain |
| High | **Thunderclap** | Priority for revenge killing |
| High | **Draco Meteor** | Opponent has Kyurem, Glowking, or Zamazenta |

**RMT logic:** *"Being able to spam Thunder and gaining Weather Ball make Raging Bolt MUCH more threatening than usual. Don't hesitate to switch out — losing Booster is not a huge deal."*

---

### Iron Treads (Hazard Setter / Sacrificial Piece)

| Priority | Action | When |
|----------|--------|------|
| Highest | **Steel Beam** | Turn 1 and opponent has Kyurem, Raging Bolt, Ninetales, or Tyranitar in preview — lead Treads and Steel Beam for ~95% |
| High | **Steel Beam** | Rocks are up and Treads is below 50% HP — commit seppuku to preserve momentum |
| High | **Earth Power** | Above condition met but Steel Beam unavailable |
| Medium | **Stealth Rock** | Rocks aren't up yet — get them up |

**RMT logic:** *"When I see Kyurem in preview, I always lead this and Steel Beam. The ability to commit seppuku with Steel Beam is invaluable. Consider it a free sac if needed once rocks are up."*

---

### Kingambit (Defensive Pivot / Secondary Wincon)

| Priority | Action | When |
|----------|--------|------|
| High | **Kowtow Cleave** / **Iron Head** | Primary attacks |
| High | **Sucker Punch** | Priority for revenge killing |
| Medium | **Swords Dance** | Early-game (turn ≤ 8) and opponent has Dragonite, Hydrapple, or Corviknight in preview |

**RMT logic:** *"You don't have to save it for the late game sweep! Tera darking/SDing early to pressure mons like Dnite or Hydrapple early game is recommended. Let your opponent throw 2-3 mons out to deal with Gambit — this is like a free 1-for-1 trade ticket."*

---

## Design Philosophy

- **Soft hints, not hard rules** — the model's action wins 80% of the time. The guide only intervenes when the model clearly picks something the RMT author would never do.
- **MCTS-ready** — `get_preferred_actions(battle)` returns a plain set of action indices. MCTS can use these as prior biases during node expansion.
- **Team-specific** — automatically disabled when using any other team file or random team sets.
