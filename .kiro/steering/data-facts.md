---
inclusion: always
---

# Data facts (measured on the full dataset, Nov 10) — read before writing any code

## Size and keys
- 8,557 pass plays, 122 games, 8,314,178 tracking rows. Every frame has exactly 22 players + 1 ball row.
- Keys join perfectly: plays (gameId, playId) ↔ tracking ↔ pffScoutingData (gameId, playId, nflId) ↔ players (nflId).
- passResult: C 4,620 · I 2,755 · S 543 · R 449 · IN 190. Throws = 7,565.

## What the tracking covers (the biggest trap)
- Tracking runs from ~0.5 s BEFORE the snap to ~0.5 s AFTER the throw / sack / scramble. Median 39 frames per play.
- There is NO catch frame, NO ball flight, NO yards after catch, NO separation at arrival in tracking.
  `pass_arrived` exists on 4.9% of throws (quick throws only); `pass_outcome_caught` on 23 plays. Never condition on them.
- Yards after catch / air yards must come from nflverse play-by-play (join `old_game_id` = gameId, `play_id` = playId) or not at all.
- Usable window: snap → release. Full route paths to release, separation at release for every route runner, depth, speed.

## Cutting a play
- Clock: `time` is truncated to whole seconds. Use `frameId / 10` (10 frames per second).
- `event` "no event" is the literal string `"None"`. The event is repeated on all 23 rows of a frame.
- Snap = first frame with `event == "ball_snap"` (25 plays lack it: drop them). Never use `autoevent_ballsnap`.
- End of dropback = earliest frame after the snap among `pass_forward` (throw), `qb_sack` / `qb_strip_sack` (sack), `run` (scramble).
  Fallback only if none: `autoevent_passforward`. One play has two `pass_forward`: take the first.
- When events and `passResult` disagree (a few dozen plays), trust `passResult`.
- Snap to throw: median 2.7 s (p10 1.9, p90 4.2). Sacks ~4.4 s, scrambles ~4.2 s.

## Cleaning rules
- Ball rows: `team == "football"`, with nflId / jerseyNumber / o / dir null. Filter them out before player math.
- Normalize direction: if `playDirection == "left"`: `x = 120 - x`, `y = 53.3 - y`, and flip `absoluteYardlineNumber` to `120 - value`.
  `absoluteYardlineNumber` is the line of scrimmage in raw tracking x.
- Use speed `s`, not `dis` (49 impossible `dis` rows).
- PFF role strings are Title Case: `Pass Route`, `Pass Block`, `Pass Rush`, `Coverage`, `Pass`. (`"Pass route"` matches nothing.)
- Use `prePenaltyPlayResult`, not `playResult` (they differ on 514 throws).
- Whitespace: only `playDescription` has trailing spaces (167 rows) and double spaces. Everything else is clean. Strip it anyway.
- `down == 0` (35 rows) are two-point tries. `yardlineSide` is an empty string at midfield.
- `players.height` is text like `6-4`. `officialPosition` is roster position, not alignment: use `pff_positionLinedUp` for alignment.
- 8 display names are shared by two players: always match on nflId, or by name only within one play's route runners.

## Targets (who the ball was thrown to)
- `plays.csv` has no target column. Parse `playDescription` (`pass ... to X.Lastname` / `intended for X.Lastname`) and match to that
  play's `Pass Route` players by first initial + last name. Matches 96.3% of throws (99.7% of completions).
- 257 throws name no receiver (throwaways, batted balls): true no-target plays.
- Do NOT infer the target as "nearest receiver to the ball": wrong 42% of the time (the ball is still near the QB when tracking ends).
- 7,287 targets to 419 players: WR 60.8%, TE 21.5%, RB 16.8%.

## Sample sizes
- Routes: 39,513 player-plays (WR 23,033 · TE 9,064 · RB 6,994). 172 players with 100+ routes (WR 103, TE 41, RB 28).
- Targets: 82 players with 30+ (WR 61, TE 15, RB 6); 144 with 20+.
- Sacks 543, scrambles 449. Pressured throws (any pff_hit or pff_hurry) 2,422 = 32% of throws.
- Chip-and-release (`pff_blockType` CH or SR on a `Pass Route` row): 1,101 TE routes, 559 RB routes. League/team level only, not per player.
- Per team: 203–339 dropbacks on offense. Per-team sack counts are small (~17): show uncertainty.

## Reference numbers (for sanity checks)
- Separation = distance from a route runner to the nearest `Coverage` defender, at the release frame.
- Completion rate by target separation at release: 0–1 yd 22.7% · 1–2 47.0% · 2–3 59.2% · 3–5 71.5% · 5–8 78.0% · 8+ 84.9%.
- Median target separation by position: RB 6.76 · TE 3.37 · WR 2.84 yd. Never compare raw separation across positions or depths.
- Some non-targeted receiver is more open than the target on 73.1% of throws: "open by distance" alone means little.
- Clean vs pressured throws: completion 67.8% vs 46.7%; yards per attempt 8.06 vs 6.00.
- Target depth past the line at release: median 5.3 yd (WR 7.8, TE 4.9, RB 0.1).
