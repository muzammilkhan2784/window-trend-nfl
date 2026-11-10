---
inclusion: always
---

# NYU Big Data Bowl — official brief (Nov 10, 2026, deadline 2:00 PM sharp)

## The challenge
Build a data tool or visualization an NFL coach, scout, or broadcaster would use.
Format: a visualization / comparison tool, OR a new metric that quantifies something not currently measured.
Focus: skill positions — TE, WR, RB routes, targets, and yards after catch.

## Judging (1–5 each)
- Football Insight: does the finding matter? Would a coach, scout or broadcaster use it? (first tiebreaker)
- Technical Execution: code works, clean, efficient, good use of libraries.
- Communication: clear output, readable visualization, README explains the value.
- Originality: non-obvious, creative framing, NOT something already found in the 2023 Big Data Bowl (pass rush / offensive line).
Triage first: Does it run? Is there a clear insight? Does the README make sense?

## Working rules
- ONE player, ONE route concept, or ONE question. Go deep. Don't boil the ocean.
- Simple and explainable beats sophisticated. The average fan must understand the metric in one sentence.
- A metric is a count or a rate, in yards, seconds or percent. "Actual vs expected" is a good pattern.
- Coach wants an actionable recommendation. Broadcaster wants a visual story. Scout wants comparative data.
- Show, don't tell: one takeaway chart whose title states the finding.
- Always compare against a simple baseline, and say where the idea does no better.
- Challenge every result: "Does this make sense? Do I have enough data to support this?" Report sample sizes.

## Data (read-only, in `bdb-data/data/`, never edit, never commit)
2021 season, Weeks 1–8, 122 games, pass plays only.
- `tracking/tracking_[gameId].csv`: 10 frames/sec; x, y, s, a, dis, o, dir, event for 22 players + ball (ball rows: nflId is NA, team == "football").
- `plays.csv` (key gameId, playId), `games.csv` (gameId), `players.csv` (nflId), `pffScoutingData.csv` (gameId, playId, nflId).
- Missing values are the string `NA`.
- `playDirection` is left/right: normalize coordinates so every play goes the same direction.
- Pass-block window = `ball_snap` to first of {`pass_forward`, `qb_sack`, `qb_strip_sack`, `run`}.
- `passResult`: C complete, I incomplete, S sack, R scramble, IN interception.
- `pff_role`: Pass route, Coverage, Pass block, Pass rush, Pass.
- Free public data may be joined (nflverse play-by-play: `old_game_id` = gameId, `play_id` = playId).

## Code conventions
- Python 3.12 (interpreter: `C:\Users\Muzammil\.venvs\bdb312\Scripts\python.exe`).
- Libraries: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, scipy (polars is fine for loading tracking).
- One reproducible entry point that runs top to bottom from raw data and regenerates every figure and number.
- Figures go in `figures/`. Label axes in football terms. No jargon in titles.
- Every number in the README must come from script output.

## Submission
One public URL (GitHub repo) with: code, output (visualization / screenshot / metric demo), and a README of 3–5 sentences covering findings, methods, assumptions and limitations. Judges read the README first.
