# Window Trend

### Two receivers are both 2.5 yards open when the ball comes out. Are they equally open?

**What we built:** Window Trend, a new per-throw metric for wide receivers, tight ends and running backs, with an interactive player and team explorer. It measures whether the gap to the nearest defender is growing or shrinking in the half-second before the throw.

**The big idea:** no. Two receivers can have the same separation when the ball comes out and still be very different throws. With 2–3 yards of separation, the pass is completed **68%** of the time if the receiver is still separating from his defender at the release, and **51%** if the defender is closing on him.

![In tight coverage, an opening window beats a closing one](figures/takeaway.png)

**Why we went this way:** the public separation stat (Next Gen Stats) is one snapshot, taken when the ball arrives. The quarterback has to decide at the release, before anyone knows how that ends. On film a coach can see whether the defender is trailing or driving on the throw; a snapshot cannot. In the tracking data, targets with the same separation kept ending differently, and the difference was which way the window was moving. That direction at the release is not a public stat.

**How it helps:** it grades the quality of the target. It shows what kind of throws wide receivers, tight ends and running backs are being given, so their catch rate and yards after the catch (YAC) can be judged fairly.

**Who would use it:** a **scout** separating a receiver's hands from the throws he was given, a **coach** getting the ball out while the receiver is still separating (or checking which coverages his defense closes in), and a **broadcaster** who wants one arrow on a replay.

**Open the tool:** [Window Trend, explained](https://muzammilkhan2784.github.io/window-trend-nfl/) · [team and coverage explorer](https://muzammilkhan2784.github.io/window-trend-nfl/team-explorer.html)

## Findings: receivers, tight ends and running backs

- **Outside receivers:** the effect is strongest for receivers split out wide. On tight-window throws (1–3 yards of separation), 72% are completed when the window is opening and 43% when it is closing.
- **Tight ends:** a closing window erases the value of the target (expected points added per target: +0.52 opening, 0.00 closing).
- **Running backs** (fullbacks not counted): 48% of their targets arrive with a defender closing (wide receivers: 30%). Catching it in an opening window is worth 4 more yards of YAC (9.3 vs 5.1).
- **Players (the throws they were given, not a skill grade):** Mark Andrews saw an opening window on 43% of his targets; Tee Higgins and DeVonta Smith on 14%. Use it to read their catch rates, not to rank them.

![2.2 yards open is not always 2.2 yards open](figures/example_plays.png)

## Methods: the metric

| Name | What it is | Unit |
|---|---|---|
| **Window Trend** | Change in the target's distance to the nearest coverage defender over the 0.5 s before release. Above +1 is *opening*, below −1 is *closing*. | yards per second, per throw |
| **Opening Rate** | Share of a player's, position's or team's targets thrown into an opening window. | percent |
| **Catches Over Expected** | A receiver's catches minus what the league catches on throws with the same space and the same window. | catches |

## About the data

- **What we were given:** player tracking for 8,557 pass plays (2021, Weeks 1–8), plus play, player and PFF role files. The tracking runs from the snap to half a second after the throw. There is no catch and no run after the catch in it.
- **What that meant:** everything is measured at the moment of release, across 7,263 targeted throws. The targeted receiver is not a column, so we read it from the play description (matched on 96% of throws).
- **What we added:** public [nflverse](https://github.com/nflverse/nflverse-data) play-by-play, joined on game and play id, for yards after the catch and expected points.

## Assumptions and limitations

- Window Trend largely forecasts where separation will be half a second after the throw. We never see the ball arrive.
- The player numbers describe the throws a receiver was **given** over eight weeks (quarterback timing, play design, coverage). They are context for his catch rate, not a grade of how well he gets open: a receiver's Opening Rate in one half of his games does not predict it in the other half, while a quarterback's does.
- Where it does no better than plain separation: receivers who are already 5+ yards open, and tight ends lined up inline (59% vs 57%).
- These are observational splits, not cause and effect. Nothing is adjusted for route type (the data has no route labels): a comeback is closing by design and a crosser is opening, so part of the gap is route mix.
- It is not a throw-depth artifact: with 2–3 yards of separation, opening-window throws travel farther in the air on average (13.1 vs 10.9 air yards) and are still completed more often.
- "Opening" and "closing" use a cut of ±1 yard per second. The gap holds with a cut of ±0.5 (66% vs 53%).
- Team samples are 143–284 targets, so team rankings are indicative only.

## Run it

Python 3.12. The raw data is not in this repo.

```bash
git clone https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data bdb-data
```

```bash
curl -L -o external/play_by_play_2021.csv.gz https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz
```

```bash
pip install -r requirements.txt
```

```bash
python src/load.py
```

```bash
python src/window_trend.py
```

```bash
python src/skill_positions.py
```

```bash
python src/make_figures.py
```

```bash
python src/build_dashboard.py
```

```bash
python src/build_tool.py
```

`load.py` prints a check table against known counts and stops if any number is off. Every figure and number above is regenerated by these scripts.

Built during the NYU Big Data Bowl build window on November 10, 2026, with Kiro.
