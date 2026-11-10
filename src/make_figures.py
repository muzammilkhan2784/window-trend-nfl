"""
src/make_figures.py - render every figure used in the README.

    figures/takeaway.png        completion rate, opening vs closing window,
                                within each band of separation at release
    figures/example_plays.png   two real plays with the same separation at
                                release: one window opening, one closing

Inputs (all produced by the other scripts, nothing is recomputed here):
    output/targets_window_trend.csv   one row per targeted throw (window_trend.py)
    cache/plays.parquet               play context (load.py)
    cache/tracking.parquet            player tracking, snap..release, offense
                                      always moving left to right (load.py)
    bdb-data/data/pffScoutingData.csv role of every player on every play
    bdb-data/data/players.csv         player names

Run from the project root:
    python src/make_figures.py

The script also prints the band table behind the chart and the two example
plays it drew, so every number in the figures can be checked in the console.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

# --------------------------------------------------------------------------- #
# Paths (resolved from this file, so the script runs from anywhere)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
TARGETS_CSV = ROOT / "output" / "targets_window_trend.csv"
PLAYS_PARQUET = ROOT / "cache" / "plays.parquet"
TRACKING_PARQUET = ROOT / "cache" / "tracking.parquet"
PFF_CSV = ROOT / "bdb-data" / "data" / "pffScoutingData.csv"
PLAYERS_CSV = ROOT / "bdb-data" / "data" / "players.csv"
FIGDIR = ROOT / "figures"

# --------------------------------------------------------------------------- #
# Style: two colours only (colourblind-safe blue / orange), greys for context
# --------------------------------------------------------------------------- #
BLUE = "#1f77b4"        # opening window
ORANGE = "#d95f02"      # closing window
BLUE_TEXT = "#17628f"   # darker shades of the same hues, for text on white
ORANGE_TEXT = "#b24c00"
INK = "#1a1a1a"
GREY = "#5f5f5f"
LIGHT = "#b9b9b9"
COLOR = {"opening": BLUE, "closing": ORANGE}
TEXT_COLOR = {"opening": BLUE_TEXT, "closing": ORANGE_TEXT}
DPI = 200

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",  # ships with matplotlib: same layout everywhere
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# --------------------------------------------------------------------------- #
# What the takeaway chart shows
# --------------------------------------------------------------------------- #
ALL_BANDS = ["0-1", "1-2", "2-3", "3-5", "5-8", "8+"]   # yards of separation
BANDS_SHOWN = ["1-2", "2-3", "3-5", "5-8"]              # bars drawn in Figure 1
WINDOWS = ["opening", "steady", "closing"]
TIGHT_BANDS = ["1-2", "2-3"]                            # "tight coverage" in the title
EXPLAINED_BAND = "2-3"                                  # its gap label is spelled out in words

# Numbers verified independently (completion %, n). The band table printed by
# this script is compared with them so a silent data change cannot slip through.
VERIFIED = {
    ("1-2", "opening"): (71.7, 127), ("1-2", "steady"): (45.8, 963), ("1-2", "closing"): (42.4, 401),
    ("2-3", "opening"): (67.7, 319), ("2-3", "steady"): (61.2, 394), ("2-3", "closing"): (51.0, 437),
    ("3-5", "opening"): (74.1, 615), ("3-5", "steady"): (72.6, 521), ("3-5", "closing"): (68.0, 710),
    ("5-8", "opening"): (77.6, 518), ("5-8", "steady"): (79.0, 347), ("5-8", "closing"): (77.7, 498),
    ("8+", "opening"): (85.3, 341), ("8+", "closing"): (80.1, 236),
}

# --------------------------------------------------------------------------- #
# The two example plays (gameId, playId).
#
# Selection rule, applied to output/targets_window_trend.csv (see
# example_candidates): Cover-1 man coverage, 2 to 3 yards of separation at
# release, target 8 to 18 yards downfield, window trend beyond +/-2 yd/s,
# opening play completed, closing play not completed. Among the plays that
# qualify we picked the best-matched pair with well-known players: both are
# 3rd & 6, both receivers are 2.2 yards from the nearest defender at release.
# make_example_plays() re-checks the rule and stops if a pick no longer fits.
# --------------------------------------------------------------------------- #
EXAMPLE_PLAYS = {
    "opening": (2021103103, 712),    # CLE vs PIT: Mayfield to Landry
    "closing": (2021091907, 2334),   # SF vs PHI: Garoppolo to Samuel
}
# Where each label sits in those two panels: (dx, dy) in yards from its anchor,
# then horizontal / vertical alignment. Hand-placed so nothing covers a player;
# a different example play needs its own four offsets.
#   gap       anchored at the midpoint between receiver and defender at release
#   earlier   anchored at the midpoint between them 0.5 s earlier
#   target / defender   anchored at that player at release
EXAMPLE_LABELS = {
    "opening": dict(gap=(0.72, -0.15, "left", "center"), earlier=(-0.75, 0.0, "right", "center"),
                    target=(0.1, -0.75, "center", "top"), defender=(0.35, 0.7, "left", "bottom")),
    "closing": dict(gap=(0.35, 0.85, "left", "bottom"), earlier=(0.0, -0.65, "center", "top"),
                    target=(-0.35, 0.75, "center", "bottom"), defender=(0.45, -0.7, "left", "top")),
}
EXAMPLE_RULE = dict(coverage="Cover-1", sep=(2.0, 3.0), depth=(8.0, 18.0), min_abs_trend=2.0)
LOOKBACK_FRAMES = 5   # 5 frames = 0.5 s at 10 frames per second
# Arrowhead on each 0.5 s path, in yards: how far short of the player it stops,
# its length and its half-width.
HEAD_GAP, HEAD_LENGTH, HEAD_HALF_WIDTH = 0.48, 0.38, 0.22


# =========================================================================== #
# Small helpers
# =========================================================================== #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n (as proportions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def text_run(fig, x: float, y: float, parts, fontsize: float) -> float:
    """
    Write several text segments on one baseline, each with its own colour and
    weight. `parts` is a list of (text, colour, weight). x, y are in figure
    fractions. Returns the x where the run ends.
    """
    renderer = fig.canvas.get_renderer()

    def width(s: str, weight: str) -> float:
        t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w

    for text, colour, weight in parts:
        fig.text(x, y, text, color=colour, fontweight=weight, fontsize=fontsize,
                 ha="left", va="baseline")
        # A sentinel keeps trailing spaces in the measured width.
        x += (width(text + "|", weight) - width("|", weight)) / fig.bbox.width
    return x


def gap_label(gap: float) -> str:
    """Opening minus closing completion rate, in percentage points, as chart text."""
    return "no gap" if abs(gap) < 1 else f"{gap:+.0f} pts".replace("-", "−")


def warn_if_clipped(fig, name: str) -> None:
    """Tell the console if any figure-level text runs off the canvas."""
    renderer = fig.canvas.get_renderer()
    for t in fig.texts:
        box = t.get_window_extent(renderer)
        if box.x0 < 0 or box.x1 > fig.bbox.width or box.y0 < 0 or box.y1 > fig.bbox.height:
            print(f"WARNING {name}: text runs off the figure: {t.get_text()[:60]!r}")


def ordinal_down(down: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(int(down), f"{down}th")


# =========================================================================== #
# Data
# =========================================================================== #
def load_targets() -> pd.DataFrame:
    """One row per targeted throw, as written by window_trend.py."""
    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run: python src/window_trend.py")
    return pd.read_csv(TARGETS_CSV)


def band_table(targets: pd.DataFrame) -> pd.DataFrame:
    """Completion rate, n and 95% Wilson interval per separation band x window."""
    rows = []
    for band in ALL_BANDS:
        for window in WINDOWS:
            sub = targets[(targets["sep_band"] == band) & (targets["window"] == window)]
            n = len(sub)
            k = int(sub["complete"].sum())
            lo, hi = wilson_interval(k, n)
            rows.append(
                dict(band=band, window=window, n=n, completions=k,
                     pct=100 * k / n if n else float("nan"), lo=100 * lo, hi=100 * hi)
            )
    return pd.DataFrame(rows)


def print_band_table(table: pd.DataFrame) -> bool:
    """Print the table behind Figure 1 and compare it with the verified numbers."""
    print("\n" + "=" * 74)
    print("Band table used by figures/takeaway.png")
    print("completion rate by window, within each band of separation at release")
    print("=" * 74)
    print(f"{'band':<6}{'window':<9}{'n':>6}{'comp%':>8}{'95% Wilson':>16}{'verified':>18}  check")
    print("-" * 74)
    all_ok = True
    for r in table.itertuples():
        ref = VERIFIED.get((r.band, r.window))
        if r.n == 0:
            line_ref, status = "", "(no throws)"
        elif ref is None:
            line_ref, status = "", ""
        else:
            ok = abs(r.pct - ref[0]) < 0.05 + 1e-9 and r.n == ref[1]
            all_ok &= ok
            line_ref, status = f"{ref[0]:.1f} ({ref[1]})", "OK" if ok else "MISMATCH"
        pct = f"{r.pct:.1f}" if r.n else "-"
        ci = f"[{r.lo:.1f}, {r.hi:.1f}]" if r.n else "-"
        shown = "*" if (r.band in BANDS_SHOWN and r.window != "steady") else " "
        print(f"{r.band:<6}{r.window:<9}{r.n:>6}{pct:>8}{ci:>16}{line_ref:>18}  {status} {shown}")
        if r.window == WINDOWS[-1]:
            print("-" * 74)
    print("* = bar drawn in the chart")
    print("All verified numbers reproduced." if all_ok else "*** MISMATCH with verified numbers ***")
    return all_ok


# =========================================================================== #
# FIGURE 1 - the takeaway chart
# =========================================================================== #
def make_takeaway(targets: pd.DataFrame, table: pd.DataFrame) -> Path:
    """Grouped bars: completion rate into closing vs opening windows, by separation band."""
    cell = {(r.band, r.window): r for r in table.itertuples()}
    fig = plt.figure(figsize=(10, 6.9), dpi=DPI)
    ax = fig.add_axes([0.085, 0.215, 0.895, 0.515])

    bar_w = 0.36
    offsets = {"closing": -bar_w / 2 - 0.015, "opening": +bar_w / 2 + 0.015}
    for i, band in enumerate(BANDS_SHOWN):
        for window in ("closing", "opening"):
            r = cell[(band, window)]
            x = i + offsets[window]
            ax.bar(x, r.pct, width=bar_w, color=COLOR[window], zorder=2)
            ax.errorbar(x, r.pct, yerr=[[r.pct - r.lo], [r.hi - r.pct]], fmt="none",
                        ecolor=INK, elinewidth=1.3, capsize=4.5, capthick=1.3, zorder=3)
            ax.text(x, r.hi + 1.6, f"{r.pct:.0f}%", ha="center", va="bottom",
                    fontsize=14.5, fontweight="bold", color=TEXT_COLOR[window])
            ax.text(x, 2.2, f"n = {r.n}", ha="center", va="bottom",
                    fontsize=10.5, color="white", fontweight="bold")

        # Gap between the two bars, written above the pair. It is drawn bold
        # when the two 95% intervals do not overlap, grey when they do.
        closing, opening = cell[(band, "closing")], cell[(band, "opening")]
        y = max(closing.hi, opening.hi) + 12.0
        clear = opening.lo > closing.hi or closing.lo > opening.hi
        colour = INK if clear else GREY
        x0, x1 = i + offsets["closing"], i + offsets["opening"]
        ax.plot([x0, x0, x1, x1], [closing.hi + 9.6, y, y, opening.hi + 9.6], color=colour,
                lw=1.8 if clear else 1.0, solid_capstyle="butt", solid_joinstyle="miter", zorder=3)
        ax.text(i, y + 1.2, gap_label(opening.pct - closing.pct), ha="center", va="bottom",
                color=colour, fontsize=15 if clear else 12, fontweight="bold" if clear else "normal")

    # Axes: football words, light horizontal grid only.
    ax.set_xlim(-0.6, len(BANDS_SHOWN) - 0.4)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=11.5, color=GREY)
    ax.set_ylabel("Passes completed", fontsize=12.5, color=INK, labelpad=6)
    ax.set_xticks(range(len(BANDS_SHOWN)))
    ax.set_xticklabels([f"{b.replace('-', '–')} yards" for b in BANDS_SHOWN],
                       fontsize=13.5, color=INK)
    ax.set_xlabel("How open the receiver was when the ball came out "
                  "(yards to the nearest coverage defender)",
                  fontsize=12.5, color=INK, labelpad=9)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)

    # Title block: the finding, the numbers behind it, then what was measured.
    def gap(band: str) -> float:
        return cell[(band, "opening")].pct - cell[(band, "closing")].pct

    def tight_range(window: str) -> str:
        rates = sorted(round(cell[(band, window)].pct) for band in TIGHT_BANDS)
        return f"{rates[0]}–{rates[-1]}%"

    tight = f"{TIGHT_BANDS[0].split('-')[0]}–{TIGHT_BANDS[-1].split('-')[1]}"
    widest = BANDS_SHOWN[-1]
    widest_gap = "no difference" if abs(gap(widest)) < 1 else gap_label(gap(widest))
    left = 0.035
    fig.text(left, 0.935, "In tight coverage, an opening window beats a closing one",
             fontsize=20, fontweight="bold", color=INK, ha="left", va="baseline")
    text_run(fig, left, 0.886, [
        (f"{tight} yards open: ", INK, "normal"), ("OPENING", BLUE_TEXT, "bold"),
        (f" completes {tight_range('opening')}, ", INK, "normal"),
        ("CLOSING", ORANGE_TEXT, "bold"),
        (f" {tight_range('closing')}. At {widest.replace('-', '–')} yards: {widest_gap}.",
         INK, "normal"),
    ], fontsize=13)
    fig.text(left, 0.845,
             "Completion rate on targeted throws by separation at release. "
             f"\"{gap_label(gap(EXPLAINED_BAND))}\" = {abs(gap(EXPLAINED_BAND)):.0f} percentage "
             "points. 2021 Weeks 1–8.",
             fontsize=11, color=GREY, ha="left", va="baseline")

    # Plain-English key (doubles as the legend).
    for y, window, word, meaning in (
        (0.793, "opening", "Opening", " = the receiver is pulling away from the nearest "
                                      "defender in the half-second before the throw"),
        (0.757, "closing", "Closing", " = the gap to the nearest defender is shrinking "
                                      "in that half-second"),
    ):
        fig.add_artist(Rectangle((left, y - 0.003), 0.017, 0.024, transform=fig.transFigure,
                                 facecolor=COLOR[window], edgecolor="none"))
        text_run(fig, left + 0.026, y, [(word, TEXT_COLOR[window], "bold"), (meaning, INK, "normal")],
                 fontsize=11.5)

    # Source and limits, with everything that is not drawn spelled out.
    shown = sum(cell[(b, w)].n for b in BANDS_SHOWN for w in ("closing", "opening"))
    far_o, far_c = cell[("8+", "opening")], cell[("8+", "closing")]
    far_overlap = far_o.lo <= far_c.hi and far_c.lo <= far_o.hi
    near_n = cell[("0-1", "opening")].n
    near_note = "no opening targets" if near_n == 0 else f"only {near_n} opening targets"
    fig.text(left, 0.062,
             f"2021 Weeks 1–8, {len(targets):,} targeted throws ({shown:,} in these bars); "
             "tracking ends 0.5 s after release. Whiskers: 95% margin of error (Wilson).",
             fontsize=9.3, color=GREY, ha="left", va="baseline")
    fig.text(left, 0.030,
             f"Not shown: steady windows (gap changing under 1 yd/s); 0–1 yd ({near_note}); "
             f"8+ yd (opening {far_o.pct:.0f}% vs closing {far_c.pct:.0f}%, margins "
             f"{'overlap' if far_overlap else 'do not overlap'}).",
             fontsize=9.3, color=GREY, ha="left", va="baseline")

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "takeaway.png"
    warn_if_clipped(fig, out.name)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# =========================================================================== #
# FIGURE 2 - two example plays
# =========================================================================== #
def example_candidates(targets: pd.DataFrame, plays: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All targeted throws that satisfy EXAMPLE_RULE, split into opening / closing."""
    # Coverage comes from plays; drop any copy already in the CSV so names stay unsuffixed.
    ctx = plays[["gameId", "playId", "pff_passCoverage"]]
    t = targets.drop(columns=["pff_passCoverage"], errors="ignore").merge(
        ctx, on=["gameId", "playId"], how="left")
    rule = EXAMPLE_RULE
    base = t[
        (t["pff_passCoverage"] == rule["coverage"])
        & t["sep_release"].between(*rule["sep"])
        & t["depth"].between(*rule["depth"])
    ]
    return {
        "opening": base[(base["trend"] > rule["min_abs_trend"]) & (base["passResult"] == "C")],
        "closing": base[(base["trend"] < -rule["min_abs_trend"]) & base["passResult"].isin(["I", "IN"])],
    }


def broken_up_by(description: str, defender_name: str) -> bool:
    """True if the play-by-play credits this defender with breaking up the pass.

    On an incompletion the official description ends "... to D.Samuel (S.Nelson)."
    where the name in brackets is the defender who got a hand on the ball.
    """
    credited = re.search(r"incomplete[^()]* to [^()]+\(([^)]+)\)", description)
    if not credited:
        return False
    last_name = defender_name.split()[-1]
    return any(name.strip().split(".")[-1] == last_name for name in credited.group(1).split(";"))


def load_example_play(game_id: int, play_id: int, targets: pd.DataFrame, plays: pd.DataFrame) -> dict:
    """Everything needed to draw one play: context, positions at release, 0.5 s trails."""
    trow = targets[(targets["gameId"] == game_id) & (targets["playId"] == play_id)].iloc[0]
    prow = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)].iloc[0]
    release = int(prow["end_frame"])

    track = (
        pl.scan_parquet(TRACKING_PARQUET)
        .filter((pl.col("gameId") == game_id) & (pl.col("playId") == play_id)
                & pl.col("frameId").is_between(release - LOOKBACK_FRAMES, release))
        .select(["nflId", "frameId", "x", "y"])
        .collect()
        .to_pandas()
    )
    roles = pd.read_csv(PFF_CSV, usecols=["gameId", "playId", "nflId", "pff_role"])
    roles = roles[(roles["gameId"] == game_id) & (roles["playId"] == play_id)][["nflId", "pff_role"]]
    names = pd.read_csv(PLAYERS_CSV, usecols=["nflId", "displayName"])
    track = track.merge(roles, on="nflId", how="left").merge(names, on="nflId", how="left")

    at_release = track[track["frameId"] == release].set_index("nflId")
    target_id = int(trow["target_nflId"])
    target_xy = at_release.loc[target_id, ["x", "y"]].to_numpy(float)

    # Nearest Coverage defender at release: the definition of separation.
    cover = at_release[at_release["pff_role"] == "Coverage"]
    dist = np.hypot(cover["x"] - target_xy[0], cover["y"] - target_xy[1])
    defender_id = int(dist.idxmin())
    if abs(dist.min() - trow["sep_release"]) > 0.02:
        raise SystemExit(f"Separation mismatch on {game_id}/{play_id}: "
                         f"{dist.min():.2f} here vs {trow['sep_release']:.2f} in the CSV")

    def path(nfl_id: int) -> np.ndarray:
        p = track[track["nflId"] == nfl_id].sort_values("frameId")
        return p[["x", "y"]].to_numpy(float)

    # The dotted "0.5 s earlier" gap is drawn to the same defender, so he must
    # also have been the nearest one then (that is what sep_before measures).
    gap_before = float(np.linalg.norm(path(target_id)[0] - path(defender_id)[0]))
    if abs(gap_before - trow["sep_before"]) > 0.02:
        raise SystemExit(f"{game_id}/{play_id}: the nearest defender changed inside the 0.5 s "
                         "window; pick an example where he is the same man throughout.")

    qb = at_release[at_release["pff_role"] == "Pass"].iloc[0]
    description = " ".join(str(prow["playDescription"]).split())
    defender_name = str(at_release.loc[defender_id, "displayName"])
    yards = int(prow["prePenaltyPlayResult"])
    result = {"C": f"complete, {yards} yards", "I": "incomplete", "IN": "intercepted"}[prow["passResult"]]
    if prow["passResult"] == "I" and broken_up_by(description, defender_name):
        result += f", broken up by {defender_name.split()[-1]}"
    return dict(
        game_id=game_id, play_id=play_id,
        description=description,
        offense=prow["possessionTeam"], defense=prow["defensiveTeam"],
        down_distance=f"{ordinal_down(prow['down'])} & {int(prow['yardsToGo'])}",
        coverage=f"{prow['pff_passCoverage']} ({str(prow['pff_passCoverageType']).lower()} coverage)",
        los=float(prow["absoluteYardlineNumber"]),
        sep_release=float(trow["sep_release"]), sep_before=float(trow["sep_before"]),
        trend=float(trow["trend"]), window=str(trow["window"]), depth=float(trow["depth"]),
        result=result,
        qb_name=str(qb["displayName"]), qb_xy=qb[["x", "y"]].to_numpy(float),
        target_name=str(at_release.loc[target_id, "displayName"]),
        defender_name=defender_name,
        target_id=target_id, defender_id=defender_id,
        target_path=path(target_id), defender_path=path(defender_id),
        players=at_release.reset_index(),
    )


def draw_field(ax, xlim: tuple[float, float], ylim: tuple[float, float], los: float) -> None:
    """Cropped top-down field: 5-yard lines, yard numbers, sidelines, line of scrimmage."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(LIGHT)
        spine.set_linewidth(1.0)

    for x in range(10, 111, 5):                       # goal line to goal line
        # a yard line hugging the crop edge would read as a double border: skip it
        if xlim[0] + 1.5 < x < xlim[1] - 1.5:
            ax.plot([x, x], [max(ylim[0], 0), min(ylim[1], 53.3)], color="#dcdcdc", lw=1.0, zorder=1)
            # yard numbers like the ones painted on the grass
            if x % 10 == 0 and 10 < x < 110:
                ax.text(x, ylim[0] + 0.55, str(50 - abs(x - 60)), ha="center", va="bottom",
                        fontsize=12, color="#c4c4c4", fontweight="bold", zorder=1)
    for y, outside in ((0.0, ylim[0]), (53.3, ylim[1])):   # sidelines, if in view
        if ylim[0] < y < ylim[1]:
            ax.axhspan(min(y, outside), max(y, outside), color="#efefef", zorder=0)
            ax.axhline(y, color=GREY, lw=2.0, zorder=1)

    ax.plot([los, los], [max(ylim[0], 0), min(ylim[1], 53.3)], color=INK, lw=2.0, zorder=2)
    ax.text(los + 0.3, ylim[1] - 0.45, "Line of scrimmage", rotation=90, ha="left", va="top",
            fontsize=9.5, color=GREY, zorder=2)
    # direction of play, centred between the two yard lines nearest the middle
    mid = 5 * round((sum(xlim) / 2 - 2.5) / 5) + 2.5
    ax.text(mid, ylim[1] - 0.45, "downfield →", ha="center", va="top", fontsize=10,
            color=GREY, zorder=2)


def draw_play_panel(ax, play: dict, xlim: tuple[float, float], ylim: tuple[float, float],
                    labels: dict) -> None:
    """
    One example play at the release frame.

    The targeted receiver is a filled circle in the window colour, his nearest
    coverage defender a black X. Faded lines are their paths over the previous
    0.5 s, each ending in an arrowhead; faded markers show where they were,
    joined by a dotted line in the window colour (the gap 0.5 s earlier).
    The solid black line is the gap at release.
    Everyone else is context.
    `labels` holds the hand-placed label offsets (see EXAMPLE_LABELS).
    """
    colour, text_colour = COLOR[play["window"]], TEXT_COLOR[play["window"]]
    draw_field(ax, xlim, ylim, play["los"])

    # --- context players ---------------------------------------------------- #
    others = play["players"]
    others = others[~others["nflId"].isin([play["target_id"], play["defender_id"]])]
    style = {
        "Pass Block": dict(marker="o", s=34, facecolors="#d9d9d9", edgecolors="none"),
        "Pass Rush": dict(marker="x", s=34, color="#cdcdcd", linewidths=1.4),
        "Pass Route": dict(marker="o", s=95, facecolors="white", edgecolors="#6f6f6f", linewidths=1.6),
        "Coverage": dict(marker="x", s=85, color="#6f6f6f", linewidths=2.0),
        "Pass": dict(marker="o", s=95, facecolors="#6f6f6f", edgecolors="none"),
    }
    for role, kw in style.items():
        grp = others[others["pff_role"] == role]
        ax.scatter(grp["x"], grp["y"], zorder=3, **kw)

    # --- the throw: the passer is usually off the left edge of the crop ----- #
    tp, dp = play["target_path"], play["defender_path"]
    t0, t1, d0, d1 = tp[0], tp[-1], dp[0], dp[-1]
    qb = play["qb_xy"]
    qb_last = play["qb_name"].split()[-1]
    if xlim[0] < qb[0] < xlim[1]:
        ax.annotate(f"{qb_last} (QB)", qb, xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=GREY)
    else:   # a short arrow just past the line of scrimmage, pointing at the target
        aim = (t1 - qb) / np.linalg.norm(t1 - qb)
        start = qb + aim * ((play["los"] + 0.7 - qb[0]) / aim[0])
        ax.annotate("", xy=start + aim * 3.8, xytext=start, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3, ls=(0, (4, 3)),
                                    shrinkA=0, shrinkB=0, mutation_scale=13))
        above = np.array([-aim[1], aim[0]]) * 0.3            # nudge the words off the arrow
        ax.text(*(start + above), f"pass from {qb_last}", rotation_mode="anchor",
                rotation=math.degrees(math.atan2(aim[1], aim[0])), ha="left", va="bottom",
                fontsize=9.5, color=GREY, zorder=2)

    # --- trails over the previous 0.5 s ------------------------------------- #
    for pts, c in ((tp, colour), (dp, INK)):
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=5, alpha=0.30, solid_capstyle="round", zorder=4)
        # arrowhead where the path meets the player, pointing the way he was moving
        heading = (pts[-1] - pts[-3]) / np.linalg.norm(pts[-1] - pts[-3])
        across = np.array([-heading[1], heading[0]])
        tip = pts[-1] - heading * HEAD_GAP
        base = tip - heading * HEAD_LENGTH
        ax.add_patch(Polygon([tip, base + across * HEAD_HALF_WIDTH, base - across * HEAD_HALF_WIDTH],
                             closed=True, facecolor=c, edgecolor="none", alpha=0.75, zorder=5))
    ax.scatter(*t0, marker="o", s=95, facecolors="white", edgecolors=colour, linewidths=1.8,
               alpha=0.9, zorder=5)
    ax.scatter(*d0, marker="x", s=75, color=INK, linewidths=2.0, alpha=0.45, zorder=5)
    ax.plot([t0[0], d0[0]], [t0[1], d0[1]], color=colour, lw=1.8, ls=(0, (1.5, 2)), zorder=4)

    # --- the two players at release, and the gap between them --------------- #
    ax.plot([t1[0], d1[0]], [t1[1], d1[1]], color=INK, lw=1.8, zorder=6)
    ax.scatter(*t1, marker="o", s=230, facecolors=colour, edgecolors="white", linewidths=1.6, zorder=7)
    ax.scatter(*d1, marker="X", s=230, facecolors=INK, edgecolors="white", linewidths=1.2, zorder=7)

    box = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8)

    def put(key: str, anchor, text: str, **kw) -> None:
        dx, dy, ha, va = labels[key]
        ax.text(anchor[0] + dx, anchor[1] + dy, text, ha=ha, va=va, zorder=8, bbox=box, **kw)

    put("gap", (t1 + d1) / 2, f"at release: {play['sep_release']:.1f} yd",
        fontsize=11.5, fontweight="bold", color=INK)
    put("earlier", (t0 + d0) / 2, f"0.5 s earlier: {play['sep_before']:.1f} yd",
        fontsize=10.5, color=text_colour)
    put("target", t1, play["target_name"].split()[-1],
        fontsize=11.5, fontweight="bold", color=text_colour)
    put("defender", d1, f"{play['defender_name'].split()[-1]}\n(nearest defender)",
        fontsize=10, color=INK, linespacing=1.05)

    # --- header and caption -------------------------------------------------- #
    ax.text(0, 1.022, f"{play['window'].upper()} WINDOW", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=14.5, fontweight="bold", color=text_colour)
    ax.text(1, 1.022, f"{play['offense']} offense vs {play['defense']}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, color=GREY)
    sign = "+" if play["trend"] > 0 else "−"
    change = "grew" if play["trend"] > 0 else "shrank"
    lines = [
        (f"{play['qb_name']} to {play['target_name']}", INK, "bold", 12.5),
        (f"{play['down_distance']}, {play['coverage']}", INK, "normal", 11.5),
        (f"Separation at release: {play['sep_release']:.1f} yd "
         f"(trend {sign}{abs(play['trend']):.1f} yd/s)", INK, "normal", 11.5),
        (f"{play['window'].capitalize()}: gap {change} {play['sep_before']:.1f} → "
         f"{play['sep_release']:.1f} yd in 0.5 s", text_colour, "bold", 12.5),
        (f"Result: {play['result']}", INK, "bold", 12.5),
    ]
    y = -0.045
    for text, c, weight, size in lines:
        ax.text(0, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=size,
                color=c, fontweight=weight)
        y -= 0.062


def make_example_plays(targets: pd.DataFrame) -> tuple[Path, list[dict]]:
    """Two real plays side by side: same separation at release, opposite window trend."""
    if not PLAYS_PARQUET.exists() or not TRACKING_PARQUET.exists():
        raise SystemExit("Missing cache/. Run: python src/load.py")
    plays = pd.read_parquet(PLAYS_PARQUET)
    candidates = example_candidates(targets, plays)
    chosen = []
    for window in ("opening", "closing"):
        game_id, play_id = EXAMPLE_PLAYS[window]
        pool = candidates[window]
        if pool[(pool["gameId"] == game_id) & (pool["playId"] == play_id)].empty:
            raise SystemExit(f"Example play {game_id}/{play_id} no longer satisfies the "
                             f"selection rule {EXAMPLE_RULE} for an {window} window.")
        chosen.append(load_example_play(game_id, play_id, targets, plays))

    # Same crop size for both panels, so a yard is the same length in each.
    width, height = 18.5, 14.5                       # yards shown: downfield x across the field
    crops = []
    for p in chosen:
        t1 = p["target_path"][-1]
        x0 = p["los"] - 1.5
        centre_y = t1[1] + float(np.clip((p["qb_xy"][1] - t1[1]) * 0.4, -3.5, 3.5))
        y0 = float(np.clip(centre_y - height / 2, -2.5, 53.3 + 2.5 - height))
        crops.append(((x0, x0 + width), (y0, y0 + height)))

    fig_w, fig_h = 11.0, 7.95
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    left, gap_w = 0.032, 0.026
    panel_w = (1 - 2 * left - gap_w) / 2
    panel_h = panel_w * fig_w * height / width / fig_h   # keeps a yard square
    top = 0.795
    for i, (window, p, (xlim, ylim)) in enumerate(zip(("opening", "closing"), chosen, crops)):
        ax = fig.add_axes([left + i * (panel_w + gap_w), top - panel_h, panel_w, panel_h])
        draw_play_panel(ax, p, xlim, ylim, EXAMPLE_LABELS[window])

    seps = [f"{p['sep_release']:.1f}" for p in chosen]
    headline = (f"{seps[0]} yards open is not always {seps[1]} yards open" if seps[0] == seps[1]
                else f"{seps[0]} yards open, {seps[1]} yards open: not the same throw")
    fig.text(left, 0.936, headline, fontsize=24, fontweight="bold", color=INK,
             ha="left", va="baseline")
    fig.text(left, 0.892,
             "Two real throws against the same coverage, frozen at the instant the ball is released. "
             "The receivers are equally open;",
             fontsize=11.3, color=GREY, ha="left", va="baseline")
    fig.text(left, 0.862,
             "what differs is whether the gap to the nearest defender was growing or shrinking "
             "over the half-second before.",
             fontsize=11.3, color=GREY, ha="left", va="baseline")

    # Legend: only the kinds of marker that are actually inside the two crops.
    in_view = set()
    for p, (xlim, ylim) in zip(chosen, crops):
        pl_ = p["players"]
        seen = pl_[pl_["x"].between(*xlim) & pl_["y"].between(*ylim)
                   & ~pl_["nflId"].isin([p["target_id"], p["defender_id"]])]
        in_view |= set(seen["pff_role"])
    handles = []
    if "Pass Route" in in_view:
        handles.append(Line2D([], [], marker="o", ls="none", ms=9, mfc="white", mec="#6f6f6f",
                              mew=1.6, label="other receivers"))
    if "Coverage" in in_view:
        handles.append(Line2D([], [], marker="x", ls="none", ms=8, color="#6f6f6f", mew=2.0,
                              label="other coverage defenders"))
    if in_view & {"Pass Block", "Pass Rush"}:
        handles.append(Line2D([], [], marker="o", ls="none", ms=5.5, mfc="#cfcfcf", mec="none",
                              label="blockers and rushers"))
    handles.append(Line2D([], [], color=INK, lw=5, alpha=0.30, solid_capstyle="round",
                          label="path over the previous 0.5 s (faded marker = where he was)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left - 0.010, 0.076),
               ncol=len(handles), frameon=False, fontsize=10, handletextpad=0.5,
               columnspacing=1.8, labelcolor=GREY)
    fig.text(left, 0.052,
             "Top-down view, offense moving left to right. NFL Big Data Bowl tracking, 2021 Weeks "
             "1–8, with PFF player roles; tracking ends 0.5 s after release.",
             fontsize=9.3, color=GREY, ha="left", va="baseline")
    rule = EXAMPLE_RULE
    fig.text(left, 0.024,
             f"Two example plays, hand-picked within a rule ({rule['coverage']}, "
             f"{rule['sep'][0]:.0f}–{rule['sep'][1]:.0f} yd separation, {rule['depth'][0]:.0f}–"
             f"{rule['depth'][1]:.0f} yd downfield); not every play ends this way. "
             f"See the chart for all {len(targets):,} throws.",
             fontsize=9.3, color=GREY, ha="left", va="baseline")

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / "example_plays.png"
    warn_if_clipped(fig, out.name)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    print("\n" + "=" * 74)
    print("Example plays drawn in figures/example_plays.png")
    print(f"rule: {EXAMPLE_RULE['coverage']}, separation {EXAMPLE_RULE['sep'][0]:.0f}-"
          f"{EXAMPLE_RULE['sep'][1]:.0f} yd, depth {EXAMPLE_RULE['depth'][0]:.0f}-"
          f"{EXAMPLE_RULE['depth'][1]:.0f} yd, |trend| > {EXAMPLE_RULE['min_abs_trend']:.0f} yd/s "
          f"-> {len(candidates['opening'])} completed opening, "
          f"{len(candidates['closing'])} not-completed closing candidates")
    print("=" * 74)
    for p in chosen:
        print(f"{p['window'].upper():<8} gameId {p['game_id']}  playId {p['play_id']}  "
              f"({p['offense']} vs {p['defense']})")
        print(f"   {p['description']}")
        print(f"   {p['qb_name']} to {p['target_name']}, {p['down_distance']}, {p['coverage']}; "
              f"nearest defender {p['defender_name']}")
        print(f"   separation at release {p['sep_release']:.2f} yd | 0.5 s earlier "
              f"{p['sep_before']:.2f} yd | trend {p['trend']:+.2f} yd/s | "
              f"target depth {p['depth']:.1f} yd | result: {p['result']}")
    return out, chosen


# =========================================================================== #
def main() -> int:
    targets = load_targets()
    table = band_table(targets)
    ok = print_band_table(table)

    paths = [make_takeaway(targets, table)]
    example_path, _ = make_example_plays(targets)
    paths.append(example_path)

    print("\nFigures written:")
    for p in paths:
        print(f"   {p.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
