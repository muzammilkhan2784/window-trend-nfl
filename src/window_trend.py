"""
src/window_trend.py — is the target's window opening or closing at the throw?

For every targeted throw we measure the target receiver's separation from the
nearest Coverage defender at two moments:

    sep_release  separation at the release frame (end of dropback)
    sep_before   separation 5 frames (0.5 s) earlier

and turn the change into a rate:

    trend = (sep_release - sep_before) / 0.5     yards per second

    opening   trend >  +1.0   (pulling away as the ball comes out)
    closing   trend <  -1.0   (defender closing as the ball comes out)
    steady    everything in between

We join nflverse play-by-play for yards_after_catch, air_yards and epa, write
one row per target to output/targets_window_trend.csv, and print four summaries.

Run:
    python src/window_trend.py
Depends on the cached parquet built by src/load.py (built on demand if missing).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import polars as pl

from load import build  # reuse the loader

ROOT = Path(__file__).resolve().parents[1]
PBP = ROOT / "external" / "play_by_play_2021.csv.gz"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)

FPS = 10              # tracking frames per second
LOOKBACK = 5          # 5 frames = 0.5 s
DT = LOOKBACK / FPS   # 0.5 s
THROW = ["C", "I", "IN"]

# Separation bands (same cut points as data-facts / load.py).
BANDS = ["0-1", "1-2", "2-3", "3-5", "5-8", "8+"]

_PASSER_RE = re.compile(r"([A-Z][A-Za-z'\-]*\.[A-Z][A-Za-z'\.\-]+)\s+pass\b")


def band_expr(col: str) -> pl.Expr:
    """Separation band from `col` (same cut points as load.py / data-facts)."""
    c = pl.col(col)
    return (
        pl.when(c < 1.0).then(pl.lit("0-1"))
        .when(c < 2.0).then(pl.lit("1-2"))
        .when(c < 3.0).then(pl.lit("2-3"))
        .when(c < 5.0).then(pl.lit("3-5"))
        .when(c < 8.0).then(pl.lit("5-8"))
        .otherwise(pl.lit("8+"))
    )


def parse_passer(desc: str | None) -> str | None:
    if desc is None:
        return None
    m = _PASSER_RE.search(desc)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Separation for the target at an arbitrary frame = end_frame - LOOKBACK
# --------------------------------------------------------------------------- #
def target_sep_before(
    tracking: pl.DataFrame, pff: pl.DataFrame, plays: pl.DataFrame
) -> pl.DataFrame:
    """
    For every targeted throw with a real release frame, the target's distance to
    the nearest Coverage defender at (end_frame - LOOKBACK). One row per
    (gameId, playId): sep_before.
    """
    tgt_plays = plays.filter(
        pl.col("target_nflId").is_not_null() & pl.col("has_release_frame")
    ).select(["gameId", "playId", "target_nflId", "end_frame"]).with_columns(
        (pl.col("end_frame") - LOOKBACK).alias("before_frame")
    )

    roles = pff.select(["gameId", "playId", "nflId", "pff_role"])

    # Positions at the before_frame for everyone on those plays.
    pos = (
        tracking.join(
            tgt_plays.select(["gameId", "playId", "before_frame", "target_nflId"]),
            on=["gameId", "playId"],
            how="inner",
        )
        .filter(pl.col("frameId") == pl.col("before_frame"))
        .select(["gameId", "playId", "nflId", "x", "y", "target_nflId"])
        .join(roles, on=["gameId", "playId", "nflId"], how="left")
    )

    target_pos = (
        pos.filter(pl.col("nflId") == pl.col("target_nflId"))
        .select(["gameId", "playId", "x", "y"])
    )
    coverage = (
        pos.filter(pl.col("pff_role") == "Coverage")
        .select(["gameId", "playId", "x", "y"])
        .rename({"x": "cx", "y": "cy"})
    )

    pairs = target_pos.join(coverage, on=["gameId", "playId"], how="left").with_columns(
        ((pl.col("x") - pl.col("cx")) ** 2 + (pl.col("y") - pl.col("cy")) ** 2)
        .sqrt()
        .alias("d")
    )
    return (
        pairs.group_by(["gameId", "playId"])
        .agg(pl.col("d").min().alias("sep_before"))
    )


# --------------------------------------------------------------------------- #
# 95% Wilson interval for a binomial proportion
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


# --------------------------------------------------------------------------- #
# Build the one-row-per-target table
# --------------------------------------------------------------------------- #
def build_target_table() -> pl.DataFrame:
    from load import load_pff

    tables = build()
    plays, routes, tracking = tables["plays"], tables["routes"], tables["tracking"]
    pff = load_pff()  # role labels; not cached as parquet

    # sep_release: the target's separation at the release frame (already in
    # routes). We also carry the target's roster position from the route table.
    sep_release = (
        routes.filter(pl.col("is_target") & pl.col("has_release_frame"))
        .select(["gameId", "playId", "nflId", "separation", "depth",
                 "officialPosition"])
        .rename({"separation": "sep_release", "nflId": "target_nflId"})
    )

    sep_before = target_sep_before(tracking, pff, plays)

    base = (
        plays.filter(
            pl.col("passResult").is_in(THROW)
            & pl.col("target_nflId").is_not_null()
            & pl.col("has_release_frame")
        )
        .select(
            [
                "gameId", "playId", "target_nflId", "target_name",
                "passResult", "playDescription", "possessionTeam",
                "defensiveTeam", "pff_passCoverage", "pff_passCoverageType",
            ]
        )
        .join(sep_release, on=["gameId", "playId", "target_nflId"], how="left")
        .join(sep_before, on=["gameId", "playId"], how="left")
        .with_columns(
            ((pl.col("sep_release") - pl.col("sep_before")) / DT).alias("trend"),
            pl.col("playDescription")
            .map_elements(parse_passer, return_dtype=pl.Utf8)
            .alias("qb"),
            (pl.col("passResult") == "C").cast(pl.Int8).alias("complete"),
            band_expr("sep_release").alias("sep_band"),
        )
        .with_columns(
            pl.when(pl.col("trend") > 1.0)
            .then(pl.lit("opening"))
            .when(pl.col("trend") < -1.0)
            .then(pl.lit("closing"))
            .otherwise(pl.lit("steady"))
            .alias("window"),
        )
    )

    # nflverse play-by-play: yards_after_catch, air_yards, epa.
    pbp = (
        pl.scan_csv(PBP, infer_schema_length=20000)
        .select(
            ["old_game_id", "play_id", "yards_after_catch", "air_yards", "epa"]
        )
        .collect()
        .rename({"old_game_id": "gameId", "play_id": "playId"})
    )
    out = base.join(pbp, on=["gameId", "playId"], how="left")

    # drop the handful with no sep_before (should be none, but be safe)
    out = out.filter(pl.col("trend").is_not_null())
    return out


# --------------------------------------------------------------------------- #
# Printing the four summaries
# --------------------------------------------------------------------------- #
def print_summaries(df: pl.DataFrame) -> None:
    # ---- (1) completion rate by window inside each separation band -------- #
    print("\n" + "=" * 72)
    print("(1) Completion rate by window trend, within each separation band")
    print("    (separation measured at release; 95% Wilson interval)")
    print("=" * 72)
    print(f"{'band':<5} {'window':<8} {'n':>5} {'comp%':>7} {'95% CI':>16}")
    print("-" * 72)
    for band in BANDS:
        sub = df.filter(pl.col("sep_band") == band)
        for win in ("opening", "steady", "closing"):
            s = sub.filter(pl.col("window") == win)
            n = s.height
            k = int(s["complete"].sum()) if n else 0
            p = (k / n * 100) if n else float("nan")
            lo, hi = wilson(k, n)
            ci = f"[{lo*100:4.1f},{hi*100:4.1f}]" if n else "n/a"
            print(f"{band:<5} {win:<8} {n:>5} {p:>6.1f}% {ci:>16}")
        print("-" * 72)

    # ---- (2) mean YAC opening vs closing, target depth < 5 yd ------------- #
    print("\n" + "=" * 72)
    print("(2) Mean yards after catch, completions only, target depth < 5 yd")
    print("=" * 72)
    shallow = df.filter(
        (pl.col("depth") < 5.0)
        & (pl.col("complete") == 1)
        & pl.col("yards_after_catch").is_not_null()
    )
    print(f"{'window':<8} {'n':>5} {'mean YAC':>10}")
    print("-" * 28)
    for win in ("opening", "closing"):
        s = shallow.filter(pl.col("window") == win)
        n = s.height
        m = s["yards_after_catch"].mean() if n else float("nan")
        print(f"{win:<8} {n:>5} {m:>9.2f}")

    # ---- (3) EPA per target opening vs closing ---------------------------- #
    print("\n" + "=" * 72)
    print("(3) EPA per target, opening vs closing")
    print("=" * 72)
    print(f"{'window':<8} {'n':>5} {'EPA/target':>12}")
    print("-" * 30)
    for win in ("opening", "closing"):
        s = df.filter((pl.col("window") == win) & pl.col("epa").is_not_null())
        n = s.height
        m = s["epa"].mean() if n else float("nan")
        print(f"{win:<8} {n:>5} {m:>11.3f}")

    # ---- (4) share of throws into opening windows, QBs with 100+ throws --- #
    print("\n" + "=" * 72)
    print("(4) Share of throws into OPENING windows, QBs with 100+ targeted throws")
    print("=" * 72)
    qb = (
        df.group_by("qb")
        .agg(
            pl.len().alias("throws"),
            (pl.col("window") == "opening").mean().alias("open_share"),
        )
        .filter(pl.col("throws") >= 100)
        .sort("open_share", descending=True)
    )
    print(f"{'QB':<14} {'throws':>7} {'open%':>7}")
    print("-" * 32)
    for r in qb.iter_rows(named=True):
        print(f"{r['qb']:<14} {r['throws']:>7} {r['open_share']*100:>6.1f}%")


# --------------------------------------------------------------------------- #
# Expected check: 2-3 yd band, opening ~67.7% (n~319) vs closing ~51.0% (n~437)
# --------------------------------------------------------------------------- #
def run_check(df: pl.DataFrame) -> bool:
    band = df.filter(pl.col("sep_band") == "2-3")
    out = []
    ok = True
    for win, ref_p, ref_n in (("opening", 67.7, 319), ("closing", 51.0, 437)):
        s = band.filter(pl.col("window") == win)
        n = s.height
        p = s["complete"].mean() * 100 if n else float("nan")
        diff = abs(p - ref_p)
        ok &= diff <= 1.0
        out.append((win, n, ref_n, p, ref_p, diff))

    print("\n" + "=" * 72)
    print("EXPECTED CHECK: 2-3 yd band completion rate")
    print("=" * 72)
    print(f"{'window':<8} {'n':>5} {'ref n':>6} {'comp%':>7} {'ref%':>7} {'diff':>7} {'status':>9}")
    print("-" * 72)
    for win, n, ref_n, p, ref_p, diff in out:
        st = "OK" if diff <= 1.0 else "OFF >1pt"
        print(f"{win:<8} {n:>5} {ref_n:>6} {p:>6.1f}% {ref_p:>6.1f}% {diff:>6.2f} {st:>9}")
    print("=" * 72)
    if ok:
        print("Within 1 point of expected on both opening and closing.")
    else:
        print("*** OFF by more than 1 point. STOPPING. ***")
    return ok


def main() -> int:
    df = build_target_table()

    out_path = OUTDIR / "targets_window_trend.csv"
    df.write_csv(out_path)
    print(f"Wrote {df.height} targeted throws -> {out_path}")

    print_summaries(df)
    ok = run_check(df)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
