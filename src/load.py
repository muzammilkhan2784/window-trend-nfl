"""
src/load.py — reproducible data loader for the NYU Big Data Bowl 2026 project.

Builds, from the raw 2021 Weeks 1-8 pass-play data, a set of cached parquet
tables that every downstream analysis reads from:

  cache/plays.parquet        one row per play, cleaned, with target receiver +
                             snap / end-of-dropback frame and type attached
  cache/routes.parquet       one row per (gameId, playId, nflId) Pass Route
                             runner, with separation from nearest Coverage
                             defender measured at the release frame
  cache/tracking.parquet     cleaned, direction-normalized tracking, players only
                             (ball rows dropped), restricted to snap->release

All rules here come straight from .kiro/steering/data-facts.md and brief.md.

Run top to bottom:
    python src/load.py                 # build caches if missing, then checks
    python src/load.py --rebuild       # force a full rebuild

The raw data is expected in <repo>/bdb-data/data (clone of
github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data).
Set the BDB_DATA environment variable to point somewhere else.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BDB_DATA", ROOT / "bdb-data" / "data"))
TRACKING_DIR = DATA / "tracking"
CACHE = ROOT / "cache"

NULLS = ["NA"]  # missing values are the literal string "NA"

# Events that end the dropback, mapped to the event type that closed the window.
# Rule: earliest frame after the snap among these wins.
# The play OUTCOME (end_type) comes from passResult, which is trusted over events
# on the few dozen plays where the two disagree.
END_EVENTS = {
    "pass_forward": "throw",
    "qb_sack": "sack",
    "qb_strip_sack": "sack",
    "run": "scramble",
}


# --------------------------------------------------------------------------- #
# 1. Load the flat (non-tracking) tables
# --------------------------------------------------------------------------- #
def load_plays() -> pl.DataFrame:
    df = pl.read_csv(DATA / "plays.csv", null_values=NULLS)
    # Only playDescription carries trailing / double spaces; strip & squeeze it.
    df = df.with_columns(
        pl.col("playDescription")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("playDescription")
    )
    return df


def load_players() -> pl.DataFrame:
    return pl.read_csv(DATA / "players.csv", null_values=NULLS)


def load_games() -> pl.DataFrame:
    return pl.read_csv(DATA / "games.csv", null_values=NULLS)


def load_pff() -> pl.DataFrame:
    return pl.read_csv(DATA / "pffScoutingData.csv", null_values=NULLS)


# --------------------------------------------------------------------------- #
# 2. Tracking: load all files, normalize direction, drop ball rows
# --------------------------------------------------------------------------- #
def load_tracking() -> pl.DataFrame:
    files = sorted(TRACKING_DIR.glob("tracking_*.csv"))
    if not files:
        raise FileNotFoundError(f"No tracking files under {TRACKING_DIR}")

    frames = []
    for f in files:
        frames.append(
            pl.scan_csv(f, null_values=NULLS).with_columns(
                # frameId / 10 == seconds (time is truncated to whole seconds).
                (pl.col("frameId") / 10.0).alias("t_sec")
            )
        )
    lf = pl.concat(frames)

    # Normalize so every offense moves left -> right.
    # When playDirection == "left" the field is rotated 180 degrees:
    # x = 120 - x, y = 53.3 - y, and the angles o / dir turn by 180.
    is_left = pl.col("playDirection") == "left"
    lf = lf.with_columns(
        pl.when(is_left).then(120.0 - pl.col("x")).otherwise(pl.col("x")).alias("x"),
        pl.when(is_left).then(53.3 - pl.col("y")).otherwise(pl.col("y")).alias("y"),
        pl.when(is_left).then((pl.col("o") + 180.0) % 360.0).otherwise(pl.col("o")).alias("o"),
        pl.when(is_left).then((pl.col("dir") + 180.0) % 360.0).otherwise(pl.col("dir")).alias("dir"),
    )

    # Drop the ball rows (team == "football"); keep 22 players per frame.
    lf = lf.filter(pl.col("team") != "football")

    return lf.collect()


# --------------------------------------------------------------------------- #
# 3. Snap frame & end-of-dropback frame/type per play
# --------------------------------------------------------------------------- #
def dropback_windows(tracking: pl.DataFrame) -> pl.DataFrame:
    """
    One row per (gameId, playId) that has a ball_snap:
        snap_frame, end_frame, end_event_type
    end_event_type is the kind of event that closed the window (throw / sack /
    scramble). Events repeat across all rows of a frame, so collapse to
    frame-level first.
    """
    frame_events = (
        tracking.select(["gameId", "playId", "frameId", "event"])
        .unique(subset=["gameId", "playId", "frameId"])
        .sort(["gameId", "playId", "frameId"])
    )

    # Snap = first frame with event == "ball_snap".
    snap = (
        frame_events.filter(pl.col("event") == "ball_snap")
        .group_by(["gameId", "playId"])
        .agg(pl.col("frameId").min().alias("snap_frame"))
    )

    fe = frame_events.join(snap, on=["gameId", "playId"], how="inner")
    after = fe.filter(pl.col("frameId") > pl.col("snap_frame"))

    # Primary end events (earliest after snap among throw/sack/scramble).
    primary = (
        after.filter(pl.col("event").is_in(list(END_EVENTS.keys())))
        .sort(["gameId", "playId", "frameId"])
        .group_by(["gameId", "playId"])
        .agg(
            pl.col("frameId").first().alias("end_frame"),
            pl.col("event").first().alias("end_event"),
        )
    )

    # Fallback: autoevent_passforward, only when no primary event exists.
    fallback = (
        after.filter(pl.col("event") == "autoevent_passforward")
        .sort(["gameId", "playId", "frameId"])
        .group_by(["gameId", "playId"])
        .agg(pl.col("frameId").first().alias("fb_frame"))
    )

    out = (
        snap.join(primary, on=["gameId", "playId"], how="left")
        .join(fallback, on=["gameId", "playId"], how="left")
        .with_columns(
            pl.coalesce(["end_frame", "fb_frame"]).alias("end_frame"),
            pl.when(pl.col("end_event").is_not_null())
            .then(pl.col("end_event").replace(END_EVENTS))
            .when(pl.col("fb_frame").is_not_null())
            .then(pl.lit("throw"))
            .otherwise(None)
            .alias("end_event_type"),
        )
        .drop("fb_frame", "end_event")
    )
    return out


# --------------------------------------------------------------------------- #
# 4. Parse targeted receiver from playDescription
# --------------------------------------------------------------------------- #
# Target phrasing in playDescription:
#   "... pass [incomplete] [deep|short] [left|right|middle] to X.Lastname ..."
#   "... pass to X.Lastname is complete"                 (two-point tries)
#   "... intended for X.Lastname INTERCEPTED ..."
# The receiver token is an initial + "." + a capitalized last name. The last
# name may carry the "St." particle (the only multi-word last name in this data,
# e.g. "A.St. Brown"). Everything else after the name (trailing stat clause,
# "INTERCEPTED", a following "PENALTY ..." sentence) must be excluded.
_TARGET_RE = re.compile(
    r"(?:intended for|pass(?:\s+incomplete)?(?:\s+(?:deep|short))?"
    r"(?:\s+(?:left|right|middle))?\s+to)\s+"
    r"([A-Z][a-z]*\.(?:St\.\s)?[A-Z][A-Za-z'\-]+)"
)

# Name suffixes to ignore when forming match keys.
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def parse_target_name(desc: str | None) -> str | None:
    """Return the receiver token (e.g. 'C.Godwin', 'A.St. Brown') or None."""
    if desc is None:
        return None
    m = _TARGET_RE.search(desc)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip()


def target_keys(name: str | None) -> tuple[str | None, str | None]:
    """
    ('C.Godwin')   -> ('C', 'godwin')
    ('A.St. Brown')-> ('A', 'st brown')
    ('Dj.Moore')   -> ('D', 'moore')     # first initial char only
    """
    if name is None:
        return (None, None)
    initial, _, last = name.partition(".")
    last = re.sub(r"\s+", " ", last).strip().lower().replace(".", "")
    toks = [t for t in last.split(" ") if t and t not in _SUFFIX]
    return (initial.strip()[:1].upper() or None, " ".join(toks) or None)


def display_keys(display_name: str | None) -> tuple[str | None, str | None]:
    """
    'Tom Brady'          -> ('T', 'brady')
    'Amon-Ra St. Brown'  -> ('A', 'st brown')
    'DJ Moore'           -> ('D', 'moore')
    """
    if display_name is None:
        return (None, None)
    dn = re.sub(r"\s+", " ", display_name).strip()
    parts = dn.split(" ")
    last_toks = [
        p.replace(".", "").lower()
        for p in parts[1:]
        if p.replace(".", "").lower() not in _SUFFIX
    ]
    return (parts[0][0].upper() if parts and parts[0] else None, " ".join(last_toks) or None)


def attach_targets(
    plays: pl.DataFrame, pff: pl.DataFrame, players: pl.DataFrame
) -> pl.DataFrame:
    """
    Add target_nflId / target_name to each play by matching the parsed
    description name to that play's Pass Route runners on first-initial + last.
    """
    parsed = plays.select(["gameId", "playId", "playDescription"]).with_columns(
        pl.col("playDescription")
        .map_elements(parse_target_name, return_dtype=pl.Utf8)
        .alias("target_name")
    ).with_columns(
        pl.col("target_name")
        .map_elements(lambda n: target_keys(n)[0], return_dtype=pl.Utf8)
        .alias("t_init"),
        pl.col("target_name")
        .map_elements(lambda n: target_keys(n)[1], return_dtype=pl.Utf8)
        .alias("t_last"),
    )

    # Candidate route runners per play, with (initial, last) keys from displayName.
    routes = (
        pff.filter(pl.col("pff_role") == "Pass Route")
        .select(["gameId", "playId", "nflId"])
        .join(players.select(["nflId", "displayName"]), on="nflId", how="left")
        .with_columns(
            pl.col("displayName")
            .map_elements(lambda d: display_keys(d)[0], return_dtype=pl.Utf8)
            .alias("r_init"),
            pl.col("displayName")
            .map_elements(lambda d: display_keys(d)[1], return_dtype=pl.Utf8)
            .alias("r_last"),
        )
    )

    matched = parsed.join(
        routes,
        left_on=["gameId", "playId", "t_init", "t_last"],
        right_on=["gameId", "playId", "r_init", "r_last"],
        how="left",
    ).select(
        ["gameId", "playId", "target_name", "nflId", "displayName"]
    ).rename({"nflId": "target_nflId", "displayName": "target_display"})

    # If a parsed name matched >1 route runner on this play, keep it ambiguous
    # (drop the nflId) rather than guess.
    counts = matched.group_by(["gameId", "playId"]).agg(
        pl.col("target_nflId").n_unique().alias("n_match")
    )
    matched = matched.join(counts, on=["gameId", "playId"], how="left")
    matched = matched.with_columns(
        pl.when(pl.col("n_match") > 1)
        .then(None)
        .otherwise(pl.col("target_nflId"))
        .alias("target_nflId")
    ).unique(subset=["gameId", "playId"]).drop("n_match")

    return plays.join(
        matched.select(["gameId", "playId", "target_name", "target_nflId"]),
        on=["gameId", "playId"],
        how="left",
    )


# --------------------------------------------------------------------------- #
# 5. Separation: each Pass Route runner -> nearest Coverage defender
# --------------------------------------------------------------------------- #
def route_separation(
    tracking: pl.DataFrame,
    pff: pl.DataFrame,
    windows: pl.DataFrame,
    frame: str = "end",
) -> pl.DataFrame:
    """
    For every Pass Route player, at the chosen frame (default = release /
    end-of-dropback), distance in yards to the nearest Coverage defender.

    Returns one row per (gameId, playId, nflId) route runner with 'separation'.
    """
    frame_col = "end_frame" if frame == "end" else "snap_frame"

    roles = pff.select(["gameId", "playId", "nflId", "pff_role"])

    # Positions at the chosen frame only.
    wf = windows.select(["gameId", "playId", frame_col]).drop_nulls()
    pos_at_frame = (
        tracking.join(wf, on=["gameId", "playId"], how="inner")
        .filter(pl.col("frameId") == pl.col(frame_col))
        .select(["gameId", "playId", "nflId", "x", "y"])
        .join(roles, on=["gameId", "playId", "nflId"], how="left")
    )

    runners = pos_at_frame.filter(pl.col("pff_role") == "Pass Route").select(
        ["gameId", "playId", "nflId", "x", "y"]
    )
    coverage = pos_at_frame.filter(pl.col("pff_role") == "Coverage").select(
        ["gameId", "playId", "x", "y"]
    ).rename({"x": "cx", "y": "cy"})

    # Pair every runner with every coverage defender on the same play (same
    # frame by construction), then take the minimum distance.
    pairs = runners.join(coverage, on=["gameId", "playId"], how="left")
    pairs = pairs.with_columns(
        (
            (pl.col("x") - pl.col("cx")) ** 2 + (pl.col("y") - pl.col("cy")) ** 2
        ).sqrt().alias("d")
    )
    sep = (
        pairs.group_by(["gameId", "playId", "nflId"])
        .agg(
            pl.col("d").min().alias("separation"),
            pl.col("x").first().alias("x"),
            pl.col("y").first().alias("y"),
        )
    )
    return sep


# --------------------------------------------------------------------------- #
# Build pipeline
# --------------------------------------------------------------------------- #
def build(rebuild: bool = False) -> dict[str, pl.DataFrame]:
    plays_p = CACHE / "plays.parquet"
    routes_p = CACHE / "routes.parquet"
    tracking_p = CACHE / "tracking.parquet"

    if (
        not rebuild
        and plays_p.exists()
        and routes_p.exists()
        and tracking_p.exists()
    ):
        print("Loading cached parquet tables from cache/ ...")
        return {
            "plays": pl.read_parquet(plays_p),
            "routes": pl.read_parquet(routes_p),
            "tracking": pl.read_parquet(tracking_p),
        }

    if not (DATA / "plays.csv").exists():
        sys.exit(
            f"Raw data not found at {DATA}. Put plays.csv, players.csv, "
            "pffScoutingData.csv and tracking/ there, or set BDB_DATA to that folder."
        )
    CACHE.mkdir(exist_ok=True)

    print("Building caches from raw data ...")
    plays = load_plays()
    players = load_players()
    pff = load_pff()

    print(f"  plays: {plays.height}  players: {players.height}  pff: {pff.height}")

    print("  loading & normalizing tracking ...")
    tracking = load_tracking()
    print(f"  tracking rows (players only): {tracking.height}")

    print("  computing dropback windows ...")
    windows = dropback_windows(tracking)

    print("  parsing targets ...")
    is_throw = pl.col("passResult").is_in(["C", "I", "IN"])
    plays = attach_targets(plays, pff, players).with_columns(
        # A target only exists on a thrown pass.
        pl.when(is_throw).then(pl.col("target_nflId")).otherwise(None).alias("target_nflId"),
        pl.when(is_throw).then(pl.col("target_name")).otherwise(None).alias("target_name"),
    )

    # 25 plays have no ball_snap: they stay in plays / routes with null
    # snap_frame, end_frame, end_type and separation, and are absent from the
    # cached tracking.
    direction = tracking.select(["gameId", "playId", "playDirection"]).unique()
    plays = (
        plays.join(windows, on=["gameId", "playId"], how="left")
        .join(direction, on=["gameId", "playId"], how="left")
        .with_columns(
            # Line of scrimmage in the same left -> right frame as tracking x.
            pl.when(pl.col("playDirection") == "left")
            .then(120 - pl.col("absoluteYardlineNumber"))
            .otherwise(pl.col("absoluteYardlineNumber"))
            .alias("absoluteYardlineNumber"),
            # Outcome of the dropback: passResult is trusted over the events.
            pl.when(pl.col("end_frame").is_null()).then(None)
            .when(is_throw).then(pl.lit("throw"))
            .when(pl.col("passResult") == "S").then(pl.lit("sack"))
            .when(pl.col("passResult") == "R").then(pl.lit("scramble"))
            .otherwise(None)
            .alias("end_type"),
        )
        .with_columns(
            # True when end_frame really is the frame the ball was released
            # (3 throws only carry a `run` event: exclude them from release math).
            ((pl.col("end_type") == "throw") & (pl.col("end_event_type") == "throw"))
            .fill_null(False)
            .alias("has_release_frame")
        )
    )

    # Restrict cached tracking to the snap -> release window (players only).
    tracking = tracking.join(
        windows.select(["gameId", "playId", "snap_frame", "end_frame"]),
        on=["gameId", "playId"],
        how="inner",
    ).filter(
        (pl.col("frameId") >= pl.col("snap_frame"))
        & (pl.col("frameId") <= pl.col("end_frame"))
    )

    print("  computing separation at release ...")
    sep = route_separation(tracking, pff, windows, frame="end")
    routes = (
        pff.filter(pl.col("pff_role") == "Pass Route")
        .select(["gameId", "playId", "nflId", "pff_positionLinedUp", "pff_blockType"])
        .join(players.select(["nflId", "displayName", "officialPosition"]),
              on="nflId", how="left")
        .join(sep, on=["gameId", "playId", "nflId"], how="left")
        .join(
            plays.select(["gameId", "playId", "target_nflId", "passResult",
                          "end_type", "has_release_frame", "absoluteYardlineNumber"]),
            on=["gameId", "playId"], how="left",
        )
        .with_columns(
            # False (not null) on plays with no matched target.
            (pl.col("nflId") == pl.col("target_nflId")).fill_null(False).alias("is_target"),
            # Yards past the line of scrimmage at the release frame.
            (pl.col("x") - pl.col("absoluteYardlineNumber")).alias("depth"),
        )
        .sort(["gameId", "playId", "nflId"])
    )

    print("  writing parquet ...")
    plays.write_parquet(plays_p)
    routes.write_parquet(routes_p)
    tracking.sort(["gameId", "playId", "nflId", "frameId"]).write_parquet(tracking_p)

    return {"plays": plays, "routes": routes, "tracking": tracking}


# --------------------------------------------------------------------------- #
# Check table against the reference numbers in data-facts.md
# --------------------------------------------------------------------------- #
REF_THROWS = 7565
REF_TARGETS = 7287
# Completion rate by target separation at release (band -> pct).
REF_SEP_BANDS = [
    ("0-1", 0.0, 1.0, 22.7),
    ("1-2", 1.0, 2.0, 47.0),
    ("2-3", 2.0, 3.0, 59.2),
    ("3-5", 3.0, 5.0, 71.5),
    ("5-8", 5.0, 8.0, 78.0),
    ("8+", 8.0, 1e9, 84.9),
]


def _band_expr() -> pl.Expr:
    e = pl.when(pl.col("separation") < 1.0).then(pl.lit("0-1"))
    e = e.when(pl.col("separation") < 2.0).then(pl.lit("1-2"))
    e = e.when(pl.col("separation") < 3.0).then(pl.lit("2-3"))
    e = e.when(pl.col("separation") < 5.0).then(pl.lit("3-5"))
    e = e.when(pl.col("separation") < 8.0).then(pl.lit("5-8"))
    return e.otherwise(pl.lit("8+"))


def run_checks(tables: dict[str, pl.DataFrame]) -> bool:
    plays = tables["plays"]
    routes = tables["routes"]

    # Throws = passes actually thrown.
    # data-facts: Throws = 7,565  (C 4,620 + I 2,755 + IN 190 = 7,565).
    throws = plays.filter(
        pl.col("passResult").is_in(["C", "I", "IN"])
    ).height

    # Targets: matched receiver on a throw.
    targets = plays.filter(
        pl.col("passResult").is_in(["C", "I", "IN"])
        & pl.col("target_nflId").is_not_null()
    ).height

    # Completion rate by separation band, among TARGETED route runners on throws.
    tgt = (
        routes.filter(
            pl.col("is_target")
            & pl.col("separation").is_not_null()
            & pl.col("passResult").is_in(["C", "I", "IN"])
        )
        .with_columns(
            _band_expr().alias("band"),
            (pl.col("passResult") == "C").cast(pl.Int32).alias("complete"),
        )
    )
    band_tbl = (
        tgt.group_by("band")
        .agg(
            pl.len().alias("n"),
            (pl.col("complete").mean() * 100).alias("comp_pct"),
        )
    ).to_dict(as_series=False)
    band_map = {
        b: (n, p)
        for b, n, p in zip(band_tbl["band"], band_tbl["n"], band_tbl["comp_pct"])
    }

    # --- print the check table -------------------------------------------- #
    rows = []
    ok = True

    def pct_off(actual: float, ref: float) -> float:
        return abs(actual - ref) / ref * 100.0

    def status(off: float) -> str:
        return "OK" if off <= 2.0 else "OFF >2%"

    o = pct_off(throws, REF_THROWS)
    ok &= o <= 2.0
    rows.append(("throws", f"{throws}", f"{REF_THROWS}", f"{o:4.2f}%", status(o)))

    o = pct_off(targets, REF_TARGETS)
    ok &= o <= 2.0
    rows.append(("targets", f"{targets}", f"{REF_TARGETS}", f"{o:4.2f}%", status(o)))

    # Exact structural counts: these test the windows and the route table.
    med_depth = routes.filter(pl.col("is_target") & pl.col("depth").is_not_null())["depth"].median()
    for name, actual, ref in [
        ("plays", plays.height, 8557),
        ("plays without ball_snap", plays["snap_frame"].null_count(), 25),
        ("route player-plays", routes.height, 39513),
        ("is_target nulls", routes["is_target"].null_count(), 0),
    ]:
        ok &= actual == ref
        rows.append((name, str(actual), str(ref), "-", "OK" if actual == ref else "MISMATCH"))
    o = pct_off(med_depth, 5.3)
    ok &= o <= 5.0
    rows.append(("median target depth (yd)", f"{med_depth:.2f}", "5.3", f"{o:4.2f}%",
                 "OK" if o <= 5.0 else "OFF >5%"))

    for band, _lo, _hi, ref_pct in REF_SEP_BANDS:
        n, p = band_map.get(band, (0, float("nan")))
        o = pct_off(p, ref_pct) if p == p else float("inf")
        ok &= o <= 2.0
        rows.append(
            (
                f"comp% sep {band}",
                f"{p:5.1f}% (n={n})" if p == p else "n/a",
                f"{ref_pct:5.1f}%",
                f"{o:4.2f}%" if o != float('inf') else "inf",
                status(o),
            )
        )

    w = [max(len(r[i]) for r in rows + [("metric", "actual", "reference", "off", "status")]) for i in range(5)]
    hdr = ("metric", "actual", "reference", "off", "status")
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print("\n" + "=" * len(line))
    print("CHECK TABLE vs data-facts reference numbers")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[i]).ljust(w[i]) for i in range(5)))
    print("=" * len(line))

    if ok:
        print("\nAll numbers within 2% of reference. Caches look good.")
    else:
        print("\n*** One or more numbers are OFF by more than 2%. STOPPING. ***")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="force rebuild caches")
    args = ap.parse_args()

    tables = build(rebuild=args.rebuild)
    ok = run_checks(tables)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
