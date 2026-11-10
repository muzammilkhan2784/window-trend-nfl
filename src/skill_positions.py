"""
src/skill_positions.py — Window Trend by skill position, alignment and receiver.

Reads the one-row-per-targeted-throw table written by window_trend.py and
answers three skill-position questions:

  1. POSITION   Do WRs, TEs and RBs get thrown opening or closing windows, and
                what happens on each?            -> output/position_summary.csv
  2. ALIGNMENT  Same question by where the target lined up (wide, slot, inline
                tight end, backfield).           -> output/alignment_summary.csv
  3. RECEIVER   For every receiver with 25+ targets: how often his window was
                opening, and how many catches he made compared with what the
                league catches on the same kind of window
                ("catches over expected").       -> output/receiver_table.csv

Definitions (all inherited from window_trend.py, nothing recomputed here)
  trend    change in the target's distance to the nearest coverage defender over
           the 0.5 s before release, in yards per second
  window   opening (trend > +1.0), closing (trend < -1.0), steady otherwise
  sep_band separation at release: 0-1, 1-2, 2-3, 3-5, 5-8, 8+ yards

Expected catch rate
  The league completion rate in the same cell (separation band x window, 18
  cells, computed on ALL targeted throws). A receiver's expected catches are the
  sum of those cell rates over his targets, so expected catches summed over the
  league equal actual catches exactly.

Honest limit, printed at the end
  Split-half stability (odd vs even games) for opening share, catch rate over
  expected, raw catch rate and closing-window catch rate. Read those numbers
  before treating any column of receiver_table.csv as a receiver skill.

Run from the project root:
    python src/skill_positions.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "output" / "targets_window_trend.csv"
ROUTES = ROOT / "cache" / "routes.parquet"
PLAYERS = ROOT / "bdb-data" / "data" / "players.csv"
OUT = ROOT / "output"

BANDS = ["0-1", "1-2", "2-3", "3-5", "5-8", "8+"]
WINDOWS = ["opening", "steady", "closing"]
TIGHT_BANDS = ["1-2", "2-3"]          # the 1-3 yd range where the effect lives
SKILL = ["WR", "TE", "RB"]
MIN_TARGETS = 25                      # receiver table cut
MIN_HALF = 12                         # targets needed in EACH half for stability
EXPECTED_TOTAL = 7263                 # targeted throws, 2021 Weeks 1-8

# Verified league figures (share opening, share closing, n) before folding FB.
REFERENCE_SHARES = {
    "WR": (26.5, 29.8, 4416),
    "TE": (29.5, 31.1, 1563),
    "RB": (21.8, 48.4, 1222),
}

# pff_positionLinedUp -> where the target lined up.
ALIGNMENT = {
    "LWR": "wide", "RWR": "wide",
    "SLWR": "slot", "SRWR": "slot",
    "SLiWR": "slot", "SRiWR": "slot", "SLoWR": "slot", "SRoWR": "slot",
    "TE-L": "inline TE", "TE-R": "inline TE",
    "TE-iL": "inline TE", "TE-iR": "inline TE",
    "TE-oL": "inline TE", "TE-oR": "inline TE",
    "HB": "backfield", "HB-L": "backfield", "HB-R": "backfield",
    "FB": "backfield", "FB-L": "backfield", "FB-R": "backfield",
}
ALIGNMENT_ORDER = ["wide", "slot", "inline TE", "backfield"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_targets() -> pd.DataFrame:
    """All targeted throws, with alignment and the league expected catch rate."""
    df = pd.read_csv(TARGETS)

    # Alignment of the targeted player on that play (one route row per player).
    routes = pd.read_parquet(
        ROUTES, columns=["gameId", "playId", "nflId", "pff_positionLinedUp"]
    ).rename(columns={"nflId": "target_nflId"})
    df = df.merge(routes, on=["gameId", "playId", "target_nflId"], how="left",
                  validate="one_to_one")
    df["alignment"] = df["pff_positionLinedUp"].map(ALIGNMENT)

    # Expected catch rate: league completion rate in the same band x window cell,
    # computed on every targeted throw (nothing left out).
    df["exp_catch"] = df.groupby(["sep_band", "window"])["complete"].transform("mean")

    # Skill position, FB folded into RB. QB / OL targets get NaN and are dropped
    # by skill_only().
    df["pos"] = df["officialPosition"].replace({"FB": "RB"})
    df.loc[~df["pos"].isin(SKILL), "pos"] = np.nan
    return df


def skill_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pos"].notna()].copy()


# --------------------------------------------------------------------------- #
# Group summary shared by the position and alignment tables
# --------------------------------------------------------------------------- #
def _pct(x: pd.Series) -> float:
    return round(100 * x.mean(), 1) if len(x) else np.nan


def summarize(g: pd.DataFrame) -> dict:
    """Window shares and outcomes for one group of targets."""
    op, cl = g[g["window"] == "opening"], g[g["window"] == "closing"]
    op_t = op[op["sep_band"].isin(TIGHT_BANDS)]
    cl_t = cl[cl["sep_band"].isin(TIGHT_BANDS)]
    op_c, cl_c = op[op["complete"] == 1], cl[cl["complete"] == 1]
    row = {
        "targets": len(g),
        "pct_opening": _pct(g["window"] == "opening"),
        "pct_steady": _pct(g["window"] == "steady"),
        "pct_closing": _pct(g["window"] == "closing"),
        "n_opening": len(op),
        "n_closing": len(cl),
        "comp_pct_opening": _pct(op["complete"]),
        "comp_pct_closing": _pct(cl["complete"]),
        "n_opening_1to3yd": len(op_t),
        "n_closing_1to3yd": len(cl_t),
        "comp_pct_opening_1to3yd": _pct(op_t["complete"]),
        "comp_pct_closing_1to3yd": _pct(cl_t["complete"]),
        "yac_opening": round(op_c["yards_after_catch"].mean(), 2),
        "yac_closing": round(cl_c["yards_after_catch"].mean(), 2),
        "epa_per_target_opening": round(op["epa"].mean(), 3),
        "epa_per_target_closing": round(cl["epa"].mean(), 3),
    }
    row["comp_gap_pts"] = round(row["comp_pct_opening"] - row["comp_pct_closing"], 1)
    row["comp_gap_1to3yd_pts"] = round(
        row["comp_pct_opening_1to3yd"] - row["comp_pct_closing_1to3yd"], 1)
    row["yac_gap"] = round(row["yac_opening"] - row["yac_closing"], 2)
    return row


def group_table(df: pd.DataFrame, by: str, order: list[str]) -> pd.DataFrame:
    rows = [{by: k, **summarize(df[df[by] == k])} for k in order]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Receiver table
# --------------------------------------------------------------------------- #
def receiver_table(df: pd.DataFrame, pos_open: dict[str, float]) -> pd.DataFrame:
    """One row per skill-position receiver with MIN_TARGETS+ targets."""
    players = pd.read_csv(PLAYERS, usecols=["nflId", "displayName"])
    names = players.set_index("nflId")["displayName"]

    rows = []
    for nfl_id, g in df.groupby("target_nflId"):
        if len(g) < MIN_TARGETS:
            continue
        cl = g[g["window"] == "closing"]
        catches, expected = int(g["complete"].sum()), g["exp_catch"].sum()
        pos = g["pos"].iloc[0]
        open_share = 100 * (g["window"] == "opening").mean()
        rows.append({
            "nflId": int(nfl_id),
            "receiver": names.get(nfl_id, g["target_name"].iloc[0]),
            "position": pos,
            "team": g["possessionTeam"].mode().iloc[0],
            "targets": len(g),
            "pct_opening": round(open_share, 1),
            "pct_closing": round(100 * (g["window"] == "closing").mean(), 1),
            "position_avg_pct_opening": pos_open[pos],
            "opening_vs_position_pts": round(open_share - pos_open[pos], 1),
            "catches": catches,
            "catch_rate": round(100 * catches / len(g), 1),
            "expected_catch_rate": round(100 * expected / len(g), 1),
            "expected_catches": round(expected, 1),
            "catches_over_expected": round(catches - expected, 1),
            "catch_rate_over_expected_pts": round(100 * (catches - expected) / len(g), 1),
            "closing_targets": len(cl),
            "closing_catch_rate": _pct(cl["complete"]),
        })
    return (pd.DataFrame(rows)
            .sort_values("catches_over_expected", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Stability: does a receiver's number repeat from one half of his games to the
# other?
# --------------------------------------------------------------------------- #
def split_half(df: pd.DataFrame) -> pd.DataFrame:
    """Split-half correlations across receivers (odd vs even games).

    Each receiver's games are ranked by gameId (chronological); odd-ranked games
    form one half, even-ranked the other. A receiver enters a row only with
    MIN_HALF+ qualifying targets in EACH half.
    """
    d = df.copy()
    d["game_rank"] = d.groupby("target_nflId")["gameId"].rank(method="dense")
    d["half"] = np.where(d["game_rank"] % 2 == 1, "odd", "even")
    d["is_open"] = (d["window"] == "opening").astype(float)
    d["over_exp"] = d["complete"] - d["exp_catch"]

    specs = [
        ("(a) share of targets with an opening window", d, "is_open", MIN_HALF),
        ("(b) catch rate over expected", d, "over_exp", MIN_HALF),
        ("(c) raw catch rate", d, "complete", MIN_HALF),
        ("(d) catch rate on closing windows", d[d["window"] == "closing"],
         "complete", MIN_HALF),
        ("(d') same, loosened to 8+ closing targets per half",
         d[d["window"] == "closing"], "complete", 8),
    ]
    rows = []
    for label, sub, col, min_n in specs:
        agg = (sub.groupby(["target_nflId", "half"])[col]
               .agg(["mean", "size"]).unstack("half"))
        agg = agg.dropna()
        agg = agg[(agg[("size", "odd")] >= min_n) & (agg[("size", "even")] >= min_n)]
        n = len(agg)
        if n >= 5:
            r, p = stats.pearsonr(agg[("mean", "odd")], agg[("mean", "even")])
            rho = stats.spearmanr(agg[("mean", "odd")], agg[("mean", "even")])[0]
        else:
            r = p = rho = np.nan
        rows.append({"measure": label, "min_targets_per_half": min_n,
                     "receivers": n, "pearson_r": round(r, 3),
                     "p_value": round(p, 3), "spearman_rho": round(rho, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Printing helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def show(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def check(label: str, ok: bool) -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    return ok


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(exist_ok=True)
    all_targets = load_targets()
    df = skill_only(all_targets)

    # ---- sanity checks ---------------------------------------------------- #
    banner("SANITY CHECKS")
    ok = check(f"total targeted throws = {len(all_targets):,} (expected {EXPECTED_TOTAL:,})",
               len(all_targets) == EXPECTED_TOTAL)
    print("  officialPosition of the target:",
          all_targets["officialPosition"].value_counts().to_dict())
    dropped = all_targets[all_targets["pos"].isna()]
    print(f"  dropped as non-skill: {len(dropped)} "
          f"({dropped['officialPosition'].value_counts().to_dict()}); "
          f"skill-position targets kept: {len(df):,}")
    for pos, (ref_open, ref_close, ref_n) in REFERENCE_SHARES.items():
        g = all_targets[all_targets["officialPosition"] == pos]
        o, c = _pct(g["window"] == "opening"), _pct(g["window"] == "closing")
        ok &= check(f"{pos} (before folding FB): opening {o}% / closing {c}% (n={len(g):,}); "
                    f"reference {ref_open}% / {ref_close}% (n={ref_n:,})",
                    (o, c, len(g)) == (ref_open, ref_close, ref_n))
    fb = all_targets[all_targets["officialPosition"] == "FB"]
    rb = df[df["pos"] == "RB"]
    print(f"  FB alone: opening {_pct(fb['window'] == 'opening')}% / "
          f"closing {_pct(fb['window'] == 'closing')}% (n={len(fb)})")
    print(f"  RB after folding in FB: opening {_pct(rb['window'] == 'opening')}% / "
          f"closing {_pct(rb['window'] == 'closing')}% (n={len(rb):,})")
    exp_sum, act_sum = all_targets["exp_catch"].sum(), int(all_targets["complete"].sum())
    ok &= check(f"sum of expected catches {exp_sum:.2f} = actual catches {act_sum:,} "
                f"(all {len(all_targets):,} targets)", abs(exp_sum - act_sum) < 0.01)
    print(f"  skill-position targets only: expected {df['exp_catch'].sum():.1f}, "
          f"actual {int(df['complete'].sum()):,}")

    banner("LEAGUE COMPLETION RATE BY CELL (separation band x window) = expected catch rate")
    cells = (all_targets.groupby(["sep_band", "window"])["complete"]
             .agg(n="size", comp_pct=lambda s: round(100 * s.mean(), 1)).reset_index())
    print("completion %")
    show(cells.pivot(index="sep_band", columns="window", values="comp_pct")
         .loc[BANDS, WINDOWS].reset_index())
    print("targets")
    show(cells.pivot(index="sep_band", columns="window", values="n")
         .loc[BANDS, WINDOWS].fillna(0).astype(int).reset_index())
    print("  (0-1 yd / opening is empty: a window that just opened by 0.5+ yd is "
          "never still under 1 yd.)")

    # ---- 1. position ------------------------------------------------------ #
    banner("1. WINDOW TREND BY SKILL POSITION (FB folded into RB)  -> output/position_summary.csv")
    pos_tab = group_table(df, "pos", SKILL).rename(columns={"pos": "position"})
    pos_tab.to_csv(OUT / "position_summary.csv", index=False)
    show(pos_tab[["position", "targets", "pct_opening", "pct_steady", "pct_closing"]])
    print()
    show(pos_tab[["position", "n_opening", "n_closing", "comp_pct_opening",
                  "comp_pct_closing", "comp_gap_pts", "n_opening_1to3yd",
                  "n_closing_1to3yd", "comp_pct_opening_1to3yd",
                  "comp_pct_closing_1to3yd", "comp_gap_1to3yd_pts"]])
    print()
    show(pos_tab[["position", "yac_opening", "yac_closing", "yac_gap",
                  "epa_per_target_opening", "epa_per_target_closing"]])
    print("  comp_pct = completion %; 1to3yd = only throws with 1-3 yd of separation at release;")
    print("  yac = mean yards after catch on completions; gaps are opening minus closing.")

    # ---- 2. alignment ----------------------------------------------------- #
    banner("2. WINDOW TREND BY WHERE THE TARGET LINED UP  -> output/alignment_summary.csv")
    print("mapping pff_positionLinedUp -> alignment:")
    for name in ALIGNMENT_ORDER:
        print(f"  {name:<10} {', '.join(k for k, v in ALIGNMENT.items() if v == name)}")
    unmapped = df[df["alignment"].isna()]["pff_positionLinedUp"].value_counts(dropna=False)
    print(f"unmapped values among skill-position targets: "
          f"{unmapped.to_dict() if len(unmapped) else 'none'}")
    al_tab = group_table(df, "alignment", ALIGNMENT_ORDER)
    al_tab.to_csv(OUT / "alignment_summary.csv", index=False)
    show(al_tab[["alignment", "targets", "pct_opening", "pct_steady", "pct_closing"]])
    print()
    show(al_tab[["alignment", "n_opening", "n_closing", "comp_pct_opening",
                 "comp_pct_closing", "comp_gap_pts", "n_opening_1to3yd",
                 "n_closing_1to3yd", "comp_pct_opening_1to3yd",
                 "comp_pct_closing_1to3yd", "comp_gap_1to3yd_pts"]])
    print()
    show(al_tab[["alignment", "yac_opening", "yac_closing", "yac_gap",
                 "epa_per_target_opening", "epa_per_target_closing"]])
    print()
    print("roster position x alignment (targets):")
    print(pd.crosstab(df["pos"], df["alignment"]).loc[SKILL, ALIGNMENT_ORDER].to_string())

    # ---- 3. receivers ----------------------------------------------------- #
    banner(f"3. RECEIVERS WITH {MIN_TARGETS}+ TARGETS  -> output/receiver_table.csv")
    pos_open = dict(zip(pos_tab["position"], pos_tab["pct_opening"]))
    print("league average share of targets with an opening window, by position "
          "(compare each receiver to his own position):")
    for p in SKILL:
        print(f"  {p}: {pos_open[p]}%")
    rec = receiver_table(df, pos_open)
    rec.to_csv(OUT / "receiver_table.csv", index=False)
    print(f"receivers in table: {len(rec)} "
          f"({rec['position'].value_counts().reindex(SKILL).to_dict()})")

    open_cols = ["receiver", "position", "team", "targets", "pct_opening",
                 "pct_closing", "position_avg_pct_opening", "opening_vs_position_pts",
                 "catch_rate"]
    coe_cols = ["receiver", "position", "team", "targets", "catches",
                "expected_catches", "catches_over_expected", "catch_rate",
                "expected_catch_rate", "catch_rate_over_expected_pts",
                "closing_targets", "closing_catch_rate"]
    by_open = rec.sort_values("pct_opening", ascending=False)
    by_coe = rec.sort_values("catches_over_expected", ascending=False)
    print("\nTOP 8 by share of targets with an opening window")
    show(by_open.head(8)[open_cols])
    print("\nBOTTOM 8 by share of targets with an opening window")
    show(by_open.tail(8).iloc[::-1][open_cols])
    print("\nTOP 8 by catches over expected")
    show(by_coe.head(8)[coe_cols])
    print("\nBOTTOM 8 by catches over expected")
    show(by_coe.tail(8).iloc[::-1][coe_cols])

    # ---- 4. stability ----------------------------------------------------- #
    banner("4. STABILITY: split-half correlation across receivers (odd vs even games)")
    print("Each receiver's games ranked by gameId; odd-ranked games vs even-ranked games.")
    print("r near 0 = the number does not repeat (not a receiver trait in 8 weeks of data);")
    print("r near 1 = it repeats.")
    show(split_half(df))

    banner("FILES WRITTEN")
    for f in ["position_summary.csv", "alignment_summary.csv", "receiver_table.csv"]:
        print(f"  {(OUT / f).relative_to(ROOT)}")
    print()
    print("ALL SANITY CHECKS PASSED" if ok else "SANITY CHECK FAILURE: see [FAIL] lines above")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
