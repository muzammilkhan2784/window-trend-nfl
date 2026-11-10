"""Build the plain-English Window Trend page.

Reads the three tables in output/ (targets_window_trend.csv, position_summary.csv,
receiver_table.csv) plus alignment_summary.csv, checks the league findings and that the
tables written by skill_positions.py agree with the target table, and writes
one self-contained HTML page (no external resources) to docs/index.html and
output/window_trend_dashboard.html.

Run from the project root:  python src/build_dashboard.py
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
TARGETS_CSV = OUT / "targets_window_trend.csv"
POSITION_CSV = OUT / "position_summary.csv"
ALIGNMENT_CSV = OUT / "alignment_summary.csv"
RECEIVER_CSV = OUT / "receiver_table.csv"
DESTINATIONS = [ROOT / "docs" / "index.html", OUT / "window_trend_dashboard.html"]

DEFAULT_PLAYER = "Cooper Kupp"
DEFAULT_TEAM = "KC"
MIN_COVERAGE_TARGETS = 15
COVERAGES = ["Cover-1", "Cover-2", "Cover-3", "Quarters", "Cover-6", "2-Man"]
TIGHT_BANDS = ["1-2", "2-3"]  # "tightly covered": 1 to 3 yards of space at the throw
CLOSE_ENOUGH = 2              # fewer catches than this either way reads "about what you would expect"

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Football Team",
}
# How the team reads inside a sentence: ("the Chiefs", plural verb) or ("Washington", singular).
TEAM_SINGULAR = {"WAS": "Washington"}

POSITION_WORDS = {
    "WR": ("wide receiver", "wide receivers", "Wide receivers"),
    "TE": ("tight end", "tight ends", "Tight ends"),
    "RB": ("running back", "running backs", "Running backs"),
}

# ----------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------
FAILURES: list[str] = []


def check(label: str, got: float, expected: float, tol: float = 0.051) -> None:
    ok = abs(float(got) - float(expected)) <= tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: got {got:.3f}, expected {expected}")
    if not ok:
        FAILURES.append(f"{label}: got {got:.3f}, expected {expected}")


def pct(mask: pd.Series) -> float:
    return float(mask.mean() * 100)


def whole(x: float) -> int:
    """Round half up, so 58.5 reads 59 (Python's own rounding gives 58)."""
    return int(math.floor(float(x) + 0.5))


def position_stats(g: pd.DataFrame) -> dict:
    """Window shares and outcomes for one position, straight from the target table."""
    o, c = g[g["window"] == "opening"], g[g["window"] == "closing"]
    o_tight, c_tight = o[o["sep_band"].isin(TIGHT_BANDS)], c[c["sep_band"].isin(TIGHT_BANDS)]
    return {
        "targets": int(len(g)),
        "open": pct(g["window"] == "opening"), "steady": pct(g["window"] == "steady"),
        "close": pct(g["window"] == "closing"),
        "yac_open": float(o.loc[o["complete"] == 1, "yards_after_catch"].mean()),
        "yac_close": float(c.loc[c["complete"] == 1, "yards_after_catch"].mean()),
        "tight_open": float(o_tight["complete"].mean() * 100), "n_tight_open": int(len(o_tight)),
        "tight_close": float(c_tight["complete"].mean() * 100), "n_tight_close": int(len(c_tight)),
    }


def about(p: float) -> str:
    """A percent as a phrase a fan would say out loud."""
    named = [(50.0, "about half"), (33.3, "about 1 in 3"), (25.0, "about 1 in 4"), (20.0, "about 1 in 5")]
    value, phrase = min(named, key=lambda kv: abs(kv[0] - p))
    if abs(value - p) <= 3.0:
        return phrase
    return f"about {round(p / 10)} in 10"


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def split_half_r(df: pd.DataFrame, key: list[str], min_per_half: int) -> tuple[float, int]:
    """Correlation of opening share between a player's odd and even games."""
    rows = []
    for _, g in df.groupby(key):
        games = {gid: i for i, gid in enumerate(sorted(g["gameId"].unique()))}
        half = g["gameId"].map(games) % 2
        a, b = g[half == 0], g[half == 1]
        if len(a) >= min_per_half and len(b) >= min_per_half:
            rows.append(((a["window"] == "opening").mean(), (b["window"] == "opening").mean()))
    if len(rows) < 3:
        return float("nan"), len(rows)
    arr = np.array(rows)
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]), len(rows)


# ----------------------------------------------------------------------------------
# Load and compute
# ----------------------------------------------------------------------------------
def compute() -> dict:
    t = pd.read_csv(TARGETS_CSV, dtype={"sep_band": str})
    pos = pd.read_csv(POSITION_CSV)
    ali = pd.read_csv(ALIGNMENT_CSV)
    rec = pd.read_csv(RECEIVER_CSV)

    op = t["window"] == "opening"
    cl = t["window"] == "closing"
    st = t["window"] == "steady"

    print("League checks")
    check("targeted throws", len(t), 7263, 0)

    def cell(band: str, mask: pd.Series) -> tuple[float, int]:
        sub = t[(t["sep_band"] == band) & mask]
        return float(sub["complete"].mean() * 100), int(len(sub))

    bands = []
    expected_bands = {
        "1-2": (71.7, 127, 42.4, 401),
        "2-3": (67.7, 319, 51.0, 437),
        "3-5": (74.1, 615, 68.0, 710),
        "5-8": (77.6, None, 77.7, None),
    }
    band_words = {
        "1-2": ("1 to 2 yards of space", "tightly covered"),
        "2-3": ("2 to 3 yards of space", "contested"),
        "3-5": ("3 to 5 yards of space", "open"),
        "5-8": ("5 to 8 yards of space", "wide open"),
    }
    for band, (eo, eno, ec, enc) in expected_bands.items():
        o, no = cell(band, op)
        c, nc = cell(band, cl)
        check(f"{band} yd opening catch %", o, eo)
        check(f"{band} yd closing catch %", c, ec)
        if eno is not None:
            check(f"{band} yd opening n", no, eno, 0)
            check(f"{band} yd closing n", nc, enc, 0)
        bands.append({"band": band, "label": band_words[band][0], "hint": band_words[band][1],
                      "open": o, "n_open": no, "close": c, "n_close": nc})

    c1 = t[t["pff_passCoverage"] == "Cover-1"]
    c1_open = float(c1.loc[c1["window"] == "opening", "complete"].mean() * 100)
    c1_close = float(c1.loc[c1["window"] == "closing", "complete"].mean() * 100)
    c1_n = (int((c1["window"] == "opening").sum()), int((c1["window"] == "closing").sum()))
    check("Cover-1 opening catch %", c1_open, 79.8)
    check("Cover-1 closing catch %", c1_close, 52.7)

    short = t[(t["depth"] < 5) & (t["complete"] == 1) & t["yards_after_catch"].notna()]
    yac_open = short.loc[short["window"] == "opening", "yards_after_catch"]
    yac_close = short.loc[short["window"] == "closing", "yards_after_catch"]
    check("short-throw yards after catch, opening", yac_open.mean(), 6.75, 0.0051)
    check("short-throw yards after catch, closing", yac_close.mean(), 3.96, 0.0051)
    check("short-throw opening n", len(yac_open), 711, 0)
    check("short-throw closing n", len(yac_close), 973, 0)

    epa_open = float(t.loc[op, "epa"].mean())
    epa_close = float(t.loc[cl, "epa"].mean())
    check("expected points per target, opening", epa_open, 0.549, 0.00051)
    check("expected points per target, closing", epa_close, 0.099, 0.00051)
    check("opening n", int(op.sum()), 1920, 0)
    check("closing n", int(cl.sum()), 2414, 0)

    share = {"open": pct(op), "steady": pct(st), "close": pct(cl)}
    check("share opening", share["open"], 26.4)
    check("share steady", share["steady"], 40.3)
    check("share closing", share["close"], 33.2)

    print("Position checks (roster position, before fullbacks are folded into running backs)")
    for p, (eo, ec, en) in {"WR": (26.5, 29.8, 4416), "TE": (29.5, 31.1, 1563), "RB": (21.8, 48.4, 1222)}.items():
        sub = t[t["officialPosition"] == p]
        check(f"{p} targets", len(sub), en, 0)
        check(f"{p} share opening", pct(sub["window"] == "opening"), eo)
        check(f"{p} share closing", pct(sub["window"] == "closing"), ec)

    print("position_summary.csv agrees with the target table (running backs include fullbacks)")
    grp = t["officialPosition"].replace({"FB": "RB"})
    for _, r in pos.iterrows():
        sub = t[grp == r["position"]]
        check(f"{r['position']} table targets", r["targets"], len(sub), 0)
        check(f"{r['position']} table share opening", r["pct_opening"], pct(sub["window"] == "opening"))
        check(f"{r['position']} table share closing", r["pct_closing"], pct(sub["window"] == "closing"))

    print("Stability (odd games against even games)")
    rec_r, rec_n = split_half_r(t[grp.isin(["WR", "TE", "RB"])], ["target_nflId"], 12)
    qb_r, qb_n = split_half_r(t, ["qb", "possessionTeam"], 40)
    print(f"  receivers, share of opening windows: r = {rec_r:.3f} ({rec_n} receivers)")
    print(f"  quarterbacks, share of opening windows: r = {qb_r:.3f} ({qb_n} quarterbacks)")
    if not abs(rec_r) < 0.15:
        FAILURES.append(f"receiver opening share is no longer unstable (r = {rec_r:.3f}); the page says it is")

    # ---- positions (roster position; throws to fullbacks are left out) ---------------
    print("Position cards against position_summary.csv (wide receivers and tight ends)")
    stats = {p: position_stats(t[t["officialPosition"] == p]) for p in POSITION_WORDS}
    table = pos.set_index("position")
    for p in ("WR", "TE"):
        check(f"{p} yards after catch, opening", stats[p]["yac_open"], table.loc[p, "yac_opening"], 0.0051)
        check(f"{p} yards after catch, closing", stats[p]["yac_close"], table.loc[p, "yac_closing"], 0.0051)
        check(f"{p} tight catch %, opening", stats[p]["tight_open"], table.loc[p, "comp_pct_opening_1to3yd"])
        check(f"{p} tight catch %, closing", stats[p]["tight_close"], table.loc[p, "comp_pct_closing_1to3yd"])
    max_close = max(stats, key=lambda p: stats[p]["close"])
    max_open = max(stats, key=lambda p: stats[p]["open"])
    positions = []
    for p, r in stats.items():
        one, many, many_cap = POSITION_WORDS[p]
        others = [q for q in stats if q != p]
        if p == max_close:
            sentence = (f"{cap(about(r['close']))} of all throws to {many} arrive with a defender closing in. "
                        f"When the window is opening instead, their catches gain {r['yac_open']:.1f} yards after the catch, "
                        f"against {r['yac_close']:.1f}.")
        elif p == max_open:
            other_bits = " and ".join(f"{stats[q]['open']:.1f}% for {POSITION_WORDS[q][1]}" for q in others)
            sentence = (f"{many_cap} get an opening window most often: {r['open']:.1f}% of throws, against {other_bits}.")
        else:
            sentence = (f"On tightly covered throws (1 to 3 yards of space), {many} catch {whole(r['tight_open'])}% "
                        f"when the window is opening and {whole(r['tight_close'])}% when a defender is closing.")
        positions.append({"pos": p, "name": one, "plural": many, "title": many_cap, "sentence": sentence, **r})

    a = ali.set_index("alignment")
    wide, inline, back = a.loc["wide"], a.loc["inline TE"], a.loc["backfield"]
    alignment = [
        f"Split out wide and tightly covered (1 to 3 yards of space): {whole(wide['comp_pct_opening_1to3yd'])}% caught when "
        f"opening, {whole(wide['comp_pct_closing_1to3yd'])}% when closing. "
        f"({int(wide['n_opening_1to3yd'])} and {int(wide['n_closing_1to3yd'])} throws)"
    ]
    if abs(inline["comp_gap_1to3yd_pts"]) < 5:
        alignment.append(
            f"Tight ends lined up next to the offensive line and tightly covered: {whole(inline['comp_pct_opening_1to3yd'])}% "
            f"against {whole(inline['comp_pct_closing_1to3yd'])}%. No real difference, so the idea does not help there. "
            f"({int(inline['n_opening_1to3yd'])} and {int(inline['n_closing_1to3yd'])} throws, a small sample)"
        )
    alignment.append(f"Out of the backfield: a defender is closing on {whole(back['pct_closing'])}% of throws.")

    # ---- players -------------------------------------------------------------------
    # Expected catches: the league catch rate on throws with the same space and the same
    # window, added up over a player's throws (the same rule as skill_positions.py).
    t["exp_catch"] = t.groupby(["sep_band", "window"])["complete"].transform("mean")
    print("Player checks against receiver_table.csv")
    players, worst = [], 0.0
    for _, r in rec.sort_values("receiver").iterrows():
        g = t[t["target_nflId"] == r["nflId"]]
        exp = float(g["exp_catch"].sum())
        catches = int(g["complete"].sum())
        if len(g) != r["targets"] or catches != r["catches"]:
            FAILURES.append(f"{r['receiver']}: targets or catches differ from receiver_table.csv")
        worst = max(worst, abs(exp - r["expected_catches"]))
        players.append({
            "name": r["receiver"], "pos": r["position"],
            "teams": g["possessionTeam"].value_counts().index.tolist(),
            "n": int(len(g)), "n_open": int((g["window"] == "opening").sum()),
            "n_close": int((g["window"] == "closing").sum()),
            "catches": catches, "exp": whole(exp), "diff": catches - whole(exp),
            "_over": catches - exp,
        })
    check("largest gap in expected catches against receiver_table.csv", worst, 0.0, 0.051)

    def pick(label: str, value) -> dict:
        return {"label": label, "i": max(range(len(players)), key=lambda i: value(players[i]))}

    picks = [pick("Most catches above a typical receiver", lambda p: p["_over"]),
             pick("Most catches below a typical receiver", lambda p: -p["_over"]),
             pick("Highest share of opening windows", lambda p: p["n_open"] / p["n"]),
             pick("Highest share of closing windows", lambda p: p["n_close"] / p["n"])]
    for p in players:
        del p["_over"]
    diffs = [p["diff"] for p in players]
    if DEFAULT_PLAYER not in {p["name"] for p in players}:
        FAILURES.append(f"default player {DEFAULT_PLAYER} is not in receiver_table.csv")

    # ---- teams ---------------------------------------------------------------------
    cov_type = (t[t["pff_passCoverage"].isin(COVERAGES)]
                .groupby("pff_passCoverage")["pff_passCoverageType"].agg(lambda s: s.mode().iloc[0]).str.lower().to_dict())

    def side(col: str, rank_window: str) -> dict:
        shares = t.groupby(col)["window"].apply(lambda s: float((s == rank_window).mean() * 100))
        ranks = shares.rank(ascending=False, method="min").astype(int)
        out = {}
        for team, g in t.groupby(col):
            rows = []
            for cov in COVERAGES:
                c = g[g["pff_passCoverage"] == cov]
                if len(c) >= MIN_COVERAGE_TARGETS:
                    rows.append({"cov": cov, "type": cov_type[cov], "n": int(len(c)),
                                 "open": pct(c["window"] == "opening"), "close": pct(c["window"] == "closing")})
            out[team] = {"n": int(len(g)), "open": pct(g["window"] == "opening"),
                         "close": pct(g["window"] == "closing"), "rank": int(ranks[team]), "cov": rows}
        return out

    offense = side("possessionTeam", "opening")
    defense = side("defensiveTeam", "closing")
    teams = sorted(offense)
    if len(teams) != 32 or set(teams) != set(TEAM_NAMES):
        FAILURES.append("team list does not match the 32 team names")
    team_n = [offense[x]["n"] for x in teams] + [defense[x]["n"] for x in teams]

    return {
        "n": int(len(t)), "share": share, "bands": bands,
        "head": next(b for b in bands if b["band"] == "2-3"),
        "beyond": {"open": cell("8+", op)[0], "close": cell("8+", cl)[0]},
        "not_skill": int((~t["officialPosition"].isin(["WR", "TE", "RB", "FB"])).sum()),
        "fullback": int((t["officialPosition"] == "FB").sum()),
        "diff_min": min(diffs), "diff_max": max(diffs),
        "c1": {"open": c1_open, "close": c1_close, "n_open": c1_n[0], "n_close": c1_n[1]},
        "yac": {"open": float(yac_open.mean()), "close": float(yac_close.mean()),
                "n_open": int(len(yac_open)), "n_close": int(len(yac_close))},
        "epa": {"open": epa_open, "close": epa_close, "n_open": int(op.sum()), "n_close": int(cl.sum())},
        "positions": positions, "alignment": alignment, "players": players, "picks": picks,
        "teams": [{"abbr": x, "name": TEAM_NAMES[x],
                   "in_sentence": TEAM_SINGULAR.get(x, "the " + TEAM_NAMES[x].split()[-1]),
                   "singular": x in TEAM_SINGULAR,
                   "off": offense[x], "def": defense[x]} for x in teams],
        "team_n_min": min(team_n), "team_n_max": max(team_n),
        "min_player_targets": int(rec["targets"].min()),
        "qb_r": qb_r, "qb_n": qb_n, "rec_r": rec_r, "rec_n": rec_n,
        "games": int(t["gameId"].nunique()),
    }


# ----------------------------------------------------------------------------------
# HTML pieces
# ----------------------------------------------------------------------------------
def e(s: object) -> str:
    return html.escape(str(s), quote=True)


def mix_strip(name: str, o: float, s: float, c: float) -> str:
    return (
        '<div class="mix">'
        f'<div class="mix-name">{e(name)}</div>'
        f'<div class="mix-bar" role="img" aria-label="{e(name)}: {o:.1f}% opening, {s:.1f}% steady, {c:.1f}% closing">'
        f'<span class="seg open" style="flex:{o:.1f} 1 0"></span>'
        f'<span class="seg steady" style="flex:{s:.1f} 1 0"></span>'
        f'<span class="seg close" style="flex:{c:.1f} 1 0"></span></div>'
        '<div class="mix-key">'
        f'<span><i class="sw open"></i>Opening {o:.1f}%</span>'
        f'<span><i class="sw steady"></i>Steady {s:.1f}%</span>'
        f'<span><i class="sw close"></i>Closing {c:.1f}%</span></div></div>'
    )


def picture(kind: str) -> str:
    """Receiver (dot) and closest defender (X) half a second before the throw and at the throw."""
    scale = 60  # pixels per yard
    before = 1.5 if kind == "open" else 3.5
    at = 2.5
    colour = "var(--open)" if kind == "open" else "var(--close)"
    rx_at, rx_before = 320, 320 - 70  # the receiver runs to the right
    rows = [("Half a second before the throw", 66, rx_before, before),
            ("When the ball is thrown", 170, rx_at, at)]
    parts = [f'<svg viewBox="0 0 400 216" role="img" aria-label="Drawing: the gap between receiver and defender goes '
             f'from {before} yards to {at} yards">']
    for i, (label, y, rx, gap) in enumerate(rows):
        dx = rx - gap * scale
        parts.append(f'<text x="14" y="{y - 40}" class="pic-row">{label}</text>')
        parts.append(f'<line x1="{dx + 15}" y1="{y}" x2="{rx - 16}" y2="{y}" stroke="{colour}" stroke-width="5" stroke-linecap="round"/>')
        parts.append(f'<text x="{(dx + rx) / 2}" y="{y - 18}" text-anchor="middle" class="pic-gap">{gap} yards</text>')
        parts.append(f'<path d="M{dx - 9} {y - 9}L{dx + 9} {y + 9}M{dx - 9} {y + 9}L{dx + 9} {y - 9}" stroke="#4a5561" stroke-width="4.5" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{rx}" cy="{y}" r="10.5" fill="#17202a"/>')
        if i == 0:
            parts.append(f'<text x="{max(dx, 34)}" y="{y + 30}" text-anchor="middle" class="pic-who">defender</text>')
            parts.append(f'<text x="{rx}" y="{y + 30}" text-anchor="middle" class="pic-who">receiver</text>')
            parts.append('<line x1="14" y1="112" x2="386" y2="112" stroke="#e3e7ec" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def band_rows(bands: list[dict]) -> str:
    out = []
    for b in bands:
        gap = whole(b["open"]) - whole(b["close"])  # the difference between the two labels as shown
        gap_text = "no real difference" if abs(gap) < 2 else f"about {gap} more catches per 100 throws"
        out.append(
            '<div class="band">'
            f'<div class="band-name">{e(b["label"])}<span>{e(b["hint"])}</span></div>'
            '<div class="bars">'
            f'<div class="bar-row"><span class="bar-lab">Opening</span><span class="track"><span class="fill open" style="width:{b["open"]:.1f}%"></span></span>'
            f'<span class="bar-val"><b>{whole(b["open"])}%</b> caught <small>of {b["n_open"]:,} throws</small></span></div>'
            f'<div class="bar-row"><span class="bar-lab">Closing</span><span class="track"><span class="fill close" style="width:{b["close"]:.1f}%"></span></span>'
            f'<span class="bar-val"><b>{whole(b["close"])}%</b> caught <small>of {b["n_close"]:,} throws</small></span></div>'
            '</div>'
            f'<div class="band-gap">{e(gap_text)}</div>'
            '</div>'
        )
    return "".join(out)


def position_cards(positions: list[dict]) -> str:
    out = []
    for p in positions:
        out.append(
            '<article class="card">'
            f'<h3>{e(p["title"])}</h3>'
            f'<p class="card-n">{p["targets"]:,} throws</p>'
            f'<p class="card-say">{e(p["sentence"])}</p>'
            '<ul class="nums">'
            f'<li><b>{p["open"]:.1f}%</b><span><i class="sw open"></i>of throws: window opening</span></li>'
            f'<li><b>{p["close"]:.1f}%</b><span><i class="sw close"></i>of throws: window closing</span></li>'
            f'<li><b>{p["yac_open"]:.1f} vs {p["yac_close"]:.1f}</b><span>yards after the catch, opening vs closing</span></li>'
            '</ul></article>'
        )
    return "".join(out)


def build_html(d: dict) -> str:
    h, c1, yac, epa, sh = d["head"], d["c1"], d["yac"], d["epa"], d["share"]
    tight, wide_open = d["bands"][0], d["bands"][-1]
    rb = next(p for p in d["positions"] if p["pos"] == "RB")

    reading = (
        f"How to read it: when the receiver is {tight['hint']} ({tight['label']}), a pass into an opening window is caught "
        f"{whole(tight['open'])} times in 100, against {whole(tight['close'])} in 100 when the window is closing. "
        f"When he is already {wide_open['hint']} ({wide_open['label'].replace(' of space', '')}), it makes no difference."
    )
    scout_qb = ""
    if d["qb_r"] >= 0.3:
        scout_qb = (" For quarterbacks it is different: how often a quarterback throws into an opening window carries over "
                    "from one half of his games to the other, so it says more about the passer and the offense than about the receiver.")

    tokens = {
        "N": f"{d['n']:,}",
        "GAMES": str(d["games"]),
        "HEAD_OPEN": str(whole(h["open"])), "HEAD_CLOSE": str(whole(h["close"])),
        "NOT_SKILL": str(d["not_skill"]),
        "BEYOND_OPEN": str(whole(d["beyond"]["open"])), "BEYOND_CLOSE": str(whole(d["beyond"]["close"])),
        "HEAD_N_OPEN": f"{h['n_open']:,}", "HEAD_N_CLOSE": f"{h['n_close']:,}",
        "PIC_OPEN": picture("open"), "PIC_CLOSE": picture("close"),
        "LEAGUE_MIX": mix_strip(f"All {d['n']:,} throws", sh["open"], sh["steady"], sh["close"]),
        "BANDS": band_rows(d["bands"]),
        "READING": e(reading),
        "YAC_OPEN": f"{yac['open']:.1f}", "YAC_CLOSE": f"{yac['close']:.1f}",
        "YAC_N": f"{yac['n_open']:,} and {yac['n_close']:,}",
        "C1_OPEN": str(whole(c1["open"])), "C1_CLOSE": str(whole(c1["close"])),
        "C1_N": f"{c1['n_open']:,} and {c1['n_close']:,}",
        "EPA_OPEN": f"{epa['open']:.2f}", "EPA_CLOSE": f"{epa['close']:.2f}",
        "CARDS": position_cards(d["positions"]),
        "ALIGNMENT": "".join(f"<li>{e(line)}</li>" for line in d["alignment"]),
        "FULLBACK": str(d["fullback"]),
        "MIN_PLAYER": str(d["min_player_targets"]),
        "N_PLAYERS": str(len(d["players"])),
        "RB_CLOSE": str(whole(rb["close"])),
        "RB_YAC_GAP": str(whole(rb["yac_open"] - rb["yac_close"])),
        "SCOUT_QB": e(scout_qb),
        "TEAM_MIN": str(d["team_n_min"]), "TEAM_MAX": str(d["team_n_max"]),
        "MIN_COV": str(MIN_COVERAGE_TARGETS),
        "DATA": json.dumps({
            "players": d["players"], "teams": d["teams"],
            "pos": {p["pos"]: {"name": p["name"], "plural": p["plural"], "title": p["title"],
                               "open": p["open"], "steady": p["steady"], "close": p["close"]} for p in d["positions"]},
            "league": {"open": sh["open"], "close": sh["close"]},
            "picks": d["picks"], "closeEnough": CLOSE_ENOUGH,
            "diffMin": d["diff_min"], "diffMax": d["diff_max"],
            "defaultPlayer": DEFAULT_PLAYER, "defaultTeam": DEFAULT_TEAM,
        }, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"),
    }
    page = TEMPLATE
    for key, value in tokens.items():
        page = page.replace("@@" + key + "@@", value)
    if "@@" in page:
        raise SystemExit("unreplaced token in template: " + page[page.index("@@"):page.index("@@") + 40])
    return page


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Window Trend: is the receiver getting more open or less open at the throw?</title>
<style>
:root{--open:#1f77b4;--close:#d95f02;--steady:#b9c0c8;--ink:#17202a;--muted:#55606c;--line:#e3e7ec;--bg:#f8f9fb;--card:#ffffff}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:0 24px 72px}
section{padding:60px 0 12px;border-top:1px solid var(--line)}
section.first{border-top:0;padding-top:56px}
h1{font-size:60px;line-height:1.05;margin:0 0 28px;letter-spacing:-.02em}
h2{font-size:31px;line-height:1.2;margin:0 0 18px;letter-spacing:-.01em}
h3{font-size:22px;line-height:1.25;margin:0}
p{margin:0 0 16px}
.kicker{font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px;font-weight:600}
.lead{font-size:23px;line-height:1.45;max-width:800px}
.muted{color:var(--muted)}
.small{font-size:16px;color:var(--muted)}
.sw{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.sw.open,.seg.open,.fill.open{background:var(--open)}
.sw.close,.seg.close,.fill.close{background:var(--close)}
.sw.steady,.seg.steady{background:var(--steady)}

/* 1. headline */
.duel{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:30px 0 14px}
.big{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px 26px 22px;border-top:8px solid var(--steady)}
.big.open{border-top-color:var(--open)}
.big.close{border-top-color:var(--close)}
.big .num{font-size:92px;line-height:1;font-weight:750;letter-spacing:-.03em}
.big.open .num{color:var(--open)}
.big.close .num{color:var(--close)}
.big .what{font-size:20px;line-height:1.35;margin:10px 0 0}
.big .what b{display:block}
.same{font-size:20px;background:#eef1f4;border-radius:10px;padding:14px 18px;margin:0 0 8px}
.jump{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin:22px 0 0;font-size:17px}
.jump a{color:var(--ink);font-weight:600;text-decoration:none;border:2px solid #c3cad2;border-radius:999px;padding:7px 16px;background:#fff}
.jump a:hover{border-color:var(--ink)}
.jump a:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
h2{scroll-margin-top:20px}

/* 2. pictures */
.pics{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:22px 0 18px}
.pic{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px 20px;margin:0}
.pic svg{display:block;width:100%;height:auto}
.pic figcaption{margin-top:6px}
.pic figcaption b{display:block;font-size:20px}
.pic-row{font-size:16px;fill:#55606c;font-weight:600}
.pic-gap{font-size:18px;fill:#17202a;font-weight:700}
.pic-who{font-size:16px;fill:#55606c}
.rule{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:0 0 8px}
.rule p:last-child{margin-bottom:0}

/* strips */
.mix{margin:14px 0 4px}
.mix-name{font-weight:650;margin-bottom:6px}
.mix-bar{display:flex;height:26px;border-radius:7px;overflow:hidden;gap:2px;background:#fff}
.seg{display:block;min-width:2px}
.mix-key{display:flex;flex-wrap:wrap;gap:4px 20px;font-size:16px;color:var(--muted);margin-top:7px}
.mix-key span{white-space:nowrap}

/* 3. bars */
.bands{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:6px 22px;margin:22px 0 14px}
.band{display:grid;grid-template-columns:200px 1fr 190px;gap:18px;align-items:center;padding:18px 0;border-top:1px solid var(--line)}
.band:first-child{border-top:0}
.band-name{font-weight:650;line-height:1.3}
.band-name span{display:block;font-weight:400;color:var(--muted);font-size:16px}
.bar-row{display:grid;grid-template-columns:72px 1fr 205px;gap:10px;align-items:center;margin:5px 0}
.bar-lab{font-size:16px;color:var(--muted)}
.track{display:block;height:22px;background:#eef1f4;border-radius:5px;overflow:hidden}
.fill{display:block;height:100%;border-radius:5px}
.bar-val{font-size:16px;white-space:nowrap}
.bar-val b{font-size:19px}
.bar-val small{color:var(--muted);font-size:14px}
.band-gap{font-size:16px;font-weight:650;text-align:right;line-height:1.3}
.facts{list-style:none;padding:0;margin:20px 0 8px;display:grid;gap:12px}
.facts li{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px}
.facts b{font-weight:700}

/* 4. cards */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:22px 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 22px 18px}
.card-n{font-size:15px;color:var(--muted);margin:2px 0 12px}
.card-say{font-size:18px;line-height:1.5}
.nums{list-style:none;margin:16px 0 0;padding:0}
.nums li{display:grid;grid-template-columns:104px 1fr;gap:10px;align-items:baseline;padding:9px 0;border-top:1px solid var(--line);font-size:15px;color:var(--muted);line-height:1.35}
.nums b{font-size:21px;color:var(--ink);white-space:nowrap}
.lineup{padding-left:24px;margin:8px 0 8px}
.lineup li{margin-bottom:8px}

/* 5, 6. lookups */
.controls{display:flex;flex-wrap:wrap;gap:18px 26px;align-items:flex-end;margin:20px 0 18px}
.field label,.switch legend{display:block;font-size:15px;font-weight:650;color:var(--muted);margin-bottom:6px;padding:0}
select{font:inherit;color:var(--ink);background:#fff;border:2px solid #8b96a3;border-radius:10px;padding:10px 14px;min-height:50px;min-width:300px;max-width:100%}
select:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
.switch{border:0;margin:0;padding:0;min-width:0}
.switch .opts{display:inline-flex;border:2px solid #8b96a3;border-radius:10px;overflow:hidden;background:#fff}
.switch label{margin:0;cursor:pointer}
.switch input{position:absolute;opacity:0;width:1px;height:1px}
.switch span{display:block;padding:10px 22px;min-height:46px;font-weight:600;color:var(--ink)}
.switch input:checked+span{background:var(--ink);color:#fff}
.switch input:focus-visible+span{outline:3px solid var(--ink);outline-offset:-5px;box-shadow:inset 0 0 0 3px #fff}
.picks{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 18px}
.picks-lab{font-size:16px;color:var(--muted)}
.picks button{font:inherit;font-size:16px;color:var(--ink);background:#fff;border:2px solid #c3cad2;border-radius:999px;padding:7px 16px;cursor:pointer;text-align:left}
.picks button:hover{border-color:var(--ink)}
.picks button:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
.report{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px 26px;margin:0 0 16px}
.report p{font-size:20px;line-height:1.55;margin:0 0 12px}
.report .verdict{font-size:22px;font-weight:700;background:#eef1f4;border-radius:10px;padding:12px 16px;margin:16px 0 18px}
.report .verdict span{display:block;font-size:17px;font-weight:400;color:var(--muted);margin-top:2px}
.report p.note{font-size:16px;color:var(--muted);margin:12px 0 0}
.report h3{font-size:17px;margin:22px 0 2px;color:var(--muted);font-weight:650}
.honest{border-left:5px solid var(--ink);background:#eef1f4;border-radius:0 10px 10px 0;padding:14px 18px;margin:0 0 8px}
table{width:100%;border-collapse:collapse;font-size:17px;margin-top:8px}
caption{text-align:left;font-size:16px;color:var(--muted);padding:0 0 8px;caption-side:top}
th,td{padding:10px 8px;border-top:1px solid var(--line);text-align:right;vertical-align:baseline}
th:first-child,td:first-child{text-align:left}
thead th{border-top:0;font-size:15px;color:var(--muted);font-weight:650;line-height:1.3;vertical-align:bottom}
td small{color:var(--muted);font-size:14px}
tr.total td{font-weight:700;border-top:2px solid #c9d0d8}

/* 7, 8 */
.who{list-style:none;padding:0;margin:18px 0 8px;display:grid;gap:12px}
.who li{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px}
.who b{display:block;font-size:20px;margin-bottom:2px}
.limits{padding-left:24px;margin:16px 0 8px}
.limits li{margin-bottom:12px}
footer{border-top:1px solid var(--line);margin-top:56px;padding-top:22px;font-size:15px;color:var(--muted)}

@media (max-width:860px){
  .band{grid-template-columns:1fr;gap:6px}
  .band-gap{text-align:left}
  .cards{grid-template-columns:1fr}
}
@media (max-width:640px){
  body{font-size:17px}
  main{padding:0 16px 56px}
  section{padding:44px 0 8px}
  section.first{padding-top:36px}
  h1{font-size:42px;margin-bottom:20px}
  h2{font-size:25px}
  .lead{font-size:20px}
  .duel{gap:10px}
  .big{padding:16px 14px}
  .big .num{font-size:58px}
  .big .what{font-size:16px}
  .same{font-size:17px}
  .pics{grid-template-columns:1fr}
  .bands{padding:4px 14px}
  .bar-row{grid-template-columns:62px 1fr;row-gap:2px}
  .bar-val{grid-column:2}
  .report{padding:18px 16px}
  .report p{font-size:18px}
  .report .verdict{font-size:19px}
  select{min-width:0;width:100%}
  .field{width:100%}
  table{font-size:15px}
  th,td{padding:9px 4px}
  thead th .sw{width:10px;height:10px;margin-right:4px}
}
</style>
</head>
<body>
<main>

<section class="first" aria-labelledby="q1">
  <p class="kicker">NFL tracking data, 2021 season, Weeks 1 to 8</p>
  <h1>Window Trend</h1>
  <h2 id="q1">What is this?</h2>
  <p class="lead">A new stat for wide receivers, tight ends and running backs: when the quarterback lets go of the ball, is the player getting <b>more open</b> or <b>less open</b>? The answer goes with a big difference in how often the pass is caught.</p>
  <div class="duel">
    <div class="big open">
      <div class="num">@@HEAD_OPEN@@%</div>
      <p class="what"><b>of passes caught when the receiver is pulling away from his defender.</b>We call this an opening window.</p>
    </div>
    <div class="big close">
      <div class="num">@@HEAD_CLOSE@@%</div>
      <p class="what"><b>of passes caught when the defender is catching up.</b>We call this a closing window.</p>
    </div>
  </div>
  <p class="same"><b>Same 2 to 3 yards of space in both cases.</b> What differs is which way the gap is moving when the ball comes out.</p>
  <p class="small">Percent of throws caught, out of @@HEAD_N_OPEN@@ opening and @@HEAD_N_CLOSE@@ closing throws where the receiver had 2 to 3 yards of space at the throw. All numbers on this page come from @@N@@ throws in @@GAMES@@ games; all but @@NOT_SKILL@@ went to a wide receiver, tight end or back.</p>
  <nav class="jump" aria-label="Jump to a section"><span class="muted">Jump to:</span><a href="#q5">Look up a player</a><a href="#q6">Look up a team</a><a href="#q7">Who would use this</a></nav>
</section>

<section aria-labelledby="q2">
  <h2 id="q2">What do opening and closing mean?</h2>
  <p>Today a receiver's openness is reported as one number, "separation": the yards between the receiver and the closest defender at the moment of the throw. But that gap is always moving. Window Trend looks at the half second before the throw and asks which way it is going.</p>
  <div class="pics">
    <figure class="pic">
      @@PIC_OPEN@@
      <figcaption><b><i class="sw open"></i>Opening: the defender is falling behind.</b>The gap is growing, so he is likely to have more space when the ball gets there.</figcaption>
    </figure>
    <figure class="pic">
      @@PIC_CLOSE@@
      <figcaption><b><i class="sw close"></i>Closing: the defender is catching up.</b>The gap is shrinking, so he is likely to have less space when the ball gets there.</figcaption>
    </figure>
  </div>
  <p>Both drawings end with the same 2.5 yards of space, so a separation number calls them the same throw. Window Trend tells them apart.</p>
  <div class="rule">
    <p><b>The exact rule.</b> We measure how fast the gap is changing over the half second before the throw. Growing by more than 1 yard per second is <b>opening</b>. Shrinking by more than 1 yard per second is <b>closing</b>. Anything in between is <b>steady</b>.</p>
    @@LEAGUE_MIX@@
  </div>
</section>

<section aria-labelledby="q3">
  <h2 id="q3">Does it actually matter?</h2>
  <p>Yes, and most when it is hard to complete a pass. Each pair of bars compares throws with the <em>same amount of space</em> at the throw. The bar length is the percent of those throws that were caught.</p>
  <div class="bands">@@BANDS@@</div>
  <p><b>@@READING@@</b></p>
  <p class="small">Not shown: with more than 8 yards of space it is @@BEYOND_OPEN@@% against @@BEYOND_CLOSE@@%.</p>
  <ul class="facts">
    <li><b>More yards after the catch.</b> On short throws (less than 5 yards downfield), the receiver gains <b>@@YAC_OPEN@@ yards</b> after the catch when the window is opening and <b>@@YAC_CLOSE@@ yards</b> when it is closing. <span class="small">(@@YAC_N@@ caught passes)</span></li>
    <li><b>Biggest against man coverage.</b> Against Cover-1, where each defender follows one receiver, <b>@@C1_OPEN@@%</b> of throws are caught when the window is opening and <b>@@C1_CLOSE@@%</b> when it is closing. <span class="small">(@@C1_N@@ throws)</span></li>
    <li><b>More expected points.</b> A throw into an opening window is worth about <b>@@EPA_OPEN@@ points</b> to the offense on average; into a closing window, about <b>@@EPA_CLOSE@@</b>. <span class="small">(Expected points: how much a play raises the score a team can expect from that drive.)</span></li>
  </ul>
</section>

<section aria-labelledby="q4">
  <h2 id="q4">What does it mean for receivers, tight ends and running backs?</h2>
  <p>The three positions get very different windows.</p>
  <div class="cards">@@CARDS@@</div>
  <p><b>Where the player lines up matters too.</b></p>
  <ul class="lineup">@@ALIGNMENT@@</ul>
  <p class="small">Positions are roster positions. The @@FULLBACK@@ throws to fullbacks are not in the running back card.</p>
</section>

<section aria-labelledby="q5">
  <h2 id="q5">Look up a player</h2>
  <p>Pick any of the @@N_PLAYERS@@ players who were thrown to at least @@MIN_PLAYER@@ times. The report compares his catches with what the league catches on throws with the same space and the same window.</p>
  <div class="controls">
    <div class="field">
      <label for="player">Player</label>
      <select id="player"></select>
    </div>
  </div>
  <div class="picks" id="picks"><span class="picks-lab">Or try one of these:</span></div>
  <div class="report" id="player-report" aria-live="polite"></div>
  <p class="honest"><b>Read this fairly.</b> These numbers describe the throws a player was <b>given</b>: quarterback timing, play design and the coverage decide whether the window is opening. Use them to judge his catch rate fairly, not as a measure of how well he gets open.</p>
  <p class="small">Counts cover the throws in the tracking data for Weeks 1 to 8, so they can run a little below box-score totals.</p>
</section>

<section aria-labelledby="q6">
  <h2 id="q6">Look up a team</h2>
  <p>Offense: is the ball coming out while the receiver is still getting open? Defense: are the defenders closing on the receiver when the ball comes out?</p>
  <div class="controls">
    <div class="field">
      <label for="team">Team</label>
      <select id="team"></select>
    </div>
    <fieldset class="switch">
      <legend>Side of the ball</legend>
      <div class="opts">
        <label><input type="radio" name="side" value="off" checked><span>Offense</span></label>
        <label><input type="radio" name="side" value="def"><span>Defense</span></label>
      </div>
    </fieldset>
  </div>
  <div class="report" id="team-report" aria-live="polite"></div>
  <p class="small">Each team has only @@TEAM_MIN@@ to @@TEAM_MAX@@ throws in these eight weeks, so treat the ranks as a rough guide. A coverage is listed only when it has @@MIN_COV@@ or more throws.</p>
</section>

<section aria-labelledby="q7">
  <h2 id="q7">Who would use this?</h2>
  <ul class="who">
    <li><b>A coach</b>To time throws so the ball leaves while the gap is still growing. Running backs are the clearest case: @@RB_CLOSE@@% of their throws arrive with a defender closing, and their catches on opening windows gain about @@RB_YAC_GAP@@ more yards after the catch. On defense, the team lookup hints at which coverages were closing on the receiver (small samples).</li>
    <li><b>A scout</b>To put a catch rate in context: did he catch more or fewer balls than the league does on throws with the same space and window?@@SCOUT_QB@@</li>
    <li><b>A broadcaster</b>To explain a replay with one color: blue if the receiver was pulling away at the throw, orange if the defender was closing. It helps explain why two throws with the same space ended differently.</li>
  </ul>
</section>

<section aria-labelledby="q8">
  <h2 id="q8">What it cannot tell you</h2>
  <ul class="limits">
    <li><b>We never see the catch.</b> The tracking data stops half a second after the throw. We know which way the gap was moving when the ball left, not how much space there was when it arrived.</li>
    <li><b>It is not a grade for getting open.</b> A receiver who got many opening windows in half of his games did not get more of them in the other half. The windows come from quarterback timing, play design and coverage.</li>
    <li><b>The samples are small.</b> This is eight weeks: @@N@@ throws in total, @@TEAM_MIN@@ to @@TEAM_MAX@@ per team, and as few as @@MIN_PLAYER@@ for a player. Team ranks and player numbers are a rough guide, not a verdict.</li>
    <li><b>It shows a pattern, not a cause.</b> Short passes, play design and the type of coverage all move the window too. Opening and closing are also a simple cut (1 yard per second), not a law of nature.</li>
  </ul>
  <footer>Built in Python from NFL player tracking data (2021 season, Weeks 1 to 8), coverage labels from PFF and public nflverse play-by-play. Every number on this page is computed by the scripts in <code>src/</code>.</footer>
</section>

</main>
<script type="application/json" id="wt-data">@@DATA@@</script>
<script>
(function () {
  var D = JSON.parse(document.getElementById('wt-data').textContent);
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }
  function pc(x) { return Math.round(Number(x)) + '%'; }
  function ord(n) {
    var s = ['th', 'st', 'nd', 'rd'], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }
  function strip(name, o, s, c) {
    return '<div class="mix"><div class="mix-name">' + esc(name) + '</div>' +
      '<div class="mix-bar" role="img" aria-label="' + esc(name) + ': ' + pc(o) + ' opening, ' + pc(s) + ' steady, ' + pc(c) + ' closing">' +
      '<span class="seg open" style="flex:' + o + ' 1 0"></span>' +
      '<span class="seg steady" style="flex:' + s + ' 1 0"></span>' +
      '<span class="seg close" style="flex:' + c + ' 1 0"></span></div>' +
      '<div class="mix-key"><span><i class="sw open"></i>Opening ' + pc(o) + '</span>' +
      '<span><i class="sw steady"></i>Steady ' + pc(s) + '</span>' +
      '<span><i class="sw close"></i>Closing ' + pc(c) + '</span></div></div>';
  }
  var teamByAbbr = {};
  D.teams.forEach(function (t) { teamByAbbr[t.abbr] = t; });

  /* ---------- player ---------- */
  var playerSel = document.getElementById('player');
  var playerOut = document.getElementById('player-report');
  ['WR', 'TE', 'RB'].forEach(function (pos) {
    var group = document.createElement('optgroup');
    group.label = D.pos[pos].title;
    D.players.forEach(function (p, i) {
      if (p.pos !== pos) { return; }
      var o = document.createElement('option');
      o.value = String(i);
      o.textContent = p.name + ' (' + p.teams.join(' / ') + ')';
      if (p.name === D.defaultPlayer) { o.selected = true; }
      group.appendChild(o);
    });
    playerSel.appendChild(group);
  });
  function renderPlayer() {
    var p = D.players[Number(playerSel.value)];
    var pos = D.pos[p.pos];
    var team = p.teams.map(function (a) { return teamByAbbr[a] ? teamByAbbr[a].name : a; }).join(' and ');
    var open = 100 * p.n_open / p.n, close = 100 * p.n_close / p.n;
    var size = Math.abs(p.diff), verdict;
    if (size < D.closeEnough) {
      verdict = 'About what you would expect from a typical receiver.';
    } else {
      verdict = 'About ' + size + (p.diff > 0 ? ' more' : ' fewer') + ' catches than a typical receiver.';
    }
    playerOut.innerHTML =
      '<p><b>' + esc(p.name) + '</b> (' + esc(pos.name) + ', ' + esc(team) + ') was thrown to <b>' + p.n + ' times</b> and caught <b>' +
      p.catches + '</b> (' + pc(100 * p.catches / p.n) + ').</p>' +
      '<p>On ' + p.n_open + ' of those throws (' + pc(open) + ') the window was opening; a typical ' + esc(pos.name) + ' gets ' + pc(pos.open) + '. ' +
      'On ' + p.n_close + ' (' + pc(close) + ') a defender was closing; typical is ' + pc(pos.close) + '.</p>' +
      '<p>A typical receiver given throws with the same space and the same windows would catch about <b>' + p.exp + '</b>.</p>' +
      '<p class="verdict">' + verdict + '<span>For scale: the ' + D.players.length + ' players here run from ' +
      Math.abs(D.diffMin) + ' fewer to ' + D.diffMax + ' more.</span></p>' +
      '<h3>His throws compared with a typical ' + esc(pos.name) + '</h3>' +
      strip(p.name, open, 100 - open - close, close) +
      strip('Typical ' + pos.name, pos.open, pos.steady, pos.close);
  }
  playerSel.addEventListener('change', renderPlayer);
  var picks = document.getElementById('picks');
  D.picks.forEach(function (k) {
    var b = document.createElement('button');
    b.type = 'button';
    b.textContent = k.label + ': ' + D.players[k.i].name;
    b.addEventListener('click', function () { playerSel.value = String(k.i); renderPlayer(); });
    picks.appendChild(b);
  });
  renderPlayer();

  /* ---------- team ---------- */
  var teamSel = document.getElementById('team');
  var teamOut = document.getElementById('team-report');
  D.teams.forEach(function (t) {
    var o = document.createElement('option');
    o.value = t.abbr;
    o.textContent = t.name;
    if (t.abbr === D.defaultTeam) { o.selected = true; }
    teamSel.appendChild(o);
  });
  function side() {
    var r = document.querySelector('input[name="side"]:checked');
    return r ? r.value : 'off';
  }
  function renderTeam() {
    var t = teamByAbbr[teamSel.value], s = side(), d = t[s], html, cap;
    var who = esc(t.in_sentence);
    if (s === 'off') {
      html = '<p>When ' + who + (t.singular ? ' throws' : ' throw') + ', the receiver is still getting more open on <b>' + pc(d.open) +
        '</b> of throws. That ranks <b>' + ord(d.rank) + ' of 32</b> teams (1st means most often; higher is better for the offense). The league average is ' + pc(D.league.open) + '.</p>' +
        '<p>A defender is closing on ' + pc(d.close) + ' of their throws (league average: ' + pc(D.league.close) + '). Based on ' + d.n + ' throws.</p>';
      cap = 'By the coverage the defense was playing against ' + who;
    } else {
      html = '<p>When teams throw on ' + who + ', a defender is closing on <b>' + pc(d.close) +
        '</b> of throws. That ranks <b>' + ord(d.rank) + ' of 32</b> teams (1st means most often; higher is better for the defense). The league average is ' + pc(D.league.close) + '.</p>' +
        '<p>The receiver is still getting more open on ' + pc(d.open) + ' of those throws (league average: ' + pc(D.league.open) + '). Based on ' + d.n + ' throws.</p>';
      cap = 'By the coverage ' + who + (t.singular ? ' was' : ' were') + ' playing';
    }
    var rows = d.cov.map(function (c) {
      return '<tr><td>' + esc(c.cov) + ' <small>(' + esc(c.type) + ')</small></td><td>' + c.n + '</td><td>' + pc(c.open) + '</td><td>' + pc(c.close) + '</td></tr>';
    }).join('');
    html += '<table><caption>' + cap + '</caption><thead><tr><th scope="col">Coverage</th><th scope="col">Throws</th>' +
      '<th scope="col"><i class="sw open"></i>% of throws opening</th><th scope="col"><i class="sw close"></i>% of throws closing</th></tr></thead><tbody>' + rows +
      '<tr class="total"><td>All throws</td><td>' + d.n + '</td><td>' + pc(d.open) + '</td><td>' + pc(d.close) + '</td></tr></tbody></table>' +
      '<p class="note">Man: each defender follows one receiver. Zone: each defender guards an area of the field.</p>';
    teamOut.innerHTML = html;
  }
  teamSel.addEventListener('change', renderTeam);
  Array.prototype.forEach.call(document.querySelectorAll('input[name="side"]'), function (r) {
    r.addEventListener('change', renderTeam);
  });
  renderTeam();
})();
</script>
</body>
</html>
"""


def main() -> None:
    for path in (TARGETS_CSV, POSITION_CSV, ALIGNMENT_CSV, RECEIVER_CSV):
        if not path.exists():
            raise SystemExit(f"missing input: {path.relative_to(ROOT)} (run src/window_trend.py and src/skill_positions.py first)")
    data = compute()
    if FAILURES:
        print("\nSTOP: the verified findings did not reproduce:")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    page = build_html(data)
    for dest in DESTINATIONS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page, encoding="utf-8", newline="\n")
        print(f"wrote {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1024:.1f} KB)")
    print(f"default player: {DEFAULT_PLAYER}   default team: {TEAM_NAMES[DEFAULT_TEAM]} (offense)")


if __name__ == "__main__":
    main()
