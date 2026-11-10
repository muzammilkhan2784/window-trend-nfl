"""
src/build_tool.py — a single self-contained interactive page.

Reads output/targets_window_trend.csv (built by src/window_trend.py) and writes
output/window_trend_tool.html: one offline HTML file (plotly.js inlined) with a
team dropdown and an Offense / Defense toggle.

Window trend (from window_trend.py): for each targeted throw, how the target's
separation from the nearest defender is changing at the moment the ball is
released.
    opening  the receiver is pulling away as the ball comes out
    closing  the defender is driving on the ball as it comes out
    steady   neither

Offense view  -> "are our throws leaving while the receiver is still getting open?"
Defense view  -> "are our defenders driving on the ball at the throw?"

For the selected team + side the page shows:
    (1) three headline tiles (opening share, closing share, completion-% gap),
        each with the league average and the team's rank out of 32;
    (2) per coverage scheme: stacked opening/steady/closing shares, with a
        league-mix reference row; schemes under 15 targets for the team hidden;
    (3) a league ranking bar chart with the team highlighted
        (Offense = opening share of its throws; Defense = closing share of
         targets against it);
    (4) a one-line plain-English read-out.

Run:
    python src/build_tool.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "output" / "targets_window_trend.csv"
OUT = ROOT / "output" / "window_trend_tool.html"

# Colours fixed by the brief.
COL_OPEN = "#1f77b4"    # blue
COL_CLOSE = "#ff7f0e"   # orange
COL_STEADY = "#9e9e9e"  # grey

SCHEMES = ["Cover-1", "Cover-2", "Cover-3", "Quarters", "Cover-6", "2-Man"]
MIN_SCHEME_N = 15       # hide a scheme with fewer team targets than this
TEAMS = None            # filled at runtime


def ordinal(r: int) -> str:
    if 10 <= r % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{r}{suf}"


def shares(df: pl.DataFrame) -> dict:
    """opening/steady/closing shares + completion% gap for a slice of targets."""
    n = df.height
    if n == 0:
        return {"n": 0, "open": None, "steady": None, "close": None,
                "comp_gap": None, "comp_open": None, "comp_close": None}
    opn = (df["window"] == "opening").mean()
    cls = (df["window"] == "closing").mean()
    std = (df["window"] == "steady").mean()
    co = df.filter(pl.col("window") == "opening")
    cc = df.filter(pl.col("window") == "closing")
    comp_open = co["complete"].mean() if co.height else None
    comp_close = cc["complete"].mean() if cc.height else None
    gap = (comp_open - comp_close) if (comp_open is not None and comp_close is not None) else None
    return {
        "n": n, "open": opn, "steady": std, "close": cls,
        "comp_open": comp_open, "comp_close": comp_close, "comp_gap": gap,
    }


def ranks(values: dict[str, float | None], descending: bool) -> dict[str, int]:
    """Rank teams by a metric; None sorts last. 1 = best given `descending`."""
    present = [(t, v) for t, v in values.items() if v is not None]
    present.sort(key=lambda kv: kv[1], reverse=descending)
    out = {t: i + 1 for i, (t, v) in enumerate(present)}
    for t, v in values.items():
        if v is None:
            out[t] = len(values)
    return out


def build_payload(df: pl.DataFrame) -> dict:
    global TEAMS
    teams = sorted(set(df["possessionTeam"].unique().to_list())
                   | set(df["defensiveTeam"].unique().to_list()))
    TEAMS = teams

    league = shares(df)
    # League scheme mix (offense == defense here; same rows, just grouped later).
    # Also carry completion rate within opening / closing windows per scheme for
    # the "what a closing window costs" panel.
    league_scheme = {}
    for sc in SCHEMES:
        scdf = df.filter(pl.col("pff_passCoverage") == sc)
        s = shares(scdf)
        league_scheme[sc] = {
            "open": s["open"], "steady": s["steady"], "close": s["close"],
            "n": s["n"],
            "comp_open": s["comp_open"], "comp_close": s["comp_close"],
            "n_open": scdf.filter(pl.col("window") == "opening").height,
            "n_close": scdf.filter(pl.col("window") == "closing").height,
        }

    payload = {
        "teams": teams,
        "colors": {"open": COL_OPEN, "steady": COL_STEADY, "close": COL_CLOSE},
        "league": {
            "open": league["open"], "steady": league["steady"],
            "close": league["close"], "comp_gap": league["comp_gap"],
            "scheme": league_scheme,
        },
        "sides": {},
    }

    for side, key in (("offense", "possessionTeam"), ("defense", "defensiveTeam")):
        per_team = {t: shares(df.filter(pl.col(key) == t)) for t in teams}

        # Ranking metric: offense -> opening share (high good);
        # defense -> closing share (high good, more disruption).
        metric = {t: (per_team[t]["open"] if side == "offense"
                      else per_team[t]["close"]) for t in teams}
        metric_rank = ranks(metric, descending=True)

        open_rank = ranks({t: per_team[t]["open"] for t in teams}, descending=True)
        # For closing share: on offense fewer closing is "better" for the offense,
        # but the tile just reports rank by closing share high->low consistently.
        close_rank = ranks({t: per_team[t]["close"] for t in teams}, descending=True)
        gap_rank = ranks({t: per_team[t]["comp_gap"] for t in teams}, descending=True)

        # Per-team scheme breakdown. We carry both the window shares and the
        # completion rate within the opening / closing windows (plus their
        # sub-sample sizes) so the "what a closing window costs" panel can
        # decide whether the team has 15+ in both groups for a scheme.
        scheme = {}
        for t in teams:
            tdf = df.filter(pl.col(key) == t)
            scheme[t] = {}
            for sc in SCHEMES:
                scdf = tdf.filter(pl.col("pff_passCoverage") == sc)
                s = shares(scdf)
                n_open = scdf.filter(pl.col("window") == "opening").height
                n_close = scdf.filter(pl.col("window") == "closing").height
                scheme[t][sc] = {
                    "n": s["n"], "open": s["open"], "steady": s["steady"],
                    "close": s["close"],
                    "comp_open": s["comp_open"], "comp_close": s["comp_close"],
                    "n_open": n_open, "n_close": n_close,
                }

        payload["sides"][side] = {
            "team": {
                t: {
                    "n": per_team[t]["n"],
                    "open": per_team[t]["open"],
                    "steady": per_team[t]["steady"],
                    "close": per_team[t]["close"],
                    "comp_open": per_team[t]["comp_open"],
                    "comp_close": per_team[t]["comp_close"],
                    "comp_gap": per_team[t]["comp_gap"],
                    "open_rank": open_rank[t],
                    "close_rank": close_rank[t],
                    "gap_rank": gap_rank[t],
                    "metric": metric[t],
                    "metric_rank": metric_rank[t],
                    "scheme": scheme[t],
                }
                for t in teams
            },
            # Full league ranking list for the bar chart.
            "ranking": sorted(
                [{"team": t, "value": metric[t]} for t in teams if metric[t] is not None],
                key=lambda d: d["value"], reverse=True,
            ),
        }
    return payload


# --------------------------------------------------------------------------- #
# Console tables
# --------------------------------------------------------------------------- #
def print_tables(payload: dict) -> None:
    lg = payload["league"]
    print("\n" + "=" * 70)
    print("LEAGUE baseline (2021 Weeks 1-8, all targeted throws)")
    print("=" * 70)
    print(f"  opening {lg['open']*100:5.1f}%   steady {lg['steady']*100:5.1f}%   "
          f"closing {lg['close']*100:5.1f}%   comp-gap {lg['comp_gap']*100:+5.1f} pts")

    for side in ("offense", "defense"):
        head = ("Offense: share of OUR throws into opening windows"
                if side == "offense" else
                "Defense: share of targets AGAINST us where the window was closing")
        print("\n" + "=" * 70)
        print(f"{side.upper()} per-team table -- {head}")
        print("=" * 70)
        print(f"{'team':<5}{'n':>5}{'open%':>8}{'steady%':>9}{'close%':>8}"
              f"{'gap(pts)':>10}{'rank':>6}")
        print("-" * 70)
        rows = payload["sides"][side]["team"]
        order = sorted(rows.items(),
                       key=lambda kv: (kv[1]["metric"] if kv[1]["metric"] is not None else -1),
                       reverse=True)
        for t, r in order:
            gap = f"{r['comp_gap']*100:+.1f}" if r["comp_gap"] is not None else "  -"
            print(f"{t:<5}{r['n']:>5}{r['open']*100:>7.1f}%{r['steady']*100:>8.1f}%"
                  f"{r['close']*100:>7.1f}%{gap:>10}{ordinal(r['metric_rank']):>6}")


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def plotly_js() -> str:
    p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return p.read_text(encoding="utf-8")


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload)
    pjs = plotly_js()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Window Trend Tool - Opening vs Closing at the Throw</title>
<style>
  :root {{ --open:{COL_OPEN}; --close:{COL_CLOSE}; --steady:{COL_STEADY}; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:0; color:#1b1f24; background:#f6f7f9; }}
  header {{ background:#0b1f33; color:#fff; padding:18px 26px; }}
  header h1 {{ margin:0; font-size:20px; font-weight:650; }}
  header p {{ margin:4px 0 0; font-size:13px; color:#b9c6d4; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:20px 26px 48px; }}
  .controls {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap;
               margin:18px 0 10px; }}
  .controls label {{ font-size:13px; font-weight:600; margin-right:6px; }}
  select {{ font-size:15px; padding:6px 10px; border-radius:8px;
            border:1px solid #c7ced6; background:#fff; }}
  .toggle {{ display:inline-flex; border:1px solid #c7ced6; border-radius:8px;
             overflow:hidden; }}
  .toggle button {{ border:0; background:#fff; padding:7px 16px; font-size:14px;
                    cursor:pointer; font-weight:600; color:#44515f; }}
  .toggle button.active {{ background:#0b1f33; color:#fff; }}
  .readout {{ font-size:17px; font-weight:600; margin:14px 0 6px;
              padding:12px 16px; background:#fff; border-left:5px solid var(--open);
              border-radius:6px; }}
  .question {{ font-size:13px; color:#5a6775; margin:0 0 14px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px;
            margin:12px 0 22px; }}
  .tile {{ background:#fff; border-radius:12px; padding:16px 18px;
           box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .tile .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.4px;
                  color:#6a7682; font-weight:700; }}
  .tile .big {{ font-size:34px; font-weight:750; margin:4px 0 2px; }}
  .tile .sub {{ font-size:12.5px; color:#6a7682; }}
  .tile .rank {{ font-size:13px; font-weight:700; margin-top:6px; }}
  .card {{ background:#fff; border-radius:12px; padding:16px 18px; margin:16px 0;
           box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .card h2 {{ font-size:15px; margin:0 0 4px; }}
  .card .note {{ font-size:12.5px; color:#6a7682; margin:0 0 8px; }}
  footer {{ font-size:12px; color:#6a7682; text-align:center; padding:8px 26px 30px; }}
  .legend span {{ font-size:12px; margin-right:14px; }}
  .sw {{ display:inline-block; width:11px; height:11px; border-radius:2px;
         margin-right:4px; vertical-align:middle; }}
</style>
</head>
<body>
<header>
  <h1>Window Trend Tool - is the receiver's window opening or closing at the throw?</h1>
  <p>For every targeted pass we measure the target's separation from the nearest
     defender at release and 0.5 s earlier. Opening = pulling away; Closing =
     defender driving on the ball; Steady = in between.</p>
</header>
<div class="wrap">
  <div class="controls">
    <div><label for="team">Team</label>
      <select id="team"></select></div>
    <div class="toggle" id="sideToggle">
      <button data-side="offense" class="active">Offense</button>
      <button data-side="defense">Defense</button>
    </div>
    <div class="legend" style="margin-left:auto">
      <span><span class="sw" style="background:var(--open)"></span>Opening</span>
      <span><span class="sw" style="background:var(--steady)"></span>Steady</span>
      <span><span class="sw" style="background:var(--close)"></span>Closing</span>
    </div>
  </div>

  <div class="readout" id="readout"></div>
  <div class="question" id="question"></div>

  <div class="tiles">
    <div class="tile"><div class="label">Opening share</div>
      <div class="big" id="t_open"></div>
      <div class="sub" id="t_open_sub"></div><div class="rank" id="t_open_rank"></div></div>
    <div class="tile"><div class="label">Closing share</div>
      <div class="big" id="t_close"></div>
      <div class="sub" id="t_close_sub"></div><div class="rank" id="t_close_rank"></div></div>
    <div class="tile"><div class="label" id="t_gap_label">Completion % gap (open - close)</div>
      <div class="big" id="t_gap"></div>
      <div class="sub" id="t_gap_sub"></div>
      <div class="sub" id="t_gap_hint"></div><div class="rank" id="t_gap_rank"></div></div>
  </div>

  <div class="card">
    <h2>By coverage scheme</h2>
    <p class="note" id="schemeNote"></p>
    <div id="schemeChart" style="height:360px"></div>
  </div>

  <div class="card">
    <h2>What a closing window costs, by coverage</h2>
    <p class="note" id="costNote"></p>
    <div id="costChart" style="height:360px"></div>
  </div>

  <div class="card">
    <h2 id="rankTitle"></h2>
    <div id="rankChart" style="height:620px"></div>
  </div>

  <footer>2021 Weeks 1-8; team samples are 140-280 targets, so rankings are indicative.</footer>
</div>

<script>{pjs}</script>
<script>
const DATA = {data_json};
const C = DATA.colors;
let side = "offense";
let team = DATA.teams.includes("KC") ? "KC" : DATA.teams[0];

const sel = document.getElementById("team");
DATA.teams.forEach(t => {{
  const o = document.createElement("option"); o.value=t; o.textContent=t; sel.appendChild(o);
}});
sel.value = team;

function pct(x) {{ return x==null ? "-" : (x*100).toFixed(1)+"%"; }}
function pts(x) {{ return x==null ? "-" : (x>=0?"+":"")+(x*100).toFixed(1); }}
function ord(r) {{
  const m100=r%100, m10=r%10;
  const suf = (m100>=10&&m100<=20)?"th":({{1:"st",2:"nd",3:"rd"}})[m10]||"th";
  return r+suf;
}}

function render() {{
  const S = DATA.sides[side];
  const T = S.team[team];
  const lg = DATA.league;

  // ---- readout + question ----
  const ro = document.getElementById("readout");
  const q = document.getElementById("question");
  if (side==="defense") {{
    ro.textContent = `${{team}} defense is closing on ${{pct(T.close)}} of targets `
      + `(${{ord(T.metric_rank)}} of 32).`;
    ro.style.borderColor = C.close;
    q.textContent = "Defense view: are our defenders driving on the ball at the throw? "
      + "Higher closing share = more defenders driving on the receiver as the ball "
      + "is released.";
  }} else {{
    ro.textContent = `${{team}} offense throws into opening windows on ${{pct(T.open)}} `
      + `of targets (${{ord(T.metric_rank)}} of 32).`;
    ro.style.borderColor = C.open;
    q.textContent = "Offense view: are our throws leaving while the receiver is still "
      + "getting open? Higher opening share = more throws to receivers still separating.";
  }}

  // ---- tiles ----
  document.getElementById("t_open").textContent = pct(T.open);
  document.getElementById("t_open_sub").textContent = `league ${{pct(lg.open)}} - n=${{T.n}}`;
  document.getElementById("t_open_rank").textContent = `${{ord(T.open_rank)}} of 32`;

  document.getElementById("t_close").textContent = pct(T.close);
  document.getElementById("t_close_sub").textContent = `league ${{pct(lg.close)}}`;
  document.getElementById("t_close_rank").textContent = `${{ord(T.close_rank)}} of 32`;

  document.getElementById("t_gap_label").textContent =
    side==="defense" ? "Completion % gap allowed (opening - closing)"
                     : "Completion % gap (opening - closing)";
  document.getElementById("t_gap_hint").textContent =
    side==="defense"
      ? "how much more often opponents complete passes when the window is opening"
      : "";
  document.getElementById("t_gap").textContent = pts(T.comp_gap);
  document.getElementById("t_gap_sub").textContent = `league ${{pts(lg.comp_gap)}} pts`;
  document.getElementById("t_gap_rank").textContent =
    T.comp_gap==null ? "n/a" : `${{ord(T.gap_rank)}} of 32`;

  // ---- scheme stacked bars ----
  const shown = [], hidden = [];
  const schemes = ["Cover-1","Cover-2","Cover-3","Quarters","Cover-6","2-Man"];
  schemes.forEach(sc => {{
    const s = T.scheme[sc];
    if (s && s.n >= {MIN_SCHEME_N}) shown.push(sc); else hidden.push(sc);
  }});

  // Build rows: team scheme rows + one league-mix reference row.
  const yLabels = shown.map(sc => `${{sc}} (n=${{T.scheme[sc].n}})`);
  yLabels.push("League mix");
  const openVals = shown.map(sc => T.scheme[sc].open);
  const steadyVals = shown.map(sc => T.scheme[sc].steady);
  const closeVals = shown.map(sc => T.scheme[sc].close);
  openVals.push(lg.open); steadyVals.push(lg.steady); closeVals.push(lg.close);

  const mk = (vals,name,color) => ({{
    type:"bar", orientation:"h", name, y:yLabels, x:vals.map(v=>v==null?0:v*100),
    marker:{{color}}, text:vals.map(v=>v==null?"":(v*100).toFixed(0)+"%"),
    textposition:"inside", insidetextanchor:"middle",
    hovertemplate:"%{{y}}<br>"+name+": %{{x:.1f}}%<extra></extra>"
  }});
  Plotly.react("schemeChart",
    [mk(openVals,"Opening",C.open), mk(steadyVals,"Steady",C.steady),
     mk(closeVals,"Closing",C.close)],
    {{barmode:"stack", margin:{{l:120,r:20,t:10,b:36}},
      xaxis:{{title:"share of targets (%)", range:[0,100]}},
      legend:{{orientation:"h", y:1.12}}, paper_bgcolor:"#fff", plot_bgcolor:"#fff"}},
    {{displayModeBar:false, responsive:true}});

  document.getElementById("schemeNote").textContent =
    (side==="defense" ? "Coverages this defense was in when targeted. " :
                        "Coverages this offense faced. ")
    + (hidden.length ? ("Hidden (under {MIN_SCHEME_N} targets): " + hidden.join(", ") + ".")
                     : "All six schemes have enough targets to show.");

  // ---- what a closing window costs: completion% opening vs closing ----
  // League line always shown. Team overlaid only where it has 15+ targets in
  // BOTH the opening and closing groups for that scheme.
  const costSchemes = schemes.slice();          // all six, league always has data
  const xcat = costSchemes;
  const lgOpen = costSchemes.map(sc => {{
    const s = lg.scheme[sc]; return s.comp_open==null?null:s.comp_open*100; }});
  const lgClose = costSchemes.map(sc => {{
    const s = lg.scheme[sc]; return s.comp_close==null?null:s.comp_close*100; }});

  const teamEnough = [];
  const tmOpen = costSchemes.map(sc => {{
    const s = T.scheme[sc];
    const ok = s && s.n_open >= {MIN_SCHEME_N} && s.n_close >= {MIN_SCHEME_N};
    if (ok) teamEnough.push(sc);
    return ok && s.comp_open!=null ? s.comp_open*100 : null;
  }});
  const tmClose = costSchemes.map(sc => {{
    const s = T.scheme[sc];
    const ok = s && s.n_open >= {MIN_SCHEME_N} && s.n_close >= {MIN_SCHEME_N};
    return ok && s.comp_close!=null ? s.comp_close*100 : null;
  }});

  const costTraces = [
    {{type:"bar", name:"League opening", x:xcat, y:lgOpen, marker:{{color:C.open}},
      opacity: teamEnough.length?0.45:1.0, offsetgroup:"lg",
      text:lgOpen.map(v=>v==null?"":v.toFixed(0)+"%"), textposition:"outside",
      hovertemplate:"%{{x}} - League opening: %{{y:.1f}}%<extra></extra>"}},
    {{type:"bar", name:"League closing", x:xcat, y:lgClose, marker:{{color:C.close}},
      opacity: teamEnough.length?0.45:1.0, offsetgroup:"lg",
      text:lgClose.map(v=>v==null?"":v.toFixed(0)+"%"), textposition:"outside",
      hovertemplate:"%{{x}} - League closing: %{{y:.1f}}%<extra></extra>"}},
  ];
  if (teamEnough.length) {{
    costTraces.push(
      {{type:"bar", name:team+" opening", x:xcat, y:tmOpen, marker:{{color:C.open}},
        offsetgroup:"tm",
        hovertemplate:"%{{x}} - "+team+" opening: %{{y:.1f}}%<extra></extra>"}},
      {{type:"bar", name:team+" closing", x:xcat, y:tmClose, marker:{{color:C.close}},
        offsetgroup:"tm",
        hovertemplate:"%{{x}} - "+team+" closing: %{{y:.1f}}%<extra></extra>"}});
  }}
  Plotly.react("costChart", costTraces,
    {{barmode:"group", margin:{{l:46,r:20,t:10,b:36}},
      yaxis:{{title:"completion %", range:[0,100]}},
      legend:{{orientation:"h", y:1.14}}, paper_bgcolor:"#fff", plot_bgcolor:"#fff"}},
    {{displayModeBar:false, responsive:true}});

  const costNote = document.getElementById("costNote");
  if (!teamEnough.length) {{
    costNote.textContent = "League only: " + team + " has fewer than {MIN_SCHEME_N} "
      + "targets in both the opening and closing groups for every scheme, so no team "
      + "bars are shown. Opening bars are completion % when the window was opening; "
      + "closing bars when it was closing.";
  }} else {{
    const miss = costSchemes.filter(sc => !teamEnough.includes(sc));
    costNote.textContent = "Lighter bars = league; solid bars = " + team + ", shown "
      + "only where it has {MIN_SCHEME_N}+ targets in both groups ("
      + teamEnough.join(", ") + ")."
      + (miss.length ? " League only for: " + miss.join(", ") + "." : "");
  }}

  // ---- league ranking bar chart ----
  const rk = S.ranking;             // sorted desc by metric
  const ys = rk.map(d => d.team).reverse();
  const xs = rk.map(d => d.value*100).reverse();
  const colors = ys.map(t => t===team ? (side==="defense"?C.close:C.open) : "#c7ced6");
  const title = side==="defense"
    ? "League ranking - share of targets where the window was CLOSING (defense)"
    : "League ranking - share of throws into OPENING windows (offense)";
  document.getElementById("rankTitle").textContent = title;
  Plotly.react("rankChart",
    [{{type:"bar", orientation:"h", y:ys, x:xs, marker:{{color:colors}},
       text:xs.map(v=>v.toFixed(1)+"%"), textposition:"outside",
       hovertemplate:"%{{y}}: %{{x:.1f}}%<extra></extra>"}}],
    {{margin:{{l:46,r:40,t:8,b:36}}, xaxis:{{title:"share of targets (%)"}},
      yaxis:{{tickfont:{{size:11}}}}, paper_bgcolor:"#fff", plot_bgcolor:"#fff"}},
    {{displayModeBar:false, responsive:true}});
}}

sel.addEventListener("change", e => {{ team = e.target.value; render(); }});
document.querySelectorAll("#sideToggle button").forEach(b => {{
  b.addEventListener("click", () => {{
    document.querySelectorAll("#sideToggle button").forEach(x=>x.classList.remove("active"));
    b.classList.add("active");
    side = b.dataset.side; render();
  }});
}});
render();
</script>
</body>
</html>
"""


def main() -> int:
    if not CSV.exists():
        import sys
        sys.exit(f"{CSV} not found. Run src/window_trend.py first.")
    df = pl.read_csv(CSV)
    payload = build_payload(df)
    print_tables(payload)

    html = build_html(payload)
    OUT.write_text(html, encoding="utf-8")
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nWrote interactive tool -> {OUT}  ({size_mb:.1f} MB, opens offline)")

    # Copy for GitHub Pages, served next to the plain-English front page
    # (docs/index.html, written by build_dashboard.py).
    docs = ROOT / "docs" / "team-explorer.html"
    docs.parent.mkdir(exist_ok=True)
    docs.write_text(html, encoding="utf-8")
    print(f"Wrote GitHub Pages copy -> {docs}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
