"""Build the figure sheet for the blog post from measured data.

Charts are hand-authored SVG against the validated palette; the GIFs are
inlined as data URIs so the page is one self-contained file.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = json.loads((ROOT / "analysis.json").read_text())
OUT = ROOT / "figures.html"

W, H = 720, 330
PAD = {"l": 58, "r": 132, "t": 22, "b": 46}


def gif(name):
    raw = (ROOT / "out" / name).read_bytes()
    return "data:image/gif;base64," + base64.b64encode(raw).decode()


def line_chart(spec, y_lo, y_hi, y_fmt=lambda v: f"{v:.2f}", y_label="", chart_id="c"):
    """Ordinal x, linear y, direct-labelled series with a hover layer."""
    xs = spec["x_labels"]
    n = len(xs)
    px = lambda i: PAD["l"] + (W - PAD["l"] - PAD["r"]) * (i / max(1, n - 1))
    py = lambda v: PAD["t"] + (H - PAD["t"] - PAD["b"]) * (1 - (v - y_lo) / (y_hi - y_lo))

    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
        f'aria-label="{spec["aria"]}" data-chart="{chart_id}">'
    ]

    ticks = 4
    for t in range(ticks + 1):
        v = y_lo + (y_hi - y_lo) * t / ticks
        y = py(v)
        parts.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" x2="{W-PAD["r"]}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD["l"]-10}" y="{y+4:.1f}" text-anchor="end">{y_fmt(v)}</text>')

    parts.append(f'<line class="axis" x1="{PAD["l"]}" y1="{py(y_lo):.1f}" x2="{W-PAD["r"]}" y2="{py(y_lo):.1f}"/>')
    for i, label in enumerate(xs):
        parts.append(f'<text class="tick" x="{px(i):.1f}" y="{H-PAD["b"]+22}" text-anchor="middle">{label}</text>')
    if spec.get("x_title"):
        parts.append(f'<text class="axis-title" x="{(PAD["l"]+W-PAD["r"])/2:.1f}" y="{H-6}" text-anchor="middle">{spec["x_title"]}</text>')
    if y_label:
        parts.append(f'<text class="axis-title" x="{PAD["l"]-44}" y="{PAD["t"]-8}" text-anchor="start">{y_label}</text>')

    for s_i, series in enumerate(spec["series"]):
        colour = f"var(--s{s_i+1})"
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(series["values"]))
        parts.append(f'<polyline class="line" points="{pts}" stroke="{colour}"/>')
        for i, v in enumerate(series["values"]):
            parts.append(
                f'<circle class="dot" cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" fill="{colour}"/>'
            )
        last = len(series["values"]) - 1
        parts.append(
            f'<text class="series-label" x="{px(last)+12:.1f}" y="{py(series["values"][last])+4:.1f}" '
            f'fill="{colour}">{series["name"]}</text>'
        )

    # Hover layer: one column band per x position
    for i, label in enumerate(xs):
        band = (W - PAD["l"] - PAD["r"]) / max(1, n - 1)
        rows = "&#10;".join(
            f'{s["name"]}: {y_fmt(s["values"][i])}' for s in spec["series"]
        )
        parts.append(
            f'<rect class="hit" x="{px(i)-band/2:.1f}" y="{PAD["t"]}" width="{band:.1f}" '
            f'height="{H-PAD["t"]-PAD["b"]}"><title>{spec["x_title"] or "x"} {label}&#10;{rows}</title></rect>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def bar_chart(rows, unit, chart_id="b", highlight_last=True):
    """Horizontal magnitude bars, value labelled at the end of each bar."""
    bw, bh, gap = 640, 34, 14
    height = len(rows) * (bh + gap) + 16
    top = 8
    max_v = max(r[1] for r in rows)
    label_w = 250
    track = bw - label_w - 96

    parts = [f'<svg viewBox="0 0 {bw} {height}" role="img" class="chart" aria-label="{chart_id}">']
    for i, (name, value, note) in enumerate(rows):
        y = top + i * (bh + gap)
        w = max(3, track * value / max_v)
        colour = "var(--s1)" if (highlight_last and i == len(rows) - 1) else "var(--bar-quiet)"
        parts.append(f'<text class="bar-label" x="0" y="{y+bh/2+5}">{name}</text>')
        parts.append(
            f'<rect class="bar" x="{label_w}" y="{y+6}" width="{w:.1f}" height="{bh-12}" '
            f'rx="4" fill="{colour}"><title>{name}: {value}{unit}</title></rect>'
        )
        parts.append(
            f'<text class="bar-value" x="{label_w + w + 10:.1f}" y="{y+bh/2+5}">{value}{unit}'
            + (f' <tspan class="bar-note">{note}</tspan>' if note else "")
            + "</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def table(headers, rows, caption):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        f'<details class="table-view"><summary>{caption}</summary>'
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        "</details>"
    )


# ── figures ──────────────────────────────────────────────────────────────

md = DATA["transfer_matched_density"]
transfer_chart = line_chart(
    {
        "aria": "At constant density, the communicating policies stay above the "
                "non-communicating one at every fleet size, and the gap widens.",
        "x_labels": [f"{a}" for a in md["agents"]],
        "x_title": "robots in the fleet",
        "series": [
            {"name": "GNN, message", "values": md["series"]["GNN message K=3"]},
            {"name": "No comms", "values": md["series"]["No communication"]},
            {"name": "GCN K=3", "values": md["series"]["GCN K=3"]},
        ],
    },
    0.45, 0.95, lambda v: f"{v*100:.0f}%", "robots reaching their goal", "transfer",
)

ha = DATA["horizon_artifact"]
horizon_chart = line_chart(
    {
        "aria": "With a fixed 42-step horizon the policy appears to collapse at large "
                "fleet sizes; with a horizon that fits the board it does not.",
        "x_labels": [str(a) for a in ha["agents"]],
        "x_title": "robots  (board grows with the fleet: 20×20 → 63×63)",
        "series": [
            {"name": "horizon 300", "values": ha["series"]["horizon 300"]},
            {"name": "horizon 42", "values": ha["series"]["horizon 42"]},
        ],
    },
    0.35, 0.90, lambda v: f"{v*100:.0f}%", "robots reaching their goal", "horizon",
)

tf = DATA["training_fleet_size"]
fleet_chart = line_chart(
    {
        "aria": "Training on ten robots instead of five does not improve performance "
                "at larger fleet sizes.",
        "x_labels": [f"{a}" for a in tf["agents"]],
        "x_title": "robots at evaluation",
        "series": [
            {"name": "trained on 5", "values": tf["series"]["trained on 5 (28x28)"]},
            {"name": "trained on 10", "values": tf["series"]["trained on 10 (28x28)"]},
            {"name": "trained on 10, dense", "values": tf["series"]["trained on 10 (20x20)"]},
        ],
    },
    0.60, 0.90, lambda v: f"{v*100:.0f}%", "robots reaching their goal", "fleet",
)

cbs = DATA["cbs_optimisation"]["steps"]
cbs_chart = bar_chart(
    [(s["stage"], s["seconds"], "") for s in cbs], "s", "cbs-optimisation", False
)

density = DATA["training_density"]
density_chart = bar_chart(
    [(d["config"], d["neighbours"], f'graph empty {d["empty_graph_pct"]}% of steps')
     for d in density],
    "", "training-density",
)

# ── page ─────────────────────────────────────────────────────────────────

HTML = f"""<title>MAPF-GNN — figures and measurements</title>
<style>
  :root {{
    color-scheme: light;
    --paper:        #f7f6f2;
    --surface:      #fcfcfb;
    --ink:          #14140f;
    --ink-2:        #52514e;
    --muted:        #898781;
    --rule:         #e1e0d9;
    --axis:         #c3c2b7;
    --s1:           #2a78d6;
    --s2:           #eb6834;
    --s3:           #1baf7a;
    --bar-quiet:    #c3c2b7;
    --good:         #0ca30c;
    --critical:     #d03b3b;
    --shadow:       0 1px 2px rgba(20,20,15,.06), 0 8px 24px -16px rgba(20,20,15,.28);
    --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --paper:     #101011;
      --surface:   #1a1a19;
      --ink:       #f5f4ef;
      --ink-2:     #c3c2b7;
      --muted:     #898781;
      --rule:      #2c2c2a;
      --axis:      #383835;
      --s1:        #3987e5;
      --s2:        #d95926;
      --s3:        #199e70;
      --bar-quiet: #383835;
      --good:      #0ca30c;
      --critical:  #e66767;
      --shadow:    0 1px 2px rgba(0,0,0,.5), 0 10px 30px -18px rgba(0,0,0,.9);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --paper:     #101011;
    --surface:   #1a1a19;
    --ink:       #f5f4ef;
    --ink-2:     #c3c2b7;
    --muted:     #898781;
    --rule:      #2c2c2a;
    --axis:      #383835;
    --s1:        #3987e5;
    --s2:        #d95926;
    --s3:        #199e70;
    --bar-quiet: #383835;
    --good:      #0ca30c;
    --critical:  #e66767;
    --shadow:    0 1px 2px rgba(0,0,0,.5), 0 10px 30px -18px rgba(0,0,0,.9);
  }}

  * {{ box-sizing: border-box; }}
  body {{
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.6;
    margin: 0;
    padding: 0 20px 120px;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  .col  {{ max-width: 62ch; }}

  header.masthead {{ padding: 72px 0 40px; border-bottom: 1px solid var(--rule); }}
  .eyebrow {{
    font-family: var(--mono); font-size: 12px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 18px;
  }}
  h1 {{
    font-size: clamp(30px, 4.4vw, 46px); line-height: 1.08; letter-spacing: -.022em;
    margin: 0 0 20px; text-wrap: balance; font-weight: 640;
  }}
  .standfirst {{ font-size: 19px; color: var(--ink-2); margin: 0; max-width: 58ch; }}

  section {{ padding-top: 64px; }}
  h2 {{
    font-size: 25px; letter-spacing: -.012em; margin: 0 0 6px;
    text-wrap: balance; font-weight: 620;
  }}
  h3 {{ font-size: 16px; margin: 34px 0 6px; font-weight: 620; }}
  .kicker {{
    font-family: var(--mono); font-size: 11.5px; letter-spacing: .13em;
    text-transform: uppercase; color: var(--s1); margin: 0 0 12px;
  }}
  p {{ margin: 0 0 16px; color: var(--ink-2); }}
  p.lead {{ color: var(--ink); font-size: 17px; }}
  strong {{ color: var(--ink); font-weight: 620; }}
  code {{ font-family: var(--mono); font-size: .89em; color: var(--ink); }}

  figure {{
    margin: 30px 0 0; padding: 20px 20px 16px;
    background: var(--surface); border: 1px solid var(--rule);
    border-radius: 10px; box-shadow: var(--shadow);
  }}
  figure.plain {{ background: none; border: 0; box-shadow: none; padding: 0; }}
  figcaption {{
    font-size: 13.5px; color: var(--muted); margin-top: 14px;
    padding-top: 12px; border-top: 1px solid var(--rule); max-width: 68ch;
  }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}

  .chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .grid {{ stroke: var(--rule); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .tick {{ fill: var(--muted); font: 12px var(--sans); font-variant-numeric: tabular-nums; }}
  .axis-title {{ fill: var(--muted); font: 11.5px var(--mono); letter-spacing: .04em; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .dot  {{ stroke: var(--surface); stroke-width: 2; }}
  .series-label {{ font: 600 12.5px var(--sans); }}
  .hit {{ fill: transparent; }}
  .hit:hover {{ fill: color-mix(in srgb, var(--ink) 5%, transparent); }}
  .bar-label {{ fill: var(--ink-2); font: 13px var(--sans); }}
  .bar-value {{ fill: var(--ink); font: 600 13px var(--mono); font-variant-numeric: tabular-nums; }}
  .bar-note {{ fill: var(--muted); font-weight: 400; }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 4px 0 14px; padding: 0; list-style: none; }}
  .legend li {{ display: flex; align-items: center; gap: 7px; font-size: 13px; color: var(--ink-2); }}
  .swatch {{ width: 11px; height: 11px; border-radius: 3px; flex: none; }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 26px 0 0; }}
  .stat {{
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
    padding: 16px 16px 14px;
  }}
  .stat .value {{
    font-size: 30px; font-weight: 640; letter-spacing: -.02em; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }}
  .stat .label {{ font-size: 12.5px; color: var(--muted); margin-top: 6px; }}
  .stat.up .value {{ color: var(--good); }}
  .stat.down .value {{ color: var(--critical); }}

  .table-view {{ margin-top: 14px; }}
  .table-view summary {{
    cursor: pointer; font-size: 13px; color: var(--muted);
    font-family: var(--mono); letter-spacing: .03em;
  }}
  .table-view summary:hover {{ color: var(--ink); }}
  .scroll {{ overflow-x: auto; margin-top: 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th, td {{
    text-align: right; padding: 7px 12px; border-bottom: 1px solid var(--rule);
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  td {{ color: var(--ink-2); }}

  .callout {{
    border-left: 3px solid var(--s2); padding: 2px 0 2px 18px; margin: 26px 0;
  }}
  .callout p:last-child {{ margin-bottom: 0; }}

  :focus-visible {{ outline: 2px solid var(--s1); outline-offset: 3px; border-radius: 3px; }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
  @media (max-width: 640px) {{ header.masthead {{ padding-top: 48px; }} }}
</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">MAPF-GNN · figure sheet</p>
  <h1>Robots that learn to talk — and what it took to measure that honestly</h1>
  <p class="standfirst">A replication of decentralised multi-robot path planning with graph neural
  networks. Every number below was measured in the repository: 200 episodes per point, seed 42,
  on an RTX 5070&nbsp;Ti.</p>
</header>

<section>
  <p class="kicker">The result, animated</p>
  <h2>Thirty robots, one instance, two policies</h2>
  <div class="col">
    <p class="lead">Both policies were trained on the same expert demonstrations and see the same
    5×5 patch of the world. The only difference is whether a robot may hear from its neighbours.</p>
    <p>Colour is state, not identity: <span style="color:var(--s1)">blue</span> is moving,
    <span style="color:var(--good)">green</span> has arrived,
    <span style="color:var(--critical)">red</span> has not moved for six steps. Thin lines are the
    communication graph the network actually consumed at that step.</p>
  </div>
  <figure class="plain">
    <img src="{gif('compare_30agents.gif')}" alt="Side-by-side animation of thirty robots. The
    non-communicating policy accumulates a large cluster of stalled robots; the communicating
    policy resolves most of them.">
    <figcaption>Same instance, same start states. Without communication the fleet forms a
    seven-robot deadlock that never clears — 17 of 30 arrive. With communication, 27 of 30 arrive.
    The stalled cluster is the failure mode that dominates every large-fleet number in this page.</figcaption>
  </figure>
</section>

<section>
  <p class="kicker">The thing that was broken</p>
  <h2>The policy was cloning a trajectory the expert never drove</h2>
  <div class="col">
    <p class="lead">Before any of the scaling questions could be asked, the training data had to
    actually contain expert behaviour. It did not.</p>
    <p>The recorder replayed each solved instance through the environment to capture what the expert
    saw — but dropped the expert's first action before replaying. Every stored observation came from
    a path that had already diverged.</p>
  </div>

  <figure>
    <svg viewBox="0 0 720 220" role="img" class="chart"
         aria-label="The recorder applied the expert's second action from the first state, so the
         replayed path left the expert's route immediately.">
      <defs>
        <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <polygon points="0,0 10,5 0,10" fill="currentColor"/>
        </marker>
        <marker id="ar-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <polygon points="0,0 10,5 0,10" fill="#d03b3b"/>
        </marker>
      </defs>

      <text x="0" y="16" class="axis-title" fill="var(--muted)">EXPERT (CBS)</text>
      <g color="var(--ink-2)">
        <circle cx="70" cy="58" r="15" fill="none" stroke="currentColor" stroke-width="1.6"/>
        <text x="70" y="63" text-anchor="middle" class="tick" fill="var(--ink)">s₀</text>
        <line x1="87" y1="58" x2="163" y2="58" stroke="currentColor" stroke-width="1.6" marker-end="url(#ar)"/>
        <text x="125" y="48" text-anchor="middle" class="tick" fill="var(--ink-2)">a₀</text>
        <circle cx="180" cy="58" r="15" fill="none" stroke="currentColor" stroke-width="1.6"/>
        <text x="180" y="63" text-anchor="middle" class="tick" fill="var(--ink)">s₁</text>
        <line x1="197" y1="58" x2="273" y2="58" stroke="currentColor" stroke-width="1.6" marker-end="url(#ar)"/>
        <text x="235" y="48" text-anchor="middle" class="tick" fill="var(--ink-2)">a₁</text>
        <circle cx="290" cy="58" r="15" fill="none" stroke="currentColor" stroke-width="1.6"/>
        <text x="290" y="63" text-anchor="middle" class="tick" fill="var(--ink)">s₂</text>
        <line x1="307" y1="58" x2="383" y2="58" stroke="currentColor" stroke-width="1.6" marker-end="url(#ar)"/>
        <text x="345" y="48" text-anchor="middle" class="tick" fill="var(--ink-2)">a₂</text>
        <circle cx="400" cy="58" r="15" fill="none" stroke="currentColor" stroke-width="1.6"/>
        <text x="400" y="63" text-anchor="middle" class="tick" fill="var(--ink)">s₃</text>
      </g>

      <text x="0" y="128" class="axis-title" fill="var(--muted)">RECORDED</text>
      <g color="var(--ink-2)">
        <circle cx="70" cy="170" r="15" fill="none" stroke="currentColor" stroke-width="1.6"/>
        <text x="70" y="175" text-anchor="middle" class="tick" fill="var(--ink)">s₀</text>
        <line x1="87" y1="170" x2="163" y2="170" stroke="#d03b3b" stroke-width="2" marker-end="url(#ar-bad)"/>
        <text x="125" y="160" text-anchor="middle" class="tick" fill="#d03b3b">a₁</text>
        <circle cx="180" cy="170" r="15" fill="none" stroke="#d03b3b" stroke-width="1.8" stroke-dasharray="3 3"/>
        <text x="180" y="175" text-anchor="middle" class="tick" fill="#d03b3b">s?</text>
        <line x1="197" y1="170" x2="273" y2="170" stroke="#d03b3b" stroke-width="2" marker-end="url(#ar-bad)"/>
        <text x="235" y="160" text-anchor="middle" class="tick" fill="#d03b3b">a₂</text>
        <circle cx="290" cy="170" r="15" fill="none" stroke="#d03b3b" stroke-width="1.8" stroke-dasharray="3 3"/>
        <text x="290" y="175" text-anchor="middle" class="tick" fill="#d03b3b">s?</text>
        <line x1="307" y1="170" x2="383" y2="170" stroke="#d03b3b" stroke-width="2" marker-end="url(#ar-bad)"/>
        <text x="345" y="160" text-anchor="middle" class="tick" fill="#d03b3b">a₃</text>
        <circle cx="400" cy="170" r="15" fill="none" stroke="#d03b3b" stroke-width="1.8" stroke-dasharray="3 3"/>
        <text x="400" y="175" text-anchor="middle" class="tick" fill="#d03b3b">s?</text>
      </g>

      <line x1="70" y1="80" x2="70" y2="148" stroke="var(--axis)" stroke-width="1" stroke-dasharray="2 4"/>
      <text x="440" y="120" class="tick" fill="var(--muted)">
        <tspan x="440" dy="0">only the first state is shared;</tspan>
        <tspan x="440" dy="17">every later pair is off the expert path</tspan>
      </text>
      <text x="440" y="58" class="tick" fill="var(--good)">31 of 31 steps match</text>
      <text x="440" y="170" class="tick" fill="#d03b3b">1 of 30 steps match</text>
    </svg>
    <figcaption>Replaying the stored actions against the CBS schedule confirmed it: unshifted, all
    31 steps reproduce the expert exactly; as recorded, one did.</figcaption>
  </figure>

  <div class="stats">
    <div class="stat up"><div class="value">14% → 75%</div><div class="label">Instances fully solved, before and after the data fixes</div></div>
    <div class="stat"><div class="value">4,945 → 30,727</div><div class="label">Usable training samples</div></div>
    <div class="stat"><div class="value">5 of ~31</div><div class="label">Timesteps per episode the loader had been using</div></div>
  </div>
</section>

<section>
  <p class="kicker">Does it scale</p>
  <h2>Trained on five robots, run on a hundred</h2>
  <div class="col">
    <p class="lead">Every weight is shared across robots and every observation is local, so the same
    policy runs on any fleet size. The board grows with the fleet here, holding robots-per-cell
    fixed — otherwise "more robots" and "more crowding" cannot be told apart.</p>
  </div>
  <figure>
    <ul class="legend">
      <li><span class="swatch" style="background:var(--s1)"></span>GNN, message passing</li>
      <li><span class="swatch" style="background:var(--s2)"></span>No communication</li>
      <li><span class="swatch" style="background:var(--s3)"></span>GCN, K=3</li>
    </ul>
    {transfer_chart}
    <figcaption>Fraction of robots reaching their goal at constant density, 200 episodes per point.
    Communication is worth 6.6 points at ten robots and 13.2 at a hundred — the advantage grows
    with the fleet, which is the claim the architecture is built on.</figcaption>
    {table(
        ["robots", "board", "neighbours", "No comms", "GCN K=3", "GNN message"],
        [[a, f"{b}×{b}", f"{n:.2f}",
          f'{md["series"]["No communication"][i]:.3f}',
          f'{md["series"]["GCN K=3"][i]:.3f}',
          f'{md["series"]["GNN message K=3"][i]:.3f}']
         for i, (a, b, n) in enumerate(zip(md["agents"], md["board"], md["neighbours"]))],
        "Show the numbers")}
  </figure>
</section>

<section>
  <p class="kicker">A measurement that lied</p>
  <h2>The horizon, not the policy</h2>
  <div class="col">
    <p class="lead">The first version of that chart showed a collapse at large fleets. It was an
    artifact of my own harness.</p>
    <p>Episodes ran for a fixed 42 steps regardless of board size. That is harmless on a fixed
    28×28 board — raising it to 200 moves results by under half a point — but under density matching
    the board grows with the fleet, and a 63×63 board takes about ninety steps just to cross.
    Robots were being scored on goals they could not physically reach.</p>
  </div>
  <figure>
    <ul class="legend">
      <li><span class="swatch" style="background:var(--s1)"></span>horizon fits the board</li>
      <li><span class="swatch" style="background:var(--s2)"></span>fixed 42 steps</li>
    </ul>
    {horizon_chart}
    <figcaption>The same checkpoint, the same episodes, two horizons. At a hundred robots the
    measured success rate moves from 40% to 67% — the difference between "the approach breaks down"
    and "it degrades gently".</figcaption>
  </figure>
  <div class="callout col">
    <p><strong>Worth stating plainly:</strong> this was not a subtle statistical issue. A stopwatch
    set too short made a working policy look broken, and the shape of the wrong curve was entirely
    plausible. Plausibility is not evidence.</p>
  </div>
</section>

<section>
  <p class="kicker">A hypothesis that died</p>
  <h2>Bigger expert teams buy nothing</h2>
  <div class="col">
    <p class="lead">The expert planner is only tractable for small teams, so it matters whether the
    policy needs to see large ones. It does not.</p>
    <p>The five-robot training set leaves the communication graph completely empty for half of every
    batch. That looked like an obvious defect — a communication mechanism trained mostly on nothing
    — so I regenerated the data at four times the density, where the graph is never empty. It made
    no difference at scale, and the sparsely trained model is if anything the best.</p>
  </div>
  <figure>
    <ul class="legend">
      <li><span class="swatch" style="background:var(--s1)"></span>trained on 5 robots</li>
      <li><span class="swatch" style="background:var(--s2)"></span>trained on 10</li>
      <li><span class="swatch" style="background:var(--s3)"></span>trained on 10, dense board</li>
    </ul>
    {fleet_chart}
    <figcaption>Message-passing model, matched-density evaluation. Three training regimes land
    within about three points of each other everywhere, and the cheapest one wins at a hundred
    robots. Scaling comes from the architecture, not from the expert.</figcaption>
  </figure>

  <h3>Training density, for reference</h3>
  <figure>
    {density_chart}
    <figcaption>Average neighbours per robot in each training set. Even the densest regime averages
    about one neighbour — and quadrupling this changed nothing downstream.</figcaption>
  </figure>
</section>

<section>
  <p class="kicker">Making the experiments affordable</p>
  <h2>The expert planner was the bottleneck, not the GPU</h2>
  <div class="col">
    <p class="lead">Conflict-Based Search took 25.8 seconds per instance. Almost all of it was one
    algorithmic mistake: every node of the high-level search replanned all N robots, when only the
    one that just received a constraint needs it.</p>
  </div>
  <figure>
    {cbs_chart}
    <figcaption>Mean seconds per instance at 20 robots on a 14×14 board. Solve and fail outcomes are
    identical at every step — nothing was traded away for speed. Total: 21.8× before parallelism,
    and about 83× with it.</figcaption>
  </figure>

  <div class="stats">
    <div class="stat"><div class="value">0.007s</div><div class="label">Median instance — the mean of 0.46s is a 5% tail</div></div>
    <div class="stat"><div class="value">149×</div><div class="label">Rollouts, after batching episodes on the GPU</div></div>
    <div class="stat"><div class="value">12×</div><div class="label">Forward pass, after removing the per-robot Python loop</div></div>
  </div>

  {table(
      ["episodes", "sequential", "batched", "speedup"],
      [[e, f"{s:.2f}s", f"{b:.2f}s", f"{s/b:.0f}×"] for e, s, b in zip(
          DATA["rollout_speedup"]["episodes"],
          DATA["rollout_speedup"]["sequential_s"],
          DATA["rollout_speedup"]["batched_s"])],
      "Rollout throughput, identical results on both paths")}

  <div class="callout col">
    <p><strong>The GPU was never the constraint.</strong> The model costs 0.65&nbsp;ms at batch 1 and
    0.61&nbsp;ms at batch 128 — that is launch latency, not arithmetic. Batching the episodes, rather
    than buying a bigger card, is what made a thousand evaluation episodes take 0.22 seconds.</p>
  </div>
</section>

<section>
  <p class="kicker">Still open</p>
  <h2>Deadlock is the remaining failure</h2>
  <div class="col">
    <p class="lead">Beyond about twenty robots, the share of instances where <em>every</em> robot
    arrives falls to zero, even with a generous horizon. Those are genuine deadlocks — the animation
    at the top shows one forming.</p>
    <p>Behaviour cloning gives the policy no signal for states the expert never visits, which is
    exactly where a deadlocked robot stands. The original paper's answer is an online expert: run the
    policy, collect the states where it jams, solve those with the planner, fold them back into the
    dataset. At 125&nbsp;ms per instance, that loop is now cheap.</p>
  </div>
  <figure class="plain">
    <img src="{gif('compare_12agents.gif')}" alt="Twelve robots on a small board, with and without
    communication.">
    <figcaption>Twelve robots on a 20×20 board — the regime the expert data was generated in.</figcaption>
  </figure>
</section>

</div>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
