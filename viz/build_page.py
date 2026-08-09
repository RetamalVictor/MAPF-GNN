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

tf = DATA["training_fleet_size"]
fleet_chart = line_chart(
    {
        "aria": "Policies trained on five robots match those trained on ten at every "
                "evaluation fleet size.",
        "x_labels": [f"{a}" for a in tf["agents"]],
        "x_title": "robots at evaluation",
        "series": [
            {"name": "expert used 5", "values": tf["series"]["trained on 5 (28x28)"]},
            {"name": "expert used 10", "values": tf["series"]["trained on 10 (28x28)"]},
            {"name": "10, dense board", "values": tf["series"]["trained on 10 (20x20)"]},
        ],
    },
    0.60, 0.90, lambda v: f"{v*100:.0f}%", "robots reaching their goal", "fleet",
)

cbs = DATA["cbs_optimisation"]["steps"]
cbs_chart = bar_chart(
    [(s["stage"], s["seconds"], "") for s in cbs], "s", "planner-throughput", False
)

# ── page ─────────────────────────────────────────────────────────────────

HTML = f"""<title>Decentralised multi-robot path planning with graph neural networks</title>
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
    line-height: 1.62;
    margin: 0;
    padding: 0 20px 120px;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  .col  {{ max-width: 64ch; }}

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

  section {{ padding-top: 68px; }}
  h2 {{
    font-size: 26px; letter-spacing: -.013em; margin: 0 0 8px;
    text-wrap: balance; font-weight: 620;
  }}
  h3 {{ font-size: 16.5px; margin: 36px 0 6px; font-weight: 620; }}
  .kicker {{
    font-family: var(--mono); font-size: 11.5px; letter-spacing: .13em;
    text-transform: uppercase; color: var(--s1); margin: 0 0 12px;
  }}
  p {{ margin: 0 0 16px; color: var(--ink-2); }}
  p.lead {{ color: var(--ink); font-size: 17px; }}
  strong {{ color: var(--ink); font-weight: 620; }}
  code {{ font-family: var(--mono); font-size: .89em; color: var(--ink); }}
  ul.body {{ color: var(--ink-2); padding-left: 20px; margin: 0 0 16px; max-width: 64ch; }}
  ul.body li {{ margin-bottom: 8px; }}

  .formula {{
    font-family: var(--mono); font-size: 14.5px; color: var(--ink);
    background: var(--surface); border: 1px solid var(--rule); border-radius: 8px;
    padding: 14px 18px; margin: 20px 0; overflow-x: auto; white-space: nowrap;
  }}
  .formula .an {{ color: var(--muted); }}

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
  .dlabel {{ font: 12px var(--sans); fill: var(--ink-2); }}
  .dlabel-mono {{ font: 11.5px var(--mono); fill: var(--muted); }}
  .dbox {{ fill: var(--surface); stroke: var(--axis); stroke-width: 1.4; }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 4px 0 14px; padding: 0; list-style: none; }}
  .legend li {{ display: flex; align-items: center; gap: 7px; font-size: 13px; color: var(--ink-2); }}
  .swatch {{ width: 11px; height: 11px; border-radius: 3px; flex: none; }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 26px 0 0; }}
  .stat {{
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
    padding: 16px 16px 14px;
  }}
  .stat .value {{
    font-size: 29px; font-weight: 640; letter-spacing: -.02em; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }}
  .stat .label {{ font-size: 12.5px; color: var(--muted); margin-top: 6px; }}

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

  .callout {{ border-left: 3px solid var(--s1); padding: 2px 0 2px 18px; margin: 26px 0; }}
  .callout p:last-child {{ margin-bottom: 0; }}

  .next {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-top: 26px; }}
  .next article {{
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 18px;
  }}
  .next h4 {{ margin: 0 0 8px; font-size: 15.5px; font-weight: 620; }}
  .next p {{ margin: 0; font-size: 14px; }}

  :focus-visible {{ outline: 2px solid var(--s1); outline-offset: 3px; border-radius: 3px; }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
  @media (max-width: 640px) {{ header.masthead {{ padding-top: 48px; }} }}
</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">Multi-robot path finding · graph neural networks</p>
  <h1>Robots that learn to talk their way through a crowd</h1>
  <p class="standfirst">A decentralised planner where each robot sees a 5×5 patch of the world,
  exchanges one vector with whoever is nearby, and picks a move. Trained by imitating a centralised
  planner on five robots; it runs a hundred.</p>
</header>

<section>
  <p class="kicker">The problem</p>
  <h2>Coordination without a coordinator</h2>
  <div class="col">
    <p class="lead">Give every robot its own goal in a shared, cluttered space and the joint problem
    explodes: five robots on a 28×28 grid have about 3×10<sup>14</sup> joint configurations, and
    optimal multi-agent path finding is NP-hard.</p>
    <p>Excellent optimal solvers exist, but they are all <strong>centralised</strong> — one process
    needs every robot's position, every goal, and the whole map, simultaneously. Real fleets are not
    like that. Each robot sees a few metres, talks to whoever is in range, and has to decide now.</p>
    <p>So the question is whether a robot can learn to <em>behave</em> as though it had a global
    planner, using only a local view and a few messages from its neighbours.</p>
  </div>
  <figure class="plain">
    <img src="{gif('compare_30agents.gif')}" alt="Two panels of thirty robots crossing a grid. In
    the left panel, which has no communication, a large cluster of robots stalls. In the right panel
    the same robots with communication mostly reach their goals.">
    <figcaption>Thirty robots, one instance, two policies trained on identical demonstrations. The
    only difference is whether a robot may hear from its neighbours. Blue is moving, green has
    arrived, red has not moved in six steps; thin lines are the live communication graph.</figcaption>
  </figure>
</section>

<section>
  <p class="kicker">The method · 1</p>
  <h2>What one robot knows</h2>
  <div class="col">
    <p class="lead">Each robot receives a 5×5 patch centred on itself, in two channels: the local map
    (obstacles and other robots) and its own goal. Fifty numbers per robot per timestep.</p>
    <p>Which raises the obvious problem — the board is 28×28 and the patch is 5×5, so
    <strong>the goal is almost never inside the field of view</strong>. Leave that channel empty and
    the robot has no idea which way to head, and imitation cannot work: the expert's action depends
    on something the network cannot see.</p>
    <p>The fix is to project the goal onto the edge of the patch. The robot doesn't learn
    <em>where</em> its goal is; it learns <em>which way</em> it lies.</p>
  </div>

  <figure>
    <svg viewBox="0 0 720 250" role="img" class="chart"
         aria-label="A goal outside the 5 by 5 field of view is projected onto the edge cell nearest
         its true direction, so the observation encodes a bearing rather than a position.">
      <defs>
        <marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <polygon points="0,0 10,5 0,10" fill="var(--s2)"/>
        </marker>
      </defs>
      <text x="0" y="16" class="dlabel-mono">WORLD · 28 × 28</text>
      <rect x="0" y="30" width="200" height="200" class="dbox" rx="4"/>
      <rect x="60" y="120" width="60" height="60" fill="none" stroke="var(--s1)" stroke-width="2" rx="3"/>
      <circle cx="90" cy="150" r="8" fill="var(--s1)"/>
      <text x="90" y="200" text-anchor="middle" class="dlabel">robot</text>
      <circle cx="168" cy="66" r="7" fill="none" stroke="var(--s2)" stroke-width="2.4"/>
      <text x="168" y="52" text-anchor="middle" class="dlabel" fill="var(--s2)">goal</text>
      <line x1="99" y1="141" x2="158" y2="76" stroke="var(--s2)" stroke-width="1.6"
            stroke-dasharray="4 4" marker-end="url(#a1)"/>

      <text x="300" y="16" class="dlabel-mono">WHAT THE NETWORK SEES · 5 × 5</text>
      <g>
        <rect x="300" y="30" width="150" height="150" class="dbox" rx="4"/>
        <line x1="330" y1="30" x2="330" y2="180" class="grid"/>
        <line x1="360" y1="30" x2="360" y2="180" class="grid"/>
        <line x1="390" y1="30" x2="390" y2="180" class="grid"/>
        <line x1="420" y1="30" x2="420" y2="180" class="grid"/>
        <line x1="300" y1="60" x2="450" y2="60" class="grid"/>
        <line x1="300" y1="90" x2="450" y2="90" class="grid"/>
        <line x1="300" y1="120" x2="450" y2="120" class="grid"/>
        <line x1="300" y1="150" x2="450" y2="150" class="grid"/>
        <circle cx="375" cy="105" r="9" fill="var(--s1)"/>
        <rect x="422" y="32" width="26" height="26" fill="var(--s2)" opacity=".9" rx="3"/>
      </g>
      <text x="300" y="200" class="dlabel">goal marker pinned to the north-east edge</text>
      <text x="300" y="220" class="dlabel-mono">channel 0 map · channel 1 goal bearing</text>

      <text x="490" y="96" class="dlabel" fill="var(--ink)">A bearing, not a position.</text>
      <text x="490" y="118" class="dlabel">The observation stays the same</text>
      <text x="490" y="136" class="dlabel">size on any board, so the policy</text>
      <text x="490" y="154" class="dlabel">transfers across map sizes.</text>
    </svg>
    <figcaption>Projecting the goal onto the patch boundary makes the observation independent of the
    map's dimensions — the same trick that lets a policy trained on 20×20 run on 63×63.</figcaption>
  </figure>
</section>

<section>
  <p class="kicker">The method · 2</p>
  <h2>Communication as a graph convolution</h2>
  <div class="col">
    <p class="lead">Robots within sensing range form a graph, capped at the four nearest neighbours
    so a robot in a crowd doesn't average its message into mush — and so a real radio has a bounded
    bandwidth to budget for.</p>
    <p>That graph changes every timestep, because the robots move. It is an <em>input</em>, not a
    structure the network learns once.</p>
    <p>A small CNN turns each robot's patch into a 64-dimensional feature. Then a graph convolution
    blends each robot's features with its neighbours', normalised so a well-connected robot in a
    crowd doesn't simply accumulate a larger vector than an isolated one:</p>
  </div>

  <div class="formula">H' = σ( Σ<sub>k=0..K-1</sub> Ã<sup>k</sup> H W<sub>k</sub> )
    <span class="an">   where  Ã = D<sup>-1/2</sup>(A + I)D<sup>-1/2</sup></span></div>

  <div class="col">
    <p>The exponent is the whole idea, and it has a physical reading: <strong>K is the number of
    communication rounds per decision.</strong> Multiply by Ã once and a robot has heard from the
    robots it can see. Twice, and it has heard about robots it cannot see, relayed by the ones it
    can.</p>
  </div>

  <figure>
    <svg viewBox="0 0 720 240" role="img" class="chart"
         aria-label="With one hop a robot uses only its own observation; with two it hears its
         direct neighbours; with three, information relays from robots it cannot see.">
      <defs>
        <marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <polygon points="0,0 10,5 0,10" fill="var(--s1)"/>
        </marker>
      </defs>
      <g>
        <text x="0" y="18" class="dlabel-mono">K = 1 · OWN VIEW</text>
        <circle cx="40" cy="70" r="16" fill="var(--s1)"/><text x="40" y="75" text-anchor="middle" class="tick" fill="#fff">A</text>
        <circle cx="120" cy="70" r="16" class="dbox"/><text x="120" y="75" text-anchor="middle" class="tick">B</text>
        <circle cx="200" cy="70" r="16" class="dbox"/><text x="200" y="75" text-anchor="middle" class="tick">C</text>
        <line x1="56" y1="70" x2="104" y2="70" stroke="var(--axis)" stroke-width="1.4"/>
        <line x1="136" y1="70" x2="184" y2="70" stroke="var(--axis)" stroke-width="1.4"/>
        <text x="0" y="118" class="dlabel">A knows only what A can see</text>
      </g>
      <g transform="translate(250,0)">
        <text x="0" y="18" class="dlabel-mono">K = 2 · ONE EXCHANGE</text>
        <circle cx="40" cy="70" r="16" fill="var(--s1)"/><text x="40" y="75" text-anchor="middle" class="tick" fill="#fff">A</text>
        <circle cx="120" cy="70" r="16" fill="var(--s1)" opacity=".55"/><text x="120" y="75" text-anchor="middle" class="tick" fill="#fff">B</text>
        <circle cx="200" cy="70" r="16" class="dbox"/><text x="200" y="75" text-anchor="middle" class="tick">C</text>
        <line x1="104" y1="70" x2="58" y2="70" stroke="var(--s1)" stroke-width="2" marker-end="url(#a2)"/>
        <line x1="136" y1="70" x2="184" y2="70" stroke="var(--axis)" stroke-width="1.4"/>
        <text x="0" y="118" class="dlabel">B's observation reaches A</text>
      </g>
      <g transform="translate(0,140)">
        <text x="0" y="18" class="dlabel-mono">K = 3 · RELAYED</text>
        <circle cx="40" cy="70" r="16" fill="var(--s1)"/><text x="40" y="75" text-anchor="middle" class="tick" fill="#fff">A</text>
        <circle cx="120" cy="70" r="16" fill="var(--s1)" opacity=".55"/><text x="120" y="75" text-anchor="middle" class="tick" fill="#fff">B</text>
        <circle cx="200" cy="70" r="16" fill="var(--s1)" opacity=".3"/><text x="200" y="75" text-anchor="middle" class="tick">C</text>
        <line x1="104" y1="70" x2="58" y2="70" stroke="var(--s1)" stroke-width="2" marker-end="url(#a2)"/>
        <line x1="184" y1="70" x2="138" y2="70" stroke="var(--s1)" stroke-width="2" marker-end="url(#a2)"/>
        <text x="0" y="118" class="dlabel">A learns about C — which it cannot see — through B</text>
      </g>
      <text x="250" y="178" class="dlabel" fill="var(--ink)">Larger K buys context.</text>
      <text x="250" y="198" class="dlabel">It also costs a round of radio traffic</text>
      <text x="250" y="216" class="dlabel">and latency inside every decision.</text>
    </svg>
    <figcaption>K sets how far information travels per decision. Every robot runs the same weights,
    so the fleet size is never baked into the network.</figcaption>
  </figure>

  <div class="col">
    <p>Each robot's post-communication feature then goes through a small MLP producing five logits —
    stay, up, down, left, right. Weights are shared across robots, which is precisely why the trained
    policy runs on any number of them.</p>
  </div>
</section>

<section>
  <p class="kicker">The method · 3</p>
  <h2>Learning from a planner that cheats</h2>
  <div class="col">
    <p class="lead">The teacher is Conflict-Based Search: a two-level optimal solver. Every robot
    plans its own shortest path with A*, ignoring the others; the high level then looks for
    collisions and, when it finds one, branches into two subproblems — one forbidding each robot
    from that cell at that time — and replans.</p>
    <p>It only pays for coordination where coordination is needed. In an empty warehouse it is barely
    more than n independent A* searches; in a tight corridor the tree explodes, which is the
    NP-hardness showing up.</p>
    <p>Its solutions become demonstrations. Each solved instance is replayed through the simulator to
    record, at every timestep, what each robot saw, who it could talk to, and what the expert did —
    then the policy is fit by cross-entropy to that action.</p>
  </div>

  <figure>
    <svg viewBox="0 0 720 210" role="img" class="chart"
         aria-label="Instances are solved by conflict-based search, replayed to record observations,
         graphs and expert actions, and used to fit the policy by imitation.">
      <defs>
        <marker id="a3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <polygon points="0,0 10,5 0,10" fill="var(--axis)"/>
        </marker>
      </defs>
      <rect x="0" y="46" width="140" height="66" class="dbox" rx="6"/>
      <text x="70" y="72" text-anchor="middle" class="dlabel" fill="var(--ink)">random instance</text>
      <text x="70" y="92" text-anchor="middle" class="dlabel-mono">starts · goals · walls</text>

      <line x1="146" y1="79" x2="188" y2="79" stroke="var(--axis)" stroke-width="1.6" marker-end="url(#a3)"/>

      <rect x="194" y="46" width="140" height="66" class="dbox" rx="6"/>
      <text x="264" y="72" text-anchor="middle" class="dlabel" fill="var(--ink)">CBS solves it</text>
      <text x="264" y="92" text-anchor="middle" class="dlabel-mono">optimal, collision-free</text>

      <line x1="340" y1="79" x2="382" y2="79" stroke="var(--axis)" stroke-width="1.6" marker-end="url(#a3)"/>

      <rect x="388" y="46" width="150" height="66" class="dbox" rx="6"/>
      <text x="463" y="72" text-anchor="middle" class="dlabel" fill="var(--ink)">replay &amp; record</text>
      <text x="463" y="92" text-anchor="middle" class="dlabel-mono">patch · graph · action</text>

      <line x1="544" y1="79" x2="586" y2="79" stroke="var(--axis)" stroke-width="1.6" marker-end="url(#a3)"/>

      <rect x="592" y="46" width="128" height="66" rx="6" fill="none" stroke="var(--s1)" stroke-width="2"/>
      <text x="656" y="72" text-anchor="middle" class="dlabel" fill="var(--s1)">policy</text>
      <text x="656" y="92" text-anchor="middle" class="dlabel-mono">cross-entropy</text>

      <text x="0" y="150" class="dlabel-mono">ONE SAMPLE</text>
      <text x="0" y="174" class="dlabel">observation (N × 2 × 5 × 5)  ·  graph (N × N)  ·  expert action (N)</text>
      <text x="0" y="194" class="dlabel">— the graph is an input, so a sample is valid for any N</text>
    </svg>
    <figcaption>The expert is centralised and offline; the student is decentralised and online. The
    student never sees the global state — only what it would have at deployment.</figcaption>
  </figure>

  <div class="callout col">
    <p><strong>The honest limitation of behaviour cloning:</strong> the policy only ever observes
    states an <em>optimal</em> planner visits. Once it errs into a state CBS would never produce, it
    has no training signal for recovering — which is exactly what a deadlock is.</p>
  </div>
</section>

<section>
  <p class="kicker">The framework</p>
  <h2>A training loop where the expensive parts are cheap</h2>
  <div class="col">
    <p class="lead">Two things dominate a project like this: generating demonstrations, and
    evaluating what you trained. Both are worth engineering, because both are how fast you can ask a
    question.</p>
    <p>On the data side, the high-level search originally replanned <em>every</em> robot at each node
    of the tree. Only the one that just received a constraint needs it — the rest of the parent's
    paths are still valid. Conflict detection then went from comparing every pair at every timestep
    to bucketing robots by cell.</p>
  </div>

  <figure>
    {cbs_chart}
    <figcaption>Mean seconds per instance, 20 robots on a 14×14 board. Solve and fail outcomes are
    identical at every stage — this is the same search, not a cheaper approximation. Instances are
    independent, so a process pool takes it a further 4.8× on 23 workers.</figcaption>
  </figure>

  <h3>Evaluation runs on the GPU, all episodes at once</h3>
  <div class="col">
    <p>The obvious way to evaluate steps one episode at a time, which makes every step a batch-1
    forward pass. But this model costs <strong>0.65 ms at batch 1 and 0.61 ms at batch 128</strong> —
    that is kernel launch latency, not arithmetic. The fix isn't a bigger card; it's giving the card
    something to do.</p>
    <p>So the simulator was rewritten to carry an episode dimension with every tensor on the device —
    same action encoding, same collision resolution order, same field of view, same graph — and
    hundreds of episodes now advance in a single forward pass.</p>
  </div>

  {table(
      ["episodes", "one at a time", "batched", "speedup"],
      [[e, f"{s:.2f}s", f"{b:.2f}s", f"{s/b:.0f}×"] for e, s, b in zip(
          DATA["rollout_speedup"]["episodes"],
          DATA["rollout_speedup"]["sequential_s"],
          DATA["rollout_speedup"]["batched_s"])],
      "Rollout throughput — both paths produce identical results")}

  <div class="stats">
    <div class="stat"><div class="value">0.22s</div><div class="label">A thousand evaluation episodes</div></div>
    <div class="stat"><div class="value">12.4s</div><div class="label">Generating 1,153 solved instances</div></div>
    <div class="stat"><div class="value">0.007s</div><div class="label">Median instance for the planner — the mean of 0.46s is a 5% tail</div></div>
  </div>

  <div class="col">
    <p style="margin-top:22px">Cheap evaluation changes what you are willing to ask. A sweep across
    five fleet sizes, three architectures and 200 episodes each is three thousand episodes — under a
    second of compute, so the question gets asked instead of estimated.</p>
  </div>
</section>

<section>
  <p class="kicker">What it buys</p>
  <h2>Trained on five robots, deployed on a hundred</h2>
  <div class="col">
    <p class="lead">Because every weight is shared across robots and every observation is local,
    the fleet size is a property of the input, not the network. The board grows with the fleet below,
    holding robots-per-cell fixed — otherwise "more robots" and "more crowding" cannot be separated.</p>
  </div>
  <figure>
    <ul class="legend">
      <li><span class="swatch" style="background:var(--s1)"></span>GNN, message passing</li>
      <li><span class="swatch" style="background:var(--s2)"></span>No communication</li>
      <li><span class="swatch" style="background:var(--s3)"></span>GCN, K=3</li>
    </ul>
    {transfer_chart}
    <figcaption>Fraction of robots reaching their goal, 200 episodes per point. Communication is
    worth 6.6 points at ten robots and 13.2 at a hundred — the benefit is not fixed, it grows with
    the fleet.</figcaption>
    {table(
        ["robots", "board", "neighbours", "No comms", "GCN K=3", "GNN message"],
        [[a, f"{b}×{b}", f"{n:.2f}",
          f'{md["series"]["No communication"][i]:.3f}',
          f'{md["series"]["GCN K=3"][i]:.3f}',
          f'{md["series"]["GNN message K=3"][i]:.3f}']
         for i, (a, b, n) in enumerate(zip(md["agents"], md["board"], md["neighbours"]))],
        "Show the numbers")}
  </figure>

  <h3>The expert does not need to be big</h3>
  <div class="col">
    <p>This matters more than it first appears. The optimal planner is only tractable for small
    teams, so if the policy needed hundred-robot demonstrations, the whole approach would be capped
    by the thing it was meant to replace.</p>
    <p>It doesn't. Doubling the expert team size — and separately quadrupling the training density —
    lands within about three points everywhere, and the cheapest training set is the best at a
    hundred robots.</p>
  </div>
  <figure>
    <ul class="legend">
      <li><span class="swatch" style="background:var(--s1)"></span>expert used 5 robots</li>
      <li><span class="swatch" style="background:var(--s2)"></span>expert used 10</li>
      <li><span class="swatch" style="background:var(--s3)"></span>10 on a dense board</li>
    </ul>
    {fleet_chart}
    <figcaption>Message-passing model, matched-density evaluation. Scaling comes from the
    architecture — shared weights and local observations — not from the demonstrations.</figcaption>
  </figure>
</section>

<section>
  <p class="kicker">Where it goes</p>
  <h2>What this opens up</h2>
  <div class="col">
    <p class="lead">The interesting part of a decentralised planner is not the benchmark number; it
    is that the same trained weights drop onto a fleet of any size, and the compute per robot never
    changes.</p>
  </div>

  <div class="next">
    <article>
      <h4>Close the deadlock gap</h4>
      <p>Beyond about twenty robots, the share of instances where <em>every</em> robot arrives falls
      away — those are genuine jams, not slow progress. Run the policy, collect the states where it
      sticks, solve those with the planner, fold them back into the data. The planner is now fast
      enough for that loop to be routine.</p>
    </article>
    <article>
      <h4>Let robots choose what to say</h4>
      <p>Today a robot averages its neighbours with fixed weights. Attention over the graph lets it
      weight the neighbour that actually matters — the one heading for the same corridor — which is
      where the message-passing variant already shows an edge as fleets grow.</p>
    </article>
    <article>
      <h4>Spend the communication budget wisely</h4>
      <p>K is a dial between context and radio traffic. On real hardware the interesting question is
      the cheapest K that holds performance at a given density, and whether a robot can decide for
      itself when a round is worth sending.</p>
    </article>
    <article>
      <h4>Leave the grid</h4>
      <p>Nothing in the communication layer assumes a grid — only that robots have local
      observations and a neighbourhood. Continuous positions, non-holonomic motion, and heterogeneous
      teams with different sensors are changes to the encoder, not to the method.</p>
    </article>
    <article>
      <h4>Imperfect communication</h4>
      <p>The graph is currently exact and instantaneous. Dropped packets, latency, and limited
      bandwidth are the difference between a simulation result and a deployable one — and the graph
      being an input makes them straightforward to model.</p>
    </article>
    <article>
      <h4>Warehouse-scale evaluation</h4>
      <p>A thousand episodes take a fifth of a second, so the natural next step is sweeping densities
      and map topologies rather than reporting a single configuration.</p>
    </article>
  </div>

  <figure class="plain" style="margin-top:34px">
    <img src="{gif('compare_12agents.gif')}" alt="Twelve robots on a small board with and without
    communication.">
    <figcaption>Twelve robots on 20×20 — the regime the demonstrations come from, and roughly the
    largest team the optimal planner handles comfortably.</figcaption>
  </figure>
</section>

</div>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
