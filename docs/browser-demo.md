# Browser demo — proposal

An interactive demo on victor-retamal.com that runs the trained policy entirely
client-side, so a reader can watch a fleet coordinate, break it, and fix it.

**Recommendation:** hand-written WGSL compute, with a TypeScript path written
first as both the fallback and the correctness reference. Three phases,
shippable after the first.

> **Building it?** The executable spec lives in the site repository, at
> `retamalvictor.github.io/docs/mapf-demo-spec.md`. It is self-contained — model
> shapes, the exact environment semantics, site conventions, phases and tests —
> so an agent working there needs no context from this repo. This document is the
> reasoning behind that spec; that one is the instruction set.

---

## 1. Why this is smaller than it sounds

The site already carries most of the infrastructure.

| Already present | Consequence |
|---|---|
| Vite + TypeScript, no framework | A demo is a class taking `containerId`. No build work |
| `BlogPost.ts` → `initializeEmbeddedDemos()` | Mount by putting `<div id="mapf-demo">` in the markdown and lazy-importing. Seven demos already follow this |
| `@webgpu/types` in devDependencies | WebGPU is already a target here |
| `@bittorch/js` — your own WebGPU inference library | Precedent for hand-written GPU kernels, not just calling a runtime |
| `public/assets/models/{depth,pelvis,transformer_new}` | Established location and convention for weights |
| three.js and several canvas demos | Rendering is solved ground |

The work is the environment, the forward pass, and the interaction — not the
plumbing.

## 2. Sizing

Measured on `gnn_msg_k3_a10b20_oe`:

```
parameters            64,117          250 KB fp32 · 125 KB fp16
per robot per step    ~181k MACs
```

At 30 fps:

| robots | throughput |
|---|---|
| 40 | 0.22 GFLOP/s |
| 100 | 0.54 GFLOP/s |
| 500 | 2.7 GFLOP/s |
| 1000 | 5.4 GFLOP/s |

**The model is five weight tensors.** Three 3×3 convolutions (2→16→16→16 on a
5×5 patch), a 400→64 encoder, a 64×3→128 graph filter with a 64→128 self term,
and a 128→5 policy head. Shipping a 2–10 MB runtime to execute 250 KB of weights
is a poor trade, especially since the environment has to be hand-written either
way.

**Plain JS covers 10–40 robots; the GPU is what reaches the hundreds** where the
project's claim actually shows.

One assumption worth checking rather than trusting: the environment's adjacency
is O(N²), which sounds like it would dominate. It doesn't — policy cost is
181,000·N against N², so the network dominates until N ≈ 181,000. The
environment is rounding error.

## 3. What the demo does

The thing worth conveying is not "a network runs in your browser". It is that a
policy trained on ten robots coordinates hundreds, and that two specific
mechanisms make that work.

### Primary controls

**Communication on / off, same instance, side by side.** Two panels, identical
starts and goals, one policy using the graph and one ignoring it. The core claim,
visible directly.

**Fleet size, 10 → 500.** At twenty robots the panels look the same. Past a
hundred the no-communication side collapses. The reader *finds* the result
instead of reading it in a table.

**Temperature, 0 → 5.** The most valuable interaction in the demo. At zero the
fleet freezes into a deadlock that never resolves however long it runs; raise it
and the same jam dissolves. Overshoot to five and the fleet wanders uselessly —
which teaches the rest of the finding, that the noise breaks symmetry rather than
solving anything.

### Secondary, if cheap

- **Deadlock preset** — loads a known-jamming instance so the frozen state is
  reachable in one click rather than by luck.
- **Live stats** — robots home, step count, mean neighbours, fps.
- **Click to place obstacles.**
- **Under the hood panel** — the ternary demo already does this well; reuse the
  pattern to show the architecture and the parameter count.

### Explicitly out of scope

Training in the browser, editing the architecture, and anything requiring more
than the shipped checkpoint.

## 4. Architecture

### Module layout

```
src/components/mapf/
  index.ts            # export { MapfDemo }
  MapfDemo.ts         # lifecycle, DOM, controls, RAF loop
  types.ts            # Config, WorldState, Weights
  weights.ts          # fetch + parse weights.bin, fp16 → f32
  env.ts              # TS reference environment
  policy.ts           # TS reference forward pass
  render2d.ts         # Canvas2D renderer (fallback path)
  gpu/
    device.ts         # adapter/device init, feature detection
    buffers.ts        # allocation and layout
    pipeline.ts       # shader modules, bind groups, dispatch order
    render.ts         # instanced WebGPU renderer
    shaders/
      fov.wgsl        # board + positions + goals -> N x 2 x 5 x 5
      adjacency.wgsl  # pairwise distance, range threshold, nearest-4
      encoder.wgsl    # conv stack + 400->64
      gnn.wgsl        # normalise, K-hop aggregate, W and W2
      action.wgsl     # 128->5, temperature sample
      step.wgsl       # apply actions, obstacles, bounds, collisions
```

Mounted exactly like the existing demos:

```ts
// BlogPost.ts → initializeEmbeddedDemos()
const mapf = document.getElementById('mapf-demo');
if (mapf) {
  import('../components/mapf').then(({ MapfDemo }) => {
    new MapfDemo({
      containerId: 'mapf-demo',
      modelPath: '/assets/models/mapf',
      defaultAgents: 40,
      defaultTemperature: 3.0,
    });
  });
}
```

### Per-frame pipeline

Both backends run the same six stages; only the implementation differs.

```
positions, goals, board
        │
        ├─ adjacency   distance, threshold at sensing range, keep nearest 4
        ├─ fov         5×5×2 patch per robot, goal projected onto the edge
        │
        ├─ encoder     conv ×3 → flatten 400 → 64                     per robot
        ├─ gnn         D^-½(A+I)D^-½, K hops, W_k and W_self → 128     over graph
        ├─ action      128 → 5 logits → sample at temperature T        per robot
        │
        └─ step        apply, revert into obstacles, clamp, revert collisions
```

State stays in GPU buffers across frames; only the render pass reads it. The
same shape as the batched environment in this repository — the demo is that
architecture, ported.

### Buffer layout (WebGPU path)

| Buffer | Type | Size at N=500 |
|---|---|---|
| `positions`, `prev` | `i32 × N × 2` | 8 KB |
| `goals` | `i32 × N × 2` | 4 KB |
| `board` | `u32 × H × W` | 16 KB @ 64² |
| `adjacency` | `f32 × N × N` | 1 MB |
| `fov` | `f32 × N × 50` | 100 KB |
| `features` | `f32 × N × 64`, `× 128` | 384 KB |
| `logits`, `actions` | `f32 × N × 5`, `u32 × N` | 12 KB |
| `weights` | `f32 × 64,117` | 250 KB |
| `rngState` | `u32 × N` | 2 KB |

Under 2 MB total. No pressure anywhere.

### Two details that need care

**Collision resolution** reverts *every* robot sharing a cell, which is a
scatter with contention. Two-pass: atomically increment an occupancy grid keyed
by cell, then each robot reads its own cell's count and reverts if it exceeds
one. Matches `env_vectorized.py` exactly, including that reverting is not
re-checked for new collisions.

**Sampling** needs per-robot reproducible randomness. A small PCG or xorshift
per robot, seeded once, advanced each step — so a given seed replays identically
and the deadlock preset is stable.

### Fallback

Feature-detect `navigator.gpu`. Without it, run the TypeScript path with a lower
default fleet and a single line explaining the cap. The TS path is written first
regardless, so this costs nothing extra.

## 5. The export interface

One script in this repository writes three files into
`public/assets/models/mapf/`. That is the entire coupling between the repos.

**`weights.bin`** — the five tensors, fp16, little-endian, in this fixed order:

| # | Tensor | Shape |
|---|---|---|
| 1 | `convLayers.0.weight` + bias | (16,2,3,3), (16) |
| 2 | `convLayers.3.weight` + bias | (16,16,3,3), (16) |
| 3 | `convLayers.6.weight` + bias | (16,16,3,3), (16) |
| 4 | `compressMLP.0.weight` + bias | (64,400), (64) |
| 5 | `GFL.0.W`, `GFL.0.W2` | (64,3,128), (64,128) |
| 6 | `actionMLP.0.weight` + bias | (5,128), (5) |

BatchNorm folds into the preceding convolution at export — the demo runs in eval
mode only, so there is no reason to carry four extra tensors and a normalisation
step.

**`config.json`** — channels, K, `sensing_range`, FOV size, pad, action encoding,
board defaults, recommended temperature.

**`fixture.json`** — one `(observation, adjacency) → logits` triple plus a small
rollout: five steps of positions and actions from a fixed seed. This is the
parity test, and it is not optional.

## 6. Plan

### Phase 1 — TypeScript reference (≈1 day)

Delivers a working demo, and the oracle everything else is checked against.

1. Export script and the three asset files
2. `weights.ts` — fetch, parse, fp16 → f32
3. `env.ts` — port `VectorizedGraphEnv`, reusing its test vectors
4. `policy.ts` — forward pass with plain loops
5. Parity test against `fixture.json`
6. `render2d.ts` — board, robots coloured by state, communication edges
7. `MapfDemo.ts` — the three controls, RAF loop, stats line

**Done when:** 40 robots at 30 fps; logits match the fixture within 1e-3; the
temperature slider visibly freezes and unfreezes the fleet.

### Phase 2 — WebGPU (≈2 days)

1. `gpu/device.ts`, `buffers.ts` — init and allocation
2. `adjacency.wgsl`, `fov.wgsl` — environment observation
3. `encoder.wgsl`, `gnn.wgsl`, `action.wgsl` — the forward pass
4. `step.wgsl` — dynamics with the two-pass collision resolution
5. Parity test again, this time GPU against the TS path on identical seeds
6. `gpu/render.ts` — instanced quads for robots and obstacles, lines for edges

**Done when:** 500 robots at 60 fps; GPU and TS paths agree step-for-step at
temperature 0 on the same seed.

### Phase 3 — polish (≈half a day)

Deadlock preset, under-the-hood panel, `prefers-reduced-motion`, mobile sizing,
a smaller default fleet on first paint.

**Ship after Phase 1 if time is short.** A 40-robot demo with a temperature
slider already carries the best finding in the project.

## 7. Testing

| Test | Guards |
|---|---|
| Logits match `fixture.json` within 1e-3 | The forward pass |
| Five-step rollout matches the fixture at T=0 | The environment dynamics |
| GPU path matches TS path, same seed, 100 steps | WGSL drift |
| Adjacency matches a brute-force TS implementation | The nearest-4 rule |
| Demo defaults equal the article's (T=3, horizon 3× board) | The demo disagreeing with the post |

The last one matters more than it looks. If the demo says something different
from the article, a reader will believe the demo — so the demo is the thing that
has to be right.

## 8. Risks

| Risk | Response |
|---|---|
| **WGSL/PyTorch numerical drift** | The fixture test, run in CI. Non-negotiable |
| **Environment diverging from the repo's** | Port from `env_vectorized.py`, which is already pinned to `GraphEnv` by an equivalence test; reuse those vectors |
| WebGPU unavailable | Feature-detect, fall back to the TS path |
| Demo contradicts the post | Ship the same checkpoint and the same defaults |
| Bundle bloat | Lazy import as the existing demos do; weights fetched on demand |
| Scope creep into a research toy | Training, architecture editing and custom maps are out of scope, written down here so it stays that way |

## 9. Estimate

| Phase | Effort | Outcome |
|---|---|---|
| 1 | ~1 day | Shippable 40-robot demo, and the reference implementation |
| 2 | ~2 days | 500 robots at 60 fps |
| 3 | ~½ day | Presets, panel, mobile |

About three and a half days for all of it, with a working demo at the end of the
first.
