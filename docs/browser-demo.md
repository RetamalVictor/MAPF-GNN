# Browser demo — analysis and proposal

Target: an interactive demo on victor-retamal.com that runs entirely client-side
and conveys what this project actually is.

---

## What the site already gives us

This is the important part of the analysis: almost none of the infrastructure
needs inventing.

| Already there | Why it matters |
|---|---|
| Vite + TypeScript, no framework | A demo is a class with a `containerId`; no build work |
| `BlogPost.ts` → `initializeEmbeddedDemos()` | Demos mount by putting `<div id="…">` in the markdown and lazy-importing the module. Seven demos already do this |
| `@webgpu/types` in devDependencies | WebGPU is already a first-class target here |
| `@bittorch/js` — your own WebGPU inference library | There is precedent for hand-written GPU inference on this site, not just a library call |
| `public/assets/models/{depth,pelvis,transformer_new}` | An established place and pattern for shipping weights |
| three.js, existing canvas demos | Rendering is solved ground |

So the question isn't "can this be done" — it's "what runs, and on what."

---

## Sizing the problem

Measured on the actual `gnn_msg_k3_a10b20_oe` checkpoint:

```
parameters            64,117        250 KB float32   125 KB float16
per robot per step    ~181k MACs
```

Which gives, at 30 fps:

| robots | throughput needed |
|---|---|
| 10 | 0.05 GFLOP/s |
| 40 | 0.22 GFLOP/s |
| 100 | 0.54 GFLOP/s |
| 500 | 2.7 GFLOP/s |
| 1000 | 5.4 GFLOP/s |

Two conclusions fall out.

**The model is small enough to hand-write.** 64k parameters across five weight
tensors — three convolutions, two matmuls, one graph filter. This is not a case
for shipping a runtime; it's five kernels.

**Plain JavaScript covers the honest range, and WebGPU covers the impressive
one.** Scalar JS manages a few hundred MFLOP/s, so it handles the 10–40 robot
regime the training data comes from. Getting to *hundreds* of robots — the regime
that demonstrates the project's actual claim — needs the GPU.

One thing worth checking before assuming otherwise: the environment's adjacency
computation is O(N²), which sounds like it would dominate. It doesn't. Policy
cost is 181,000·N against N² for the graph, so the network dominates until
N ≈ 181,000. The policy is the workload; the environment is rounding error.

---

## What the demo should let a visitor do

The complexity worth conveying is not "a neural network runs in your browser" —
it's that a policy trained on ten robots coordinates hundreds, and that two
mechanisms make that work. Three interactions carry it:

**1. Communication on / off, side by side, same instance.**
The core claim, visible directly. Two panels, identical starts, one policy with
the graph and one without.

**2. A fleet-size slider, 10 → 500.**
At 20 robots the panels look the same. Push it past 100 and the
no-communication side collapses. The reader *finds* the result rather than
reading it in a table.

**3. A temperature slider.**
The most valuable second of the whole demo: drag to zero and watch the fleet
freeze into a deadlock that never resolves; drag it up and watch the same jam
dissolve. That is the fixed-point finding, made tangible in a way no chart
manages. It also invites the reader to overshoot — at T=5 the fleet wanders
uselessly, which teaches that the noise is breaking symmetry, not solving the
problem.

Secondary, if cheap: click to drop obstacles, a "deadlock preset" button that
loads a known-jamming instance, and a live stat line (robots home, steps,
fps, mean neighbours).

---

## Three technical routes

### A. Hand-written WGSL — recommended

Weights exported as a flat fp16 binary; five compute shaders for the forward
pass; the environment step in WGSL too, so state never leaves the GPU. Render
with instanced quads on the same device.

- **For:** matches the `@bittorch/js` precedent exactly. Total bundle is the
  125 KB of weights plus a few KB of shaders. Scales to the fleet sizes that
  make the point. And it repeats the repository's own central lesson — the big
  win there was moving the whole batched environment onto the GPU and deleting
  the per-agent loops; the demo would be that same architecture, ported.
- **Against:** the most work, and needs a numerical parity check against PyTorch.

### B. ONNX Runtime Web

Export to ONNX, run with the WASM or WebGPU execution provider.

- **For:** fastest route to something running.
- **Against:** 2–10 MB of runtime for a 250 KB model; the environment still has
  to be written by hand regardless, which is most of the work; and dynamic agent
  counts through ONNX are fiddly. The saving is smaller than it first looks.

### C. Pure TypeScript, no GPU

- **For:** works everywhere, no fallback needed, trivial to debug.
- **Against:** caps around 40–100 robots, which is exactly where the interesting
  behaviour starts.

**Recommendation: A, with C as the fallback path** rather than a separate
project — the TS implementation is worth writing first anyway, as the reference
the WGSL is checked against.

---

## Plan

**Phase 1 — the interaction, in TypeScript (about a day).**
Port `VectorizedGraphEnv` to TS, implement the forward pass with plain loops,
render on Canvas2D, wire the three controls. Capped at ~40 robots. This settles
the interaction design and doubles as the correctness reference.

**Phase 2 — WebGPU (about two days).**
Move the policy and the environment into WGSL, keep state in storage buffers,
render with instanced draws. Target 500 robots at 60 fps. Feature-detect and
fall back to Phase 1's path.

**Phase 3 — polish.**
Deadlock preset, stats overlay, reduced-motion and mobile handling, the
"under the hood" panel the ternary demo already does well.

Ship after Phase 1 if time is short: a 40-robot demo with a temperature slider
already carries the best finding in the project.

---

## Risks, and what to do about them

| Risk | Mitigation |
|---|---|
| **WGSL/PyTorch numerical drift** — the demo behaves unlike the reported numbers | Export a fixed `(observation, graph) → logits` pair from PyTorch as a test fixture; assert the browser matches within tolerance. Non-negotiable, and cheap |
| **The environment silently diverging from the repo's** | Port from `env_vectorized.py`, which is already pinned to `GraphEnv` by an equivalence test. Reuse the same test vectors |
| WebGPU unavailable | Feature-detect, fall back to the TS path with a smaller default fleet and a one-line notice |
| Demo contradicts the post's numbers | Ship the same checkpoint and the same defaults (T=3, horizon 3× board). If the demo says something different from the article, the demo is the bug |
| Bundle bloat | Lazy-import as the existing demos do; weights fetched on demand, not bundled |

---

## What to export from this repo

A small script producing, into `public/assets/models/mapf/`:

- `weights.bin` — the five tensors, fp16, in a fixed documented order
- `config.json` — channel sizes, K, sensing range, FOV, action encoding
- `fixture.json` — one `(observation, graph, logits)` triple for the parity test

That is the entire interface between the two repositories.
