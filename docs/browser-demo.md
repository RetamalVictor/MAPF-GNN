# Browser demo — proposal, and what the build changed

An interactive demo on victor-retamal.com that runs the trained policy entirely
client-side, so a reader can watch a fleet coordinate, break it, and fix it.

**Built.** The implementation and its reference documentation live in the site
repository, at `retamalvictor.github.io/docs/mapf-demo-spec.md`. That one is
current and self-contained. This document is the reasoning that led to it, kept
because two of its predictions were wrong in ways worth recording.

---

## 1. Why this was smaller than it sounded

The site already carried most of the infrastructure.

| Already present | Consequence |
|---|---|
| Vite + TypeScript, no framework | A demo is a class taking `containerId`. No build work |
| `BlogPost.ts` → `initializeEmbeddedDemos()` | Mount by putting `<div id="mapf-demo">` in the markdown and lazy-importing. Seven demos already followed this |
| `@webgpu/types` in devDependencies | WebGPU was already a target there |
| `@bittorch/js` — a hand-written WebGPU inference library | Precedent for writing kernels rather than calling a runtime |
| `public/assets/models/` | Established location and convention for weights |
| Canvas demos with a theme bridge | Rendering was solved ground |

The work was the environment, the forward pass, and the interaction — not the
plumbing. That held.

## 2. Shipping a runtime would have been the wrong trade

The policy is a handful of tensors: three 3×3 convolutions on a 5×5 patch, a
400→64 encoder, a 192→128 graph filter, and a 128→5 head. Two models together
are 340 KB.

Shipping a 2–10 MB inference runtime to execute that is a poor trade, especially
since the environment has to be hand-written either way — no runtime knows what
a nearest-four communication graph is. So: hand-written TypeScript, with the
option of WGSL later.

One assumption worth checking rather than trusting: the environment's adjacency
is O(N²), which sounds like it would dominate. It doesn't. Policy cost is
~181,000·N against N², so the network dominates until N ≈ 181,000. Confirmed in
the measurements below — cost is near-linear in N through 200 robots.

## 3. Where the proposal was wrong

### Plain JavaScript is much faster than predicted

The estimate was that the TypeScript path would cover 10–40 robots and that
WebGPU was what would reach the hundreds "where the project's claim actually
shows".

Measured, both panels, one policy step each:

| robots | board | ms/step | steps/s |
|---|---|---|---|
| 10 | 20 | 3.0 | 329 |
| 40 | 40 | 11.7 | 86 |
| 100 | 63 | 29.3 | 34 |
| 200 | 89 | 75.4 | 13 |
| 500 | 141 | 181.3 | 5.5 |

100 robots — the fleet size where communication becomes decisive, and therefore
the whole point — runs at 34 steps/s. The demo simulates at 12. WebGPU would
only buy 350 and up, which is past where the demo has anything left to say.

Phase 2 was cancelled rather than deferred, and the three-phase estimate
collapsed to the first phase.

The lesson is the ordinary one: the proposal reasoned about throughput in
GFLOP/s and concluded the GPU was necessary, when the actual question was how
many steps per second a *watchable* demo needs, which is about 12.

### float16 was a false economy

The plan was to ship fp16 to halve the weights. Two reasons not to: folding
BatchNorm into a convolution scales its weights by `1/sqrt(running_var + eps)`,
which can be large and is not known until export; and the parity test wants
headroom well under its 1e-3 tolerance. float32 costs 170 KB extra, which is
nothing, and removes a class of debugging.

## 4. What the build found

Reimplementing the forward pass is a test in itself, and it surfaced a defect
that every training run and benchmark to date had missed: the message-passing
variant's self-term was not local, and not permutation-equivariant. It is in
`findings.md` §5. The short version is that reimplementation forced the question
"does this obey the property the architecture claims?", which no loss curve or
success rate can answer — a policy given extra information gets *better* at both.

That is why the demo ships the GCN for both panels, and why
`scripts/export_web_model.py` refuses the message architecture.

Two smaller things the port had to get right, neither of them in the original
plan:

- The adjacency is **not symmetric** — nearest-four is a per-row rule — and the
  Python multiplies `(F, N) @ (N, N)`, so aggregation is over the **transpose**.
- The fixture instance has to be **crowded**. The first one used 8 robots on
  20×20, where both policies choose identical actions; it would have passed for
  an implementation that ignored the communication graph entirely.

## 5. The export interface

One script, `scripts/export_web_model.py`, writes four files into the site's
asset directory. That is the entire coupling between the repositories.

- **`comm.bin`, `nocomm.bin`** — float32, little-endian, BatchNorm folded,
  matrices pre-transposed so the browser does a plain `x @ W`
- **`config.json`** — per-tensor offsets and shapes, plus the environment
  constants and the defaults that must match the article
- **`fixture.json`** — a deterministic rollout: field of view, adjacency,
  logits, actions and positions at every step, so a parity failure localises to
  a stage instead of just saying the demo is wrong

## 6. What the demo does

Two panels, one instance — identical obstacles, starts, goals *and* per-robot
random streams — differing only in whether the policy may use the graph.

**Fleet size, 10 → 200.** At twenty robots the panels are indistinguishable.
Past a hundred the no-communication side collapses. The reader finds the result
instead of reading it in a table.

**Temperature, 0 → 5.** The most valuable interaction. At zero the fleet freezes
into a deadlock that never resolves however long it runs; raise it and the same
jam dissolves. Overshoot to five and the fleet wanders uselessly — which teaches
the rest of the finding, that the noise breaks symmetry rather than solving
anything.

Colour encodes robot *state*, not identity: two hundred arbitrary hues would be
decoration, whereas moving / arrived / stuck is what makes a deadlock legible.

Out of scope, deliberately: training in the browser, editing the architecture,
custom maps.
