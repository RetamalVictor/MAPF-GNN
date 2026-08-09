# Findings

Each entry: what was observed, what it turned out to be, and what was done about
it. Numbers live in `viz/analysis.json`; this file is the reasoning.

---

## 1. Deadlock was a decoding artifact, not a learning failure

**Observed.** Success rate α — every robot home — sat at 0.000 beyond about 20
robots, while the fraction of robots reaching their goal was a healthy 0.75–0.85.
So most robots arrived and a handful never did, permanently.

**First wrong theory.** That the policy was bad at recovering from crowded
states, because behaviour cloning never shows it any. Reasonable, and it led to
building the online expert — which did help, but nowhere near enough.

**What it actually was.** Checking whether failed episodes' joint states repeat:

| | |
|---|---|
| Failed episodes | 159 / 200 |
| **Completely frozen** — one state repeating | **156 (98%)** |
| Short cycles | 3 (2%) |
| Anything else | **0** |

The policy takes an `argmax`; the environment is deterministic. Once the joint
state repeats, it repeats **forever**. Two robots wanting the same cell, running
identical shared weights on symmetric observations, choose the same action,
collide, revert, and do it again for eternity.

This also explains an earlier puzzle: raising the horizon from 42 to 200 steps
moved results by under half a point. Of course it did — no horizon escapes a
fixed point.

**Solution.** Sample from the policy instead of taking the argmax. One line, at
inference, no retraining:

| decoding | α @20 | α @40 | mean steps @40 |
|---|---|---|---|
| argmax | 0.205 | 0.015 | 57 |
| sample T=1 | 0.635 | 0.295 | 72 |
| sample T=2 | 0.960 | 0.880 | 70 |
| **sample T=3** | **0.980** | **0.980** | 71 |
| sample T=5 | 0.695 | 0.020 | 109 |
| **uniform random** | **0.000** | **0.000** | — |

**Why this isn't just "randomness solves it".** Uniform random scores **zero**.
The policy does all the work; the noise only breaks ties. And there is a genuine
optimum — by T=5 the policy's own preference is drowned out and success
collapses. Successful paths get ~25% longer for a 65× higher success rate.

**Caveat.** Every result measured before this was under argmax and understates
the policy.

---

## 2. The online expert helps, and is not a substitute for the above

**Built.** Roll the policy out, take the configuration at the moment progress
stopped, ask CBS what to do from there, merge the answer, retrain. The expert is
queried from states the policy actually reached — the part plain cloning cannot
provide.

**Result over three rounds** (argmax, collection at the 10-robot training fleet):

| round | OE cases | α@10 | α@20 | α@40 |
|---|---|---|---|---|
| 0 | 0 | 0.440 | 0.090 | 0.000 |
| 3 | 383 | 0.600 | 0.205 | 0.015 |

The cleanest evidence it works is independent of α: the failure rate at the
collection fleet fell **56% → 45% → 38%** across rounds. The policy genuinely
jams less in the regime it is corrected in.

**Under sampled decoding**, evaluated at a board-proportional horizon, three
rounds and 117 relabelled cases against 1,181 offline ones:

| robots | before | after | gain |
|---|---|---|---|
| 10 | 0.950 | 0.960 | +1.0 |
| 20 | 0.895 | 0.960 | +6.5 |
| 40 | 0.860 | 0.945 | +8.5 |
| 60 | 0.785 | **0.935** | **+15.0** |
| 100 | 0.695 | 0.755 | +6.0 |

The gain is largest in the middle of the range, where the fleet is dense enough
to jam but the policy is not yet overwhelmed. Collection now uses the same
decoding as deployment — relabelling states the policy never reaches teaches the
wrong thing — so only 51 of 600 rollouts fail rather than 169 of 300, and every
case in the pool is a residual failure rather than a tie-breaking artifact.

**The two interventions compose.** At 40 robots:

| | argmax | sample T=1 |
|---|---|---|
| offline only | 0.000 | 0.040 |
| + online expert | 0.015 | **0.295** |

Neither substitutes for the other, and together they reach 0.880 at T=2.

**Incidental finding.** CBS solves **85%** of real jammed configurations at 55 ms
each. Jammed states are harder than random ones, as expected — but not so much
harder that the ECBS suboptimality bound budgeted for in the plan was needed.

---

## 3. Horizon and decoding interact, and it is a measurement trap

**Observed.** Re-running the transfer comparison at T=3 with the previous
300-step horizon gave α ≈ 0.98–0.995 for *every* architecture including the
non-communicating baseline. The comparison had become meaningless.

**What happened.** Under argmax the horizon is nearly irrelevant, because a fixed
point never resolves however long you wait. Under sampling a generous horizon
becomes a brute-force budget: keep shaking and something eventually works.

**Solution.** Scale the horizon to the board — 3× the board width — so it grows
with the map but never turns into unlimited retries.

Worth stating plainly: two different horizons, both defensible in isolation,
produced opposite conclusions about whether communication matters. The horizon
has to be reported alongside α.

**And it caught me twice.** The online expert loop's own evaluation still had a
hardcoded 300-step horizon, so its first sampled run reported α ≈ 1.0 at every
fleet size from round 0 and appeared to show the online expert doing nothing.
Same trap, second occurrence, in code I had written after identifying it.

---

## 4. What communication is actually worth

With symmetry breaking in place and a fair horizon, matched density, trained on
10 robots:

| robots | board | No comms | GCN K=3 | GNN message |
|---|---|---|---|---|
| 10 | 20×20 | 0.965 | 0.950 | 0.950 |
| 20 | 28×28 | 0.950 | 0.935 | 0.895 |
| 40 | 40×40 | 0.935 | 0.930 | 0.860 |
| 60 | 49×49 | 0.770 | 0.850 | 0.785 |
| **100** | **63×63** | **0.210** | **0.600** | **0.695** |

**Communication buys nothing up to ~40 robots and becomes decisive at 100**,
where the non-communicating policy collapses to 0.21 while communicating ones
hold 0.60–0.70. A reactive policy plus a tie-break is genuinely sufficient at
small scale.

This is a sharper claim than the earlier one. Under argmax, communication looked
uniformly helpful — but that was measuring which architecture entered fewer
fixed points, not which coordinates better.

---

## 5. Scaling comes from the architecture, not the demonstrations

Doubling the expert's fleet size (5 → 10 robots) and separately quadrupling
training density changed results by under three points everywhere, and the
cheapest training set was best at 100 robots.

This matters because the expert is the part that does not scale — CBS is
comfortable to about 20 robots and falls apart by 30. If the policy needed
hundred-robot demonstrations, the approach would be capped by the thing it
replaces. It doesn't: shared weights and local observations are what generalise.

---

## 6. Engineering that changed what was askable

None of this is the method, but all of it decided how many questions could be
asked.

- **CBS: 25.8s → 1.18s per instance (21.8×).** Almost all of it one mistake — the
  high-level search replanned every robot at each node, when only the constrained
  one needs it. Then O(T·N) conflict detection, and not allocating a throwaway
  constraint object per validity check. Identical solve/fail outcomes throughout.
- **Rollouts: 149× via batching episodes on the GPU.** 1000 episodes in 0.22s.
- **Forward pass: 12× at batch 128** by folding the agent axis into the batch
  instead of looping in Python.
- The model costs 0.65 ms at batch 1 and 0.61 ms at batch 128 — **latency, not
  compute**. A CUDA graph gives 5.2× at 100 episodes and 1.1× at 1000, which is
  the whole lesson about when fusion pays.

The online expert loop needs exactly the two things above — finding failures and
solving them. Before this work a round would have taken over an hour; it takes
seconds, and training dominates.

---

## Open

- **Fleet-size bucketing.** Collection is pinned to the training fleet size
  because a sample's tensors are shaped `(agents, …)`. Bucketing batches by agent
  count would let the online expert collect where the policy actually jams.
- **Peer modelling.** Temperature breaks symmetry *randomly*. A model of what a
  neighbour will do next would break it *deliberately* — likely better on path
  quality rather than success rate. A cheap first version is an auxiliary loss
  predicting neighbours' next actions from the aggregated message.
- **Memory.** The policy is purely reactive and cannot represent "I have been
  stuck for six steps". A recurrent node state is the prerequisite for anything
  belief-shaped.
- A **world model** in the usual sense is not the missing piece: the dynamics are
  known and trivial, and all the uncertainty is about other agents.
