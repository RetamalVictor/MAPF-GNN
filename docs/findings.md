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

**One column of that table is not what it appears to be.** See §5.

---

## 5. The message-passing variant was not decentralised

**Observed.** Building the browser demo meant reimplementing the forward pass,
which meant checking a property nothing in training had ever tested: a fleet is
an unordered set, so relabelling the robots must relabel the outputs and change
nothing else.

It does not. Permuting the agents and the adjacency identically, then undoing
the permutation on the output:

| N | GNN message | GCN K=3 |
|---|---|---|
| 6 | 2.48 | 0.000004 |
| 10 | 3.44 | 0.000004 |
| 40 | 5.45 | 0.000002 |
| 100 | 6.31 | 0.000002 |

Logits are on a scale of about 10, so the message model's error is 25–60% — not
a numerical artifact. The GCN's is float noise.

**What it actually was.** One line in `MessagePassingLayer.forward`:

```python
node_feats_not_shared = node_feats.reshape([batch_size, self.n_nodes, self.in_features])
```

`node_feats` arrives as `(B, F, N)`. Turning that into `(B, N, F)` needs a
transpose; `reshape` reinterprets the buffer instead. The two agree only in
trivial cases. Replacing it with `.permute(0, 2, 1)` drives the equivariance
error to **exactly 0.0** at every fleet size.

**Why it matters more than a wrong number.** The self-term is meant to be the
robot's *own* features. What it actually reads is a fixed stride through the
whole fleet's features — so it is a global channel. Perturbing the observation
of a robot that is not connected to anything, on an **empty** adjacency matrix:

| | N=6 | N=10 | N=40 |
|---|---|---|---|
| GNN message: change in agent 0's logits | 0.82 | 0.94 | 0.16 |
| GCN K=3 | **0.000000** | **0.000000** | **0.000000** |

Information reaches a robot from robots it cannot communicate with. That is not
a decentralised policy, and the whole premise — local observations, a bounded
communication radius, shared weights — does not hold for it.

**Consequences.**

- The "GNN message" column in §4 measures something else and should not be read
  as a decentralised result. Its 0.695 at 100 robots is plausibly *helped* by
  the leak, since global information is exactly what a crowded board rewards.
  The **GCN K=3 column stands**, and it still carries the claim: 0.60 against
  0.21 for no communication.
- §2's online-expert numbers were all measured on this checkpoint. The
  before/after comparison is internally consistent — same architecture on both
  sides — so the method's effect is real, but the magnitudes should be
  re-measured on the GCN before being quoted as the method's.
- The browser demo ships the **GCN** for both panels. A demo about
  decentralised control cannot run a policy that secretly reads every robot.

**Solution.** `.reshape` → `.permute(0, 2, 1)`, plus
`tests/test_equivariance.py`, which checks three things the rest of the suite
cannot: that permuting the fleet permutes the output, that the graph layers
alone do so, and that an unconnected robot cannot change another's logits. The
last one fails loudly on the old code.

The weights are the part that cannot be fixed in a line — they were trained
against the defect and need a retrain to mean anything.
`scripts/export_web_model.py` refuses the message architecture with a pointer
here in the meantime.

**Worth noting how it survived.** Every test was a loss curve or a success rate,
and both were healthy — a leak that hands the policy extra information would
*improve* them. What caught it was asking whether the code obeyed a property the
architecture claims, which no aggregate metric can answer.

---

## 6. Most of "communication matters at scale" was a training-data artifact

**Observed.** §4 reported that at 100 robots a non-communicating policy collapses
to 0.21 while a communicating one holds 0.60, and read that as communication
becoming decisive with scale. Two changes to how the policies are *trained*
mostly erase it.

The first is the neighbour cap. The graph joined each robot to its four nearest
neighbours within range — a bandwidth limit inherited from the original design.
But `D^-½(A+I)D^-½` is precisely what makes a varying degree safe to aggregate
over, so the cap was throwing away the thing the architecture exists to handle.
At 100 robots with a radius of 8 it discarded more than half the available
links (8.5 neighbours available, 4.0 kept). It also made a wider radio
pointless: with the cap on, widening swaps near neighbours for far ones rather
than adding any, which is why raising the range at inference *hurt* (0.60 at 4,
0.34 at 14).

The second is obstacle density. Training used eight obstacles on a 20×20 board —
2%, and nothing else, ever.

**Result.** Retraining with the cap removed, a radius of 8, and four obstacle
densities (2/5/10/15%), at 100 robots and matched density:

| policy | training data | α @100 |
|---|---|---|
| no comms | 1,182 cases, 2% only | 0.210 |
| no comms | 1,670 cases, 2% only | 0.300 |
| no comms | 1,937 cases, four densities | **0.790** |
| **comms**, radius 8, uncapped | 1,937 cases, four densities | **0.935** |

The middle row is the control that matters. More data alone buys +0.09; the
density variety buys **+0.49**; communication then adds **+0.145** on top.

**So communication does still help, and by a solid margin — but it is worth a
third of what §4 implied, because §4's baseline was undertrained.** The
non-communicating policy was not collapsing under scale so much as failing on
maps it had never been shown.

**And the old models are worse than "worse" off-distribution.** At 10%
obstacles, both previously shipped policies score exactly **0.000** at 40 and
100 robots. The retrained pair manages 0.39 and 0.10. Anything that lets a
reader vary obstacle density — the browser demo does — was showing a cliff that
belonged to the training set, not to the method.

**Caveat.** The retrained pair changes three things at once (cap, radius,
density curriculum). The volume control above isolates data quantity from
density, but not the cap from the radius; those two are entangled by design,
since a wider radio does nothing while the cap is on.

---

## 7. Scaling comes from the architecture, not the demonstrations

Doubling the expert's fleet size (5 → 10 robots) and separately quadrupling
training density changed results by under three points everywhere, and the
cheapest training set was best at 100 robots.

This matters because the expert is the part that does not scale — CBS is
comfortable to about 20 robots and falls apart by 30. If the policy needed
hundred-robot demonstrations, the approach would be capped by the thing it
replaces. It doesn't: shared weights and local observations are what generalise.

---

## 8. Engineering that changed what was askable

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

- **Retrain the message variant.** The self-term fix (§5) is one line, but the
  weights it was trained with encode the defect. Until it is retrained there is
  no honest message-passing number, and no way to tell how much of its apparent
  advantage over the GCN was the leak.
- **Re-measure the online expert on the GCN.** §2's magnitudes were collected on
  the defective checkpoint.
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
