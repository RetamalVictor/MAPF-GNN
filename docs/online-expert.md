# Online expert — implementation proposal

## Why

Behaviour cloning fits the policy to the **expert's** state distribution. At
deployment the policy visits its **own**. The first small error puts it somewhere
CBS would never have gone, where there is no training signal at all, so errors
compound instead of correcting.

In this domain that has a specific shape: a cluster of robots that jams and never
clears. We know these are genuine deadlocks rather than slow episodes, because
raising the evaluation horizon from 42 to 200 steps moves results by under half a
point.

The gap it causes, on the metric the literature actually reports —
success rate α, the fraction of cases where *every* robot reaches its goal:

| robots | Li et al., GNN(OE) K=3 | this repo, no OE |
|---|---|---|
| 20 | ~0.90 | 0.05 – 0.08 |
| 40 | ~0.91 | 0.000 |
| 60 | ~0.94 | 0.000 |

Every result in the paper and in the presentation uses OE. It is not a refinement
on top of the method; it is part of the method.

## The loop

```
  offline CBS demonstrations
            │
            ▼
      ┌──> train policy
      │         │
      │         ▼
      │   roll out N episodes          ← batched GPU env, 1000 episodes in 0.22s
      │         │
      │   collect failures
      │         │
      │         ▼
      │   for each stuck configuration:
      │     starts  := robot positions now
      │     goals   := original goals
      │     solve with CBS             ← ~125 ms, parallel across 23 cores
      │         │
      │         ▼
      │   record + append to dataset
      │         │
      └─────────┘
```

The load-bearing detail is that the expert is queried **from states the policy
actually reached**, not from states the expert would have chosen. That is what
teaches recovery, and it is the only mechanism here that can.

## Design

### Where it lives

```
data_generation/online_expert.py    # collection + relabelling
scripts/train_oe.py                 # the outer loop: train → collect → merge → repeat
tests/test_online_expert.py
```

Nothing in `train.py` changes. The outer loop calls it per round, so a plain
offline run stays exactly as it is today.

### Step 1 — find failures

Run the current policy on the batched environment. An episode is a failure if any
robot is not on its goal at timeout.

Rather than taking the final state, take the state where **progress stopped**:
the earliest timestep after which the set of robots at their goals never grows.
That is closer to the decision that caused the jam than the final frame, and it
avoids relabelling a configuration that is already hopelessly tangled.

Record `(positions, goals, obstacles, timestep)` for each failure.

### Step 2 — turn a stuck state into a CBS instance

Current positions become starts, original goals stay goals, obstacles unchanged.

Two cases need care:

- **Robots already on their goal.** Keep them in the instance — they are moving
  obstacles that may have to step aside. Dropping them produces a solution that
  walks straight through them.
- **Duplicate positions.** Cannot occur; the environment reverts colliding moves.

### Step 3 — solve, and expect some to fail

CBS from a jammed configuration is *harder* than from a random one, because the
robots are packed exactly where conflicts are dense. Budget for a lower solve
rate than the offline generator's 96%.

Mitigations in order of preference:

1. `cbs_time_limit` (already implemented) — drop the tail, keep the throughput.
2. If yield is poor, add a **suboptimality bound** (ECBS-style: accept any
   solution within `w ×` optimal cost). The paper used ECBS for exactly this
   reason. This is the one piece of genuinely new algorithm work, so it is
   deliberately not in the first version.

**Measure the solve rate before building ECBS.** If optimal CBS clears, say, 70%
of failure states inside 2 s, that is plenty and ECBS is unnecessary.

### Step 4 — record and merge

Replay the new solution through the environment exactly as the offline recorder
does, so the samples are identical in form: `(observation, graph, expert action)`
per timestep.

Two policy decisions worth being explicit about:

- **Mixing ratio.** Keep all offline data and append OE data each round. If OE
  data comes to dominate, the policy sees mostly recovery states and may get
  worse at ordinary progress. Cap OE at roughly a third of the set and log the
  ratio every round.
- **Deduplication.** Different failures often jam in similar configurations. Key
  on the sorted `(positions, goals)` tuple and skip repeats, so one common
  failure mode doesn't flood the dataset.

### Step 5 — retrain

Retrain from scratch each round rather than fine-tuning. Rounds are cheap and
fine-tuning on an appended dataset makes the result depend on round order, which
is a reproducibility problem for a small gain.

Three rounds to start.

## What to measure

Per round, so the loop's effect is visible rather than assumed:

| Metric | Why |
|---|---|
| **α at 10 / 20 / 40 / 60 robots** | The headline. Comparable to the paper and the deck. |
| Fraction of robots home | The softer metric reported so far; keep for continuity. |
| Failures collected / solved / merged | Is the loop actually feeding itself? |
| OE share of the dataset | Guards the mixing ratio. |
| Flowtime increase δ_FT | The paper's second metric; worth adding for comparability. |

Stop when α stops improving between rounds, or after five rounds.

## Cost

| Step | Estimate |
|---|---|
| Roll out 1000 episodes | 0.22 s |
| Solve ~200 failure states | ~2 s across 23 workers |
| Record | ~2 s |
| Retrain 50 epochs | ~5 min |
| **One round** | **~5 min, dominated by training** |
| Three rounds × 3 architectures | ~45 min |

Worth noting: the two parts of this loop that would have been expensive — finding
failures and solving them — are precisely the two things already optimised. Before
that work, one round would have been roughly nine minutes of rollouts plus 25 s
per failure state, i.e. over an hour per round instead of seconds.

## Risks

| Risk | Signal | Response |
|---|---|---|
| CBS can't solve jammed states | Solve rate under ~50% | Add ECBS suboptimality bound |
| Policy gets worse at ordinary progress | α up at 40 robots, down at 10 | Lower the OE mixing cap |
| Same failure mode floods the data | Dedup rejection rate very high | Sample failures across episodes, not the worst ones |
| No improvement at all | α flat after two rounds | Check that relabelled samples actually differ from offline data — a bug here looks exactly like "OE doesn't help" |

That last row is the one to watch. A silent no-op in the collection step would
produce a flat curve and a wrong conclusion about the method.

## Order of work

1. Failure collection + stuck-state extraction, with a test that a known jam is
   detected and converted into a valid CBS instance
2. Relabelling and merge, with a test that merged samples have the right shape
   and are not duplicates of offline data
3. The outer loop script, logging the metrics table above
4. Three rounds on `gnn_msg_k3`, measure α
5. Only if yield demands it: ECBS

Step 4 is the decision point. If α at 40 robots moves off 0.000, the approach is
confirmed and it is worth running for the other architectures. If it doesn't,
stop and find out why before spending more compute.
