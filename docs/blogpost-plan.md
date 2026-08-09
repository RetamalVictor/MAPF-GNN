# Blog post plan — tone, diagrams, and the questions to answer

Reference material: the IROS presentation (41 slides), the existing posts on
victor-retamal.com (BitTorch, TinyLM), and the questions asked while building this.

---

## 1. Tone

### The voice, in one line

A builder's log written in first person, which explains a mechanism before it
names it, attaches every number to the conditions that produced it, and asks the
reader real questions rather than rhetorical ones.

### Where it comes from

Two sources already establish it, and they agree.

**The published posts.** The BitNet post opens: *"The first time I ran my CUDA
kernel, it was 20× slower than PyTorch."* Not a claim, not a summary — a specific
moment with a number in it. That is the register.

**The presentation.** The deck builds from the ground up (agents and environments
→ control cycle → multi-agent setup → the mess) and stops twice at slides titled
**"Question time"** to put genuinely open questions to the room. It teaches rather
than announces.

### Rules

| Do | Don't |
|---|---|
| First person singular, past tense: *"I measured", "it turned out"* | The corporate "we" for solo work |
| Open a section with a concrete number or moment | Open with a definition |
| Explain the mechanism, then name it | Name-drop then explain |
| Attach conditions to every number: *"0.72 at 60 robots, 200 episodes, matched density"* | Bare figures with no setup |
| State a limitation the moment it's relevant | Save caveats for a footnote |
| Ask a real question and leave it open | Rhetorical questions with obvious answers |
| Short declaratives for the load-bearing sentences | Stacked subordinate clauses |
| Use the domain's own words: vertex conflict, flowtime, makespan | Generic ML vocabulary that flattens the domain |

### The reader

Someone who knows machine learning but has never touched multi-agent path
finding. They do not need graph convolutions explained from scratch, but they do
need to be told *why a robot's goal is almost never in its field of view* — that
is the domain knowledge that makes the design choices legible.

Corollary: never explain backprop; always explain a swapping conflict.

### Phrases to keep

From the deck, already well put:

- **"Centralised training, decentralised deployment"** — the thesis in four words.
- **"Multi-agent mess"** — the honest name for what happens when you put n robots in one space.
- **Vertex conflict / swapping conflict** — name both; the branching rule follows from them.

### Phrases to avoid

"Leverage", "seamless", "powerful", "state-of-the-art", "revolutionise", and any
sentence that would survive being pasted into a different post about a different
subject.

---

## 2. Diagrams

Numbered by build order. The first four carry the argument; the rest support it.

### Already built (`viz/build_page.py`)

| # | Diagram | What it shows | Status |
|---|---|---|---|
| D1 | **Goal projection** | Goal outside the 5×5 patch, marker pinned to the patch edge; world view beside network view | Done |
| D2 | **K-hop propagation** | A ← B ← C over three panels; K as communication rounds | Done |
| D3 | **Imitation pipeline** | instance → CBS → replay & record → policy, with the sample shape | Done |

### To recreate from the deck

| # | Deck slide | Diagram | Why it earns its place |
|---|---|---|---|
| D4 | 8 | **Vertex conflict vs swapping conflict** | Two small grids, two robots each. This is the *entire* branching rule of CBS — without it, "looks for collisions" is hand-waving. Highest value of anything on this list. |
| D5 | 9 | **Centralised vs decentralised** | Side by side: one brain with every arrow into it, versus n brains with only local arrows. Draw the *difference*, not two labelled boxes. |
| D6 | 11–17 | **The CBS constraint tree** | The deck spends 7 slides here. Compress to one animated or stepped figure: root → conflict found → branch into two constrained subproblems → replan one agent each. Must show that *only the constrained agent replans* — that is both the algorithm and the 22× speed-up. |
| D7 | 20–21 | **Communication as a dynamic graph** | Same fleet at t and t+1 with different edges. The point is that the graph is an *input that changes every step*, not a learned structure. |
| D8 | 37 | **Generalisation matrix** | Heatmap, agents-at-train × agents-at-test. This is the paper's headline claim and the one to reproduce. Reuse the layout; regenerate with our own numbers. |

### New, needed for this post

| # | Diagram | What it shows |
|---|---|---|
| D9 | **The distribution gap** | Two overlapping regions: states the expert visits, states the policy reaches. The non-overlapping part is where cloning has no signal — and where deadlock lives. This is the entire argument for the online expert, and it is one picture. |
| D10 | **The OE loop** | train → roll out → collect failures → query CBS from the stuck state → merge → retrain. A cycle, not a pipeline; the arrow back is the whole idea. |

### Animations already rendered (`viz/out/`)

- `compare_30agents.gif` — the deadlock cluster forming, with and without communication. The hero.
- `compare_12agents.gif` — the training regime.
- `gnn_msg_k3_a10b20_40agents.gif` — a fleet 8× the training size.

Worth adding once OE exists: **the same instance before and after OE**, so the
reader watches a jam that used to persist get resolved.

---

## 3. The questions the post should answer

These came up naturally while building. A reader will have the same ones at the
same moments, which makes them section prompts rather than an FAQ appendix.

**On the method**

1. *Why can't you just run the optimal planner on the robots?* — centralised vs
   decentralised, and the NP-hardness that makes it moot at scale.
2. *If the goal is off-screen, how does the robot know where to go?* — the
   projection trick. The single most illuminating detail in the architecture.
3. *What does K actually mean?* — communication rounds per decision, and what a
   round costs on real hardware.
4. *Can this work with different kinds of robots?* — from the deck's Question
   time. Leave genuinely open.
5. *Is a vanilla GCN the right model here?* — the deck's other open question, and
   the natural setup for attention / MAGAT.

**On the training framework**

6. *How much expert data do you need, and how big does the expert have to be?* —
   measured: the expert stays small, which is what makes the approach viable.
7. *Are we actually using the GPU?* — the answer is more interesting than yes:
   0.65 ms at batch 1 versus 0.61 ms at batch 128 is latency, not compute.
8. *Is there any gain in fusing custom CUDA kernels?* — measured both ways: 5.2×
   when launch-bound, 1.1× when not. The lesson is about *when* to reach for it.
9. *Should the simulator be in JAX, or C++?* — the reasoning about which tool
   fits which bottleneck, and why a CPU rewrite would be a step backwards here.

**On what's missing**

10. **What is the online expert?** — the distribution gap, why cloning cannot fix
    it alone, and what querying the expert on *your own* mistakes buys.
11. *Which metric are we even reporting?* — success rate α (all robots home)
    versus fraction of robots home. They tell very different stories, and the
    literature uses α.

---

## 4. Structure

1. **The mess** — n robots, one space; vertex and swapping conflicts (D4)
2. **Two ways to solve it** — centralised optimality vs decentralised locality (D5)
3. **The teacher** — CBS and its constraint tree (D6)
4. **What one robot knows** — the patch, and the goal projection (D1)
5. **Talking** — the dynamic graph (D7) and K-hop convolution (D2)
6. **Learning to imitate** — the pipeline (D3), and the honest limitation
7. **The framework** — making data generation and evaluation cheap enough to ask questions
8. **Does it scale** — the generalisation matrix (D8)
9. **The gap** — distribution shift (D9), the online expert (D10), before/after
10. **Open questions** — the deck's two, plus communication budget

---

## 5. Notes to self

- Report **α as the headline metric**. The fraction-of-robots number is softer and
  not what the literature quotes; leading with it invites a fair objection.
- Every chart needs its conditions in the caption: episodes, seed, density regime,
  horizon. The horizon in particular changes conclusions.
- The post should say plainly which results are reproductions and which are this
  repository's own measurements.
