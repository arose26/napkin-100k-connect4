# napkin-100k-connect4

Train a neural network by self-play on a laptop, quantise it into a **single source
file of at most 100,000 bytes**, and climb a live public bot ladder with it — no
hand-written position evaluation anywhere in the agent.

This is the second instance of that question. The first is
[napkin-100k-tictac](https://github.com/arose26/napkin-100k-tictac), which put a
self-play net at **global rank 1,017 of 10,071** in CodinGame's Ultimate Tic-Tac-Toe
arena (Gold league). The 100,000-byte cap turned out to be **platform-wide** —
verified on seven arenas — so the toolchain ports and only the environment changes.
Hence one repo per game. This one is CodinGame's
[Connect 4](https://www.codingame.com/multiplayer/bot-programming/connect-4).

**Status: submitted 2026-08-20, first ladder reading is provisional and BAD** — see
*First contact with the ladder*. The engine, the GPU self-play, and the packer are
verified (below). A first training run is complete — **4,643,317 self-play games in 88 minutes
on one laptop GPU** — and its headline result is a *negative* one: six times the
self-play bought about 0.085 of score, and my registered prediction (H4) was not met.
The net is now on the arena and the early number does not flatter it.

## Why this game is not "just connect 4"

Two things make the venue's variant its own problem:

- The board is **7 rows × 9 columns**, not 7×6. Sixty-three cells, 126 possible
  four-in-a-rows, and a wider board than the solved-by-1988 standard game.
- **The STEAL.** The second player's first action may be `-2` instead of a column.
  It places nothing; it repaints the first player's single chip as its own. That is
  the pie rule, and it means the opening move is a *bid*. The action space is nine
  columns plus one, and the second player's first decision is whether the first
  player's opening was too good.

Both league levels ship byte-identical statements, so there is one ruleset from
Wood upward.

## The agent

```
       305 inputs                  trunk                heads
  ┌──────────────────────┐   ┌──────────────┐   ┌──────────────────┐
  │ 4 × 63 cell planes   │   │ 305 → 160    │   │ policy → 10      │  move ordering
  │ 5 × 9  column blocks │ → │ 160 → 112    │ → │ value  → 1 tanh  │  leaf scoring
  │ 8 scalars            │   │ ReLU, ReLU   │   └──────────────────┘
  └──────────────────────┘   └──────────────┘
        67,952 int8 weights → base85 → 98,457 bytes of C++
```

Self-play generates everything. Both training targets come from the net's own play:

- **policy target** — softmax over Q from an exhaustive shallow search whose leaves
  are the net's own value head, with exact terminals
- **value target** — the actual outcome of the self-play game

At inference the packed C++ runs iterative-deepening negamax with alpha-beta, where
the **leaf score is the value head** and the **move ordering is the policy head**.
The only non-net values in the search are exact terminals. There is no hand-tuned
evaluation function.

### Two-ply policy targets

The tic-tac-toe repo improved its policy one ply deep: play all 81 actions, score
the children. Here the action space is 10, so **all 100 action pairs** fit in the
same GPU batch, and the target becomes exact minimax — *my* move, *their* best
reply, net value at the leaf. That matters because Connect 4 is decided by
one-move-deep tactics ("that drop hands them the cell directly above"), and a
depth-1 target is systematically blind to exactly that.

Measured, at equal games: value loss **0.763 → 0.675**, at 1.55× the cost per game.
Eighty-one actions made this unaffordable in the previous repo; nine make it cheap.

### Derived input features

The encoder feeds threat squares, per-column win/block/trap flags, column heights
and threat parity alongside raw occupancy. These are **inputs, not an evaluation** —
the net still decides what they are worth. The previous repo measured that adding
derived features was the only change that moved its value loss (0.90 → 0.826); this
one is built that way from the start, in Connect 4's own idiom.

## Everything is on the GPU, including the game

The standing rule from the previous repo, learned the expensive way: putting only
the *network* on the GPU bought nothing, because the bottleneck was Python stepping
one game at a time. So the environment is batched tensors too — occupancy planes,
legality, win detection, and every derived feature are matrix products against a
126×63 line-membership matrix.

| | measured |
|---|---|
| reference engine (Python, single game) | 567,657 plies/s |
| tensor engine + 2-ply expansion, batch 2048 | 39,166 game-plies/s |
| full training loop (self-play + 48 updates/iter) | ~840 games/s |

## Verification, before any claim about strength

An engine that is subtly not the venue's engine invalidates every number
downstream, so this comes first and is reproducible.

| gate | result |
|---|---|
| `fuzz` — vs a clean-room second engine | 50,000 games, **1,185,095 plies, 0 divergences** |
| **vs the official Java referee**, headless | 320 games, **0 parity mismatches** |
| … of which exercised the STEAL | 142 |
| `gpu-parity --check-encode` — tensor vs reference | 512 games: legal sets, outcomes, **all 305 features** identical |
| `gpu-parity` — outcomes only, larger | 2,048 games, 0 divergences |
| `check-pack` — emitted C++ vs independent numpy int8 | 137 positions, max value drift **1e-6**, policy argmax **137/137** |
| `check-bot` — the packed binary | 102 moves all legal, forced wins **16/16** |
| `pack` | **98,458 bytes**, 1,542 under the cap, blob round trip bit-exact |

The second engine is deliberately written in the referee's own idiom — a 7×9
character grid, a column scanned downward, win detection by walking outward from the
changed cell — so it shares no representation and no line table with the bitboard
version. Two implementations agreeing on a million plies is evidence; one
implementation agreeing with itself is not.

### Budget arithmetic (measured, not estimated)

```
67,952 int8 weights × 1.25 chars    = 84,940
base85 chunk quoting, 170 × 5 chars       850
282 float biases as decimal text        3,327
C++ inference + search + encoder        9,340
                                       ------
                                       98,457  of 100,000
```

Trunk width is a budget decision, not a taste one. 160→112 is the widest that packs
under the cap with this harness — established by emitting at six widths, where
160→120 overshoots by 266 bytes.

## Registered hypotheses

Written before the measurements they refer to, per the series protocol. Open ones
stay open in this file until they are answered, including if the answer is no.

**H1 — venue parity.** The offline engine reproduces the referee exactly, STEAL
included. *Answered: 320 games against the official Java referee, 0 mismatches, 142
of them using STEAL; plus 1,185,095 plies against a clean-room engine.*

**H2 — the packer is faithful.** The emitted C++ forward pass matches an independent
int8 reference implementation. *Answered: max value drift 1e-6 over 129 positions,
policy argmax 129/129.*

**H3 — two-ply targets beat one-ply.** Exact minimax over 100 action pairs produces a
better value signal than one-ply children, at a cost the GPU absorbs. *Answered for
the value signal: 0.763 → 0.675 at equal games. NOT yet answered for playing
strength — that needs the past-self measurement below.*

**H4 — does more self-play still buy strength?** Prediction registered before
looking: the net will beat its own 1M-game checkpoint at better than 0.60 after
another 3M games. **Answered: prediction NOT MET, though I registered it badly.** The
final 4.64M-game net scores 0.585 (k=4) / 0.549 (k=8) against its 0.77M-game self and
is indistinguishable from its 4.02M self. The threshold was registered without fixing
the measurement protocol, and the verdict turns out to depend on it. See *The plateau*.

**H5 — the ladder, and only the ladder, is the verdict.** Prediction registered
before submission: the packed net clears Wood and Bronze on placement and finishes
above the median of the arena's ranked bots. **Open — nothing has been submitted.**

**H6 — does the net learn to steal?** Prediction registered before looking: the
trained net will use STEAL in a clear majority of games where it is offered.
**Answered: YES, and more sharply than predicted** — 80% after a random opening ply,
100% after its own preferred opening. See *What the net learned about the pie rule*.


## First contact with the ladder

Submitted 2026-08-20 (test session 41168163). Before submitting, the bot was run
through CodinGame's own sandbox endpoint, which compiles and plays a real game
**without entering the arena** — the pre-submission check this README promised:

| sandbox check | result |
|---|---|
| compiles on CG's g++ 11.2 with the AVX2 `target` pragma | **yes, no compilation error** |
| runs without faulting (i.e. AVX2 really is present on the judges) | **yes, 12 frames** |
| result of the test game | **won 10–0** |
| value head at the empty board, on the judge | `v=0.281395` |
| value head at the empty board, on this laptop | `v=0.281395` |

That last pair is the useful one: the packed forward pass is numerically identical on
the venue's hardware, so the quantisation and the base85 decode survive the trip.

**First ladder snapshot (provisional — placement may still be settling):**

```
arena connect-4     926 ranked bots
Napkin100k          global rank 669, score 22.43, league 0 (Wood)
top of the ladder   RoboStac / _Royale / MrSubZero, score 46.07
```

Rank 669 of 926 is **below the median**, and still in Wood. H5's registered
prediction was "clears Wood and Bronze on placement and finishes above the median."
On this reading it is heading for falsified, which is recorded here now rather than
after the fact. The league split is 259 bots in Wood and 668 above it, so global 669
is the top of Wood — a promotion may still be pending. **Not resolving H5 until the
score stops moving.**

### The pragma is worth 5.4× and the judge is 3.9× slower

CodinGame compiles C++ at `-O0` unless the source says otherwise, so this matters.
Same source, same position, one mid-game turn at the 85 ms budget:

| build | depth reached | nodes |
|---|---|---|
| `-O0`, pragmas stripped — *what a naive submission gets* | 5 | 2,144 |
| `-O0`, with the pragmas — **what this bot ships as** | **7** | **11,616** |
| `-O3`, pragmas stripped | 7 | 11,360 |

The `optimize` pragma recovers full `-O3` behaviour under the venue's `-O0` default:
**5.4× the nodes and two extra plies**, for four lines of source.

And the judge's own speed, measured on the *identical* empty-board position at the
same 900 ms budget:

| | depth | nodes |
|---|---|---|
| this laptop (Core 5 210H) | 6 | 10,048 |
| the CodinGame judge | 5 | 2,592 |

**~3.9× slower**, so plan for one ply less than local benchmarking suggests. (The
previous repo measured ~1.6× for a smaller net; this is worse, and worth carrying.)

### A misreading I corrected before it became a conclusion

The sandbox stderr showed `d=3 e=224` on mid-game turns, and my first reading was that
the judge was ~50× slower than this laptop. It is not. The root search loop contains

```c
if(bv>=1.f) break;    /* a win is proven; stop searching */
```

so a small node count means the bot **found a forced win quickly**, not that it
searched slowly. The 3.9× figure above comes from comparing the *same* position on
both machines, which is the only comparison that was ever going to be meaningful.

## The plateau (H4) — and a prediction I registered badly

One run, 6,000 iterations, **4,643,317 self-play games in 5,253 s** (884 games/s
end-to-end, including 288,000 optimiser steps).

### First, is the instrument trustworthy?

A flat reading is worthless if the measurement is blind, so the protocol is validated
before it is used. Paired 8-ply or 4-ply random openings, both sides at identical
search depth, 2,048 games:

| control | k=4 | k=8 | what it establishes |
|---|---|---|---|
| **the same net on both seats** | **0.5000** (0.478–0.522) | **0.5000** (0.478–0.522) | pairing is exactly unbiased; seat advantage cancels |
| same net, one side crippled to depth-1 search | 0.736 (0.716–0.754) | 0.661 (0.640–0.681) | a known handicap is resolved clearly |
| final vs a randomly initialised net | — | 0.906 (0.887–0.923) | coarse resolution |
| final vs the 10,500-game smoke net | — | 0.750 (0.723–0.776) | coarse resolution |

Two things follow. The protocol is sound — an identical net against itself reads
0.5000 to four places. And **the deeper opening book compresses differences**: every
gap is smaller at k=8 than at k=4. So k=8 *understates* strength differences, and
**k=4 is the right setting for past-self comparisons.**

### The curve

| final net (4.64M games) vs its own self at | k=4 | k=8 |
|---|---|---|
| 0.77M games | **0.585** (0.563–0.606) | 0.549 (0.519–0.580) |
| 1.55M games | 0.516 (0.494–0.537) | 0.517 (0.486–0.547) |
| 2.33M games | 0.540 (0.518–0.561) | 0.505 (0.475–0.536) |
| 4.02M games | 0.498 (0.476–0.519) | 0.511 (0.481–0.542) |

And the era when learning was supposedly still happening, 0.77M → 1.55M games, is
itself only **0.548** (0.526–0.570) at k=4.

Three honest readings:

1. **Total progress over the run is small.** From 0.77M to 4.64M games — a 6× increase
   in self-play — the net gains about 0.085 of score. The last 0.6M games gain nothing
   (0.498, interval spanning 0.5).
2. **The intermediate ordering is non-monotonic.** The final net beats its 2.33M self
   more clearly (0.540) than its 1.55M self (0.516). A strictly improving lineage
   cannot do that. This is either noise at the floor or genuine non-transitivity
   between checkpoints, which self-play produces routinely; either way it is a warning
   against reading a single past-self number as "progress".
3. **I registered H4 badly.** The prediction was ">0.60 versus the 1M-game checkpoint"
   with *no protocol fixed*. The point estimate is below 0.60 under both settings, but
   only the k=8 interval excludes 0.60 — at k=4 the interval reaches 0.606. So: the
   prediction is **not met**, and it is decisively falsified only under the protocol
   that compresses differences most. A registered threshold without a registered
   measurement is a half-registered hypothesis, and that is my error, recorded here
   rather than resolved in my favour.

The scripted yardstick agrees by saying nothing: 0.866 (0.844–0.886) at 1M games
versus 0.881 (0.860–0.900) at 4.64M — overlapping intervals.

### Why it plateaued — the policy head is already converged

The policy loss sat at ~1.94 for the entire run, which looks stuck. It is not. The
cross-entropy floor is the **target's own entropy**, and measured on the final net:

```
target entropy       1.9400   <- cross-entropy cannot go below this
achieved cross-ent   1.9880
excess (KL)          0.0479   <- all that training can still reduce
uniform over 10      2.3026
```

The policy head has fit its teacher to within 0.048 nats. The teacher itself is
blurry: at `tau=0.5` over Q values bounded in [-1, 1] the softmax spread is at most 4
logits, so the target sits close to uniform over 10 actions by construction. **The
net is faithfully imitating a deliberately soft teacher, and more games cannot fix
that.**

Scope this claim carefully. It explains why the *policy head* stopped improving; it
does not by itself explain the strength plateau, because the deployed search uses the
policy only for **move ordering** and the **value head** for leaf scoring. A blurry
policy costs search depth, not move choice. The value loss (0.6155, from 0.6510 at 1M
games) is the other half and is not explained by this measurement. Target temperature
is therefore the *next experiment*, not the established cause.

## What the net learned about the pie rule

Nothing here is coded. These are the net's own 2-ply valuations at the empty board,
with the opponent's steal already priced into the lookahead:

```
opening column:    0      1      2      3      4      5      6      7      8
net's Q:        0.003  0.062  0.022 -0.210 -0.175 -0.221 -0.060  0.014  0.037
                       ^^^^^ picks this        ^^^^^^ centre, and it hates it
```

Standard Connect 4 opening theory says play the centre. This net believes the centre
is the **worst** opening available — and its own steal valuations say why:

| after p0 opens | Q(steal) for p1 | Q(best drop) | net's choice |
|---|---|---|---|
| column 1 (off-centre) | -0.150 | -0.276 | STEAL |
| column 4 (centre) | **+0.017** | -0.538 | STEAL |

Opening in the centre hands the opponent a chip worth stealing. Opening off-centre
hands over much less. The net steals in both cases — correctly, since the steal beats
every drop — so it opens with the *least valuable* move it can. That is pie-rule
strategy, arrived at by self-play from an encoder that was told only which cells are
occupied and which lines are threatened.

Stated as interpretation, not proof: these are the net's own valuations, not ground
truth, and a net that plateaued at this strength may simply be wrong about the
centre. What is verified is the internal consistency — its opening choice and its
steal valuations agree with each other, and its behaviour is what the pie rule
predicts.

## A methodology finding, recorded because it nearly cost me the experiment

The training loop reported "vs-greedy 0.70" for a thousand iterations and it was
**meaningless twice over**:

1. The evaluation randomised the *net's* opening by sampling from its own soft
   policy, while the scripted opponent was never made to play badly. An asymmetric
   handicap: the net threw away roughly 29% of games before playing a real move.
2. Removing the handicap produced a clean **1.000** — so I counted distinct final
   positions. **Five, across 1,024 games.** Two deterministic players play one game
   per seat no matter how large the batch; the opponent's tie-break jitter almost
   never overrode its own win/block rules. "1024 out of 1024" meant "won five
   games."

The fix is to randomise the **position**, not anyone's policy: eight uniformly
random legal plies, with games `2i` and `2i+1` given the *same* opening and
*opposite* seats, so the comparison is paired and seat advantage cancels exactly.
Every score this repo prints now carries the number of distinct final positions
beside it, because a benchmark that has quietly collapsed looks exactly like a
benchmark that works.

Scores are only comparable at a fixed opening depth — a deeper random book more
often hands one side a lost position, and paired seats then contribute 0.5
mechanically. **Eight plies is the standard here.** For the 1M-game checkpoint:

| opponent | opening plies | score (95% Wilson) | distinct final positions |
|---|---|---|---|
| greedy | 0 | 1.000 | **2** ← the bug, reproduced |
| greedy | 4 | 0.962 (0.949–0.972) | 884 |
| greedy | **8** | **0.866** (0.844–0.886) | 938 |
| greedy | 12 | 0.774 (0.747–0.798) | 842 |
| random | 8 | 0.974 (0.962–0.982) | 1008 |

The ceiling of this instrument is not 1.0 and is not yet known, so 0.866 is a
tracking number, not a strength estimate. The carried lesson from the previous repo
stands and got sharper: a single scripted opponent has almost no resolution. Once it
saturates, the only yardstick with headroom is net-versus-past-self, which is why
`train-gpu` snapshots on the way past.

## Honest limitations

- **No ladder result.** Nothing has been submitted. H5 is open.
- ~~The AVX2 target pragma is unverified at the venue.~~ **Resolved:** it compiles
  and runs on the judges, and the packed forward pass is numerically identical there.
  See *First contact with the ladder*.
- **Strength is measured against one scripted opponent and the net's own past
  selves.** Neither is a population of independent opponents, and past-self scores
  cannot detect a whole lineage being stuck in the same local optimum. The ladder is
  the only real test, which is why H5 is the one that counts.
- **`greedy`, `random`, `ab` and `steal` are offline measuring sticks and are never
  submitted.** Standing series rule: every submission is the net. CodinGame leagues
  never demote, so a scripted promotion would permanently raise the net's starting
  floor and eat the climb that is the point.

## Repro

One file is the whole project. Requires PyTorch (CUDA optional), `g++`, and a JDK
for the venue-parity harness.

```bash
python napkin_c4.py selfcheck                       # ~60 asserts on the rules and encoders
python napkin_c4.py fuzz --other blind_engine.py --games 30000
python napkin_c4.py gpu-parity --games 512 --check-encode
python napkin_c4.py bench                           # reference-engine plies/s
```

Venue parity against the official referee. Guice 4.0 needs the module opened on
modern JDKs, and the runner's `GameResult` lives under `runner.simulate`:

```bash
git clone --depth 1 https://github.com/AshKcg/cg-multi-connect4 && cd cg-multi-connect4
mvn -q -B -DskipTests compile dependency:build-classpath -Dmdep.outputFile=cp.txt
CP="target/classes:$(cat cp.txt)"
javac -cp "$CP" -d target/fuzz ../FuzzMain.java
NAPKIN_C4_PY=../napkin_c4.py java --add-opens java.base/java.lang=ALL-UNNAMED \
  -cp "target/fuzz:$CP" FuzzMain 200 1000        # expect: TOTAL parity_mismatch_lines=0
```

Train, pack, verify, measure:

```bash
python napkin_c4.py train-gpu --iters 6000 --depth 2 --batch-games 2048
python napkin_c4.py pack --net out/gpunet.pt --out out/c4_bot.cpp
python napkin_c4.py check-pack --net out/gpunet.pt --cpp out/c4_bot.cpp --games 20
python napkin_c4.py check-bot  --cpp out/c4_bot.cpp --games 12
python napkin_c4.py eval-net --net out/gpunet.pt --vs-net out/gpunet_it01000.pt
```

`out/*.pt`, `out/*.log` and `out/*.cpp` are gitignored — a committed checkpoint
makes a diff binary and unreviewable, and 98 KB of base85 is not a reviewable diff
either. The consequence, stated plainly: **the exact deployed weights are not
recoverable from this repository alone.** Training is seeded, so `train-gpu` with
the same flags reproduces the pipeline, not the byte-identical net. The bot that
actually gets submitted will be committed at that point, because a disclosed bot
whose source nobody can read is not disclosed.

## Disclosure

The bot, when submitted, will carry a comment naming this repository, and the
CodinGame profile will say it is a trained network. One account, no alternates.
Arena ladders only — never a timed contest without a fresh rules review.
