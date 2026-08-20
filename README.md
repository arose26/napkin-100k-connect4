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

**Status: not submitted.** The engine, the GPU self-play, and the packer are
verified (below). Training is in progress. There is no ladder number yet, and this
README will not carry one until there is.

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
| `check-pack` — emitted C++ vs independent numpy int8 | 129 positions, max value drift **1e-6**, policy argmax **129/129** |
| `check-bot` — the packed binary | all moves legal, forced wins **12/12** |
| `pack` | **98,457 bytes**, 1,543 under the cap, blob round trip bit-exact |

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

**H4 — does more self-play still buy strength?** Prediction registered now, before
looking: the net will beat its own 1M-game checkpoint at better than 0.60 after
another 3M games. The previous repo's conclusion was that its plateau needed a
different order of training compute rather than another knob; this measures whether
that transfers. **Open.**

**H5 — the ladder, and only the ladder, is the verdict.** Prediction registered
before submission: the packed net clears Wood and Bronze on placement and finishes
above the median of the arena's ranked bots. **Open — nothing has been submitted.**

**H6 — does the net learn to steal?** The pie rule is the one decision here with no
analogue in the previous game. Prediction: the trained net will use STEAL in a
clear majority of games where it is offered, because a random opening book makes the
first move informative. **Open.**

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
- **The AVX2 target pragma is unverified at the venue.** CodinGame compiles C++ at
  `-O0` unless the source says otherwise, so the optimise pragma is mandatory; the
  `target` pragma that lets g++ vectorise the dot products assumes AVX2 on the
  judges. That will be smoke-tested in the sandbox — CodinGame's `TestSession/play`
  endpoint compiles and runs a bot **without** entering the arena — before any
  submission, not after.
- **Strength is measured against one scripted opponent and a 1M-game past self.**
  Neither is a population. Treat all of it as a smoke test.
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
