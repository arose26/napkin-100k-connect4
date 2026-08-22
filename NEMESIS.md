# napkin-nemesis — outline sketch

> **Superseded.** This sketch became its own series:
> https://github.com/arose26/napkin-nemesis (see PLAN.md there). Kept for
> provenance — including the contaminated-run disclosure below.

Status: **not started.** Sketch only, for a fresh session to pick up. Nothing here is
a result. One thing below IS already contaminated and is flagged as such.

## The question

Every napkin experiment so far has maximised strength against a ladder. None has asked
how *fragile* the result is, and we now have a measured reason to care.

Two nets that beat the deployed v2 under our own offline protocol both scored **worse**
on the actual ladder:

| net | offline vs v2 (paired, k=4) | ladder score |
|---|---|---|
| v2 | — | **32.24** |
| v3peak | 0.537 | 31.65 |
| soup (avg of last 10 ckpts) | **0.594** | 31.46 |

Self-play's opponent distribution is exactly one policy — the current net. Getting
better at beating yourself while getting *worse* against 926 other bots is what
self-play overfitting looks like. This experiment tests that directly, in two phases:

- **Phase A (diagnostic).** Train a best response ("nemesis") against a frozen target and
  measure how far it climbs. That number is the target's exploitability.
- **Phase B (the payoff).** If Phase A shows real exploitability, train against a
  **pool** (past selves + nemeses + scripted baselines) and ask whether the result holds
  its offline edge *and* stops losing ladder score.

Phase B is the one that could produce a better competitor. Phase A is the cheap gate
that says whether Phase B is worth building.

## Preregistered hypotheses

Numeric, with 80% intervals so they can be scored later for calibration. The protocol is
fixed **here**, before running — the H4 lesson from this repo was that a threshold
registered without a protocol produces a verdict that depends on the protocol.

**N1 — exploitability magnitude.** A nemesis warm-started from its target and trained
2,500 iterations (~2.5M games) reaches **≥0.75** against that frozen target.
80% interval for the peak: **[0.68, 0.92]**, point estimate 0.82.
⚠️ **CONTAMINATED — see "What I already saw".** Treat N1 as exploratory, not registered.

**N2 — rank does not predict fragility.** Exploitability is *not* monotone in ladder
score across our lineage (v1, v2, v3peak, soup). Specifically: the **soup is the least
exploitable** (weight averaging smooths decision boundaries), and v2 — our best ladder
score — is **not** the least exploitable.
Prediction (directional): peak-nemesis-score(soup) < peak-nemesis-score(v2).
80% interval on the gap `v2 − soup`: **[−0.02, +0.16]**, point estimate +0.06.
The interval deliberately admits negative values — I think the soup is less exploitable
but I am not confident, and an interval that excluded the outcome I consider possible
would be a dishonest one. Scored as: directional hit if the gap is > 0, plus an interval
hit if the observed gap lands inside [−0.02, +0.16]. Clean, nothing observed.

**N3 — the payoff.** A net fine-tuned against a pool beats the deployed net offline
**and** does not lose ladder score, unlike the last two self-play-only nets.
Prediction: offline ≥0.55 vs deployed, and ladder score within −0.2 of it or better.
80% interval on the ladder score delta: **[−0.6, +1.2]**. Clean.

**N4 — sparse reward beats from-scratch best response.** Under an equal budget, a
from-scratch nemesis reaches a lower peak than a warm-started one.
Partly informed by something I already saw (below), so: **weakly held, disclose when
scoring.**

**N5 — the honest null, and what it is NOT allowed to conclude.** If N2 and N3 both
fail, that does **not** establish that self-play overfitting is absent. A null here is
equally consistent with a badly built test: too small a pool, an unlucky pool
composition, a best response that never converged, or noise swamping a real effect. The
inference is only licensed if the tests first pass their own validity checks:

- the null control reads ~0.500 through the training harness (not just through `eval-net`)
- the nemesis actually climbs on *some* target — i.e. the best-response procedure is
  demonstrably capable of finding an exploit
- the pool contains ≥4 genuinely distinct opponents, verified by pairwise scores away
  from 0.500 (a "pool" of near-identical nets is still one policy)
- the observed effect size is larger than the measured noise floor (this repo's is
  ~0.03 for a single 1024-game reading)

If those hold and N2/N3 still fail, *then* the offline/ladder gap needs a different
explanation — the leading candidate being that the population plays strategies absent
from our self-play distribution altogether, which is napkin-mirror, not this experiment.
If they do not hold, the result is "test inconclusive", and saying so is the finding.

## Protocol, fixed in advance

- **Measurement:** `eval-net`, paired random openings (games 2i / 2i+1 share an opening
  with opposite seats), **k=4**, distinct-final-position count reported on every score.
- **Curve:** 1024 games, seed 1, one point per 100-iteration snapshot.
- **Confirmation:** 3 seeds × 2048 games. No claim off a single seed.
- **Null control required** before any comparison: target vs itself must read ~0.500.
- **Do NOT report max-over-snapshots as the peak.** This repo measured a 32-point curve
  whose spread (stdev 0.017) was *entirely* inside single-reading noise (CI half-width
  ~0.030), and I wrongly called one sample a "peak". Report **both** the 3-seed mean at
  the argmax snapshot **and** the mean over the last 5 snapshots.
- **Warm start from the target** for the headline nemesis, so it begins at 0.500 by
  construction and the climb is interpretable.
- Value loss is **not** an outcome measure. This repo measured value loss improving while
  strength decayed.

## Resolve this first — the run I started was broken

I launched Phase A before writing any of this (my error, and the reason this file
exists). It produced four curve points and they are **impossible**:

```
it00100  0.008 (0.004..0.015)
it00200  0.017 (0.010..0.026)
it00300  0.008 (0.004..0.015)
it00400  0.020 (0.013..0.030)
```

A nemesis **warm-started from its own target** must start at ~0.500 against it. Reading
0.008 means the net was destroyed within 100 iterations. That is a bug, not a result.

Candidate causes, cheapest first:

1. **Value-target sign.** `selfcheck` asserts the *recording* invariants of
   `best_response_step()` (only learner plies recorded, right action per seat) but does
   **not** assert the sign of `z` end-to-end in opponent mode. A flipped sign would train
   the net to lose. This is the same bug class that cost the tictac repo two rounds
   ("negating is right in pure self-play but wrong against a fixed opponent"). **Write
   that assert first.**
2. **Collapse from a degenerate reward.** Warm-started from the target, early games are
   ~50/50, but if the buffer fills with mostly-losing games at lr 5e-4 the net may
   diverge. Check by logging the z distribution per iteration.
3. **`plies` / staging desync.** In opponent mode `plies` counts only learner plies while
   `MAXP` still assumes a full game. Verify slot bookkeeping.

Do not interpret any nemesis number until the target-vs-itself null control reads 0.500
**through the same training harness**.

## What exists vs what needs building

Exists and tested:
- `train-gpu --opponent CKPT` (frozen opponent, only learner plies recorded)
- `best_response_step()` + its four selfcheck invariants
- `train-gpu --init` warm start, `--snapshot-every`
- `eval-net` with paired openings, Wilson intervals, distinct-position counts
- `unpack-az` (recover a deployed net from its committed `.cpp`) — in the tictac repo

Needs building:
- the `z`-sign assert for opponent mode (blocking, see above)
- **pool** training for Phase B: sample the opponent per-game from a set rather than one
  frozen net. This is the actual new machinery, and the single-nemesis version is *not* a
  substitute — training against one adversary just swaps one degenerate opponent
  distribution for another.
- an exploitability summary command, so N2's four numbers come out of one invocation

## Cost

- Phase A, one nemesis: ~2,500 iters ≈ 35 min GPU, plus ~10 min of eval.
- N2 needs four targets: ~3 h total.
- Phase B: one pool run ≈ 2 h, plus a ladder submission (~2 h placement, and note
  leagues never demote, so each submission replaces the previous one).

## Housekeeping owed before this starts

I pre-committed that if the soup failed to beat rank 34 I would roll back to v2. It
failed (31.46 vs 32.24). **The rollback is owed and has not been done.** Do it before or
alongside Phase A, so the ladder reflects our best known net while this runs.
