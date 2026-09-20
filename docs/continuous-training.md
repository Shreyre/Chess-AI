# Automatic teacher refresh and best-model selection

Approved scope: keep teacher examples current and let the screen player use a
latest saved model while retaining a separately tested best model. Preserve the latest resumable trainer checkpoint.

- [x] Reuse one batched paired-match implementation for manual evaluation and the
  gate. Check legal moves, swapped colors, deterministic openings and cancellation.
- [x] Sample recent PGN games efficiently; label a larger candidate pool, retaining
  practice positions, difficult examples and a random sample. Keep the stable
  held-out partition and mix bounded older teacher examples into each refresh.
- [x] Run refresh and selection at saved-iteration boundaries and on resume when
  due. Persist configuration and completed-iteration markers; save datasets and
  best-model changes atomically. Cancellation/errors retain the prior best/data.
- [x] Require completed paired matches, at least 60% against the incumbent and
  at least 50% against each of two older available checkpoints. Twenty games per
  opponent is a screening rule, not statistical proof or an Elo estimate.
- [x] Follow latest.pt in the screen player; preserve explicit historical selection
  and completed-screen-game learning. Show the selected model's iteration.
- [x] Verify regression checks, a real Stockfish refresh and a batched match smoke test.
- [ ] Complete the live handoff after iteration 445 saves; the one-time launcher
  is waiting and will enable both features, then resume toward iteration 539.

Default intervals for the active run: five iterations; 1,024 fresh teacher examples
from the latest 2,000 games, plus up to 1,024 older examples; 20,000 Stockfish nodes
per position; 20 games per gate opponent at 64 simulations per move. Existing runs
remain opt-in. The bootstrap incumbent is the previously tested iteration 444.

## Usage

Resume with `--teacher-refresh-every 5 --gate-every 5 --teacher-engine PATH --teacher-fens docs/practice-positions.fen`. These settings and completed iteration markers persist in `latest.pt`; zero disables either feature. Stop requests during maintenance cancel safely and retry on resume. Refreshes use 20,000 nodes per position, a recent 2,000-game window, practice positions, a mixture of difficult and random examples, and up to 1,024 retained older examples.

The gate plays 20 games per available opponent at 64 searches per move, with paired openings and swapped colors. Promotion requires at least 60% against the incumbent and 50% against each of up to two older checkpoints, with no unfinished games. Results and PGNs are in `evaluations/`. This is a regression screen, not proof of strength or an Elo estimate. Both jobs run sequentially between saved iterations and add time to training.

The screen player's normal `model.pt`/`latest.pt` selections follow the latest saved iteration, checking for updates before each AI move. Selection tests still maintain `best.pt`, which can be chosen explicitly. Historical checkpoint selections remain pinned. Completed screen games train the latest model, and saved updates reach normal play without waiting for selection. Restart an already-open screen player once to load this behavior.

## Teaching upgrade

The new teaching controls preserve existing network dimensions and checkpoints.
`--priority-fraction 0.5` mixes uniform and error-based sampling; `--curriculum-every 1`
unlocks easy, intermediate and advanced teacher examples over three saved learning
iterations. `--postgame-nodes 20000` adds offline Stockfish corrections for completed
screen games, including older unreviewed saves. `--teacher-capacity 20000` retains
older examples across the smaller periodic refreshes, up to 20,000 total.

The active-run handoff is `runs/activate-teaching.py`, with progress in
`runs/main/teaching-upgrade-status.json`. It waits for the expanded dataset and the
requested checkpoint stop, backs up the full checkpoint, enables the controls,
runs three short curriculum iterations in a separate trial, reviews saved screen games, assesses
held-out teacher agreement, and runs 20 paired games at 64 searches per move
against a frozen pre-upgrade model. Trial weights replace the active model only
with no truncated games, at least 60% match score, and no newer active checkpoint.
It then resumes the remaining requested
self-play iterations. Reports are saved as `teaching-*-assessment.json` and
`teaching-comparison.json`; these are measurements, not an Elo estimate or a
guarantee of stronger play. The handoff log records any failure without deleting
the saved checkpoint or completed games.

## Laptop tuning

For the 16 GB RAM / RTX 4060 laptop, the queued configuration is 100 games per iteration, all 100 concurrent, with 64 searches per move and the existing 512-ply ceiling. Teacher refresh and candidate testing move from every five large iterations to every 25 smaller iterations, keeping roughly the same cadence per generated game. The one-time `runs/tune-laptop.py` handoff preserves the active iteration and computes the new iteration target from the remaining game budget (63,036 total games, allowing up to 99 extra games for rounding or screen-game imports). Its status is recorded in `runs/main/laptop-tuning-status.json`.

Training now releases discarded batch records before generating the next batch, and drops the duplicate replay-list reference after restoring a checkpoint. A regression test verifies that only retained replay positions survive into the next self-play call. The CUDA opening-position microbenchmark is recorded in `runs/laptop-benchmark.log`; it diagnoses per-search overhead and does not predict full-iteration time.
