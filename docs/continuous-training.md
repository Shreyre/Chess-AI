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
