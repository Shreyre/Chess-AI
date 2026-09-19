# Hybrid training

Approved direction: use engine-labelled examples and focused practice alongside
self-play, retain representative experience, and measure strength against older
checkpoints. Keep the existing network and 500-game concurrency.

Implementation and verification:

- [x] Sample incoming experience before the bounded replay buffer discards it;
  preserve a sample of old experience and scale optimizer steps to new positions.
  Test early/late games, deterministic sampling, old checkpoint compatibility.
- [x] Generate a bounded, reproducible dataset with the installed python-chess
  UCI interface and Stockfish. Store all legal move indices, teacher policy and
  side-to-move value targets. Validate data at load; keep held-out positions out
  of training. Test target perspective, legal moves and invalid datasets.
- [x] Add teacher-only warmup and teacher/self-play batch mixing. Persist dataset
  settings in checkpoints so dashboard resumes keep the hybrid configuration.
  Test warmup, resume and screen-game learning compatibility.
- [x] Mix varied openings and focused tactical/endgame examples. Evaluate held-out
  teacher agreement and paired matches at equal search budgets; do not interpret
  a small match as an Elo estimate.
- [x] Preserve the pre-change checkpoint, finish the active iteration safely,
  generate labels, run a bounded warmup/evaluation, and resume the main run.

No repetition penalty and no engine calls in the screen player. Engine binaries
and generated datasets stay under ignored runs/. Source examples and CLI usage
remain in the repository. No new Python dependency is required.

