# Chess AI implementation plan

**Goal:** Deliver a runnable local chess learner and a portable engine interface.
**Architecture:** Policy/value network plus PUCT self-play, batched across games;
shared model/search for training and playing. Atomic training snapshots keep
resume coherent.
**Tech stack:** Python, PyTorch, NumPy, python-chess, stdlib unittest and argparse.
**Spec:** [design.md](design.md)

## Tasks

- [x] Implement `chess_ai/model.py`: canonical board/move encoding, residual
  network, legal-move inference, and checkpoint loading.
- [x] Implement `chess_ai/search.py`: PUCT, alternating value backup, root noise,
  visit policies, batched leaves across independent games, cancellation.
- [x] Implement `chess_ai/training.py`: self-play, bounded sparse replay, masked
  policy/value optimization, atomic save/resume, metrics and PGN.
- [x] Implement `chess_ai/__main__.py` and `chess_ai/uci.py`: validated CLI,
  analysis/play/evaluation, asynchronous UCI protocol and local launchers.
- [x] Run `python -m unittest discover -s tests -v`: rule edge cases, forced mate,
  gradient updates, resume determinism, and real UCI subprocess round trips.
- [x] Run a short CUDA training job, resume it, and evaluate its checkpoint.
- [x] Document installation, normal training/resume commands, outputs, engine
  connection, and measured limitations in `README.md`.
