# Chess AI design

Build a local, from-scratch chess learner for the user's RTX 4060 laptop (8 GB
VRAM, 16 GB RAM). Use a compact policy/value residual network with PUCT Monte
Carlo Tree Search and self-play, inspired by AlphaZero. Basic DQN has a weaker
fit for this large, adversarial move space; supervised engine imitation would
violate the requested from-scratch learning. No pretrained models, opening books,
engine teachers, or hand-written material rewards.

## Training

Python 3.10+; PyTorch, NumPy, and python-chess only. python-chess supplies rules,
legal moves, and PGN, not evaluations. Encode positions from the current player's
perspective. Distinguish castling, en passant, repetition, and halfmove clock.
Use a 73-plane move representation including underpromotions. Mask illegal moves
during both search and policy training. Search backs up values with alternating
signs. Root noise and early-game sampling drive exploration. Terminal results
train the value head; positions in games stopped at the move limit have no value
target, rather than being mislabeled draws. Automatic draws use the chess rules;
optional draw claims are not actions in this first version.

Bound memory with a replay buffer. Save model, optimizer, replay, counters, and
random generator states in atomic checkpoints. Resume after completed iterations;
Ctrl+C preserves that checkpoint. Log losses, completed outcomes, truncations,
and sample games. Batch independent self-play games for GPU inference. Keep
network size, game count, simulations, and training batch size configurable.

## Playing and validation

Provide CLI training, evaluation against random or another saved model, terminal
play, FEN analysis, and a responsive UCI interface with time limits and stop.
Application/website integration is deferred until the user chooses a platform.
Support legal moves, castling, en passant, all promotions, terminal positions,
search sign correctness, parameter updates, checkpoint resume, and UCI through a
small runnable unittest suite and an actual GPU smoke run. Evaluation reports
truncations separately and does not claim an Elo or guaranteed improvement.

## Limits

This is a compact educational implementation, not a reproduction of DeepMind's
compute scale. Chess learning from sparse results is slow. Short runs verify
operation, not playing strength. The encoder exposes current repetition counts
but not the full position history; search retains the actual board history.

References: https://arxiv.org/abs/1712.01815,
https://python-chess.readthedocs.io/en/stable/,
https://pytorch.org/get-started/locally/,
https://backscattering.de/chess/uci/.
