"""Chess encoding, policy/value network, and checkpoint I/O."""

import os
from pathlib import Path
import tempfile

import chess
import numpy as np
import torch
from torch import nn

INPUT_PLANES = 20
ACTION_SIZE = 73 * 64
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1))
KNIGHTS = ((1, 2), (2, 1), (2, -1), (1, -2),
           (-1, -2), (-2, -1), (-2, 1), (-1, 2))


def square_for_turn(square, turn):
    return square if turn == chess.WHITE else square ^ 56


def encode(board):
    """Friendly pieces first; the player to move always advances up the ranks."""
    state = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        square = square_for_turn(square, board.turn)
        plane = piece.piece_type - 1 + (0 if piece.color == board.turn else 6)
        state[plane, square // 8, square % 8] = 1
    for offset, color in enumerate((board.turn, not board.turn)):
        state[12 + offset * 2].fill(board.has_kingside_castling_rights(color))
        state[13 + offset * 2].fill(board.has_queenside_castling_rights(color))
    if board.ep_square is not None:
        square = square_for_turn(board.ep_square, board.turn)
        state[16, square // 8, square % 8] = 1
    state[17].fill(min(board.halfmove_clock, 150) / 150)
    state[18].fill(board.is_repetition(2))
    state[19].fill(board.is_repetition(3))
    return state


def move_index(move, turn):
    source = square_for_turn(move.from_square, turn)
    target = square_for_turn(move.to_square, turn)
    dx, dy = target % 8 - source % 8, target // 8 - source // 8
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        plane = 64 + (move.promotion - chess.KNIGHT) * 3 + dx + 1
    elif (dx, dy) in KNIGHTS:
        plane = 56 + KNIGHTS.index((dx, dy))
    else:
        distance = max(abs(dx), abs(dy))
        direction = (int(np.sign(dx)), int(np.sign(dy)))
        plane = DIRECTIONS.index(direction) * 7 + distance - 1
    return plane * 64 + source


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ChessNet(nn.Module):
    def __init__(self, channels=64, blocks=3):
        super().__init__()
        if channels < 8 or channels % 8 or blocks < 1:
            raise ValueError("channels must be a positive multiple of 8; blocks >= 1")
        self.config = {"channels": channels, "blocks": blocks}
        self.body = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.ReLU(),
            *(Residual(channels) for _ in range(blocks)))
        self.policy = nn.Conv2d(channels, 73, 1)
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
                                   nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        features = self.body(x)
        return self.policy(features).flatten(1), self.value(features).squeeze(1)


def choose_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Install CUDA PyTorch or use --device cpu.")
    return torch.device(name)


@torch.inference_mode()
def evaluate_boards(model, boards):
    """One GPU batch, with probabilities normalized over legal moves only."""
    device = next(model.parameters()).device
    states = torch.from_numpy(np.stack([encode(board) for board in boards])).to(device)
    logits, values = model(states)
    logits, values = logits.cpu().numpy(), values.cpu().numpy()
    results = []
    for board, scores, value in zip(boards, logits, values):
        moves = list(board.legal_moves)
        selected = scores[[move_index(move, board.turn) for move in moves]]
        if len(moves):
            probabilities = np.exp(selected - selected.max())
            probabilities /= probabilities.sum()
        else:
            probabilities = np.empty(0, dtype=np.float32)
        if not np.isfinite(value) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Network produced non-finite predictions")
        results.append((moves, probabilities, float(value)))
    return results


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                           dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_snapshot(model, iteration=0):
    return {"format_version": 1, "network": model.config,
            "model": model.state_dict(), "iteration": iteration}


def load_model(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    model = ChessNet(**data["network"]).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, data
