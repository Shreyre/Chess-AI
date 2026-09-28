"""Supervised tactical policy practice without changing the value network."""
import chess
import torch

from .model import encode, move_index
from .puzzles import load_puzzles, puzzle_position
from .teacher import position_group


def load_tactics(path, benchmark=None):
    rows = load_puzzles(path)
    reserved, games = set(), set()
    def game(row):
        return row.get('GameId', '').split('/')[0].split('#')[0]
    if benchmark:
        for row in load_puzzles(benchmark):
            if game(row):
                games.add(game(row))
            board, line = puzzle_position(row)
            for index, move in enumerate(line):
                if index % 2 == 0:
                    reserved.add(position_group(board))
                board.push(move)
    records = []
    for row in rows:
        if game(row) and game(row) in games:
            raise ValueError('Tactical lessons overlap a benchmark source game')
        board, line = puzzle_position(row)
        for index, move in enumerate(line):
            if index % 2 == 0:
                if position_group(board) in reserved:
                    raise ValueError('Tactical lessons overlap a benchmark position')
                variants = [(board, move)]
                if not board.castling_rights:
                    variants.append((board.transform(chess.flip_horizontal),
                        chess.Move(move.from_square ^ 7, move.to_square ^ 7, promotion=move.promotion)))
                for position, best in variants:
                    moves = list(position.legal_moves)
                    if best not in moves or not position.is_valid():
                        raise ValueError('Invalid tactical move transformation')
                    records.append((torch.from_numpy(encode(position)).half(),
                        torch.tensor([move_index(m, position.turn) for m in moves]),
                        torch.tensor([float(m == best) for m in moves]), 0.0, False))
            board.push(move)
    return records


def practice_tactics(model, optimizer, records, steps, learning_rate, rng, live=None):
    from .training import train_batch
    if not records or steps < 1 or not 0 < learning_rate <= 1:
        raise ValueError('Tactical practice needs examples, positive steps and a learning rate in (0, 1]')
    flags = [parameter.requires_grad for parameter in model.parameters()]
    # The accepted model's value function stays exactly intact during this phase.
    model.zero_grad(set_to_none=True)
    model.body.requires_grad_(False)
    model.value.requires_grad_(False)
    model.policy.requires_grad_(True)
    policy_optimizer = torch.optim.AdamW(model.policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    try:
        model.train()
        for step in range(steps):
            samples = [records[i] for i in rng.choice(len(records), 128, replace=len(records) < 128)]
            loss = train_batch(model, policy_optimizer, samples)
            if live:
                live.update(phase='learning', maintenance='Learning tactical solution moves',
                            maintenance_progress=f'{step + 1}/{steps}')
        # Old moments no longer describe the updated policy parameters.
        for parameter in model.policy.parameters():
            optimizer.state.pop(parameter, None)
        return dict(updates=steps, **loss)
    finally:
        for parameter, enabled in zip(model.parameters(), flags):
            parameter.requires_grad_(enabled)
        model.zero_grad(set_to_none=True)
        if live:
            live.update(force=True, maintenance=None, maintenance_progress=0)
