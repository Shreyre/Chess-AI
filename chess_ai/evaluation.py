"""Batched matches with identical openings and swapped colors."""
import chess
import chess.pgn
import numpy as np

from .search import Search, run_searches


def play_match(model, opponent, games=20, simulations=64, max_plies=512,
               opening_plies=4, seed=42, stop=lambda: False, progress=lambda ply: None):
    rng = np.random.default_rng(seed)
    boards = []
    for index in range(games):
        if index % 2 == 0:
            board = chess.Board()
            for _ in range(opening_plies):
                if board.is_game_over():
                    break
                moves = list(board.legal_moves)
                board.push(moves[int(rng.integers(len(moves)))])
            opening = board
        boards.append(opening.copy())
    for ply in range(max_plies):
        if stop():
            raise InterruptedError("Match cancelled; incumbent retained")
        active = [i for i, board in enumerate(boards) if not board.is_game_over()]
        if not active:
            break
        pending = []
        for candidate, current in ((True, model), (False, opponent)):
            indices = [i for i in active if (boards[i].turn == (i % 2 == 0)) == candidate]
            if current is None:
                for i in indices:
                    moves = list(boards[i].legal_moves)
                    pending.append((i, moves[int(rng.integers(len(moves)))]))
            else:
                searches = [Search(boards[i]) for i in indices]
                if searches:
                    run_searches(current, searches, simulations, rng)
                for i, search in zip(indices, searches):
                    moves, policy = search.policy(0)
                    pending.append((i, moves[int(policy.argmax())]))
        for i, move in pending:
            boards[i].push(move)
        progress(ply + 1)
    counts = dict(wins=0, losses=0, draws=0, truncated=0)
    pgns = []
    for i, board in enumerate(boards):
        outcome = board.outcome()
        result = ("truncated" if outcome is None else "draws" if outcome.winner is None
                  else "wins" if outcome.winner == (i % 2 == 0) else "losses")
        counts[result] += 1
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "candidate" if i % 2 == 0 else "opponent"
        game.headers["Black"] = "opponent" if i % 2 == 0 else "candidate"
        pgns.append(str(game))
    completed = games - counts["truncated"]
    score = counts["wins"] + counts["draws"] / 2
    return dict(**counts, completed=completed, score=score / games,
                score_on_completed=score / completed if completed else None,
                simulations=simulations, opening_plies=opening_plies, seed=seed), pgns
