"""Explicit local UCI assistance for terminal play, analysis and puzzle scoring."""
from contextlib import contextmanager
import subprocess
import sys

import chess.engine


@contextmanager
def open_engine(path):
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    with chess.engine.SimpleEngine.popen_uci(str(path), creationflags=flags) as engine:
        engine.configure({'Threads': 1, 'Hash': 64})
        yield engine


def engine_move(engine, board, nodes=20000):
    if nodes < 1 or not board.is_valid():
        raise ValueError('Engine needs a positive node budget and a valid board')
    if board.is_game_over():
        return None
    # Reset engine state per decision so puzzle order cannot prime later searches.
    move = engine.play(board, chess.engine.Limit(nodes=nodes), game=object()).move
    if move not in board.legal_moves:
        raise ValueError('Local engine returned an illegal or missing move')
    return move
