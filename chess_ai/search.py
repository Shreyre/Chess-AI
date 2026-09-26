"""PUCT search. Values in every node are from that node's player's perspective."""

from dataclasses import dataclass, field
import math
import time

import numpy as np

from .model import evaluate_boards
from .mate import shortest_mate


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    total: float = 0.0
    children: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.total / self.visits if self.visits else 0.0


def terminal_value(board):
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def expand(node, prediction):
    moves, priors, _ = prediction
    node.children = {move: Node(float(prior)) for move, prior in zip(moves, priors)}


def backup(path, value):
    for node in reversed(path):
        node.visits += 1
        node.total += value
        value = -value


class Search:
    def __init__(self, board, c_puct=1.5):
        self.board = board.copy(stack=True)
        self._working_board = self.board.copy(stack=True)
        self.root = Node()
        self.c_puct = c_puct
        self.mate_move = None
        self.mate_in = None
        self.mate_nodes = 0

    def leaf(self):
        # Rewind the previous branch instead of copying the entire game each simulation.
        # The returned board is valid until the next leaf() call on this search.
        board, node = self._working_board, self.root
        while len(board.move_stack) > len(self.board.move_stack):
            board.pop()
        path = [node]
        while node.children:
            scale = self.c_puct * math.sqrt(node.visits + 1)
            move, node = max(node.children.items(), key=lambda item:
                             -item[1].value + scale * item[1].prior / (1 + item[1].visits))
            board.push(move)
            path.append(node)
        return board, node, path

    def policy(self, temperature=1.0):
        moves = list(self.root.children)
        if not moves:
            return [], np.empty(0, dtype=np.float32)
        if self.mate_move is not None:
            return moves, np.array([move == self.mate_move for move in moves], dtype=np.float32)
        counts = np.array([self.root.children[move].visits for move in moves], dtype=float)
        if counts.sum() == 0:
            counts = np.array([self.root.children[move].prior for move in moves], dtype=float)
        if temperature <= 0:
            probs = np.zeros(len(moves))
            probs[counts.argmax()] = 1.0
        else:
            logs = np.log(np.maximum(counts, 1e-30)) / temperature
            probs = np.exp(logs - logs.max())
            probs /= probs.sum()
        return moves, probs.astype(np.float32)


def run_searches(model, searches, simulations, rng=None, noise=False,
                 stop=None, deadline=None, mate_moves=3, mate_nodes=512):
    """Batch one leaf per independent game; training needs no worker processes."""
    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    if not 0 <= mate_moves <= 10 or mate_nodes < 1:
        raise ValueError("mate moves must be in 0..10; mate nodes must be positive")
    active = [search for search in searches if terminal_value(search.board) is None]
    for search in active:
        search.mate_move, search.mate_in, search.mate_nodes = shortest_mate(
            search.board, mate_moves, mate_nodes, stop, deadline)
        if search.mate_move is not None:
            search.root = Node(children={move: Node() for move in search.board.legal_moves})
            # One completed proof, not fabricated MCTS visits.
            backup([search.root, search.root.children[search.mate_move]], -1.0)
    active = [search for search in active if search.mate_move is None]
    if not active:
        return
    predictions = evaluate_boards(model, [search.board for search in active])
    for search, prediction in zip(active, predictions):
        expand(search.root, prediction)
        if noise:
            if rng is None:
                raise ValueError("Exploration requires a random generator")
            values = rng.dirichlet(np.full(len(search.root.children), 0.3))
            for child, value in zip(search.root.children.values(), values):
                child.prior = 0.75 * child.prior + 0.25 * value
    # ponytail: rebuild each move; retain subtrees if profiling shows search reuse matters.
    for _ in range(simulations):
        if ((stop is not None and stop.is_set()) or
                (deadline is not None and time.monotonic() >= deadline)):
            break
        leaves = []
        for search in active:
            board, node, path = search.leaf()
            value = terminal_value(board)
            if value is None:
                leaves.append((board, node, path))
            else:
                backup(path, value)
        if leaves:
            predictions = evaluate_boards(model, [leaf[0] for leaf in leaves])
            for (_, node, path), prediction in zip(leaves, predictions):
                expand(node, prediction)
                backup(path, prediction[2])


def select_move(model, board, simulations=128, **kwargs):
    search = Search(board)
    run_searches(model, [search], simulations, **kwargs)
    moves, probabilities = search.policy(0)
    return (moves[int(probabilities.argmax())] if moves else None), search
