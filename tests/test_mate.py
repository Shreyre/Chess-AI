"""Run: python -m unittest discover -s tests -p test_mate.py -v"""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess
import chess.engine
import chess.pgn
import numpy as np

from chess_ai.mate import shortest_mate
from chess_ai.model import move_index
from chess_ai.search import Search, run_searches
from chess_ai.teacher import correct_screen_game, teacher_record


class MateChecks(unittest.TestCase):
    def test_shortest_mates_in_both_colors_including_quiet_moves(self):
        positions = [('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1', 1),
                     ('7k/8/5K2/8/8/8/8/3R4 w - - 0 1', 2),
                     ('7k/8/8/4K3/8/8/8/3Q4 w - - 0 1', 3)]

        # Independent full-width reference: compare worst-case mate distance.
        def distance(board, attacker, depth):
            if board.is_checkmate():
                return 0 if board.turn != attacker else float('inf')
            if depth == 0 or board.is_game_over():
                return float('inf')
            values = []
            for move in board.legal_moves:
                child = board.copy(); child.push(move)
                values.append(1 + distance(child, attacker, depth - 1))
            return (min if board.turn == attacker else max)(values)

        for fen, expected in positions:
            for board in (chess.Board(fen), chess.Board(fen).mirror()):
                original = board.fen(), list(board.move_stack)
                move, mate, nodes = shortest_mate(board, 3, 10000)
                self.assertEqual(mate, expected)
                self.assertLessEqual(nodes, 10000)
                self.assertEqual((board.fen(), board.move_stack), original)
                child = board.copy(); child.push(move)
                self.assertEqual(distance(child, board.turn, 2 * expected - 2), 2 * expected - 2)
                self.assertIsNone(shortest_mate(board, expected - 1, 10000)[0])
                if expected > 1:
                    self.assertFalse(board.gives_check(move))

    def test_draws_history_cancellation_and_exhausted_budget(self):
        mate2 = chess.Board('7k/8/5K2/8/8/8/8/3R4 w - - 0 1')
        self.assertEqual(shortest_mate(mate2, 3, 1), (None, None, 1))
        stopped = threading.Event(); stopped.set()
        self.assertEqual(shortest_mate(mate2, stop=stopped), (None, None, 0))
        self.assertEqual(shortest_mate(mate2, deadline=time.monotonic()-1), (None, None, 0))
        self.assertIsNone(shortest_mate(chess.Board(), 3, 1000)[0])
        # A claimable draw refutes the otherwise forced mate in two.
        mate2.halfmove_clock = 99
        self.assertIsNone(shortest_mate(mate2, 2, 10000)[0])
        # Checkmate still takes precedence over the automatic 75-move draw.
        mate1 = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 149 1')
        self.assertEqual(shortest_mate(mate1)[1], 1)
        stale = chess.Board('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1')
        self.assertTrue(stale.is_stalemate())
        self.assertIsNone(shortest_mate(stale)[0])
        repeated = chess.Board('7k/8/5K2/8/8/8/8/3R4 w - - 0 1')
        for token in ['d1e1', 'h8h7', 'e1d1', 'h7h8'] * 2:
            repeated.push_uci(token)
        original = repeated.fen(), list(repeated.move_stack)
        shortest_mate(repeated, 3, 1000)
        self.assertEqual((repeated.fen(), repeated.move_stack), original)
        for args in [(-1, 10), (11, 10), (1, 0)]:
            with self.assertRaises(ValueError):
                shortest_mate(mate1, *args)

    def test_proofs_override_noise_and_train_the_policy_without_a_network(self):
        board = chess.Board('7k/8/5K2/8/8/8/8/3R4 w - - 0 1')
        search = Search(board)
        with patch('chess_ai.search.evaluate_boards', side_effect=AssertionError('Unneeded network')):
            run_searches(None, [search], 1, np.random.default_rng(7), noise=True)
        self.assertEqual(search.mate_in, 2)
        self.assertEqual(search.root.value, 1)
        for temperature in (0, 1, 10):
            moves, policy = search.policy(temperature)
            self.assertEqual(moves, list(board.legal_moves))
            self.assertEqual(policy.sum(), 1)
            self.assertEqual(moves[policy.argmax()], search.mate_move)

    def test_teacher_prefers_mate_distance_and_preserves_outcome_values(self):
        for board in (chess.Board(), chess.Board().mirror()):
            moves = list(board.legal_moves)
            for scores, expected in [([chess.engine.Mate(5), chess.engine.Mate(1), chess.engine.Mate(1)], [0, .5, .5]),
                                     ([chess.engine.Mate(-1), chess.engine.Mate(-5), chess.engine.Mate(-3)], [0, 1, 0]),
                                     ([chess.engine.Mate(-5), chess.engine.Cp(-500), chess.engine.Mate(-1)], [0, 1, 0]),
                                     ([chess.engine.Cp(20000), chess.engine.Mate(3), chess.engine.Cp(0)], [0, 1, 0])]:
                infos = [dict(pv=[move], score=chess.engine.PovScore(score, board.turn))
                         for move, score in zip(moves, scores)]
                row = teacher_record(board, infos)
                np.testing.assert_allclose(row[2][:3].numpy(), expected)
                self.assertEqual(row[2].sum(), 1)
                if max(scores).is_mate():
                    self.assertEqual(row[3], 1 if max(scores).mate() > 0 else -1)

    def test_postgame_corrects_slower_mates_below_centipawn_threshold(self):
        board = chess.Board('7k/8/5K2/8/8/8/8/3R4 w - - 0 1')
        slow, fast = chess.Move.from_uci('d1e1'), chess.Move.from_uci('f6f7')
        infos = [dict(pv=[move], score=chess.engine.PovScore(chess.engine.Mate(n), board.turn))
                 for move, n in [(slow, 5), (fast, 2)]]
        row = teacher_record(board, [infos[0]])
        board.push(slow)
        game = dict(records=[row], pgn=str(chess.pgn.Game.from_board(board)))
        with patch('chess_ai.teacher.is_held_out', return_value=False):
            rows, report = correct_screen_game(game, SimpleNamespace(analyse=lambda *a, **k: infos), 100)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['best'], fast.uci())
        self.assertTrue(report[0]['mate_distance'])
        self.assertEqual(int(rows[0][1][rows[0][2].argmax()]), move_index(fast, chess.WHITE))


if __name__ == '__main__':
    unittest.main()
