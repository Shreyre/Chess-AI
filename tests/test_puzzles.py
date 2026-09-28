"""Puzzle setup, full-line scoring, validation and alternative mates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import chess

from chess_ai.puzzles import assess_puzzles, load_puzzles, puzzle_position
from chess_ai.search import Node
from chess_ai.engine import engine_move, open_engine


ROW = dict(PuzzleId='000hf', GameId='71ygsFeE', Rating=1575,
           FEN='r1bqk2r/pp1nbNp1/2p1p2p/8/2BP4/1PN3P1/P3QP1P/3R1RK1 b kq - 0 19',
           Moves='e8f7 e2e6 f7f8 e6f7', Themes=['mate', 'mateIn2', 'middlegame'])


class PuzzleChecks(unittest.TestCase):
    def test_local_engine_scoring_validation_and_cleanup(self):
        engine = Mock(id={'name': 'test engine'})
        engine.play.side_effect = [Mock(move=chess.Move.from_uci(m)) for m in ('e2e6', 'e6f7')]
        with patch('chess_ai.puzzles.run_searches') as neural:
            report = assess_puzzles(None, [ROW], engine=engine, engine_nodes=123)
        neural.assert_not_called()
        self.assertEqual(report['solved'], 1)
        self.assertEqual(report['decision_method'], 'local_uci_engine')
        self.assertIsNone(report['simulations'])
        self.assertEqual(report['engine_nodes'], 123)
        calls = engine.play.call_args_list
        self.assertEqual(calls[0].args[1].nodes, 123)
        self.assertIsNot(calls[0].kwargs['game'], calls[1].kwargs['game'])
        board, _ = puzzle_position(ROW)
        original = board.fen(), list(board.move_stack)
        engine.play.side_effect = None
        engine.play.return_value.move = chess.Move.from_uci('a1a8')
        with self.assertRaisesRegex(ValueError, 'illegal'):
            engine_move(engine, board)
        self.assertEqual((board.fen(), board.move_stack), original)
        with self.assertRaises(ValueError):
            engine_move(engine, board, 0)
        terminal = chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
        self.assertIsNone(engine_move(engine, terminal))
        with patch('chess_ai.engine.chess.engine.SimpleEngine.popen_uci') as popen:
            with self.assertRaisesRegex(ValueError, 'failure'):
                with open_engine('engine.exe') as opened:
                    opened.configure.assert_called_once_with({'Threads': 1, 'Hash': 64})
                    raise ValueError('failure')
            popen.return_value.__exit__.assert_called_once()

    def test_setup_full_line_and_mismatch(self):
        board, line = puzzle_position(ROW)
        self.assertEqual(board.peek().uci(), 'e8f7')
        self.assertEqual(board.turn, chess.WHITE)
        self.assertEqual([m.uci() for m in line], ['e2e6','f7f8','e6f7'])
        decisions = []
        def search(model, searches, simulations):
            for s in searches:
                decisions.append(s.board.fen())
                selected = 'e2e6' if len(decisions) == 1 else 'e6f7'
                s.root.children = {m: Node(visits=int(m.uci() == selected)) for m in s.board.legal_moves}
        with patch('chess_ai.puzzles.run_searches', side_effect=search):
            report = assess_puzzles(None, [ROW], 2)
        self.assertEqual(report['solved'], 1)
        self.assertEqual(report['first_move_correct'], 1)
        self.assertEqual(report['results'][0]['decisions'], 2)
        self.assertEqual(report['by_theme']['mate']['solved'], 1)
        self.assertEqual(report['by_rating']['1500-1999']['puzzles'], 1)
        board.push_uci('e2e6'); board.push_uci('f7f8')
        self.assertEqual(decisions[1], board.fen())
        def wrong(model, searches, simulations):
            for s in searches:
                selected = next(m for m in s.board.legal_moves if m.uci() != 'e2e6')
                s.root.children = {m: Node(visits=int(m == selected)) for m in s.board.legal_moves}
        with patch('chess_ai.puzzles.run_searches', side_effect=wrong):
            report = assess_puzzles(None, [ROW], 2)
        self.assertEqual(report['solved'], 0)
        self.assertEqual(report['results'][0]['decisions'], 1)
        self.assertEqual(report['results'][0]['expected'], 'e2e6')

    def test_alternative_immediate_mate_is_accepted(self):
        row = ROW | dict(FEN='6k1/5Q2/6K1/8/8/8/8/8 b - - 0 1',
                         Moves='g8h8 f7g7', Themes=['mate','mateIn1'])
        board, _ = puzzle_position(row)
        alternative = chess.Move.from_uci('f7h7')
        board.push(alternative)
        self.assertTrue(board.is_checkmate())
        def search(model, searches, simulations):
            for s in searches:
                s.root.children = {m: Node(visits=int(m == alternative)) for m in s.board.legal_moves}
        with patch('chess_ai.puzzles.run_searches', side_effect=search):
            self.assertEqual(assess_puzzles(None, [row])['solved'], 1)

    def test_invalid_and_duplicate_inputs(self):
        for fields in [dict(Moves='e8f7'), dict(Moves='e8e1 e2e6'), dict(FEN='bad'),
                       dict(Themes='mate'), dict(Rating=True), dict(Moves='e8f7 e2e6')]:
            with self.assertRaises(ValueError):
                puzzle_position(ROW | fields)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'puzzles.jsonl'
            path.write_text(json.dumps(ROW)+'\n')
            self.assertEqual(load_puzzles(path), [ROW])
            path.write_text((json.dumps(ROW)+'\n')*2)
            with self.assertRaisesRegex(ValueError, 'line 2.*Duplicate'):
                load_puzzles(path)
            path.write_text('')
            with self.assertRaises(ValueError):
                load_puzzles(path)


if __name__ == '__main__':
    unittest.main()
