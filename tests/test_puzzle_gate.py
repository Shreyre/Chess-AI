"""Strict selection, optimizer rollback, checkpoint resume and frozen benchmark."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import chess

from chess_ai.puzzle_gate import PuzzleGate
from chess_ai.telemetry import LiveStatus
from chess_ai.training import train


def puzzles(root):
    path = root / 'puzzles.jsonl'
    row = dict(FEN='6k1/5Q2/6K1/8/8/8/8/8 b - - 0 1',
               Moves='g8h8 f7g7', Themes=['mate', 'mateIn1'], Rating=500)
    path.write_text(''.join(json.dumps(row | {'PuzzleId': str(i)})+'\n' for i in range(3)))
    return path


class GateChecks(unittest.TestCase):
    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_regression_tie_improvement_error_and_file_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = puzzles(root)
            config = {'puzzle_data': str(path)}
            gate = PuzzleGate(config, LiveStatus(root))
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            with patch('chess_ai.puzzle_gate.assess_puzzles', return_value={'solved': 2}):
                gate.initialize(model, 10)
            self.assertEqual(config['puzzle_gate']['accepted_rate'], 2/3)
            for result, status in ((1, 'rejected'), (2, 'rejected'), (3, 'accepted'),
                                   (RuntimeError('scoring failed'), 'error')):
                before = deepcopy((model.state_dict(), optimizer.state_dict()))
                optimizer.zero_grad()
                model(torch.ones(1, 1)).sum().backward()
                optimizer.step()
                candidate = deepcopy((model.state_dict(), optimizer.state_dict()))
                kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': {'solved': result}}
                with patch('chess_ai.puzzle_gate.assess_puzzles', **kwargs):
                    decision = gate.check(model, optimizer, before, 11)
                self.assertEqual(decision['status'], status)
                self.assert_state_equal((model.state_dict(), optimizer.state_dict()),
                                        candidate if status == 'accepted' else before)
            self.assertEqual(config['puzzle_gate']['accepted_solved'], 3)
            before = deepcopy((model.state_dict(), optimizer.state_dict()))
            path.write_text(path.read_text()+'\n')
            self.assertEqual(gate.check(model, optimizer, before, 12)['status'], 'error')
            with self.assertRaisesRegex(ValueError, 'changed'):
                PuzzleGate(config, LiveStatus(root))

    def test_training_saves_accepted_weights_and_resumes_gate(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lessons = root/'lessons.jsonl'
            lessons.write_text(json.dumps(dict(PuzzleId='opening', FEN=chess.STARTING_FEN,
                Moves='e2e4 e7e5', Themes=['opening'], Rating=1000))+'\n')
            args = SimpleNamespace(run_dir=str(root), resume=None, iterations=1,
                puzzle_data=puzzles(root), games=1, parallel_games=1, simulations=1,
                max_plies=2, train_steps=1, batch_size=2, channels=8, blocks=1,
                mate_moves=0, mate_nodes=1, tactics_data=lessons, tactics_steps=2)
            with patch('chess_ai.puzzle_gate.assess_puzzles', side_effect=[{'solved': 2}, {'solved': 1}]):
                train(args, torch.device('cpu'))
            data = torch.load(root/'latest.pt', weights_only=True)
            initial = torch.load(root/'initial.pt', weights_only=True)
            self.assert_state_equal(data['model'], initial['model'])
            self.assertEqual(data['optimizer']['state'], {})
            self.assertEqual(data['config']['puzzle_gate']['status'], 'rejected')
            self.assertEqual(data['config']['puzzle_gate']['accepted_iteration'], 0)
            self.assertEqual(data['iteration'], 1)
            self.assertEqual(len(data['replay']), 2)
            for name in ('model.pt', 'best.pt'):
                self.assert_state_equal(torch.load(root/name, weights_only=True)['model'], initial['model'])
            args.resume, args.puzzle_data = str(root/'latest.pt'), None
            with patch('chess_ai.puzzle_gate.assess_puzzles', return_value={'solved': 3}) as score:
                train(args, torch.device('cpu'))
            self.assertEqual(score.call_count, 1)  # Persisted baseline was reused.
            data = torch.load(root/'latest.pt', weights_only=True)
            self.assertEqual(data['config']['puzzle_gate']['status'], 'accepted')
            self.assertEqual(data['config']['puzzle_gate']['accepted_iteration'], 2)
            self.assertTrue(data['optimizer']['state'])
            self.assertTrue(any(not torch.equal(value, initial['model'][key]) for key, value in data['model'].items()))


if __name__ == '__main__':
    unittest.main()
