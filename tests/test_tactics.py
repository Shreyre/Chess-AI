"""Tactical supervision preserves value features and excludes scoring positions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from chess_ai.model import ChessNet
from chess_ai.tactics import load_tactics, practice_tactics
from chess_ai.training import train_batch


class TacticsChecks(unittest.TestCase):
    def test_supervision_freezing_optimizer_and_overlap(self):
        torch.set_num_threads(1)
        torch.manual_seed(14)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = dict(PuzzleId='lesson', GameId='training-game',
                FEN='6k1/5Q2/6K1/8/8/8/8/8 b - - 0 1', Moves='g8h8 f7g7',
                Themes=['mate', 'mateIn1'], Rating=500)
            path = root/'lessons.jsonl'
            path.write_text(json.dumps(row)+'\n')
            records = load_tactics(path)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r[4] is False and r[2].sum() == 1 for r in records))
            model = ChessNet(8, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            train_batch(model, optimizer, [records[0][:3] + (0.5, True)])
            before = deepcopy(model.state_dict())
            moments = deepcopy(optimizer.state[next(model.body.parameters())])
            result = practice_tactics(model, optimizer, records, 3, 0.01, np.random.default_rng(7))
            self.assertEqual(result['updates'], 3)
            self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()
                                if not key.startswith('policy.')))
            self.assertFalse(torch.equal(model.policy.weight, before['policy.weight']))
            self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertTrue(all(p not in optimizer.state for p in model.policy.parameters()))
            for key, value in moments.items():
                self.assertTrue(torch.equal(value, optimizer.state[next(model.body.parameters())][key]))
            benchmark = root/'benchmark.jsonl'
            benchmark.write_text(json.dumps(row | {'PuzzleId':'reserved', 'GameId':'other-game'})+'\n')
            with self.assertRaisesRegex(ValueError, 'benchmark position'):
                load_tactics(path, benchmark)
            benchmark.write_text(json.dumps(row | {'PuzzleId':'reserved'})+'\n')
            with self.assertRaisesRegex(ValueError, 'source game'):
                load_tactics(path, benchmark)


if __name__ == '__main__':
    unittest.main()
