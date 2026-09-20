"""Hybrid-training contracts, without downloading or requiring an engine."""
from collections import deque
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
import chess.engine
import numpy as np
import torch

from chess_ai.model import ChessNet, atomic_save, encode, move_index
from chess_ai.training import update_replay, learning_steps, DEFAULTS, save_training, train, train_batch
from chess_ai.teacher import teacher_record, load_teacher, position_group


class HybridChecks(unittest.TestCase):
    def test_previous_batch_is_released_before_next_selfplay(self):
        import weakref
        references = []
        def games(*args, **kwargs):
            if references:
                self.assertLessEqual(sum(ref() is not None for ref in references), 1)
            rows = []
            for _ in range(3):
                row = teacher_record(chess.Board(), [dict(pv=[chess.Move.from_uci('e2e4')],
                    score=chess.engine.PovScore(chess.engine.Cp(0), True))])
                references.append(weakref.ref(row[0]))
                rows.append(row)
            return rows, dict(white_wins=0, black_wins=0, draws=1, truncated=0), []
        with tempfile.TemporaryDirectory() as folder:
            options = dict.fromkeys(DEFAULTS)
            options.update(channels=8, blocks=1, games=1, replay_size=1, batch_size=1,
                           train_steps=1)
            args = SimpleNamespace(**options, resume=None, run_dir=folder, iterations=2)
            with patch('chess_ai.training.self_play', side_effect=games):
                train(args, 'cpu')

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_replay_samples_all_games_and_keeps_old_experience(self):
        incoming = list(range(10000))  # Ordered by game, just like self_play().
        replay = deque(range(-200, 0), maxlen=200)
        update_replay(replay, incoming, np.random.default_rng(7))
        self.assertEqual(len(replay), 200)
        self.assertEqual(sum(x < 0 for x in replay), 100)
        self.assertTrue(any(0 <= x < 1000 for x in replay))
        self.assertTrue(any(x >= 9000 for x in replay))
        again = deque(range(-200, 0), maxlen=200)
        update_replay(again, incoming, np.random.default_rng(7))
        self.assertEqual(replay, again)
        update_replay(replay, [], np.random.default_rng(7))
        self.assertEqual(replay, again)
        config = DEFAULTS | dict(train_steps=100, batch_size=128)
        self.assertEqual(learning_steps(config, 128), 100)
        self.assertEqual(learning_steps(config, 50000, 32), 521)

    def test_teacher_targets_and_rejects_invalid_data(self):
        for board in (chess.Board(), chess.Board().mirror()):
            moves = list(board.legal_moves)
            infos = [dict(pv=[moves[0]], score=chess.engine.PovScore(chess.engine.Cp(300), board.turn)),
                     dict(pv=[moves[1]], score=chess.engine.PovScore(chess.engine.Cp(-200), board.turn))]
            record = teacher_record(board, infos)
            self.assertGreater(record[3], 0)
            self.assertTrue(record[4])
            self.assertEqual(record[1].tolist(), [move_index(m, board.turn) for m in moves])
            self.assertGreater(record[2][0], record[2][1])
            self.assertAlmostEqual(record[2].sum().item(), 1, places=6)
            np.testing.assert_array_equal(record[0].float().numpy(), encode(board))
        with self.assertRaises(ValueError):
            teacher_record(chess.Board(), [dict(pv=[chess.Move.from_uci('a1a8')],
                                               score=chess.engine.PovScore(chess.engine.Cp(0), True))])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'teacher.pt'
            atomic_save(dict(format_version=1, kind='teacher', records=[record],
                             validation_records=[], validation_fens=[]), path)
            self.assertEqual(len(load_teacher(path)['records']), 1)
            bad = list(record)
            bad[2] = torch.full_like(bad[2], float('nan'))
            atomic_save(dict(format_version=1, kind='teacher', records=[tuple(bad)],
                             validation_records=[], validation_fens=[]), path)
            with self.assertRaises(ValueError):
                load_teacher(path)

    def test_teacher_warmup_then_hybrid_resume_excludes_validation(self):
        board = chess.Board()
        row = teacher_record(board, [dict(pv=[chess.Move.from_uci('e2e4')],
                                          score=chess.engine.PovScore(chess.engine.Cp(300), True))])
        row = (*row[:3], 0.75, True)
        validation = (*row[:3], -0.75, True)
        self.assertEqual(position_group(board), position_group(board.mirror()))
        config = DEFAULTS | dict(channels=8, blocks=1, games=2, parallel_games=2, simulations=2,
                                 max_plies=4, train_steps=2, batch_size=4, opening_plies=2)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            dataset, checkpoint = directory / 'teacher.pt', directory / 'latest.pt'
            atomic_save(dict(format_version=1, kind='teacher', records=[row],
                             validation_records=[validation], validation_fens=[board.fen()]), dataset)
            model = ChessNet(8, 1)
            save_training(checkpoint, model, torch.optim.AdamW(model.parameters()), [], 0,
                          config, np.random.default_rng(7), dict(games=0, positions=0, updates=0), {})
            pending = directory / 'screen-games' / 'pending.pt'
            atomic_save(dict(result='unfinished'), pending)
            args = SimpleNamespace(**dict.fromkeys(DEFAULTS), resume=str(checkpoint), run_dir=None,
                                   iterations=1, teacher_only=True, teacher_data=dataset)
            with patch('chess_ai.training.train_batch', wraps=train_batch) as batches:
                train(args, 'cpu')
                self.assertTrue(all(sample[3] == 0.75 for call in batches.call_args_list
                                    for sample in call.args[2]))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['totals'], dict(games=0, positions=0, updates=2))
            self.assertEqual(saved['config']['teacher_data'], str(dataset.resolve()))
            self.assertEqual(saved['screen_games'], [])
            pending.unlink()
            args.teacher_only, args.teacher_data = False, None
            with patch('chess_ai.training.train_batch', wraps=train_batch) as batches:
                train(args, 'cpu')
                for call in batches.call_args_list:
                    self.assertEqual(sum(sample[3] == 0.75 for sample in call.args[2]), 1)
                    self.assertFalse(any(sample[3] == -0.75 for sample in call.args[2]))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['totals'], dict(games=2, positions=8, updates=5))
            self.assertEqual(saved['metrics']['teacher_positions'], 1)


if __name__ == '__main__':
    unittest.main()
