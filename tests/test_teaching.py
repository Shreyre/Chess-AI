"""Teaching improvements: corrections, curriculum, sampling and resume."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from chess_ai.model import ChessNet, atomic_save
from chess_ai.teacher import correct_screen_game, curriculum_stage, teacher_record, load_teacher, generate
from chess_ai.training import DEFAULTS, priority_probabilities, save_training, train, train_batch


class TeachingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_priority_mixes_ordinary_examples_and_tracks_error(self):
        probabilities = priority_probabilities(np.array([0., 1., 100.]), .5)
        self.assertAlmostEqual(probabilities.sum(), 1)
        self.assertTrue(np.all(probabilities >= 1 / 6))
        self.assertGreater(probabilities[2], probabilities[1])
        np.testing.assert_allclose(priority_probabilities(np.zeros(3), .5), [1/3]*3)
        for bad in (np.array([np.nan]), np.array([-1.])):
            with self.assertRaises(ValueError):
                priority_probabilities(bad, .5)
        board = chess.Board()
        row = teacher_record(board, [dict(pv=[chess.Move.from_uci('e2e4')],
            score=chess.engine.PovScore(chess.engine.Cp(0), board.turn))])
        model = ChessNet(8, 1)
        stats, errors = train_batch(model, torch.optim.AdamW(model.parameters()), [row], return_errors=True)
        self.assertTrue(np.isfinite(errors).all())
        self.assertGreater(errors[0], 0)
        self.assertIn('loss', stats)

    def test_curriculum_unlocks_and_never_uses_validation(self):
        config = dict(curriculum_every=5, curriculum_start=100)
        self.assertEqual([curriculum_stage(config, i) for i in (100, 104, 105, 110)], [0, 0, 1, 2])
        self.assertEqual(curriculum_stage(dict(curriculum_every=0), 0), 2)
        board = chess.Board()
        row = teacher_record(board, [dict(pv=[chess.Move.from_uci('e2e4')],
            score=chess.engine.PovScore(chess.engine.Cp(0), board.turn))])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / 'teacher.pt'
            atomic_save(dict(format_version=1, kind='teacher', records=[(*row[:3], value, True)
                for value in (.1, .2, .3)], levels=[0, 1, 2], priorities=[1., 2., 3.],
                validation_records=[(*row[:3], -.9, True)], validation_fens=[board.fen()]), dataset)
            model = ChessNet(8, 1)
            config = DEFAULTS | dict(channels=8, blocks=1, batch_size=32, train_steps=1,
                teacher_data=str(dataset), priority_fraction=.5, curriculum_every=1)
            save_training(root/'latest.pt', model, torch.optim.AdamW(model.parameters()), [], 0,
                config, np.random.default_rng(7), dict(games=0, positions=0, updates=0), {})
            args = SimpleNamespace(**dict.fromkeys(DEFAULTS), resume=str(root/'latest.pt'),
                run_dir=None, iterations=1, teacher_only=True)
            for stage in range(3):
                with patch('chess_ai.training.train_batch', wraps=train_batch) as batches:
                    train(args, 'cpu')
                values = {round(row[3], 1) for row in batches.call_args.args[2]}
                self.assertTrue(values <= set((.1, .2, .3)[:stage+1]))
                self.assertIn((.1, .2, .3)[stage], values)
                saved = torch.load(root/'latest.pt', weights_only=True)
                self.assertEqual(saved['config']['curriculum_start'], 0)
                self.assertEqual(saved['metrics']['curriculum_stage'], stage)

    def test_postgame_corrects_confirmed_mistake_but_not_held_out(self):
        # A legal missed capture. Keep a full PGN so old screen saves also work.
        board = chess.Board('4k3/8/8/3q4/2P5/8/8/4K3 w - - 0 1')
        move, best = chess.Move.from_uci('e1f1'), chess.Move.from_uci('c4d5')
        original = teacher_record(board, [dict(pv=[move], score=chess.engine.PovScore(chess.engine.Cp(-800), True))])
        after = board.copy(); after.push(move)
        game = dict(records=[original], result='0-1', pgn=str(chess.pgn.Game.from_board(after)))
        def analyse(position, limit, **kwargs):
            chosen = move if kwargs.get('root_moves') else best
            info = dict(pv=[chosen], score=chess.engine.PovScore(chess.engine.Cp(-800 if chosen == move else 100), True))
            return [info] if kwargs.get('multipv') else info
        engine = SimpleNamespace(analyse=analyse)
        with patch('chess_ai.teacher.is_held_out', return_value=False):
            rows, report = correct_screen_game(game, engine, 100)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['loss_cp'], 900)
        self.assertEqual(rows[0][1][rows[0][2].argmax()], teacher_record(board, [analyse(board, None)])[1][
            teacher_record(board, [analyse(board, None)])[2].argmax()])
        with patch('chess_ai.teacher.is_held_out', return_value=True):
            rows, report = correct_screen_game(game, engine, 100)
        self.assertIs(rows[0], original)
        self.assertEqual(report, [])
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_save(game, root/'screen-games/game.pt')
            model = ChessNet(8, 1)
            config = DEFAULTS | dict(channels=8, blocks=1, train_steps=1, batch_size=4,
                                    teacher_engine='unused', postgame_nodes=100, priority_fraction=.5)
            save_training(root/'latest.pt', model, torch.optim.AdamW(model.parameters()), [original], 0,
                config, np.random.default_rng(7), dict(games=1, positions=1, updates=0), {}, ['game.pt'])
            args = SimpleNamespace(**dict.fromkeys(DEFAULTS), resume=str(root/'latest.pt'),
                                   run_dir=None, iterations=1, external_only=True)
            with patch('chess_ai.teacher.chess.engine.SimpleEngine.popen_uci', side_effect=OSError('teacher failed')):
                with self.assertRaises(OSError):
                    train(args, 'cpu')
            self.assertEqual(torch.load(root/'latest.pt', weights_only=True)['iteration'], 0)
            context = MagicMock(); context.__enter__.return_value = context
            context.analyse.side_effect = analyse
            with patch('chess_ai.teacher.chess.engine.SimpleEngine.popen_uci', return_value=context), \
                    patch('chess_ai.teacher.is_held_out', return_value=False):
                train(args, 'cpu')
            saved = torch.load(root/'latest.pt', weights_only=True)
            self.assertEqual(saved['totals']['games'], 1)
            self.assertEqual(saved['metrics']['corrected_positions'], 1)
            self.assertEqual(saved['config']['reviewed_screen_games'], ['game.pt'])
            from chess_ai.model import move_index
            self.assertTrue(all(int(row[1][row[2].argmax()]) == move_index(best, chess.WHITE)
                                for row in saved['replay']))
            with patch('chess_ai.teacher.chess.engine.SimpleEngine.popen_uci') as launch:
                train(args, 'cpu')
                launch.assert_not_called()
            self.assertEqual(torch.load(root/'latest.pt', weights_only=True)['iteration'], 1)
        game['pgn'] = 'broken game'
        with self.assertRaises(ValueError):
            correct_screen_game(game, engine, 100)

    def test_generated_metadata_and_partition_survive_refresh(self):
        board = chess.Board()
        def analyse(position, limit, **kwargs):
            return [dict(pv=[next(iter(position.legal_moves))],
                         score=chess.engine.PovScore(chess.engine.Cp(0), position.turn))]
        from unittest.mock import MagicMock
        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.id = {'name': 'test teacher'}
        engine.analyse.side_effect = analyse
        with tempfile.TemporaryDirectory() as folder:
            fens = Path(folder) / 'practice.fen'
            positions = [board.copy()]
            for move in list(board.legal_moves):
                child = board.copy(); child.push(move); positions.append(child)
            for token in ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6'):
                board.push_uci(token); positions.append(board.copy())
            import random
            rng = random.Random(7)
            for _ in range(100):
                if board.is_game_over():
                    board.reset()
                board.push(rng.choice(list(board.legal_moves)))
                if not board.is_game_over():
                    positions.append(board.copy())
            fens.write_text('\n'.join(b.fen() + ' # level=1' for b in positions))
            args = SimpleNamespace(pgn=None, fens=fens, output=Path(folder)/'teacher.pt',
                engine='unused', samples=100, nodes=100, seed=7, max_records=100)
            with patch('chess_ai.teacher.chess.engine.SimpleEngine.popen_uci', return_value=engine):
                generate(args)
            data = load_teacher(args.output)
            self.assertEqual(len(data['levels']), len(data['records']))
            self.assertTrue(set(data['levels']) <= {0, 1, 2})
            from chess_ai.teacher import position_group
            self.assertFalse(set(map(position_group, map(chess.Board, data['fens']))) &
                             set(map(position_group, map(chess.Board, data['validation_fens']))))
            args.previous, args.output = args.output, Path(folder)/'refreshed.pt'
            args.samples, args.max_records = 30, 150
            with patch('chess_ai.teacher.chess.engine.SimpleEngine.popen_uci', return_value=engine):
                generate(args)
            refreshed = load_teacher(args.output)
            self.assertEqual(len(refreshed['records']), 150)
            self.assertEqual(len(refreshed['levels']), 150)


if __name__ == '__main__':
    unittest.main()
