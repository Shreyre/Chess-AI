"""A disk corpus rotates through training without loading every shard."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.engine
import numpy as np
import torch

from chess_ai.maintenance import maintenance
from chess_ai.model import ChessNet, atomic_save
from chess_ai.teacher import load_teacher, teacher_record
from chess_ai.telemetry import LiveStatus
from chess_ai.training import DEFAULTS, save_training, train


class CorpusCheck(unittest.TestCase):
    def test_rotation_training_resume_and_refresh(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / "corpus"
            corpus.mkdir()
            with self.assertRaisesRegex(ValueError, "no completed"):
                load_teacher(corpus)
            board = chess.Board()
            row = teacher_record(board, [dict(pv=[chess.Move.from_uci("e2e4")],
                score=chess.engine.PovScore(chess.engine.Cp(0), True))])
            for i, value in enumerate((-.5, .5)):
                atomic_save(dict(format_version=1, kind="teacher",
                    records=[(*row[:3], value, True)], levels=[0],
                    validation_records=[row], validation_fens=[board.fen()]),
                    corpus / f"{i}.pt")
            self.assertEqual([load_teacher(corpus, i)["records"][0][3]
                              for i in range(3)], [-.5, .5, -.5])
            (corpus / "incomplete.tmp").write_text("not a dataset")
            model = ChessNet(8, 1)
            config = DEFAULTS | dict(channels=8, blocks=1, batch_size=1,
                train_steps=1, teacher_data=str(corpus))
            save_training(root / "latest.pt", model, torch.optim.AdamW(model.parameters()),
                [], 0, config, np.random.default_rng(7),
                dict(games=0, positions=0, updates=0), {})
            args = SimpleNamespace(**dict.fromkeys(DEFAULTS), resume=str(root / "latest.pt"),
                run_dir=None, iterations=2, teacher_only=True)
            values = []
            def learn(model, optimizer, samples):
                values.append(samples[0][3])
                return dict(loss=0., policy_loss=0., value_loss=0.)
            with patch("chess_ai.training.train_batch", side_effect=learn):
                train(args, "cpu")
            self.assertEqual(values, [-.5, .5])
            saved = torch.load(root / "latest.pt", weights_only=True)
            self.assertEqual(saved["config"]["teacher_data"], str(corpus))
            config.update(teacher_refresh_every=1, teacher_engine="test")
            with patch("chess_ai.maintenance.generate") as generate, \
                 patch("chess_ai.maintenance.load_teacher"):
                maintenance(model, root, config, 2, LiveStatus(root))
            self.assertEqual(Path(config["teacher_data"]), corpus)
            self.assertEqual(generate.call_args.args[0].output.parent, corpus)
            self.assertIsNone(generate.call_args.args[0].previous)


if __name__ == "__main__":
    unittest.main()
