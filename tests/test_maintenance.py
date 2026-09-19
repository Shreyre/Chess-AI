"""Selection must fail closed and persist its resume markers."""
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import torch

from chess_ai.evaluation import play_match
from chess_ai.maintenance import maintenance
from chess_ai.model import ChessNet, atomic_save, load_model, model_snapshot
from chess_ai.screen_player import ScreenPlayer
from chess_ai.telemetry import LiveStatus
from chess_ai.teacher import sample_positions


class MaintenanceCheck(unittest.TestCase):
    def test_selection_refresh_and_screen_follow_latest_model(self):
        torch.set_num_threads(1)
        model = ChessNet(8, 1).eval()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, number in (("best.pt", 10), ("model.pt", 11),
                                 ("latest.pt", 11), ("model-000008.pt", 8), ("model-000006.pt", 6)):
                atomic_save(model_snapshot(model, number), root / name)
            player = ScreenPlayer.__new__(ScreenPlayer)
            player.model = player.model_stamp = None
            player.messages = queue.Queue()
            first = player.refresh_model(root / "model.pt")
            self.assertEqual(player.messages.get_nowait(), ("model", 11))
            config = dict(gate_every=5, seed=7)
            live = LiveStatus(root)
            def match(score, truncated=0):
                return dict(score=score, truncated=truncated), []
            with patch("chess_ai.maintenance.play_match", side_effect=[match(.7), match(.4), match(.8)]):
                maintenance(model, root, config, 11, live)
            self.assertEqual(config["gate_iteration"], 11)
            self.assertIs(player.refresh_model(root / "model.pt"), first)
            with patch("chess_ai.maintenance.play_match") as matches:
                maintenance(model, root, config, 12, live)
                matches.assert_not_called()
            with patch("chess_ai.maintenance.play_match", side_effect=[match(.8, 1), match(.8), match(.8)]):
                maintenance(model, root, config, 16, live)
            self.assertEqual(load_model(root / "best.pt", "cpu")[1]["iteration"], 10)
            with patch("chess_ai.maintenance.play_match", side_effect=InterruptedError):
                maintenance(model, root, config, 21, live)
            self.assertEqual(config["gate_iteration"], 16)
            with patch("chess_ai.maintenance.play_match", side_effect=[match(.6), match(.5), match(.5)]):
                maintenance(model, root, config, 21, live)
            self.assertIs(player.refresh_model(root / "model.pt"), first)
            self.assertTrue(player.messages.empty())
            player.refresh_model(root / "best.pt")
            self.assertEqual(player.messages.get_nowait(), ("model", 21))
            player.refresh_model(root / "model-000008.pt")
            self.assertEqual(player.messages.get_nowait(), ("model", 8))
            config.update(teacher_refresh_every=5, teacher_engine="missing", teacher_data="original")
            with patch("chess_ai.maintenance.generate", side_effect=OSError("engine missing")):
                maintenance(model, root, config, 22, live)
            self.assertEqual(config["teacher_data"], "original")
            self.assertNotIn("teacher_refresh_iteration", config)
            with patch("chess_ai.maintenance.generate"), patch("chess_ai.maintenance.load_teacher"):
                maintenance(model, root, config, 22, live)
            self.assertEqual(config["teacher_refresh_iteration"], 22)

    def test_batched_matches_pair_openings_and_cancel(self):
        model = ChessNet(8, 1).eval()
        report, pgns = play_match(model, model, games=2, simulations=1, max_plies=2)
        self.assertEqual(report["truncated"], 2)
        import io
        games = [chess.pgn.read_game(io.StringIO(pgn)) for pgn in pgns]
        moves = [list(game.mainline_moves()) for game in games]
        self.assertEqual(moves[0][:4], moves[1][:4])
        self.assertEqual([len(m) for m in moves], [6, 6])
        with self.assertRaises(InterruptedError):
            play_match(model, model, stop=lambda: True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.pgn"
            path.write_text("\n\n".join(pgns), encoding="utf-8")
            self.assertEqual(len(sample_positions(path, 20, 7, recent_games=1)), 1)


if __name__ == "__main__":
    unittest.main()
