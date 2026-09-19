"""Run with: python -m unittest discover -s tests -v"""

from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch

from chess_ai.model import ACTION_SIZE, ChessNet, atomic_save, encode, evaluate_boards, model_snapshot, move_index
from chess_ai.search import Node, Search, backup, select_move, terminal_value
from chess_ai.training import DEFAULTS, restore_training, save_screen_game, save_training, self_play, train, train_batch
from chess_ai.uci import parse_position, search_limits

ROOT = Path(__file__).resolve().parents[1]


class ChessChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.model = ChessNet(8, 1).eval()

    def test_encoding_and_special_moves(self):
        boards = [chess.Board(),
                  chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
                  chess.Board("7k/P7/8/8/8/8/7p/4K3 w - - 0 1"),
                  chess.Board("7k/P7/8/8/8/8/7p/4K3 b - - 0 1"),
                  chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
                  chess.Board("1r5k/P7/8/8/8/8/8/4K3 w - - 0 1"),
                  chess.Board("7k/8/8/8/8/8/p7/1R2K3 b - - 0 1")]
        self.assertEqual(len([m for m in boards[2].legal_moves if m.promotion]), 4)
        self.assertTrue(any(boards[4].is_en_passant(m) for m in boards[4].legal_moves))
        self.assertEqual(sum(boards[1].is_castling(m) for m in boards[1].legal_moves), 2)
        for board in boards[5:]:
            self.assertEqual(sum(bool(move.promotion) and board.is_capture(move)
                                 for move in board.legal_moves), 4)
        rng = np.random.default_rng(8)
        random_board = chess.Board()
        for _ in range(160):
            boards.append(random_board.copy())
            if random_board.is_game_over():
                random_board.reset()
            moves = list(random_board.legal_moves)
            random_board.push(moves[int(rng.integers(len(moves)))])
        for board in boards:
            indices = [move_index(move, board.turn) for move in board.legal_moves]
            self.assertEqual(len(set(indices)), len(indices))
            self.assertTrue(all(0 <= index < ACTION_SIZE for index in indices))
            np.testing.assert_array_equal(encode(board), encode(board.mirror()))
        repeated = chess.Board()
        for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            repeated.push_uci(move)
        self.assertEqual(encode(repeated)[19, 0, 0], 1)
        self.assertIsNone(terminal_value(repeated))  # Claims are optional, not automatic.
        for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            repeated.push_uci(move)
        self.assertEqual(terminal_value(repeated), 0)

    def test_search_perspective_mate_and_legality(self):
        parent, child = Node(), Node()
        backup([parent, child], -1)
        self.assertEqual(parent.value, 1)
        self.assertEqual(child.value, -1)
        for parameter in self.model.parameters():
            parameter.data.zero_()
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        original = board.fen()
        move, search = select_move(self.model, board, 128)
        self.assertEqual(board.fen(), original)
        self.assertEqual(search.root.visits, 128)
        board.push(move)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(terminal_value(board), -1)
        self.assertIsNone(select_move(self.model, board, 2)[0])
        black = chess.Board(original).mirror()
        move, _ = select_move(self.model, black, 128)
        black.push(move)
        self.assertTrue(black.is_checkmate())
        board = chess.Board()
        moves, probabilities, value = evaluate_boards(self.model, [board])[0]
        self.assertAlmostEqual(float(probabilities.sum()), 1, places=6)
        self.assertTrue(all(move in board.legal_moves for move in moves))
        self.assertTrue(-1 <= value <= 1)

    def test_search_reuses_working_board_without_losing_history(self):
        repeated = chess.Board()
        for token in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            repeated.push_uci(token)
        boards = [repeated,
                  chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
                  chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
                  chess.Board("7k/P7/8/8/8/8/8/4K3 w - - 0 1")]
        for board in boards:
            search = Search(board)
            original = board.fen(), list(board.move_stack)
            working_board = None
            for move in list(board.legal_moves):
                # Visit sibling branches, including castling, en passant and promotion.
                search.root.children = {move: Node()}
                leaf, node, path = search.leaf()
                expected = board.copy()
                expected.push(move)
                self.assertEqual(leaf.fen(), expected.fen())
                self.assertEqual(leaf.move_stack, expected.move_stack)
                np.testing.assert_array_equal(encode(leaf), encode(expected))
                self.assertEqual(terminal_value(leaf), terminal_value(expected))
                self.assertEqual(path, [search.root, node])
                if working_board is not None:
                    self.assertIs(leaf, working_board)
                working_board = leaf
            self.assertEqual((board.fen(), board.move_stack), original)
            self.assertEqual((search.board.fen(), search.board.move_stack), original)

    def test_learning_truncation_and_resume(self):
        rng = np.random.default_rng(17)
        config = DEFAULTS | dict(games=2, parallel_games=2, simulations=2, max_plies=4,
                                 train_steps=2, batch_size=4, channels=8, blocks=1)
        records, outcomes, pgns = self_play(self.model, config, rng, progress=None)
        self.assertEqual(outcomes["truncated"], 2)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(not row[4] for row in records))
        self.assertTrue(all('[Result "*"]' in pgn for pgn in pgns))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        before = self.model.policy.weight.detach().clone()
        stats = train_batch(self.model, optimizer, records)
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertEqual(stats["value_loss"], 0)
        self.assertFalse(torch.equal(before, self.model.policy.weight))
        # Real terminal labels must reach the value head as well.
        labeled = [(row[0], row[1], row[2], 1.0, True) for row in records]
        before = self.model.value[-2].weight.detach().clone()
        stats = train_batch(self.model, optimizer, labeled)
        self.assertGreater(stats["value_loss"], 0)
        self.assertFalse(torch.equal(before, self.model.value[-2].weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            totals = dict(games=2, positions=8, updates=2)
            save_training(path, self.model, optimizer, deque(records), 1, config, rng, totals, stats)
            expected_random = rng.integers(1000000)
            train_batch(self.model, optimizer, records)
            restored, opt, replay, restored_config, restored_rng, data = restore_training(path, "cpu", {})
            self.assertEqual(restored_rng.integers(1000000), expected_random)
            self.assertEqual(data["iteration"], 1)
            self.assertEqual(restored_config, config)
            self.assertEqual(len(replay), 8)
            with self.assertRaises(ValueError):
                restore_training(path, "cpu", {"seed": 99})
            train_batch(restored, opt, records)
            for actual, expected in zip(restored.parameters(), self.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_positions_and_time_limits(self):
        board = parse_position("startpos moves e2e4 e7e5".split())
        self.assertEqual(len(board.move_stack), 2)
        for tokens in ("startpos moves e2e5", "startpos moves 0000", "fen bad"):
            with self.assertRaises(ValueError):
                parse_position(tokens.split())
        self.assertEqual(search_limits(["nodes", "12"], True, 128), (12, None))
        self.assertIsNotNone(search_limits(["wtime", "1000", "winc", "50"], True, 128)[1])
        with self.assertRaises(ValueError):
            search_limits(["movetime", "-1"], True, 128)

    def test_screen_games_update_once_and_resume(self):
        board, searches = chess.Board(), {}
        for ply, token in enumerate(("f2f3", "e7e5", "g2g4", "d8h4")):
            _, search = select_move(self.model, board, 2)
            moves, policy = search.policy(1)
            searches[ply] = (token, torch.tensor([move_index(m, board.turn) for m in moves]), torch.from_numpy(policy))
            board.push_uci(token)
        searches[2] = ("h2h4", searches[2][1], searches[2][2])  # An unconfirmed attempt must be excluded.
        searches[4] = searches[0]  # A pending move beyond the observed history is excluded too.
        config = DEFAULTS | dict(channels=8, blocks=1, games=2, parallel_games=2, simulations=2,
                                 max_plies=4, train_steps=2, batch_size=4)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            checkpoint = directory / "latest.pt"
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
            save_training(checkpoint, self.model, optimizer, [], 0, config, np.random.default_rng(7),
                          dict(games=0, positions=0, updates=0), {})
            game = save_screen_game(directory, board, searches)
            records = torch.load(game, weights_only=True)["records"]
            self.assertEqual([row[3] for row in records], [-1.0, 1.0, 1.0])
            self.assertTrue(all(row[4] for row in records))
            with self.assertRaises(ValueError):
                save_screen_game(directory, chess.Board(), searches)
            with self.assertRaises(ValueError):
                save_screen_game(directory, board, searches, "1-0")
            args = SimpleNamespace(**dict.fromkeys(DEFAULTS), resume=str(checkpoint), run_dir=None,
                                   iterations=1, external_only=True)
            with patch("chess_ai.training.self_play", side_effect=AssertionError("Unexpected self-play")):
                train(args, "cpu")
                learned = torch.load(checkpoint, weights_only=True)
                self.assertEqual(learned["totals"], dict(games=1, positions=3, updates=2))
                self.assertEqual(learned["screen_games"], [game.name])
                self.assertFalse(torch.equal(learned["model"]["policy.weight"], self.model.policy.weight))
                self.assertFalse(torch.equal(learned["model"]["value.5.weight"], self.model.value[-2].weight))
                exported = torch.load(directory / "model.pt", weights_only=True)
                torch.testing.assert_close(exported["model"]["policy.weight"], learned["model"]["policy.weight"])
                train(args, "cpu")
                repeated = torch.load(checkpoint, weights_only=True)
                self.assertEqual(repeated["iteration"], 1)
                self.assertEqual(repeated["totals"], learned["totals"])
            # Dashboard self-play also imports new external games, without importing the first twice.
            ongoing = board.root()
            ongoing.push_uci("f2f3")
            second = save_screen_game(directory, ongoing, searches, "1/2-1/2")
            args.external_only = False
            train(args, "cpu")
            resumed = torch.load(checkpoint, weights_only=True)
            # Nine new positions need three four-position updates under scaled learning.
            self.assertEqual(resumed["totals"], dict(games=4, positions=12, updates=5))
            self.assertEqual(set(resumed["screen_games"]), {game.name, second.name})
            self.assertEqual(resumed["metrics"]["screen_games"], 1)
            self.assertFalse((directory / ".training.lock").exists())

    def test_cli_training_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "chess_ai", "train", "--iterations", "1",
                       "--run-dir", directory, "--games", "2", "--parallel-games", "2",
                       "--simulations", "2", "--max-plies", "4", "--train-steps", "2",
                       "--batch-size", "4", "--channels", "8", "--blocks", "1", "--device", "cpu"]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
            checkpoint = Path(directory) / "latest.pt"
            resumed = subprocess.run([sys.executable, "-m", "chess_ai", "train", "--resume", str(checkpoint),
                                      "--iterations", "1", "--device", "cpu"], cwd=ROOT, check=True,
                                     capture_output=True, text=True, timeout=60)
            self.assertIn("Starting at iteration 1", resumed.stdout)
            data = torch.load(checkpoint, weights_only=True)
            self.assertEqual(data["iteration"], 2)
            self.assertEqual(data["totals"], dict(games=4, positions=16, updates=4))
            self.assertFalse((Path(directory) / ".training.lock").exists())
            metrics = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["iteration"] for row in metrics], [1, 2])

    def test_uci_handshake_play_stop_and_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_save(model_snapshot(self.model), path)
            process = subprocess.Popen([sys.executable, "-m", "chess_ai", "uci", "--checkpoint", str(path),
                                        "--device", "cpu", "--simulations", "4"], cwd=ROOT,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, bufsize=1)
            lines = queue.Queue()

            def read():
                for line in process.stdout:
                    lines.put(line.strip())
                lines.put("PROCESS_ENDED")

            reader = threading.Thread(target=read, daemon=True)
            reader.start()

            def send(text):
                process.stdin.write(text + "\n")
                process.stdin.flush()

            def until(prefix):
                for _ in range(30):
                    line = lines.get(timeout=30)
                    if line.startswith(prefix):
                        return line
                    if line == "PROCESS_ENDED":
                        self.fail(process.stderr.read())
                self.fail(f"Missing response: {prefix}")

            try:
                send("uci")
                until("uciok")
                send("isready")
                until("readyok")
                send("position startpos moves e2e4")
                send("go nodes 4")
                move = chess.Move.from_uci(until("bestmove").split()[1])
                board = chess.Board()
                board.push_uci("e2e4")
                self.assertIn(move, board.legal_moves)
                send("go infinite")
                send("isready")
                until("readyok")
                send("stop")
                self.assertIn(chess.Move.from_uci(until("bestmove").split()[1]), board.legal_moves)
                send("go movetime 20")
                self.assertIn(chess.Move.from_uci(until("bestmove").split()[1]), board.legal_moves)
                send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
                send("go nodes 4")
                self.assertEqual(until("bestmove"), "bestmove 0000")
                send("quit")
                process.wait(timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr.read())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                reader.join(timeout=2)
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
