"""Screen recognition and tracking checks; never click the user's desktop."""

from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
try:
    from PIL import Image, ImageDraw
except ImportError:
    raise unittest.SkipTest("Install the screen extra to run screen-player checks")

from chess_ai.screen_vision import BoardArea, GameTracker, PieceReader, UncertainBoard, align_board_area, board_labels, matching_move, screen_square


def render(board, white_bottom=True, assets=None, highlighted=()):
    image = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(image)
    for row in range(8):
        for col in range(8):
            square = screen_square(row, col, white_bottom)
            background = (235, 237, 210) if (row+col) % 2 == 0 else (116, 149, 82)
            if square in highlighted:
                background = (246, 246, 105) if (row+col) % 2 == 0 else (186, 202, 68)
            x, y = col*64, row*64
            draw.rectangle((x, y, x+63, y+63), fill=background)
            piece = board.piece_at(square)
            if piece:
                if assets:
                    name = ("w" if piece.color else "b") + piece.symbol().lower() + ".png"
                    with Image.open(assets / name) as source:
                        sprite = source.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
                    image.paste(sprite, (x, y), sprite)
                else:
                    color = (249, 249, 249) if piece.color else (30, 30, 30)
                    # Distinct calibration glyphs with contrasting outlines on both square colors.
                    draw.rectangle((x+15, y+18, x+48, y+47), fill=color, outline=(90, 90, 90), width=2)
                    for bit in range(3):
                        if piece.piece_type & (1 << bit):
                            draw.rectangle((x+18+bit*10, y+7, x+23+bit*10, y+20), fill=color)
    return image


class ScreenChecks(unittest.TestCase):
    def test_drag_alignment_before_recognition(self):
        for white_bottom in (True, False):
            board = chess.Board()
            screen = Image.new("RGB", (740, 740), (49, 46, 43))
            screen.paste(render(board, white_bottom).resize((690, 690)), (20, 20))
            for selected in (BoardArea(19, 17, 710, 715), BoardArea(22, 18, 708, 712)):
                area = align_board_area(screen, selected)
                for actual, expected in zip(area.bbox, (20, 20, 710, 710)):
                    self.assertLessEqual(abs(actual - expected), 1)
                reader = PieceReader(screen.crop(area.bbox), white_bottom)
                after = board.copy()
                after.push_uci("h2h4")
                after.push_uci("b8c6")
                moved = screen.copy()
                moved.paste(render(after, white_bottom, highlighted=(chess.B8, chess.C6)).resize((690, 690)), (20, 20))
                self.assertEqual(reader.read(moved.crop(area.bbox)), board_labels(after))
        with self.assertRaises(ValueError):
            align_board_area(Image.new("RGB", (512, 512), "white"), BoardArea(0, 0, 512, 512))

    def test_region_coordinates_on_both_orientations(self):
        area = BoardArea(-800, 200, -288, 712)
        self.assertEqual(area.center(chess.A8), (-768, 232))
        self.assertEqual(area.center(chess.H1), (-320, 680))
        self.assertEqual(area.center(chess.A8, False), (-320, 680))
        self.assertEqual(area.center(chess.H1, False), (-768, 232))
        for bounds in ((0, 0, 50, 50), (0, 0, 400, 700), (50, 50, 0, 0)):
            with self.assertRaises(ValueError):
                BoardArea(*bounds)

    def check_vision(self, assets=None):
        for white_bottom in (True, False):
            board = chess.Board()
            reader = PieceReader(render(board, white_bottom, assets), white_bottom)
            for token in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"):
                move = chess.Move.from_uci(token)
                board.push(move)
                image = render(board, white_bottom, assets, (move.from_square, move.to_square))
                self.assertEqual(reader.read(image), board_labels(board))
            promotion = chess.Board("7k/P7/8/8/8/8/8/4K3 w - - 0 1")
            for piece in ("q", "r", "b", "n"):
                candidate = promotion.copy()
                candidate.push_uci("a7a8"+piece)
                self.assertEqual(reader.read(render(candidate, white_bottom, assets)), board_labels(candidate))

    def test_calibrated_recognition_and_highlights(self):
        self.check_vision()
        with self.assertRaises(ValueError):
            PieceReader(Image.new("RGB", (512, 512), "white"))
        moved = chess.Board()
        moved.push_uci("e2e4")
        with self.assertRaises(ValueError):
            PieceReader(render(moved))
        reader = PieceReader(render(chess.Board()))
        obscured = render(chess.Board())
        ImageDraw.Draw(obscured).rectangle((64, 0, 127, 63), fill=(150, 30, 180))
        self.assertNotEqual(reader.read(obscured), board_labels(chess.Board()))

    def test_confirm_moves_and_fast_opponent(self):
        tracker = GameTracker(chess.WHITE)
        tracker.pending = chess.Move.from_uci("e2e4")
        self.assertEqual(tracker.observe(board_labels(tracker.board)), "pending")
        after = tracker.board.copy()
        after.push_uci("e2e4")
        after.push_uci("e7e5")
        self.assertEqual(tracker.observe(board_labels(after)), "reply")
        self.assertEqual(tracker.board.fen(), after.fen())
        self.assertEqual(len(tracker.board.move_stack), 2)
        self.assertIsNone(tracker.pending)
        # An unexpected manual move on our turn must never be silently accepted.
        after.push_uci("g1f3")
        with self.assertRaises(UncertainBoard):
            tracker.observe(board_labels(after))
        black = GameTracker(chess.BLACK)
        initial = chess.Board()
        initial.push_uci("d2d4")
        self.assertEqual(black.observe(board_labels(initial)), "opponent_move")

    def test_castling_en_passant_and_promotions(self):
        for fen, tokens in (
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e1c1"]),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
            ("7k/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]),
        ):
            board = chess.Board(fen)
            for token in tokens:
                after = board.copy()
                after.push_uci(token)
                self.assertEqual(matching_move(board, board_labels(after)), chess.Move.from_uci(token))
                self.assertEqual(board.fen(), fen)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "runs/screen-validation/wp.png").exists(),
                         "Chess.com graphic samples have not been downloaded")
    def test_chess_com_piece_graphics(self):
        assets = Path(__file__).resolve().parents[1] / "runs/screen-validation"
        self.check_vision(assets)
        for size in (424, 517, 640):
            board = chess.Board()
            reader = PieceReader(render(board, assets=assets).resize((size, size)))
            for move in ("d2d4", "g8f6", "c1g5", "e7e6", "b1c3"):
                board.push_uci(move)
                self.assertEqual(reader.read(render(board, assets=assets).resize((size, size))), board_labels(board))

    @unittest.skipUnless(sys.platform == "win32", "Windows input contract")
    def test_input_is_bounded_and_cancellable(self):
        from chess_ai.windows_input import WindowsInput
        events = []

        def send(count, values, size):
            if count == 1:
                events.append(values._obj.data.mi.dwFlags)
            else:
                events.extend(value.data.mi.dwFlags for value in values)
            return count

        windows = WindowsInput.__new__(WindowsInput)
        windows.api = SimpleNamespace(GetAsyncKeyState=lambda _: 0, SendInput=send)
        windows.window_at = lambda _: 17
        windows.foreground = lambda: 17
        windows.desktop = lambda: (-500, 0, 1500, 1000)
        stop = threading.Event()
        with patch("chess_ai.windows_input.time.sleep"):
            windows.click((-100, 100), 17, stop)
        self.assertEqual(events, [0x8000 | 0x4000 | 1, 2, 4])
        events.clear()
        stop.set()
        with self.assertRaises(InterruptedError):
            windows.click((-100, 100), 17, stop)
        self.assertFalse(events)
        stop.clear()
        windows.foreground = lambda: 99
        with self.assertRaises(InterruptedError):
            windows.click((-100, 100), 17, stop)
        self.assertFalse(events)

    @unittest.skipUnless(sys.platform == "win32", "Windows UI layout check")
    def test_native_controller_builds_without_input(self):
        import ctypes
        import tkinter as tk
        import torch
        from chess_ai.screen_player import ScreenPlayer
        from chess_ai.model import move_index
        from chess_ai.windows_input import Input
        self.assertEqual(ctypes.sizeof(Input), 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        root = tk.Tk()
        root.withdraw()
        try:
            controller = ScreenPlayer(root, Path("runs/main/model.pt"))
            root.update_idletasks()
            self.assertTrue(controller.start_button.instate(["disabled"]))
            self.assertEqual(controller.color.get(), "White")
            self.assertIsNone(controller.reader)
            self.assertTrue(controller.learn.get())
            board = chess.Board()
            moves = list(board.legal_moves)
            controller.searches = {0: ("f2f3", torch.tensor([move_index(m, board.turn) for m in moves]),
                                        torch.full((len(moves),), 1 / len(moves)))}
            for token in ("f2f3", "e7e5", "g2g4", "d8h4"):
                board.push_uci(token)
            controller.tracker = GameTracker(chess.WHITE)
            controller.tracker.board = board
            controller.result.set("Black won")
            with tempfile.TemporaryDirectory() as directory:
                controller.game_checkpoint = Path(directory) / "model.pt"
                (Path(directory) / "latest.pt").touch()
                controller.manual_finish()
                controller.manual_finish()
                self.assertEqual(len(list((Path(directory) / "screen-games").glob("*.pt"))), 1)
                self.assertTrue(controller.game_saved)
                self.assertTrue(controller.start_button.instate(["disabled"]))
                with patch("chess_ai.screen_player.subprocess.Popen") as launch:
                    launch.return_value.poll.return_value = None
                    controller.poll()
                    self.assertIn("--external-only", launch.call_args.args[0])
                    self.assertTrue(launch.call_args.kwargs["creationflags"])
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
