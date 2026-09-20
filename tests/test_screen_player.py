"""Screen recognition and tracking checks; never click the user's desktop."""

from pathlib import Path
import queue
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


def render(board, white_bottom=True, assets=None, highlighted=(), plain_background=None):
    image = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(image)
    for row in range(8):
        for col in range(8):
            square = screen_square(row, col, white_bottom)
            background = (235, 237, 210) if (row+col) % 2 == 0 else (116, 149, 82)
            if plain_background is not None:
                background = plain_background
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
    def test_black_bottom_opening_selection(self):
        from chess_ai.screen_player import ScreenPlayer
        player = ScreenPlayer.__new__(ScreenPlayer)
        player.color = SimpleNamespace(get=lambda: "Black")
        player.tolerance = SimpleNamespace(get=lambda: 0.18)
        player.result = SimpleNamespace(set=lambda _: None)
        player.windows = SimpleNamespace(window_at=lambda _: 17)
        player.session_path = None
        with Image.open(Path(__file__).parent / "fixtures/black-bottom-e4.png") as reported:
            for white_bottom in (False, True):
                player.orientation = SimpleNamespace(get=lambda: "White at bottom" if white_bottom else "Black at bottom")
                for token in (None, "e2e4", "d2d4", "g1f3", "b1c3"):
                    with self.subTest(white_bottom=white_bottom, move=token):
                        board = chess.Board()
                        if token:
                            board.push_uci(token)
                        image = reported.copy() if not white_bottom and token == "e2e4" else render(
                            board, white_bottom, highlighted=() if token is None else
                            (board.peek().from_square, board.peek().to_square))
                        screen = Image.new("RGB", (image.width + 40, image.height + 40), (49, 46, 43))
                        screen.paste(image, (20, 20))
                        area = align_board_area(screen, BoardArea(19, 17, image.width + 20, image.height + 23))
                        for actual, expected in zip(area.bbox, (20, 20, image.width + 20, image.height + 20)):
                            self.assertLessEqual(abs(actual - expected), 1)
                        player.attach_position(screen.crop(area.bbox), area)
                        self.assertEqual(player.tracker.board.move_stack, board.move_stack)
                        self.assertEqual(player.tracker.board.fen(), board.fen())
                        self.assertEqual(player.reader.read(image), board_labels(board))
                        self.assertEqual(player.tracker.color, chess.BLACK)
                # A later game requires an explicit FEN; failed selection preserves the current game.
                saved = player.tracker
                board.push_uci("e7e5")
                with self.assertRaises(ValueError):
                    player.attach_position(render(board, white_bottom), BoardArea(0, 0, 512, 512))
                self.assertIs(player.tracker, saved)

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

    def test_legal_move_dots_are_empty_squares(self):
        for white_bottom in (True, False):
            board = chess.Board()
            board.push_uci("a2a3")
            reader = PieceReader(render(chess.Board(), white_bottom), white_bottom)
            image = render(board, white_bottom, highlighted=(chess.A3,))
            draw = ImageDraw.Draw(image)
            for row in range(8):
                for col in range(8):
                    if board.piece_at(screen_square(row, col, white_bottom)) is None:
                        x, y = col * 64, row * 64
                        draw.ellipse((x+22, y+22, x+42, y+42), fill=(90, 90, 90))
            self.assertEqual(reader.read(image), board_labels(board))
            # Actual reported square: the rank digit protrudes beside the move dot.
            with Image.open(Path(__file__).parent / "fixtures/legal-move-dot.png") as dot:
                image.paste(dot.resize((64, 64)), (0, 4*64))
            self.assertEqual(reader.read(image), board_labels(board))
            # Larger overlays still pause recognition instead of hiding a piece.
            ImageDraw.Draw(image).rectangle((64*3+8, 64*3+8, 64*4-8, 64*4-8), fill=(150, 30, 180))
            with self.assertRaises(UncertainBoard):
                reader.read(image)

    def test_white_king_on_matching_light_background(self):
        fixtures = Path(__file__).parent / "fixtures"
        with Image.open(fixtures / "white-king-dark-square.png") as dark, \
                Image.open(fixtures / "white-king-light-square.png") as light:
            for white_bottom in (True, False):
                board = chess.Board()
                initial = render(board, white_bottom)
                area = BoardArea(0, 0, 512, 512)
                x, y = area.center(chess.E1, white_bottom)
                initial.paste(dark.resize((64, 64)), (x-32, y-32))
                reader = PieceReader(initial, white_bottom)
                board.set_piece_at(chess.H3, board.remove_piece_at(chess.E1))
                moved = render(board, white_bottom)
                x, y = area.center(chess.H3, white_bottom)
                moved.paste(light.resize((64, 64)), (x-32, y-32))
                self.assertEqual(reader.read(moved), board_labels(board))

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

    def test_resume_rechecks_pending_move_before_allowing_one_retry(self):
        for tokens, result in (((), "retry"), (("e2e4",), "own_move"),
                               (("e2e4", "e7e5"), "reply")):
            with self.subTest(tokens=tokens):
                tracker = GameTracker(chess.WHITE)
                tracker.pending = chess.Move.from_uci("e2e4")
                screen = chess.Board()
                for token in tokens:
                    screen.push_uci(token)
                self.assertEqual(tracker.observe(board_labels(screen), retry_pending=True), result)
                self.assertIsNone(tracker.pending)
                self.assertEqual(tracker.board.move_stack, screen.move_stack)
        tracker.pending = chess.Move.from_uci("g1f3")
        with self.assertRaises(UncertainBoard):
            tracker.observe((".",) * 64, retry_pending=True)
        self.assertEqual(tracker.pending.uci(), "g1f3")
        tracker.board = chess.Board("7k/P7/8/8/8/8/8/4K3 w - - 0 1")
        tracker.pending = chess.Move.from_uci("a7a8n")
        self.assertEqual(tracker.observe(board_labels(tracker.board), retry_pending=True), "retry")
        self.assertIsNone(tracker.pending)

    def test_automatic_promotion_menu_and_confirmation(self):
        import numpy as np
        from chess_ai.screen_player import ScreenPlayer
        assets = Path(__file__).resolve().parents[1] / "runs/screen-validation"
        assets = assets if (assets / "wp.png").exists() else None
        for white_bottom in (True, False):
            for color in (chess.WHITE, chess.BLACK):
                for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    with self.subTest(white_bottom=white_bottom, color=color, piece=piece):
                        player = ScreenPlayer.__new__(ScreenPlayer)
                        player.tracker = GameTracker(color)
                        # Capture promotions exercise the destination file, not the pawn's file.
                        board = chess.Board("1r5k/P7/8/8/8/8/7p/1R2K3 w - - 0 1")
                        if not color:
                            board = board.mirror()
                        move = chess.Move(chess.A7 if color else chess.A2,
                                          chess.B8 if color else chess.B1, promotion=piece)
                        self.assertIn(move, board.legal_moves)
                        player.tracker.board = board
                        player.searches = {}
                        player.stop_signal = threading.Event()
                        player.messages = queue.Queue()
                        player.area = BoardArea(-700, 100, -188, 612)
                        player.target = 17
                        player.reader = PieceReader(render(chess.Board(), white_bottom, assets), white_bottom)
                        menu = board.copy()
                        menu.remove_piece_at(move.from_square)
                        choices = {}
                        for offset, option in enumerate((chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP)):
                            square = move.to_square + (-8 if color else 8) * offset
                            menu.set_piece_at(square, chess.Piece(option, color))
                            choices[option] = square
                        background = (255, 255, 255) if assets else (220, 220, 220)
                        menu_image = render(menu, white_bottom, assets, plain_background=background)
                        self.assertEqual(player.reader.promotion_square(menu_image, move, color), choices[piece])
                        self.assertIsNone(player.reader.promotion_square(render(board, white_bottom, assets), move, color))
                        after = board.copy()
                        after.push(move)
                        clicks = []
                        frames = [render(board, white_bottom, assets), menu_image,
                                  render(after, white_bottom, assets, highlighted=(move.from_square, move.to_square))]

                        def capture(**_):
                            return frames[0 if len(clicks) < 2 else 1 if len(clicks) == 2 else 2]

                        player.windows = SimpleNamespace(
                            stopped=lambda _: player.stop_signal.is_set() or bool(player.tracker.board.move_stack),
                            foreground=lambda: 17, park_pointer=lambda *_: None, window_at=lambda _: 17,
                            click=lambda point, *_: clicks.append(point))
                        search = SimpleNamespace(policy=lambda _: ([move], np.array([1.0], dtype=np.float32)))
                        with patch.object(ScreenPlayer, "refresh_model", return_value=None), \
                                patch("chess_ai.screen_player.select_move", return_value=(move, search)), \
                                patch("chess_ai.screen_player.ImageGrab.grab", side_effect=capture), \
                                patch.object(player.stop_signal, "wait", return_value=False):
                            player.play_loop(Path("unused.pt"), 1, 0.2)
                        self.assertEqual(clicks, [player.area.center(square, white_bottom) for square in
                                                 (move.from_square, move.to_square, choices[piece])])
                        self.assertEqual(player.tracker.board.fen(), after.fen())
                        self.assertIsNone(player.tracker.pending)

    def test_reported_promotion_menu_with_saved_calibration(self):
        import numpy as np
        fixtures = Path(__file__).parent / "fixtures"
        reader = PieceReader.__new__(PieceReader)
        with np.load(fixtures / "promotion-calibration.npz") as saved:
            reader.templates = dict(saved)
        reader.symbols = tuple(reader.templates)
        reader.white_bottom, reader.tolerance = True, 0.18
        with Image.open(fixtures / "promotion-menu.png") as image:
            for piece, square in (("q", chess.G8), ("n", chess.G7),
                                  ("r", chess.G6), ("b", chess.G5)):
                move = chess.Move.from_uci("g7g8" + piece)
                self.assertEqual(reader.promotion_square(image, move, chess.WHITE), square)
            # Menu tolerance must not weaken normal board recognition.
            with self.assertRaises(UncertainBoard):
                reader.read(image)
            obscured = image.copy()
            ImageDraw.Draw(obscured).rectangle((555, 278, 647, 369), fill="purple")
            self.assertIsNone(reader.promotion_square(obscured, move, chess.WHITE))

    def test_pending_promotion_resumes_safely(self):
        from chess_ai.screen_player import ScreenPlayer
        for scenario in ("menu", "ignored", "unrecognized", "stopped", "fast_reply", "autoqueen"):
            with self.subTest(scenario=scenario):
                player = ScreenPlayer.__new__(ScreenPlayer)
                player.tracker = GameTracker(chess.WHITE)
                board = chess.Board("7k/P7/8/8/8/8/8/4K3 w - - 0 1")
                move = chess.Move.from_uci("a7a8n")
                player.tracker.board = board
                player.tracker.pending = move
                player.searches = {}
                player.stop_signal = threading.Event()
                player.messages = queue.Queue()
                player.area = BoardArea(0, 0, 512, 512)
                player.target = 17
                player.reader = PieceReader(render(chess.Board()))
                menu = board.copy()
                menu.remove_piece_at(chess.A7)
                for square, kind in zip((chess.A8, chess.A7, chess.A6, chess.A5),
                                        (chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP)):
                    menu.set_piece_at(square, chess.Piece(kind, chess.WHITE))
                menu_image = render(menu, plain_background=(220, 220, 220))
                if scenario == "unrecognized":
                    menu_image = Image.new("RGB", (512, 512), "purple")
                after = board.copy()
                after.push_uci("a7a8q" if scenario == "autoqueen" else "a7a8n")
                if scenario == "fast_reply":
                    after.push_uci("h8g8")
                clicks = []

                def click(point, *_):
                    if scenario == "stopped":
                        raise InterruptedError("Stopped with F8")
                    clicks.append(point)

                def capture(**_):
                    return render(after) if scenario in ("fast_reply", "autoqueen") or (
                        clicks and scenario == "menu") else menu_image

                player.windows = SimpleNamespace(
                    stopped=lambda _: player.stop_signal.is_set() or bool(player.tracker.board.move_stack),
                    foreground=lambda: 17, park_pointer=lambda *_: None, click=click)
                with patch.object(ScreenPlayer, "refresh_model", return_value=None), \
                        patch("chess_ai.screen_player.select_move") as choose, \
                        patch("chess_ai.screen_player.ImageGrab.grab", side_effect=capture), \
                        patch("chess_ai.screen_player.time.monotonic", side_effect=range(1, 100)), \
                        patch.object(player.stop_signal, "wait", return_value=False):
                    player.play_loop(Path("unused.pt"), 1, 0.2)
                choose.assert_not_called()
                self.assertEqual(clicks, [(32, 96)] if scenario in ("menu", "ignored") else [])
                self.assertEqual(player.tracker.board.fen(), after.fen() if scenario in ("menu", "fast_reply") else board.fen())
                self.assertEqual(player.tracker.pending, None if scenario in ("menu", "fast_reply") else move)

    def test_resume_retries_missed_clicks_once_then_pauses(self):
        import numpy as np
        from chess_ai.screen_player import ScreenPlayer
        player = ScreenPlayer.__new__(ScreenPlayer)
        player.tracker = GameTracker(chess.WHITE)
        move = chess.Move.from_uci("e2e4")
        player.tracker.pending = move
        player.searches = {0: "old unconfirmed target"}
        player.stop_signal = threading.Event()
        player.messages = queue.Queue()
        player.area = BoardArea(0, 0, 512, 512)
        player.target = 17
        clicks = []
        checks = iter([False] * 30 + [True])
        player.windows = SimpleNamespace(stopped=lambda _: next(checks), foreground=lambda: 17,
                                         park_pointer=lambda *_: None, window_at=lambda _: 17,
                                         click=lambda point, *_: clicks.append(point))
        player.reader = PieceReader(render(chess.Board()))
        search = SimpleNamespace(policy=lambda _: ([move], np.array([1.0], dtype=np.float32)))
        # The game ignores both clicks: resume may try once, never loop clicking.
        with patch.object(ScreenPlayer, "refresh_model", return_value=None), \
                patch("chess_ai.screen_player.select_move", return_value=(move, search)), \
                patch("chess_ai.screen_player.ImageGrab.grab", return_value=render(chess.Board())), \
                patch("chess_ai.screen_player.time.monotonic", side_effect=range(1, 100)), \
                patch.object(player.stop_signal, "wait", return_value=False):
            player.play_loop(Path("unused.pt"), 1, 0.2)
        self.assertEqual(clicks, [(288, 416), (288, 288)])
        self.assertEqual(player.tracker.board.move_stack, [])
        self.assertEqual(player.tracker.pending, move)
        self.assertEqual(player.searches[0][0], "e2e4")
        self.assertTrue(player.stop_signal.is_set())

    def test_undo_recovers_an_exact_earlier_position(self):
        tracker = GameTracker(chess.WHITE)
        for token in ("e2e4", "e7e5", "g1f3", "b8c6"):
            tracker.board.push_uci(token)
        tracker.pending = chess.Move.from_uci("f1c4")
        before = tracker.board.copy()
        before.pop()
        before.pop()
        self.assertEqual(tracker.observe(board_labels(before)), "rewind")
        self.assertEqual(tracker.board.move_stack, before.move_stack)
        self.assertIsNone(tracker.pending)
        # A repetition is a forward move, not an undo to the same old position.
        tracker = GameTracker(chess.WHITE)
        for token in ("g1f3", "g8f6", "f3g1"):
            tracker.board.push_uci(token)
        self.assertEqual(tracker.observe(board_labels(chess.Board())), "opponent_move")
        self.assertEqual(len(tracker.board.move_stack), 4)

    def test_undo_discards_learning_targets_for_removed_moves(self):
        from chess_ai.screen_player import ScreenPlayer
        player = ScreenPlayer.__new__(ScreenPlayer)
        player.tracker = GameTracker(chess.BLACK)
        for token in ("e2e4", "e7e5", "g1f3", "b8c6"):
            player.tracker.board.push_uci(token)
        earlier = player.tracker.board.copy()
        earlier.pop()
        earlier.pop()
        player.searches = {1: "keep", 3: "undone", 4: "unconfirmed"}
        player.stop_signal = threading.Event()
        player.messages = queue.Queue()
        player.area = BoardArea(0, 0, 512, 512)
        player.target = 17
        checks = iter((False, False, True))
        player.windows = SimpleNamespace(stopped=lambda _: next(checks), foreground=lambda: 17,
                                         park_pointer=lambda *_: None)
        player.reader = SimpleNamespace(read=lambda _: board_labels(earlier))
        with patch.object(ScreenPlayer, "refresh_model", return_value=None), \
                patch("chess_ai.screen_player.ImageGrab.grab", return_value=render(earlier)):
            player.play_loop(Path("unused.pt"), 1, 0.2)
        self.assertEqual(player.searches, {1: "keep"})
        self.assertEqual(player.tracker.board.move_stack, earlier.move_stack)

    def test_screen_model_follows_latest_saved_iteration(self):
        import torch
        from chess_ai.model import ChessNet, atomic_save, model_snapshot
        from chess_ai.screen_player import ScreenPlayer

        player = ScreenPlayer.__new__(ScreenPlayer)
        player.model = player.model_stamp = None
        player.messages = queue.Queue()
        model = ChessNet(channels=8, blocks=1)
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "model.pt"
            latest = selected.with_name("latest.pt")
            atomic_save(model_snapshot(model, 10), selected)
            atomic_save(model_snapshot(model, 11), latest)
            atomic_save(model_snapshot(model, 9), selected.with_name("best.pt"))
            first = player.refresh_model(selected)
            self.assertEqual(player.messages.get_nowait(), ("model", 11))
            self.assertIs(player.refresh_model(selected), first)
            self.assertTrue(player.messages.empty())

            with torch.no_grad():
                next(model.parameters()).fill_(0.125)
            atomic_save(model_snapshot(model, 12), latest)
            updated = player.refresh_model(selected)
            self.assertIsNot(updated, first)
            torch.testing.assert_close(next(updated.parameters()), next(model.parameters()))
            self.assertEqual(player.messages.get_nowait(), ("model", 12))

            # Explicit historical models must not switch to the run's latest model.
            historical = selected.with_name("model-000010.pt")
            atomic_save(model_snapshot(first, 10), historical)
            player.refresh_model(historical)
            self.assertEqual(player.messages.get_nowait(), ("model", 10))

            # A broken save pauses the player rather than silently using old weights.
            latest.write_bytes(b"incomplete checkpoint")
            with self.assertRaises(Exception):
                player.refresh_model(selected)

    def test_reposition_preserves_game_and_rejects_unrelated_board(self):
        from chess_ai.screen_player import ScreenPlayer
        player = ScreenPlayer.__new__(ScreenPlayer)
        player.reader = PieceReader(render(chess.Board()))
        player.tracker = GameTracker(chess.WHITE)
        for token in ("e2e4", "e7e5"):
            player.tracker.board.push_uci(token)
        player.tracker.pending = chess.Move.from_uci("g1f3")
        player.searches = {0: "confirmed", 2: "pending"}
        player.windows = SimpleNamespace(window_at=lambda _: 17)
        original = player.tracker.board.copy()
        area = BoardArea(200, 100, 1008, 908)
        player.reposition_board(render(original).resize((808, 808)), area)
        self.assertEqual(player.area, area)
        self.assertEqual(player.tracker.board.move_stack, original.move_stack)
        self.assertEqual(player.tracker.pending.uci(), "g1f3")
        self.assertEqual(player.searches, {0: "confirmed", 2: "pending"})
        # Accept the already-sent move and its reply at the new location.
        after = original.copy()
        after.push_uci("g1f3")
        after.push_uci("b8c6")
        player.reposition_board(render(after).resize((808, 808)), area)
        self.assertEqual(player.tracker.board.move_stack, after.move_stack)
        self.assertIsNone(player.tracker.pending)
        with self.assertRaises(UncertainBoard):
            unrelated = after.copy()
            unrelated.push_uci("f1c4")
            player.reposition_board(render(unrelated), BoardArea(0, 0, 512, 512))
        self.assertEqual(player.area, area)
        self.assertEqual(player.tracker.board.move_stack, after.move_stack)
        player.reposition_board(render(original), area)
        self.assertEqual(player.searches, {0: "confirmed"})

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
        windows.park_pointer(BoardArea(-300, 200, 212, 712), 17, stop)
        self.assertEqual(events, [0x8000 | 0x4000 | 1])
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
    def test_current_game_import_and_session_restore(self):
        import tkinter as tk
        import torch
        from chess_ai.screen_player import ScreenPlayer
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "session.pt"
                player = ScreenPlayer(root, Path("runs/main/model.pt"), path)
                board = chess.Board("NN6/8/8/1k6/6b1/8/8/K7 w - - 1 110")
                with patch("chess_ai.screen_player.simpledialog.askstring", return_value=board.fen()), \
                        patch.object(player, "select_area") as select:
                    player.continue_game()
                    self.assertEqual(select.call_args.kwargs["position"].fen(), board.fen())
                for text in (None, "bad fen", board.board_fen()):
                    with patch("chess_ai.screen_player.simpledialog.askstring", return_value=text), \
                            patch.object(player, "select_area") as select:
                        player.continue_game()
                        select.assert_not_called()
                player.windows = SimpleNamespace(window_at=lambda _: 17)
                player.attach_position(render(board), BoardArea(0, 0, 512, 512), board)
                player.set_running(False)
                self.assertTrue(player.start_button.instate(["!disabled"]))
                self.assertEqual(player.reader.read(render(board)), board_labels(board))
                # Save history, an unconfirmed move, and learning targets across restart.
                for token in ("a8c7", "b5c5"):
                    player.tracker.board.push_uci(token)
                player.tracker.pending = chess.Move.from_uci("b8a6")
                player.searches = {0: ("a8c7", torch.tensor([0]), torch.tensor([1.0]))}
                player.game_checkpoint = Path("runs/main/model.pt").resolve()
                player.save_session()
                saved = player.tracker.board.copy(stack=True)
                restored = ScreenPlayer(root, Path("unused.pt"), path)
                self.assertEqual(restored.tracker.board.fen(), saved.fen())
                self.assertEqual(restored.tracker.board.move_stack, saved.move_stack)
                self.assertEqual(restored.tracker.board.root().fen(), board.fen())
                self.assertEqual(restored.tracker.pending.uci(), "b8a6")
                self.assertEqual(restored.searches[0][0], "a8c7")
                self.assertEqual(restored.game_checkpoint, player.game_checkpoint)
                self.assertTrue(restored.start_button.instate(["disabled"]))
                self.assertTrue(restored.reposition_button.instate(["!disabled"]))
                self.assertIsNone(restored.target)
                restored.windows = SimpleNamespace(window_at=lambda _: 17)
                restored.reposition_board(render(saved), BoardArea(100, 100, 612, 612))
                restored.set_running(False)
                self.assertTrue(restored.start_button.instate(["!disabled"]))
                after = saved.copy()
                after.push(restored.tracker.pending)
                restored.reposition_board(render(after), BoardArea(100, 100, 612, 612))
                self.assertIsNone(restored.tracker.pending)
                self.assertEqual(restored.tracker.board.fen(), after.fen())
                # Failed imports leave the existing game available.
                with self.assertRaises(ValueError):
                    restored.attach_position(render(board), restored.area, chess.Board.empty())
                self.assertEqual(restored.tracker.board.fen(), after.fen())
                path.write_bytes(b"broken session")
                broken = ScreenPlayer(root, Path("unused.pt"), path)
                self.assertIsNone(broken.reader)
                self.assertIn("Could not restore", broken.status.get())
        finally:
            root.destroy()

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
            self.assertTrue(root.attributes("-topmost"))
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
