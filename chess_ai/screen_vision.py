"""Calibrate visible 2D chess pieces, then match observed legal transitions."""

from dataclasses import dataclass

import chess
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class BoardArea:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self):
        width, height = self.right - self.left, self.bottom - self.top
        if min(width, height) < 160 or max(width, height) > 4000:
            raise ValueError("Select a board between 160 and 4,000 pixels wide")
        if abs(width - height) / max(width, height) > 0.04:
            raise ValueError("Select only the square 8 by 8 board, without labels or borders")

    @property
    def bbox(self):
        return self.left, self.top, self.right, self.bottom

    def center(self, square, white_bottom=True):
        file, rank = chess.square_file(square), chess.square_rank(square)
        col, row = (file, 7 - rank) if white_bottom else (7 - file, rank)
        return (round(self.left + (col + 0.5) * (self.right - self.left) / 8),
                round(self.top + (row + 0.5) * (self.bottom - self.top) / 8))


def screen_square(row, col, white_bottom=True):
    return chess.square(col, 7 - row) if white_bottom else chess.square(7 - col, row)


def board_labels(board):
    return tuple(board.piece_at(square).symbol() if board.piece_at(square) else "."
                 for square in chess.SQUARES)


def align_board_area(image, area):
    """Fit square edges across the middle ranks, tolerating an opening move."""
    pixels = np.asarray(image.crop(area.bbox).convert("RGB"), dtype=np.float32)
    height = pixels.shape[0]

    def edges(lines, indices):
        # Vote across scanlines so a moved piece cannot replace a square edge.
        contrast = np.median(np.max(np.abs(np.diff(lines, axis=0)), axis=2), axis=1)
        positions = []
        for index in indices:
            start = round(len(lines) * (index / 8 - 1 / 32))
            end = round(len(lines) * (index / 8 + 1 / 32))
            peak = start + int(np.argmax(contrast[start:end]))
            if contrast[peak] < 20:
                raise ValueError("Cannot locate the square edges. Select the starting board again.")
            positions.append(peak + 1)
        return np.asarray(positions)

    files, ranks = np.arange(1, 8), np.arange(3, 6)
    vertical = edges(pixels[height // 4:3 * height // 4].transpose(1, 0, 2), files)
    horizontal = edges(pixels, ranks)
    size, left = np.polyfit(files, vertical, 1)
    top = np.median(horizontal - ranks * size)
    if (np.max(np.abs(vertical - (left + files * size))) > 2 or
            np.max(np.abs(horizontal - (top + ranks * size))) > 2):
        raise ValueError("Cannot locate an even chess grid. Select the starting board again.")
    aligned = BoardArea(round(area.left + left), round(area.top + top),
                        round(area.left + left + 8 * size), round(area.top + top + 8 * size))
    if aligned.left < 0 or aligned.top < 0 or aligned.right > image.width or aligned.bottom > image.height:
        raise ValueError("Keep the whole board visible before selecting it")
    return aligned


def tile_features(image):
    """Suppress flat backgrounds so last-move highlights do not look like pieces."""
    tile = np.asarray(image.resize((40, 40), Image.Resampling.BILINEAR).convert("RGB"), dtype=np.float32) / 255
    edge = np.concatenate((tile[:3].reshape(-1, 3), tile[-3:].reshape(-1, 3),
                           tile[:, :3].reshape(-1, 3), tile[:, -3:].reshape(-1, 3)))
    background = np.median(edge, axis=0)
    mask = np.clip((np.max(np.abs(tile - background), axis=2) - 0.02) / 0.055, 0, 1)
    # Coordinates in the corners and antialiased board borders are excluded.
    mask[:3] = mask[-3:] = False
    mask[:, :3] = mask[:, -3:] = False
    mask[:7, :7] = mask[:7, -7:] = mask[-7:, :7] = mask[-7:, -7:] = False
    # Keep outlined piece interiors when their fill matches the square's color.
    enclosed = Image.fromarray((mask > 0.2).astype(np.uint8) * 255).copy()
    ImageDraw.floodfill(enclosed, (0, 0), 128)
    mask[np.asarray(enclosed) == 0] = 1
    # ponytail: pieces must extend beyond the central 45%; use theme templates for miniature pieces.
    outside = mask.copy()
    outside[11:29, 11:29] = 0
    outside[:12, :12] = outside[:12, -12:] = outside[-12:, :12] = outside[-12:, -12:] = 0
    if not np.any(outside) and np.all(mask[18:22, 18:22] > 0.5):
        mask[11:29, 11:29] = 0
    pixels = tile * mask[:, :, None]
    return np.concatenate((mask[:, :, None] * 2, pixels), axis=2).astype(np.float32).ravel()


def image_tiles(image, white_bottom=True):
    width, height = image.size
    for row in range(8):
        for col in range(8):
            tile = image.crop((round(col * width / 8), round(row * height / 8),
                               round((col + 1) * width / 8), round((row + 1) * height / 8)))
            yield screen_square(row, col, white_bottom), tile_features(tile)


class UncertainBoard(ValueError):
    pass


class PieceReader:
    def __init__(self, image, white_bottom=True, tolerance=0.18, board=None):
        self.white_bottom = white_bottom
        self.tolerance = tolerance
        expected = board_labels(chess.Board() if board is None else board)
        self.symbols = tuple(".PNBRQKpnbrqk")
        templates = {symbol: [] for symbol in self.symbols}
        for square, features in image_tiles(image, white_bottom):
            templates[expected[square]].append(features)
        self.templates = {symbol: np.stack(rows) for symbol, rows in templates.items() if rows}
        self.symbols = tuple(self.templates)
        if self.read(image) != expected:
            raise ValueError("The selected board must match the supplied position and use clear, flat 2D pieces")

    def read(self, image):
        labels = ["."] * 64
        for square, features in image_tiles(image, self.white_bottom):
            labels[square] = self.read_piece(square, features)
        return tuple(labels)

    def read_piece(self, square, features, symbols=None, tolerance=None):
        tolerance = self.tolerance if tolerance is None else tolerance
        symbols = self.symbols if symbols is None else tuple(s for s in symbols if s in self.templates)
        distances = np.array([np.min(np.mean((self.templates[symbol] - features) ** 2, axis=1)) ** 0.5
                              for symbol in symbols])
        order = distances.argsort()
        best, second = distances[order[0]], distances[order[1]]
        if best > tolerance or second - best < 0.015:
            raise UncertainBoard(f"Cannot confidently read {chess.square_name(square)}")
        return symbols[order[0]]

    def promotion_square(self, image, move, color):
        # ponytail: recognize four square-sized choices down the destination file;
        # add a menu-specific detector if another app uses a different layout.
        squares = {move.to_square + (-8 if color else 8) * offset for offset in range(4)}
        choices = {}
        # The menu belongs to the promoting side; its background can otherwise
        # make the two queen colors ambiguous against their board templates.
        symbols = ".PNBRQK" if color else ".pnbrqk"
        try:
            for square, features in image_tiles(image, self.white_bottom):
                if square in squares:
                    # White menu backgrounds change antialiased edges: the reported
                    # menu scores 0.20–0.23 against board templates. Require all four
                    # distinct choices below, while leaving board tolerance unchanged.
                    choices[self.read_piece(square, features, symbols,
                                            tolerance=max(self.tolerance, 0.25))] = square
        except UncertainBoard:
            return None
        if set(choices) == set("QRBN" if color else "qrbn"):
            return choices[chess.Piece(move.promotion, color).symbol()]
        return None


def matching_move(board, observed):
    """Only accept a unique legal successor, preserving castling/EP/history."""
    matches = []
    for move in list(board.legal_moves):
        board.push(move)
        matches_position = board_labels(board) == observed
        board.pop()
        if matches_position:
            matches.append(move)
    return matches[0] if len(matches) == 1 else None


class GameTracker:
    def __init__(self, color):
        self.board = chess.Board()
        self.color = color
        self.pending = None

    def observe(self, observed, retry_pending=False):
        if self.pending is not None:
            after = self.board.copy(stack=True)
            after.push(self.pending)
            if board_labels(after) == observed:
                self.board.push(self.pending)
                self.pending = None
                return "own_move"
            reply = matching_move(after, observed)
            if reply is not None:
                # The opponent may reply before the next screen capture.
                self.board.push(self.pending)
                self.board.push(reply)
                self.pending = None
                return "reply"
            if board_labels(self.board) == observed:
                # Only an explicit resume can retry an unchanged, verified board.
                if retry_pending:
                    self.pending = None
                    return "retry"
                return "pending"
            return self.rewind(observed, "The screen does not match the move just sent")
        if board_labels(self.board) == observed:
            return "same"
        if self.board.turn == self.color:
            return self.rewind(observed, "Position changed on the AI's turn. Recalibrate for a new game.")
        move = matching_move(self.board, observed)
        if move is None:
            return self.rewind(observed, "The screen is not a legal continuation of this game")
        self.board.push(move)
        return "opponent_move"

    def rewind(self, observed, error):
        earlier = self.board.copy(stack=True)
        while earlier.move_stack:
            earlier.pop()
            if board_labels(earlier) == observed:
                self.board = earlier
                self.pending = None
                return "rewind"
        raise UncertainBoard(error)
