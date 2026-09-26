"""UCI transport: keep stdin responsive while a worker searches."""

import sys
import threading
import time

import chess

from .model import load_model
from .search import select_move


def parse_position(tokens):
    if not tokens:
        raise ValueError("position needs startpos or fen")
    if tokens[0] == "startpos":
        board, remaining = chess.Board(), tokens[1:]
    elif tokens[0] == "fen" and len(tokens) >= 7:
        board, remaining = chess.Board(" ".join(tokens[1:7])), tokens[7:]
    else:
        raise ValueError("position needs startpos or six FEN fields")
    if not board.is_valid():
        raise ValueError("Invalid chess position")
    if remaining:
        if remaining[0] != "moves":
            raise ValueError("Expected moves after position")
        for token in remaining[1:]:
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move: {token}")
            board.push(move)
    return board


def search_limits(tokens, turn, default_simulations):
    values = {}
    for key in ("nodes", "movetime", "wtime", "btime", "winc", "binc", "movestogo"):
        if key in tokens:
            index = tokens.index(key)
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for {key}")
            values[key] = int(tokens[index + 1])
            if values[key] < 0:
                raise ValueError(f"{key} must be nonnegative")
    if "infinite" in tokens:
        return sys.maxsize, None
    simulations = max(1, values.get("nodes", default_simulations))
    milliseconds = values.get("movetime")
    clock = "wtime" if turn else "btime"
    if milliseconds is None and clock in values:
        remaining = max(1, values[clock] - 30)
        increment = values.get("winc" if turn else "binc", 0)
        milliseconds = min(remaining, remaining / max(1, values.get("movestogo", 30)) + increment * 0.8)
    if milliseconds is not None:
        # Reserve a small margin for returning the move to the GUI.
        deadline = time.monotonic() + max(0, milliseconds - 10) / 1000
        return max(1, values.get("nodes", sys.maxsize)), deadline
    return simulations, None


def run_uci(model, device, checkpoint, simulations=128):
    board = chess.Board()
    stopped = threading.Event()
    worker = None
    output_lock = threading.Lock()

    def emit(message):
        with output_lock:
            print(message, flush=True)

    def stop_search():
        nonlocal worker
        stopped.set()
        if worker is not None:
            worker.join()
            worker = None

    def think(position, budget, deadline):
        start = time.monotonic()
        try:
            move, search = select_move(model, position, budget, stop=stopped, deadline=deadline)
            elapsed = int((time.monotonic() - start) * 1000)
            score = f" score mate {search.mate_in}" if search.mate_in is not None else ""
            emit(f"info nodes {search.root.visits + search.mate_nodes} time {elapsed}{score}")
            emit(f"bestmove {move.uci() if move else '0000'}")
        except Exception as error:
            emit("info string search failed: " + str(error).replace("\n", " "))
            fallback = next(iter(position.legal_moves), None)
            emit(f"bestmove {fallback.uci() if fallback else '0000'}")

    try:
        for line in sys.stdin:
            words = line.split()
            if not words:
                continue
            command, tokens = words[0], words[1:]
            try:
                if command == "uci":
                    emit("id name Chess AI Self Play")
                    emit("id author Local Chess AI")
                    emit(f"option name Simulations type spin default {simulations} min 1 max 1000000")
                    emit(f"option name ModelPath type string default {checkpoint}")
                    emit("uciok")
                elif command == "isready":
                    emit("readyok")
                elif command == "setoption":
                    stop_search()
                    if not tokens or tokens[0] != "name" or "value" not in tokens:
                        raise ValueError("Expected setoption name ... value ...")
                    split = tokens.index("value")
                    name, value = " ".join(tokens[1:split]), " ".join(tokens[split + 1:])
                    if name == "Simulations":
                        number = int(value)
                        if not 1 <= number <= 1000000:
                            raise ValueError("Simulations must be in 1..1000000")
                        simulations = number
                    elif name == "ModelPath":
                        loaded, _ = load_model(value, device)
                        model, checkpoint = loaded, value
                elif command == "ucinewgame":
                    stop_search()
                    board = chess.Board()
                elif command == "position":
                    stop_search()
                    board = parse_position(tokens)
                elif command == "go":
                    stop_search()
                    if "searchmoves" in tokens or "ponder" in tokens:
                        raise ValueError("searchmoves and ponder are not supported")
                    budget, deadline = search_limits(tokens, board.turn, simulations)
                    if "depth" in tokens or "mate" in tokens:
                        emit("info string MCTS uses node/time limits; depth and mate limits are ignored")
                    stopped.clear()
                    worker = threading.Thread(target=think, args=(board.copy(stack=True), budget, deadline))
                    worker.start()
                elif command == "stop":
                    stop_search()
                elif command == "quit":
                    break
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                emit("info string error: " + str(error).replace("\n", " "))
    finally:
        stop_search()
