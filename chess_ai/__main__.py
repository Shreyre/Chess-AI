"""Command-line training, playing, analysis, and evaluation."""

import argparse
import json
from pathlib import Path
import sys

import chess
import torch

from .model import choose_device, load_model
from .search import select_move
from .training import DEFAULTS, train
from .uci import run_uci


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def learning_rate(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def fraction(value):
    number = float(value)
    if not 0 <= number < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return number


def valid_board(fen):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("FEN does not describe a valid standard chess position")
    return board


def play(args, model):
    board = valid_board(args.fen)
    human = args.color == "white"
    while not board.is_game_over():
        print("\n" + str(board) + "\n  a b c d e f g h")
        if board.turn == human:
            try:
                text = input("Your move (SAN or UCI; quit to exit): ").strip()
            except EOFError:
                return
            if text.lower() in ("quit", "exit"):
                return
            try:
                move = board.parse_san(text)
                if move not in board.legal_moves:
                    raise ValueError("Move is not legal")
            except ValueError as error:
                print(error)
                continue
        else:
            move, _ = select_move(model, board, args.simulations)
            print("AI:", board.san(move))
        board.push(move)
    print("\n" + str(board) + "\nResult: " + board.result())


def evaluate(args, model, device):
    opponent = load_model(args.opponent, device)[0] if args.opponent != "random" else None
    from .evaluation import play_match
    report, pgns = play_match(model, opponent, args.games, args.simulations,
                             args.max_plies, args.opening_plies, args.seed)
    report["opponent"] = args.opponent
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        path.with_suffix(".pgn").write_text("\n\n".join(pgns) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Learn chess from scratch with neural self-play and MCTS.")
    sub = parser.add_subparsers(dest="command", required=True)
    dashboard = sub.add_parser("dashboard", help="Watch and control training in a local browser")
    dashboard.add_argument("--run-dir", default="runs/main")
    dashboard.add_argument("--port", type=positive, default=8765)
    trainer = sub.add_parser("train", help="Generate self-play and update the network")
    trainer.add_argument("--iterations", type=positive, default=10, help="Additional iterations to run")
    trainer.add_argument("--run-dir", help="Output directory (default: runs/main or resume file's folder)")
    trainer.add_argument("--resume", help="Resume a full latest.pt checkpoint")
    trainer.add_argument("--external-only", action="store_true", help="Learn from pending completed screen games, without new self-play")
    trainer.add_argument("--teacher-data", type=Path, help="Offline teacher dataset; saved for subsequent resumes")
    trainer.add_argument("--teacher-only", action="store_true", help="Warm up on teacher examples without generating self-play")
    trainer.add_argument("--teacher-engine", type=Path)
    trainer.add_argument("--teacher-fens", type=Path)
    for name, default in DEFAULTS.items():
        kind = (fraction if name in ("teacher_fraction", "priority_fraction") else learning_rate if name == "learning_rate"
                else nonnegative if name in ("seed", "temperature_plies", "opening_plies", "teacher_refresh_every", "gate_every", "curriculum_every", "postgame_nodes") else positive)
        trainer.add_argument("--" + name.replace("_", "-"), type=kind,
                             help=f"Default for new runs: {default}; otherwise keep checkpoint setting")
    teacher = sub.add_parser("teach", help="Generate offline training targets with a UCI Stockfish engine")
    teacher.add_argument("--engine", required=True, help="Path to the Stockfish executable")
    teacher.add_argument("--pgn", type=Path, help="Sample openings, middlegames and endings from games")
    teacher.add_argument("--fens", type=Path, help="Additional practice FENs, one per line")
    teacher.add_argument("--samples", type=positive, default=1024)
    teacher.add_argument("--max-records", type=positive, help="Maximum training examples, including retained older examples")
    teacher.add_argument("--nodes", type=positive, default=20000)
    teacher.add_argument("--seed", type=nonnegative, default=7)
    teacher.add_argument("--output", type=Path, required=True)
    for command in ("play", "analyze", "evaluate", "uci", "assess"):
        child = sub.add_parser(command)
        child.add_argument("--checkpoint", required=True, help="Saved .pt model or training checkpoint")
        child.add_argument("--simulations", type=positive, default=128)
        if command in ("play", "analyze"):
            child.add_argument("--fen", default=chess.STARTING_FEN)
        if command == "play":
            child.add_argument("--color", choices=("white", "black"), default="white")
        if command == "assess":
            child.add_argument("--teacher-data", type=Path, required=True)
            child.add_argument("--output", type=Path)
        if command == "evaluate":
            child.add_argument("--opponent", default="random", help="random or another checkpoint path")
            child.add_argument("--games", type=positive, default=20)
            child.add_argument("--max-plies", type=positive, default=512)
            child.add_argument("--opening-plies", type=nonnegative, default=4,
                               help="Shared random opening per pair; colors swapped")
            child.add_argument("--seed", type=nonnegative, default=42)
            child.add_argument("--output", help="Write JSON report and companion PGN")
    for child in sub.choices.values():
        child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        child.add_argument("--threads", type=positive, default=2, help="PyTorch CPU threads")
    args = parser.parse_args()
    try:
        if args.command == "dashboard":
            from .dashboard import serve
            serve(args.run_dir, args.port)
            return
        torch.set_num_threads(args.threads)
        if args.command == "teach":
            from .teacher import generate
            generate(args)
            return
        device = choose_device(args.device)
        if args.command == "train":
            train(args, device)
        else:
            model, data = load_model(args.checkpoint, device)
            if args.command == "uci":
                run_uci(model, device, args.checkpoint, args.simulations)
            elif args.command == "play":
                play(args, model)
            elif args.command == "evaluate":
                evaluate(args, model, device)
            elif args.command == "assess":
                from .teacher import assess
                assess(args, model)
            else:
                board = valid_board(args.fen)
                move, search = select_move(model, board, args.simulations)
                print(json.dumps(dict(move=move.uci() if move else None,
                                      san=board.san(move) if move else None,
                                      value=search.root.value, visits=search.root.visits,
                                      iteration=data["iteration"], result=board.result()), indent=2))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
