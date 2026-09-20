"""Offline UCI teacher labels and held-out checks; never used by the screen player."""

import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from .model import ACTION_SIZE, INPUT_PLANES, atomic_save, encode, evaluate_boards, move_index


def teacher_record(board, infos):
    if not board.is_valid() or board.is_game_over() or not infos:
        raise ValueError("Teacher positions must be valid, unfinished games")
    moves = list(board.legal_moves)
    policy = np.zeros(len(moves), dtype=np.float32)
    scores, choices = [], []
    for info in infos:
        if not info.get("pv") or info["pv"][0] not in moves or "score" not in info:
            raise ValueError("Teacher returned an illegal or unscored move")
        choices.append(moves.index(info["pv"][0]))
        scores.append(info["score"].pov(board.turn).score(mate_score=10000))
    if len(set(choices)) != len(choices):
        raise ValueError("Teacher returned duplicate moves")
    scores = np.asarray(scores, dtype=np.float64)
    # ponytail: 100-centipawn policy temperature; tune using held-out match results.
    weights = np.exp(np.clip((scores - scores.max()) / 100, -60, 0))
    policy[choices] = weights / weights.sum()
    best = infos[int(scores.argmax())]
    wdl = (best["wdl"].pov(board.turn) if "wdl" in best else
           best["score"].pov(board.turn).wdl(model="sf16", ply=board.ply()))
    return (torch.from_numpy(encode(board)).half(),
            torch.tensor([move_index(move, board.turn) for move in moves]),
            torch.from_numpy(policy), float(2 * wdl.expectation() - 1), True)


def load_teacher(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1 or data.get("kind") != "teacher" or not data.get("records"):
        raise ValueError("Expected a nonempty teacher dataset")
    for row in data["records"] + data.get("validation_records", []):
        if not isinstance(row, (tuple, list)) or len(row) != 5:
            raise ValueError("Invalid teacher record")
        state, indices, policy, value, known = row
        if (not all(isinstance(t, torch.Tensor) for t in (state, indices, policy)) or
                state.shape != (INPUT_PLANES, 8, 8) or indices.ndim != 1 or
                indices.dtype != torch.int64 or policy.shape != indices.shape or
                not 1 <= len(indices) <= 218 or len(indices.unique()) != len(indices) or
                not torch.all((indices >= 0) & (indices < ACTION_SIZE)) or
                not torch.isfinite(state).all() or not torch.isfinite(policy).all() or
                not torch.all(policy >= 0) or abs(policy.sum().item() - 1) > 1e-5 or
                not isinstance(value, (float, int)) or not -1 <= value <= 1 or known is not True):
            raise ValueError("Invalid teacher tensors or targets")
    fens = data.get("validation_fens", [])
    if len(fens) != len(data.get("validation_records", [])):
        raise ValueError("Validation positions and targets differ in length")
    for fen in fens:
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over():
            raise ValueError("Invalid held-out position")
    count = len(data["records"])
    for name, default in (("levels", 2), ("priorities", 1.0), ("fens", None)):
        data.setdefault(name, [default] * count)
        if len(data[name]) != count:
            raise ValueError(f"Teacher {name} and records differ in length")
    if any(level not in (0, 1, 2) for level in data["levels"]):
        raise ValueError("Curriculum levels must be 0, 1 or 2")
    if not np.isfinite(data["priorities"]).all() or np.any(np.asarray(data["priorities"]) < 0):
        raise ValueError("Invalid teacher priorities")
    return data


def sample_positions(pgn, samples, seed, recent_games=0, stop=lambda: False, progress=lambda n: None, per_game=1):
    """Scan headers, then parse only selected games from the recent window."""
    from collections import deque
    rng, selected = random.Random(seed), []
    if pgn is None:
        return selected
    offsets = deque(maxlen=recent_games or None)
    with Path(pgn).open(encoding="utf-8-sig") as stream:
        while True:
            if stop():
                raise InterruptedError("Teacher refresh cancelled")
            offset = stream.tell()
            if chess.pgn.read_headers(stream) is None:
                break
            offsets.append(offset)
            if len(offsets) % 1000 == 0:
                progress(0)
        for offset in rng.sample(list(offsets), min(samples, len(offsets))):
            if stop():
                raise InterruptedError("Teacher refresh cancelled")
            stream.seek(offset)
            game = chess.pgn.read_game(stream)
            if game.errors:
                raise ValueError(f"Invalid PGN game: {game.errors[0]}")
            phases = [[], [], []]
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), 1):
                board.push(move)
                phase = 0 if ply <= 20 else 2 if len(board.piece_map()) <= 10 else 1
                phases[phase].append(board.fen())
            for bucket in phases:
                rng.shuffle(bucket)
            # Cycle through openings, middlegames and endings instead of sampling mostly long endings.
            available = [i for i, bucket in enumerate(phases) if bucket]
            rng.shuffle(available)
            for index in range(per_game):
                if not available or len(selected) >= samples:
                    break
                phase = available[index % len(available)]
                candidate = chess.Board(phases[phase].pop())
                if candidate.is_valid() and not candidate.is_game_over():
                    selected.append(candidate)
                if not phases[phase]:
                    available.remove(phase)
            if len(selected) >= samples:
                break
    return selected


def position_group(board):
    variants = (board, board.mirror(), board.transform(chess.flip_horizontal),
                board.mirror().transform(chess.flip_horizontal))
    return min(" ".join(b.fen().split()[:4]) for b in variants)


def is_held_out(board):
    return int(hashlib.sha256(position_group(board).encode()).hexdigest()[:8], 16) % 10 == 0


def curriculum_stage(config, iteration):
    every = config.get("curriculum_every", 0)
    return min(2, max(0, (iteration - config.get("curriculum_start", iteration)) // every)) if every else 2


def correct_screen_game(game, engine, nodes, threshold=100):
    """Replace confirmed blunder targets with teacher alternatives; keep original saves intact."""
    parsed = chess.pgn.read_game(io.StringIO(game.get("pgn", "")))
    if parsed is None or parsed.errors or not list(parsed.mainline_moves()):
        raise ValueError("Post-game teaching needs a valid saved PGN")
    rows, report = list(game["records"]), []
    board, index = parsed.board(), 0
    for move in parsed.mainline_moves():
        if index < len(rows) and torch.equal(rows[index][0], torch.from_numpy(encode(board)).half()):
            if not is_held_out(board):
                infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=3, game=object())
                actual = next((info for info in infos if info["pv"][0] == move), None)
                if actual is None:
                    actual = engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[move], game=object())
                best = max(info["score"].pov(board.turn).score(mate_score=10000) for info in infos)
                loss = best - actual["score"].pov(board.turn).score(mate_score=10000)
                if loss >= threshold:
                    rows[index] = teacher_record(board, infos)
                    report.append(dict(index=index, fen=board.fen(), played=move.uci(),
                                       best=infos[0]["pv"][0].uci(), loss_cp=loss))
            index += 1
        board.push(move)
    if index != len(rows):
        raise ValueError("Saved screen targets do not match the PGN; game retained for review")
    return rows, report


def generate(args, stop=lambda: False, progress=lambda n: None, model=None):
    if not args.pgn and not args.fens:
        raise ValueError("Provide --pgn and/or --fens for teacher positions")
    if Path(args.output).exists():
        raise ValueError("Teacher dataset already exists; choose a new --output")
    boards = [(board, "selfplay", 1 if len(board.piece_map()) <= 10 else 2) for board in
              sample_positions(args.pgn, args.samples * (2 if model is not None else 1), args.seed,
              getattr(args, "recent_games", 0), stop, progress, per_game=3)]
    if args.fens:
        for line in Path(args.fens).read_text(encoding="utf-8-sig").splitlines():
            fen = line.split("#", 1)[0].strip()
            if fen:
                board = chess.Board(fen)
                if not board.is_valid() or board.is_game_over():
                    raise ValueError(f"Invalid practice FEN: {fen}")
                comment = line.partition("#")[2]
                level = int(comment.split("level=", 1)[1].split()[0]) if "level=" in comment else 0
                if level not in (0, 1, 2):
                    raise ValueError("Practice level must be 0, 1 or 2")
                variants = [board, board.mirror()]
                if not board.castling_rights:
                    variants.extend((board.transform(chess.flip_horizontal), board.mirror().transform(chess.flip_horizontal)))
                boards.extend((b, "practice", level) for b in variants if b.is_valid() and not b.is_game_over())
    unique = {}
    for board, source, level in boards:
        key = " ".join(board.fen().split()[:4])
        unique[key] = (board, source, level)
    if len(unique) < 2:
        raise ValueError("Provide at least two distinct positions")
    records, validation, validation_fens, validation_sources = [], [], [], []
    difficulties, practice, levels, fens = [], [], [], []
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with chess.engine.SimpleEngine.popen_uci(args.engine, creationflags=flags) as engine:
        engine.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
        teacher_id = engine.id
        for index, (key, (board, source, level)) in enumerate(unique.items()):
            if stop():
                raise InterruptedError("Teacher refresh cancelled")
            infos = engine.analyse(board, chess.engine.Limit(nodes=args.nodes), multipv=3, game=object())
            record = teacher_record(board, infos)
            # Stable split keeps a repeated position in the same partition across dataset refreshes.
            held_out = is_held_out(board)
            if held_out:
                validation.append(record)
                validation_fens.append(board.fen())
                validation_sources.append(source)
            else:
                records.append(record)
                practice.append(source == "practice")
                levels.append(level)
                fens.append(board.fen())
                if model is not None:
                    _, probabilities, value = evaluate_boards(model, [board])[0]
                    difficulties.append(float(-(record[2].numpy() * np.log(
                        np.maximum(probabilities, 1e-9))).sum() + (value - record[3]) ** 2))
                else:
                    difficulties.append(1.0)
            progress(index + 1)
            if (index + 1) % 50 == 0:
                print(f"Labelled {index + 1}/{len(unique)} positions", flush=True)
    if not records or not validation:
        raise ValueError("Need both training and held-out positions; supply more examples")
    if model is not None and len(records) > args.samples:
        keep = [i for i, flag in enumerate(practice) if flag]
        others = sorted((i for i, flag in enumerate(practice) if not flag),
                        key=lambda i: difficulties[i], reverse=True)
        budget = max(0, args.samples - len(keep))
        hard = budget // 2
        keep.extend(others[:hard])
        keep.extend(random.Random(args.seed).sample(others[hard:], min(budget - hard, len(others) - hard)))
        records = [records[i] for i in keep]
        levels, fens, difficulties = ([values[i] for i in keep] for values in (levels, fens, difficulties))
    data = dict(format_version=1, kind="teacher", records=records, validation_records=validation,
                validation_fens=validation_fens, validation_sources=validation_sources,
                levels=levels, fens=fens, priorities=difficulties,
                teacher=teacher_id, nodes=args.nodes, seed=args.seed,
                source_pgn=str(args.pgn), source_fens=str(args.fens))
    previous = getattr(args, "previous", None)
    capacity = getattr(args, "max_records", None) or 2 * args.samples
    if previous:
        old = load_teacher(previous)
        groups = {position_group(chess.Board(fen)) for fen in fens}
        available = [i for i, fen in enumerate(old["fens"]) if fen is None or
                     (not is_held_out(chess.Board(fen)) and position_group(chess.Board(fen)) not in groups)]
        keep = random.Random(args.seed).sample(available, min(len(available), max(0, capacity - len(records))))
        for name in ("records", "levels", "fens", "priorities"):
            data[name].extend(old[name][i] for i in keep)
    if len(data["records"]) > capacity:
        # Keep all elementary lessons before sampling the larger pool.
        easy = [i for i, level in enumerate(data["levels"]) if level == 0]
        other = [i for i, level in enumerate(data["levels"]) if level != 0]
        rng = random.Random(args.seed)
        keep = rng.sample(easy, min(len(easy), capacity))
        keep += rng.sample(other, min(len(other), capacity - len(keep)))
        for name in ("records", "levels", "fens", "priorities"):
            data[name] = [data[name][i] for i in keep]
    atomic_save(data, args.output)
    print(json.dumps(dict(training_positions=len(data["records"]), held_out_positions=len(validation),
                          teacher=teacher_id, output=str(args.output))), flush=True)


def assess(args, model):
    from .search import Search, run_searches
    data = load_teacher(args.teacher_data)
    boards = [chess.Board(fen) for fen in data["validation_fens"]]
    if not boards:
        raise ValueError("Dataset has no held-out positions")
    direct, searched, errors = 0, 0, []
    for offset in range(0, len(boards), 64):
        batch = boards[offset:offset + 64]
        searches = [Search(board) for board in batch]
        predictions = evaluate_boards(model, batch)
        run_searches(model, searches, args.simulations)
        for i, (board, search, prediction) in enumerate(zip(batch, searches, predictions), offset):
            _, indices, policy, value, _ = data["validation_records"][i]
            best = set(indices[policy >= policy.max() - 1e-6].tolist())
            moves, probs, predicted_value = prediction
            direct += move_index(moves[int(probs.argmax())], board.turn) in best
            moves, probs = search.policy(0)
            searched += move_index(moves[int(probs.argmax())], board.turn) in best
            errors.append((predicted_value - value) ** 2)
    report = dict(positions=len(boards), policy_agreement=direct / len(boards),
                  search_agreement=searched / len(boards), value_mse=float(np.mean(errors)),
                  simulations=args.simulations, checkpoint=str(args.checkpoint),
                  note="Held-out teacher agreement, not an Elo or playing-strength measurement")
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
