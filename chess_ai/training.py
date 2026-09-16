"""Self-play, replay training, and resumable iterations."""

from collections import deque
import json
from pathlib import Path
import time
import uuid

import chess
import chess.pgn
import numpy as np
import torch
from torch.nn import functional as F

from .model import ChessNet, atomic_save, encode, load_model, model_snapshot, move_index
from .search import Search, run_searches
from .run_lock import acquire_training_lock
from .telemetry import LiveStatus

DEFAULTS = dict(games=16, parallel_games=8, simulations=64, max_plies=512,
                temperature_plies=30, replay_size=20000, batch_size=128,
                train_steps=100, learning_rate=0.001, channels=64, blocks=3,
                seed=7, save_every=25)


def save_screen_game(directory, board, searches, result=None):
    """Save search targets only for AI moves confirmed in the tracked game."""
    result = result or board.result()
    if result not in ("1-0", "0-1", "1/2-1/2"):
        raise ValueError("Choose the finished game's result before learning")
    if board.is_game_over() and result != board.result():
        raise ValueError("The chosen result disagrees with the tracked final position")
    winner = {"1-0": chess.WHITE, "0-1": chess.BLACK, "1/2-1/2": None}[result]
    position, records = board.root(), []
    for ply, move in enumerate(board.move_stack):
        if ply in searches and searches[ply][0] == move.uci():
            _, indices, policy = searches[ply]
            value = 0.0 if winner is None else 1.0 if winner == position.turn else -1.0
            records.append((torch.from_numpy(encode(position)).half(), indices, policy, value, True))
        position.push(move)
    if not records:
        raise ValueError("There are no confirmed AI moves to learn from yet")
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Chess AI screen game"
    game.headers["Result"] = result
    path = Path(directory) / "screen-games" / f"{uuid.uuid4().hex}.pt"
    atomic_save(dict(records=records, result=result, pgn=str(game)), path)
    return path


def self_play(model, config, rng, progress=print, live=None):
    records, games = [], []
    outcomes = {"white_wins": 0, "black_wins": 0, "draws": 0, "truncated": 0}
    for offset in range(0, config["games"], config["parallel_games"]):
        count = min(config["parallel_games"], config["games"] - offset)
        boards = [chess.Board() for _ in range(count)]
        histories = [[] for _ in boards]
        if live:
            live.boards(boards, offset, 0, force=True)
        for ply in range(config["max_plies"]):
            active = [i for i, board in enumerate(boards) if board.outcome() is None]
            if not active:
                break
            searches = [Search(boards[i]) for i in active]
            run_searches(model, searches, config["simulations"], rng, noise=True)
            for i, search in zip(active, searches):
                board = boards[i]
                moves, policy = search.policy(1)
                indices = torch.tensor([move_index(move, board.turn) for move in moves])
                histories[i].append((torch.from_numpy(encode(board)).half(), indices,
                                     torch.from_numpy(policy), board.turn))
                if ply < config["temperature_plies"]:
                    selected = int(rng.choice(len(moves), p=policy.astype(float) / policy.sum(dtype=float)))
                else:
                    best = np.flatnonzero(policy == policy.max())
                    selected = int(rng.choice(best))
                board.push(moves[selected])
            if live:
                live.boards(boards, offset, ply + 1)
            if progress and (ply + 1) % 32 == 0:
                progress(f"  games {offset + 1}-{offset + count}: ply {ply + 1}, "
                         f"{len(active)} still playing", flush=True)
        if live:
            live.boards(boards, offset, max(len(board.move_stack) for board in boards), force=True)
        for board, history in zip(boards, histories):
            outcome = board.outcome()
            if outcome is None:
                outcomes["truncated"] += 1
            elif outcome.winner is None:
                outcomes["draws"] += 1
            else:
                outcomes["white_wins" if outcome.winner else "black_wins"] += 1
            for state, indices, policy, turn in history:
                value = (0.0 if outcome is None or outcome.winner is None
                         else 1.0 if outcome.winner == turn else -1.0)
                records.append((state, indices, policy, value, outcome is not None))
            game = chess.pgn.Game.from_board(board)
            game.headers["Event"] = "Chess AI self-play"
            game.headers["White"] = game.headers["Black"] = "Chess AI"
            game.headers["Termination"] = (outcome.termination.name.lower()
                                           if outcome else "move limit; no value target")
            games.append(str(game))
    return records, outcomes, games


def train_batch(model, optimizer, samples):
    """Sparse legal policy targets avoid storing 4672 floats per replay position."""
    device = next(model.parameters()).device
    states = torch.stack([row[0] for row in samples]).to(device, dtype=torch.float32)
    width = max(len(row[1]) for row in samples)
    indices = torch.zeros((len(samples), width), dtype=torch.long)
    targets = torch.zeros((len(samples), width))
    legal = torch.zeros((len(samples), width), dtype=torch.bool)
    for i, row in enumerate(samples):
        length = len(row[1])
        indices[i, :length], targets[i, :length], legal[i, :length] = row[1], row[2], True
    logits, values = model(states)
    selected = logits.gather(1, indices.to(device)).masked_fill(~legal.to(device), -1e9)
    policy_loss = -(targets.to(device) * F.log_softmax(selected, dim=1)).sum(1).mean()
    value_targets = torch.tensor([row[3] for row in samples], device=device)
    known = torch.tensor([row[4] for row in samples], device=device)
    value_loss = F.mse_loss(values[known], value_targets[known]) if known.any() else values.sum() * 0
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss; previous checkpoint is intact")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


def save_training(path, model, optimizer, replay, iteration, config, rng, totals, metrics, screen_games=()):
    data = model_snapshot(model, iteration)
    data.update(optimizer=optimizer.state_dict(), replay=list(replay), config=config,
                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                totals=totals, metrics=metrics, screen_games=sorted(screen_games))
    atomic_save(data, path)


def restore_training(path, device, overrides):
    model, data = load_model(path, device)
    if "optimizer" not in data or "replay" not in data:
        raise ValueError("This is an inference model. Resume from latest.pt instead.")
    config = data["config"] | overrides
    if any(config[key] != model.config[key] for key in ("channels", "blocks")):
        raise ValueError("Cannot change network dimensions when resuming")
    if config["seed"] != data["config"]["seed"]:
        raise ValueError("Resume restores the saved random state; use a new run to change the seed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    optimizer.load_state_dict(data["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = config["learning_rate"]
    rng = np.random.default_rng()
    rng.bit_generator.state = data["rng"]
    torch.set_rng_state(data["torch_rng"])
    if torch.cuda.is_available() and len(data["cuda_rng"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    return (model, optimizer, deque(data["replay"], maxlen=config["replay_size"]),
            config, rng, data)


def train(args, device):
    external_only = getattr(args, "external_only", False)
    if external_only and not args.resume:
        raise ValueError("Screen-game learning needs --resume with a full latest.pt checkpoint")
    overrides = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None}
    directory = Path(args.run_dir or (Path(args.resume).parent if args.resume else "runs/main")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    if latest.exists() and not args.resume:
        raise ValueError(f"Run already exists. Use --resume \"{latest}\" or a new --run-dir.")
    if args.resume and latest.exists() and Path(args.resume).resolve() != latest:
        raise ValueError("Use a new --run-dir when branching from a different checkpoint")
    lock = directory / ".training.lock"
    stop_request = directory / ".stop-request"
    live = LiveStatus(directory)
    while True:
        try:
            guard = acquire_training_lock(directory)
            break
        except (BlockingIOError, FileExistsError, PermissionError):
            if not external_only:
                raise ValueError(f"Training is already running or its lock cannot be verified: {lock}") from None
            time.sleep(1)  # Let the active trainer finish saving before loading its latest weights.
    try:
        stop_request.unlink(missing_ok=True)
        live.update(force=True, phase="initializing", device=str(device))
        if args.resume:
            model, optimizer, replay, config, rng, data = restore_training(args.resume, device, overrides)
            iteration, totals = data["iteration"], data["totals"]
            screen_games = set(data.get("screen_games", []))
        else:
            config = DEFAULTS | overrides
            torch.manual_seed(config["seed"])
            rng = np.random.default_rng(config["seed"])
            model = ChessNet(config["channels"], config["blocks"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
            replay = deque(maxlen=config["replay_size"])
            iteration, totals = 0, {"games": 0, "positions": 0, "updates": 0}
            screen_games = set()
            atomic_save(model_snapshot(model), directory / "initial.pt")
            atomic_save(model_snapshot(model), directory / "model.pt")
            save_training(latest, model, optimizer, replay, 0, config, rng, totals, {})
        print(f"Device: {device}; network: {model.config}; "
              f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
        print(f"Run: {directory}\nStarting at iteration {iteration}; Ctrl+C keeps the last completed iteration.", flush=True)
        live.update(force=True, phase="selfplay", iteration=iteration, saved_iteration=iteration,
                    target_iteration=iteration + args.iterations, config=config, totals=totals,
                    parameters=sum(p.numel() for p in model.parameters()))
        for _ in range(args.iterations):
            started = time.monotonic()
            model.eval()
            print(f"Iteration {iteration + 1}: {'screen-game learning' if external_only else 'self-play'}", flush=True)
            live.update(force=True, phase="learning" if external_only else "selfplay", iteration=iteration + 1,
                        started_at=time.time(), train_step=0)
            if external_only:
                records, pgns = [], []
                outcomes = dict(white_wins=0, black_wins=0, draws=0, truncated=0)
            else:
                records, outcomes, pgns = self_play(model, config, rng, live=live)
            imported, external_records = [], []
            for path in sorted((directory / "screen-games").glob("*.pt")):
                if path.name in screen_games:
                    continue
                game = torch.load(path, map_location="cpu", weights_only=True)
                if game["result"] not in ("1-0", "0-1", "1/2-1/2") or not game["records"]:
                    raise ValueError(f"Invalid completed screen game: {path}")
                external_records.extend(game["records"])
                outcomes[{"1-0": "white_wins", "0-1": "black_wins", "1/2-1/2": "draws"}[game["result"]]] += 1
                imported.append(path.name)
            if external_only and not imported:
                atomic_save(model_snapshot(model, iteration), directory / "model.pt")
                live.update(force=True, phase="completed", iteration=iteration, target_iteration=iteration)
                return
            records.extend(external_records)
            replay.extend(records)
            losses = []
            model.train()
            # Convert once: random indexing into a deque is linear.
            population = list(replay)
            steps = min(10, config["train_steps"]) if external_only else config["train_steps"]
            live.update(force=True, phase="learning", outcomes=outcomes, replay_positions=len(replay),
                        config=config | {"train_steps": steps}, screen_games=len(imported))
            for step in range(steps):
                fresh = min(len(external_records), max(1, config["batch_size"] // 4)) if step < 10 else 0
                count = config["batch_size"] - fresh
                indices = rng.choice(len(population), count, replace=len(population) < count)
                samples = [population[i] for i in indices]
                if fresh:
                    samples.extend(external_records[i] for i in rng.choice(len(external_records), fresh, replace=False))
                losses.append(train_batch(model, optimizer, samples))
                live.update(phase="learning", train_step=step + 1, loss=losses[-1])
            iteration += 1
            totals = {"games": totals["games"] + (0 if external_only else config["games"]) + len(imported),
                      "positions": totals["positions"] + len(records),
                      "updates": totals["updates"] + steps}
            metrics = dict(iteration=iteration, **totals, **outcomes, replay_positions=len(replay),
                           screen_games=len(imported),
                           value_positions=sum(row[4] for row in records),
                           seconds=round(time.monotonic() - started, 2),
                           **{key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]})
            live.update(force=True, phase="saving", train_step=steps)
            screen_games.update(imported)
            save_training(latest, model, optimizer, replay, iteration, config, rng, totals, metrics, screen_games)
            atomic_save(model_snapshot(model, iteration), directory / "model.pt")
            if iteration % config["save_every"] == 0:
                atomic_save(model_snapshot(model, iteration), directory / f"model-{iteration:06d}.pt")
            # The checkpoint is authoritative; logs may lag it if interrupted during writing.
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(metrics) + "\n")
            if pgns:
                with (directory / "selfplay.pgn").open("a", encoding="utf-8") as output:
                    output.write("\n\n".join(pgns) + "\n\n")
            print(json.dumps(metrics), flush=True)
            print(f"Saved {latest}", flush=True)
            live.update(force=True, saved_iteration=iteration, totals=totals, metrics=metrics)
            if stop_request.exists():
                print("Dashboard stop requested. Checkpoint saved.", flush=True)
                break
        live.update(force=True, phase="completed")
    except KeyboardInterrupt:
        live.update(force=True, phase="stopped")
        print(f"\nStopped. Resume the last completed iteration from {latest}", flush=True)
    except Exception as error:
        live.update(force=True, phase="error", error=str(error))
        raise
    finally:
        try:
            lock.unlink(missing_ok=True)
            stop_request.unlink(missing_ok=True)
        finally:
            guard.close()
