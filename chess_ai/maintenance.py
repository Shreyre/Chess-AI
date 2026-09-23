"""Checkpoint-bound teacher refresh and conservative playing-model selection."""
import json
from pathlib import Path
from types import SimpleNamespace

from .evaluation import play_match
from .model import atomic_save, load_model, model_snapshot
from .teacher import generate, load_teacher


def maintenance(model, directory, config, iteration, live):
    directory = Path(directory)
    stop = lambda: (directory / ".stop-request").exists()
    if stop():
        return
    def progress(stage, count=0):
        live.update(force=True, phase="learning", maintenance=stage, maintenance_progress=count)
    model.eval()
    try:
        every = config.get("teacher_refresh_every", 0)
        if every and iteration - config.get("teacher_refresh_iteration", -every) >= every:
            progress("Refreshing teacher examples")
            corpus = Path(config["teacher_data"]) if config.get("teacher_data") else None
            sharded = corpus is not None and corpus.is_dir()
            output = (corpus if sharded else directory / "teacher") / f"teacher-{iteration:06d}.pt"
            if not output.exists():
                generate(SimpleNamespace(engine=config["teacher_engine"],
                         pgn=directory / "selfplay.pgn", fens=config.get("teacher_fens"),
                         samples=1024, recent_games=2000, nodes=20000,
                         seed=config["seed"] + iteration, output=output,
                         max_records=config.get("teacher_capacity", 20000),
                         previous=None if sharded else config.get("teacher_data")), stop,
                         lambda n: progress("Refreshing teacher examples", n), model=model)
            load_teacher(output)
            config.update(teacher_data=str((corpus if sharded else output).resolve()),
                          teacher_refresh_iteration=iteration)
        every = config.get("gate_every", 0)
        if not every or iteration - config.get("gate_iteration", -every) < every or stop():
            return
        best = directory / "best.pt"
        if not best.exists():
            baseline = model_snapshot(model, iteration)
            baseline["selection"] = {"status": "initial baseline; not yet tested"}
            atomic_save(baseline, best)
        device = next(model.parameters()).device
        incumbent, data = load_model(best, device)
        if data["iteration"] >= iteration:
            config["gate_iteration"] = iteration
            return
        archive = directory / "best-history" / f"model-{data['iteration']:06d}.pt"
        if not archive.exists():
            atomic_save(data, archive)
        paths = sorted(set(directory.glob("model-*.pt")) |
                       set((directory / "best-history").glob("model-*.pt")),
                       key=lambda p: int(p.stem.split("-")[-1]), reverse=True)
        opponents, seen = [(best, incumbent, data["iteration"])], {data["iteration"]}
        for path in paths:
            number = int(path.stem.split("-")[-1])
            if number < iteration and number not in seen:
                opponents.append((path, None, number))
                seen.add(number)
            if len(opponents) == 3:
                break
        results = []
        for path, opponent, number in opponents:
            progress(f"Testing against iteration {number}")
            opponent = opponent if opponent is not None else load_model(path, device)[0]
            report, pgns = play_match(model, opponent, games=20, simulations=64,
                seed=config["seed"] + iteration, stop=stop,
                progress=lambda n: progress(f"Testing against iteration {number}", n))
            report["opponent_iteration"] = number
            results.append(report)
            destination = directory / "evaluations" / f"{iteration:06d}-vs-{number:06d}.pgn"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("\n\n".join(pgns) + "\n", encoding="utf-8")
            del opponent
        # ponytail: 20 games per opponent screen regressions; increase for statistical strength claims.
        passed = all(not r["truncated"] and r["score"] >= (0.6 if i == 0 else 0.5)
                     for i, r in enumerate(results))
        report = dict(candidate=iteration, incumbent=data["iteration"], promoted=passed,
                      matches=results, note="Screening gate, not an Elo estimate")
        if stop():
            raise InterruptedError("Selection cancelled")
        if passed:
            snapshot = model_snapshot(model, iteration)
            snapshot["selection"] = report
            atomic_save(snapshot, best)
        destination = directory / "evaluations" / f"gate-{iteration:06d}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        config["gate_iteration"] = iteration
        print(json.dumps(report), flush=True)
    except InterruptedError as error:
        print(str(error), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        # Fail closed: keep the old dataset/model and retry at the next saved boundary.
        live.update(force=True, maintenance_error=str(error))
        print(f"Maintenance deferred: {error}", flush=True)
    finally:
        live.update(force=True, maintenance=None, maintenance_progress=0)
