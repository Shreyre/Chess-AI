"""Local Lichess puzzle validation and repeatable solution-line evaluation."""
import hashlib
import json
from pathlib import Path
import time

import chess

from .search import Search, run_searches
from .engine import engine_move


def puzzle_position(row):
    """Validate the entire line; return the solver's board after the setup move."""
    if (not isinstance(row, dict) or not isinstance(row.get('PuzzleId'), str) or
            not row['PuzzleId'] or not isinstance(row.get('FEN'), str) or
            not isinstance(row.get('Moves'), str) or
            not isinstance(row.get('Themes'), list) or
            not all(isinstance(theme, str) for theme in row['Themes']) or
            type(row.get('Rating')) is not int or row['Rating'] < 0):
        raise ValueError('Invalid Lichess puzzle fields')
    board = chess.Board(row['FEN'])
    if not board.is_valid():
        raise ValueError(f"Invalid puzzle position: {row['PuzzleId']}")
    tokens = row['Moves'].split()
    if len(tokens) < 2 or len(tokens) % 2:
        raise ValueError('Puzzle needs a setup move and a line ending on a solver move')
    moves = []
    for token in tokens:
        if board.is_game_over():
            raise ValueError('Puzzle continues after the game is over')
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal puzzle move: {token}")
        moves.append(move)
        board.push(move)
    if 'mate' in row['Themes'] and not board.is_checkmate():
        raise ValueError('A mate puzzle must end in checkmate')
    board = chess.Board(row['FEN'])
    board.push(moves[0])
    return board, moves[1:]


def load_puzzles(path):
    rows, seen = [], set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                puzzle_position(row)
                if row['PuzzleId'] in seen:
                    raise ValueError('Duplicate PuzzleId')
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'Puzzle line {number}: {error}') from error
            seen.add(row['PuzzleId'])
            rows.append(row)
    if not rows:
        raise ValueError('Puzzle file is empty')
    return rows


def assess_puzzles(model, rows, simulations=64, engine=None, engine_nodes=20000, progress=None):
    """Play every solver turn; use dataset replies only while the line matches.

    Immediate alternative checkmates count as correct. Other alternatives count
    as solution mismatches, not necessarily chess blunders. No Elo is inferred.
    """
    started = time.monotonic()
    results = []
    # ponytail: batches of 32 bound CPU/VRAM use; tune only after profiling.
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        states = [puzzle_position(row) for row in batch]
        outcomes = [dict(id=row['PuzzleId'], rating=row['Rating'], themes=row['Themes'],
                         first_move_correct=False, solved=False, decisions=0,
                         expected=None, played=None) for row in batch]
        active = list(range(len(batch)))
        step = 0
        while active:
            if engine is None:
                searches = [Search(states[i][0]) for i in active]
                run_searches(model, searches, simulations)
                choices = []
                for search in searches:
                    moves, policy = search.policy(0)
                    choices.append(moves[int(policy.argmax())])
            else:
                choices = [engine_move(engine, states[i][0], engine_nodes) for i in active]
            following = []
            for i, move in zip(active, choices):
                board, line = states[i]
                expected = line[step]
                board.push(move)
                correct = move == expected or board.is_checkmate()
                outcome = outcomes[i]
                outcome['decisions'] += 1
                if step == 0:
                    outcome['first_move_correct'] = correct
                if not correct:
                    outcome.update(expected=expected.uci(), played=move.uci())
                elif board.is_checkmate() or step + 1 == len(line):
                    outcome['solved'] = True
                else:
                    board.push(line[step + 1])
                    following.append(i)
            active, step = following, step + 2
        results.extend(outcomes)
        if progress:
            progress(len(results), len(rows))

    def summary(items):
        return dict(puzzles=len(items), first_move_correct=sum(r['first_move_correct'] for r in items),
                    solved=sum(r['solved'] for r in items),
                    solve_rate=sum(r['solved'] for r in items) / len(items))

    if not results:
        raise ValueError('No puzzles to assess')
    themes = sorted({theme for row in rows for theme in row['Themes']})
    bands = sorted({row['Rating'] // 500 * 500 for row in rows})
    return dict(**summary(results), simulations=simulations if engine is None else None,
                decision_method='neural_mcts' if engine is None else 'local_uci_engine',
                engine_id=None if engine is None else engine.id,
                engine_nodes=None if engine is None else engine_nodes,
                elapsed_seconds=round(time.monotonic() - started, 3),
                by_theme={t: summary([r for r in results if t in r['themes']]) for t in themes},
                by_rating={f'{b}-{b+499}': summary([r for r in results if b <= r['rating'] < b+500])
                           for b in bands}, results=results,
                note='Solution-line benchmark, not Elo; alternate immediate mates accepted. '
                     'Other alternatives are mismatches, not proven blunders. '
                     'Prior training exposure must be checked separately.')


def assess(args, model, iteration, engine=None):
    report = assess_puzzles(model, load_puzzles(args.puzzles), args.simulations,
                            engine, args.engine_nodes)
    report.update(checkpoint=str(args.checkpoint), iteration=iteration,
                  puzzle_file=str(args.puzzles),
                  puzzle_sha256=hashlib.sha256(Path(args.puzzles).read_bytes()).hexdigest())
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
