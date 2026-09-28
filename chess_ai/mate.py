"""Bounded exhaustive mate proofs, independent of network predictions."""
import time


def immediate_mate_blunders(board, stop=None, deadline=None):
    """Proven losing moves to exclude, only after verifying a safe alternative.

    Interrupted candidates remain unclassified. Automatic draws and our own
    checkmates are safe. Optional draw claims do not prevent an opponent mating.
    """
    position = board.copy(stack=True)
    bad, safe = set(), False

    def cancelled():
        return ((stop is not None and stop.is_set()) or
                (deadline is not None and time.monotonic() >= deadline))

    if position.is_game_over():
        return bad
    # ponytail: only immediate mate threats; deeper defense remains MCTS's job.
    for move in list(position.legal_moves):
        if cancelled():
            break
        position.push(move)
        try:
            if position.is_game_over():
                safe = True
                continue
            for reply in position.legal_moves:
                if cancelled():
                    return bad if safe else set()
                position.push(reply)
                try:
                    mated = position.is_checkmate()
                finally:
                    position.pop()
                if mated:
                    bad.add(move)
                    break
            else:
                safe = True
        finally:
            position.pop()
    return bad if safe else set()


def shortest_mate(board, max_moves=3, max_nodes=512, stop=None, deadline=None):
    """Return (move, mate-in-N, nodes); None means unproven, never 'no mate'.

    Iterative deepening rules out every shorter mate before accepting a move.
    Attacker needs one continuation; every legal defense must lose. Draw claims
    are available to the defender. History is retained for repetition rules.
    """
    if not 0 <= max_moves <= 10 or max_nodes < 1:
        raise ValueError("mate moves must be in 0..10; mate nodes must be positive")
    position = board.copy(stack=True)
    attacker, nodes = position.turn, 0

    def tick():
        nonlocal nodes
        if (nodes >= max_nodes or (stop is not None and stop.is_set()) or
                (deadline is not None and time.monotonic() >= deadline)):
            raise InterruptedError
        nodes += 1

    def forced(remaining):
        if remaining == 0:
            return position.turn != attacker and position.is_checkmate()
        outcome = position.outcome(claim_draw=False)
        if outcome is not None:
            return outcome.winner == attacker
        attacking = position.turn == attacker
        if not attacking and position.can_claim_draw():
            return False
        for move in position.legal_moves:
            tick()
            position.push(move)
            try:
                won = forced(remaining - 1)
            finally:
                position.pop()
            if won == attacking:
                return attacking
        return not attacking

    if max_moves == 0 or position.is_game_over():
        return None, None, nodes
    # ponytail: no transposition cache; a correct cache must include draw history.
    try:
        for distance in range(1, max_moves + 1):
            for move in position.legal_moves:
                tick()
                position.push(move)
                try:
                    won = forced(2 * distance - 2)
                finally:
                    position.pop()
                if won:
                    return move, distance, nodes
    except InterruptedError:
        pass
    return None, None, nodes
