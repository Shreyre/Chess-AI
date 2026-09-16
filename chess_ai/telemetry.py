"""Small, best-effort live snapshots; training checkpoints remain authoritative."""

import json
import os
from pathlib import Path
import tempfile
import time


class LiveStatus:
    def __init__(self, directory):
        self.path = Path(directory) / "live.json"
        self.state = {"pid": os.getpid(), "boards": []}
        self.last_write = 0.0
        self.warned = False

    def update(self, force=False, **fields):
        self.state.update(fields)
        now = time.time()
        if not force and now - self.last_write < 0.5:
            return
        self.state["updated_at"] = now
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as output:
                temporary = output.name
                json.dump(self.state, output, allow_nan=False)
            for attempt in range(3):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.01)  # Windows readers can briefly hold the old file open.
            self.last_write = now
        except OSError as error:
            if not self.warned:
                print(f"Live dashboard update unavailable: {error}", flush=True)
                self.warned = True
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def boards(self, boards, offset, ply, force=False):
        if not force and time.time() - self.last_write < 0.5:
            return
        snapshots = []
        for index, board in enumerate(boards):
            last = board.peek() if board.move_stack else None
            previous = board.copy(stack=True)
            if last:
                previous.pop()
            snapshots.append(dict(id=offset + index + 1, fen=board.fen(),
                                  ply=len(board.move_stack), result=board.result(),
                                  last_move=last.uci() if last else None,
                                  last_san=previous.san(last) if last else None))
        self.update(force=force, phase="selfplay", boards=snapshots, ply=ply)
