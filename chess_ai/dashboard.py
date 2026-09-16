"""Local dashboard using the standard library HTTP server."""

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

import chess
import chess.svg

from .run_lock import training_pid


def read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def read_metrics(path):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            # ponytail: scan the small log; cache by mtime if runs reach millions of iterations.
            for line in deque(stream, maxlen=200):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # A writer may not have finished the last line yet.
    except OSError:
        pass
    return rows


class Dashboard:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.guard = threading.Lock()

    def state(self, game=1, flipped=False):
        state = read_json(self.directory / "live.json", {})
        metrics = read_metrics(self.directory / "metrics.jsonl")
        owner = training_pid(self.directory)
        locked = owner is not None
        starting = self.process is not None and self.process.poll() is None and not locked
        active = locked or starting
        phase = state.get("phase", "idle")
        if starting:
            phase = "starting"
        elif locked:
            if owner != state.get("pid"):
                phase = "initializing"
        elif not active and phase in ("initializing", "selfplay", "learning", "saving"):
            phase = "stopped"
        snapshots = state.get("boards", [])
        selected = next((board for board in snapshots if board["id"] == game),
                        snapshots[0] if snapshots else None)
        board = chess.Board(selected["fen"]) if selected else chess.Board()
        last = chess.Move.from_uci(selected["last_move"]) if selected and selected["last_move"] else None
        svg = chess.svg.board(board, lastmove=last, flipped=flipped, size=560,
                              colors={"square light": "#e7ecf1", "square dark": "#7191aa",
                                      "square light lastmove": "#c5d9ad",
                                      "square dark lastmove": "#a0bb7c", "margin": "#f6f8fa",
                                      "coord": "#526579"})
        return dict(state, phase=phase, active=active, locked=locked,
                    run_name=self.directory.name, metrics=metrics, selected=selected,
                    board_svg=svg, stop_requested=locked and (self.directory / ".stop-request").exists(),
                    checkpoint_exists=(self.directory / "latest.pt").exists(),
                    stale=active and phase != "starting" and time.time() - state.get("updated_at", 0) > 30)

    def start(self, iterations):
        if type(iterations) is not int or not 1 <= iterations <= 10000:
            raise ValueError("Choose between 1 and 10,000 iterations")
        with self.guard:
            if self.state()["active"]:
                raise ValueError("Training is already running or its run lock still exists")
            command = [sys.executable, "-m", "chess_ai", "train", "--run-dir", str(self.directory),
                       "--iterations", str(iterations)]
            checkpoint = self.directory / "latest.pt"
            if checkpoint.exists():
                command += ["--resume", str(checkpoint)]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            with (self.directory / "dashboard-training.log").open("a", encoding="utf-8") as output:
                self.process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, creationflags=flags)

    def stop(self):
        with self.guard:
            state = self.state()
            if not state["locked"] or state["phase"] not in ("selfplay", "learning", "saving"):
                raise ValueError("Wait for training to start before requesting a stop")
            (self.directory / ".stop-request").write_text("stop after checkpoint\n", encoding="utf-8")


def handler_for(dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def valid_host(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in allowed

        def send(self, data, content_type="application/json", code=200):
            body = data.encode("utf-8") if isinstance(data, str) else json.dumps(data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.valid_host():
                return self.send({"error": "Invalid host"}, code=403)
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8"), "text/html")
            if url.path == "/api/state":
                try:
                    query = parse_qs(url.query)
                    return self.send(dashboard.state(int(query.get("game", ["1"])[0]),
                                                     query.get("flipped", ["0"])[0] == "1"))
                except (ValueError, KeyError):
                    return self.send({"error": "Invalid board selection"}, code=400)
            self.send({"error": "Not found"}, code=404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected = "http://" + self.headers.get("Host", "")
            if not self.valid_host() or (origin is not None and origin != expected):
                return self.send({"error": "Only local dashboard requests are allowed"}, code=403)
            if self.headers.get("Content-Type") != "application/json":
                return self.send({"error": "Expected JSON"}, code=415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/start":
                    dashboard.start(body.get("iterations"))
                elif self.path == "/api/stop":
                    dashboard.stop()
                else:
                    return self.send({"error": "Not found"}, code=404)
                self.send({"ok": True})
            except (ValueError, OSError) as error:
                self.send({"error": str(error)}, code=400)

    return Handler


def serve(directory, port=8765):
    dashboard = Dashboard(directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(dashboard))
    print(f"Training dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard closed. Any running training continues.", flush=True)
    finally:
        server.server_close()
