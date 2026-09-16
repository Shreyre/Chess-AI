"""Small integration checks for live training and local dashboard controls."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import chess

from chess_ai.dashboard import Dashboard, handler_for
from chess_ai.run_lock import acquire_training_lock
from chess_ai.telemetry import LiveStatus


class DashboardChecks(unittest.TestCase):
    def test_run_lock_excludes_writers_and_recovers_after_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = "\n".join([
                "import os, sys",
                "from pathlib import Path",
                "from chess_ai.run_lock import acquire_training_lock",
                "guard = acquire_training_lock(sys.argv[1])",
                "print('ready', flush=True)",
                "sys.stdin.readline()",
                "(Path(sys.argv[1]) / '.training.lock').unlink()",
                "print('removed', flush=True)",
                "sys.stdin.readline()",
                "os._exit(1)",
            ])
            # Use the interpreter directly so there is no Windows venv launcher child.
            with subprocess.Popen([sys._base_executable, "-c", code, directory],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as process:
                try:
                    self.assertEqual(process.stdout.readline().strip(), "ready")
                    self.assertTrue(Dashboard(root).state()["active"])
                    with self.assertRaises(OSError):
                        acquire_training_lock(root)
                    process.stdin.write("remove\n")
                    process.stdin.flush()
                    self.assertEqual(process.stdout.readline().strip(), "removed")
                    # The OS lock still protects the checkpoint without the PID marker.
                    with self.assertRaises(OSError):
                        acquire_training_lock(root)
                    process.communicate("exit\n", timeout=30)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            with acquire_training_lock(root):
                self.assertTrue(Dashboard(root).state()["active"])

    def test_exited_trainer_does_not_block_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Reproduce a trainer exiting without its normal finally cleanup.
            subprocess.run([sys.executable, "-c", "\n".join([
                "import os, sys",
                "from pathlib import Path",
                "from chess_ai.telemetry import LiveStatus",
                "root = Path(sys.argv[1])",
                "(root / '.training.lock').write_text(str(os.getpid()))",
                "(root / '.stop-request').write_text('stop after checkpoint')",
                "LiveStatus(root).update(force=True, phase='selfplay')",
                "os._exit(1)",
            ]), directory], check=False, timeout=30)
            state = Dashboard(root).state()
            self.assertFalse(state["active"])
            self.assertEqual(state["phase"], "stopped")
            self.assertFalse(state["stop_requested"])
            command = [sys.executable, "-m", "chess_ai", "train", "--run-dir", directory,
                       "--iterations", "1", "--games", "1", "--parallel-games", "1",
                       "--simulations", "1", "--max-plies", "2", "--train-steps", "1",
                       "--batch-size", "2", "--channels", "8", "--blocks", "1", "--device", "cpu"]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            self.assertEqual(Dashboard(root).state()["saved_iteration"], 1)

    def test_live_board_and_local_http_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = LiveStatus(root)
            board = chess.Board()
            board.push_uci("e2e4")
            live.boards([board], 0, 1, force=True)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Dashboard(root)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            address = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(address + "/api/state") as response:
                    state = json.load(response)
                self.assertEqual(state["selected"]["last_san"], "e4")
                self.assertEqual(state["selected"]["fen"], board.fen())
                self.assertIn("<svg", state["board_svg"])
                self.assertFalse(state["active"])
                with urlopen(address) as response:
                    self.assertIn(b"Training room", response.read())
                request = Request(address + "/api/start", data=b'{"iterations":1}',
                                  headers={"Content-Type": "application/json", "Origin": "https://example.com"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
                request = Request(address + "/api/start", data=b'{"iterations":0}',
                                  headers={"Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown()
                worker.join()
                server.server_close()

    def test_start_and_stop_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, "-m", "chess_ai", "train", "--run-dir", directory,
                       "--iterations", "1", "--games", "2", "--parallel-games", "2",
                       "--simulations", "2", "--max-plies", "4", "--train-steps", "2",
                       "--batch-size", "4", "--channels", "8", "--blocks", "1", "--device", "cpu"]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            state = Dashboard(root).state()
            self.assertEqual(state["phase"], "completed")
            self.assertEqual(state["saved_iteration"], 1)
            self.assertEqual(state["selected"]["ply"], 4)
            dashboard = Dashboard(root)
            dashboard.start(100)
            try:
                with self.assertRaises(ValueError):
                    dashboard.start(1)
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    state = dashboard.state()
                    if state["phase"] in ("selfplay", "learning", "saving") and state["locked"]:
                        dashboard.stop()
                        break
                    time.sleep(0.05)
                else:
                    self.fail("Training did not publish its live status")
                dashboard.process.wait(timeout=60)
                self.assertEqual(dashboard.process.returncode, 0)
                final = dashboard.state()
                self.assertEqual(final["phase"], "completed")
                self.assertGreaterEqual(final["saved_iteration"], 2)
                self.assertLess(final["saved_iteration"], 101)
                self.assertFalse(final["active"])
                self.assertFalse(final["stop_requested"])
                self.assertIn("Checkpoint saved", (root / "dashboard-training.log").read_text())
            finally:
                if dashboard.process.poll() is None:
                    dashboard.process.kill()
                    dashboard.process.wait()


if __name__ == "__main__":
    unittest.main()
