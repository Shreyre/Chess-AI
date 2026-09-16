"""Standalone Windows controller: select a visible board and play a saved model."""

import argparse
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import chess
from PIL import ImageGrab, ImageTk
import torch

from .model import load_model, move_index
from .screen_vision import BoardArea, GameTracker, PieceReader, UncertainBoard, align_board_area, board_labels
from .search import select_move
from .training import save_screen_game
from .windows_input import WindowsInput, enable_dpi_awareness


class ScreenPlayer:
    def __init__(self, root, checkpoint):
        self.root = root
        self.windows = WindowsInput()
        self.area = self.reader = self.tracker = None
        self.target = None
        self.worker = None
        self.searches = {}
        self.game_saved = False
        self.game_checkpoint = None
        self.learner = None
        self.learning_queue = {}
        self.stop_signal = threading.Event()
        self.messages = queue.Queue()
        self.model_path = tk.StringVar(value=str(checkpoint))
        self.color = tk.StringVar(value="White")
        self.orientation = tk.StringVar(value="White at bottom")
        self.simulations = tk.IntVar(value=128)
        self.tolerance = tk.DoubleVar(value=0.18)
        self.delay = tk.DoubleVar(value=0.2)
        self.learn = tk.BooleanVar(value=True)
        self.result = tk.StringVar(value="Result…")
        self.learning_status = tk.StringVar(value="Learns after each completed game; the next game uses the saved model.")
        self.status = tk.StringVar(value="Open a new game, then select the 8 by 8 board.")
        self.position = tk.StringVar(value="No board selected")
        self.root.title("Chess AI — Screen player")
        self.root.geometry("470x710")
        self.root.minsize(440, 610)
        self.root.configure(bg="#f3f6f9")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6f9")
        style.configure("TLabel", background="#f3f6f9", foreground="#193247", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=7)
        style.configure("Heading.TLabel", font=("Segoe UI", 20, "bold"))
        viewport = tk.Canvas(root, background="#f3f6f9", highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=viewport.yview)
        scrollbar.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)
        viewport.configure(yscrollcommand=scrollbar.set)
        panel = ttk.Frame(viewport, padding=20)
        content = viewport.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _: viewport.configure(scrollregion=viewport.bbox("all")))
        viewport.bind("<Configure>", lambda event: viewport.itemconfigure(content, width=event.width))
        root.bind("<MouseWheel>", lambda event: viewport.yview_scroll(-int(event.delta / 120), "units"))
        ttk.Label(panel, text="Play on your screen", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Works over a visible chess board in an app or browser.", wraplength=420).pack(anchor="w", pady=(3, 13))
        row = ttk.Frame(panel)
        row.pack(fill="x")
        self.color_box = ttk.Combobox(row, textvariable=self.color, values=("White", "Black"), width=10, state="readonly")
        self.color_box.pack(side="left")
        ttk.Label(row, text="AI side", padding=(6, 0, 18, 0)).pack(side="left")
        self.orientation_box = ttk.Combobox(row, textvariable=self.orientation,
                                          values=("White at bottom", "Black at bottom"), width=19, state="readonly")
        self.orientation_box.pack(side="left")
        for box in (self.color_box, self.orientation_box):
            box.bind("<<ComboboxSelected>>", lambda _: self.clear_calibration())
        self.select_button = ttk.Button(panel, text="Select board area…", command=self.select_area)
        self.select_button.pack(fill="x", pady=(12, 5))
        ttk.Label(panel, text="Drag around the squares only. Start with all pieces in their original places.", wraplength=420).pack(anchor="w")
        self.preview = ttk.Label(panel, text="Your selected board will appear here", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=12)
        ttk.Label(panel, textvariable=self.position, wraplength=420).pack(anchor="w")
        self.status_label = ttk.Label(panel, textvariable=self.status, wraplength=420, foreground="#266997")
        self.status_label.pack(anchor="w", pady=(8, 12))
        tuning = ttk.Frame(panel)
        tuning.pack(fill="x")
        self.tuning = []
        for text, variable, lower, upper, increment in (
                ("Searches", self.simulations, 1, 10000, 32),
                ("Vision tolerance", self.tolerance, 0.08, 0.3, 0.01),
                ("Click delay (s)", self.delay, 0.1, 1, 0.1)):
            column = ttk.Frame(tuning)
            column.pack(side="left", expand=True, fill="x")
            ttk.Label(column, text=text).pack(anchor="w")
            spin = ttk.Spinbox(column, textvariable=variable, from_=lower, to=upper,
                               increment=increment, width=10)
            spin.pack(anchor="w", pady=3)
            self.tuning.append(spin)
        model_row = ttk.Frame(panel)
        model_row.pack(fill="x", pady=(8, 10))
        ttk.Label(model_row, text="Model").pack(side="left", padx=(0, 6))
        self.model_entry = ttk.Entry(model_row, textvariable=self.model_path)
        self.model_entry.pack(side="left", fill="x", expand=True)
        self.browse_button = ttk.Button(model_row, text="Browse", command=self.choose_model)
        self.browse_button.pack(side="left", padx=(5, 0))
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="Start / resume", command=self.start, state="disabled")
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(buttons, text="Stop now (F8)", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.learn_button = ttk.Checkbutton(panel, text="Learn from completed games", variable=self.learn,
                                             command=lambda: self.set_running(False))
        self.learn_button.pack(anchor="w", pady=(10, 3))
        ttk.Label(panel, textvariable=self.learning_status, wraplength=420).pack(anchor="w")
        result_row = ttk.Frame(panel)
        result_row.pack(fill="x", pady=(8, 3))
        self.result_box = ttk.Combobox(result_row, textvariable=self.result,
                                       values=("White won", "Black won", "Draw"), state="readonly", width=13)
        self.result_box.pack(side="left")
        self.finish_button = ttk.Button(result_row, text="Finish & learn", command=self.manual_finish, state="disabled")
        self.finish_button.pack(side="left", padx=(8, 0))
        ttk.Label(panel, text="For resignation, timeout or a covered final board: stop, choose the result, then Finish & learn.",
                  wraplength=420).pack(anchor="w")
        ttk.Label(panel, text="F8 stops from any window. Keep the board visible and in place.", wraplength=420).pack(anchor="w", pady=(10, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.poll)

    def clear_calibration(self):
        self.reader = self.tracker = None
        self.finish_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.status.set("Side or orientation changed. Select and calibrate the starting board again.")

    def choose_model(self):
        path = filedialog.askopenfilename(title="Select your trained chess model", filetypes=[("PyTorch model", "*.pt")])
        if path:
            self.model_path.set(path)

    def select_area(self):
        self.root.withdraw()
        self.root.after(350, self.overlay)

    def overlay(self):
        try:
            screenshot = ImageGrab.grab(all_screens=True, include_layered_windows=True)
            left, top, width, height = self.windows.desktop()
        except OSError as error:
            self.status.set(f"Cannot capture the screen: {error}")
            self.root.deiconify()
            return
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.geometry(f"{width}x{height}+0+0")
        canvas = tk.Canvas(overlay, width=width, height=height, highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True)
        photo = ImageTk.PhotoImage(screenshot)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo
        canvas.create_rectangle(20, 20, 680, 70, fill="#193247", outline="")
        canvas.create_text(38, 45, text="Drag around the 8 × 8 board. Escape cancels.",
                           anchor="w", font=("Segoe UI", 15), fill="white")
        overlay.update_idletasks()
        handle = self.windows.api.GetAncestor(overlay.winfo_id(), 2)
        self.windows.api.SetWindowPos(handle, -1, left, top, width, height, 0x0040)
        origin = []

        def cancel(_=None):
            overlay.destroy()
            self.root.deiconify()

        def down(event):
            origin[:] = [event.x, event.y]

        def drag(event):
            if not origin:
                return
            x1, y1 = origin
            x2, y2 = event.x, event.y
            canvas.delete("selection")
            canvas.create_rectangle(x1, y1, x2, y2, outline="#42e9ba", width=3, tags="selection")
            for step in range(1, 8):
                x, y = x1 + (x2-x1)*step/8, y1 + (y2-y1)*step/8
                canvas.create_line(x, y1, x, y2, fill="#42e9ba", tags="selection")
                canvas.create_line(x1, y, x2, y, fill="#42e9ba", tags="selection")

        def up(event):
            if not origin:
                return
            x1, x2 = sorted((max(0, min(width, origin[0])), max(0, min(width, event.x))))
            y1, y2 = sorted((max(0, min(height, origin[1])), max(0, min(height, event.y))))
            overlay.destroy()
            try:
                selected = align_board_area(screenshot, BoardArea(x1, y1, x2, y2))
                area = BoardArea(left+selected.left, top+selected.top, left+selected.right, top+selected.bottom)
                cropped = screenshot.crop(selected.bbox)
                tolerance = self.tolerance.get()
                if not 0.08 <= tolerance <= 0.3:
                    raise ValueError("Vision tolerance must be in 0.08–0.30")
                reader = PieceReader(cropped, self.orientation.get() == "White at bottom", tolerance)
                self.area, self.reader = area, reader
                self.tracker = GameTracker(self.color.get() == "White")
                self.searches = {}
                self.game_saved = False
                self.game_checkpoint = None
                self.result.set("Result…")
                self.finish_button.configure(state="disabled")
                self.target = self.windows.window_at(area.center(chess.D4))
                if not self.target:
                    raise ValueError("No application window was found under the board")
                self.show_image(cropped)
                self.position.set(f"Board: {area.right-area.left} × {area.bottom-area.top} pixels at ({area.left}, {area.top})")
                self.status.set("Starting position calibrated. Keep this board visible, then press Start.")
                self.start_button.configure(state="normal")
            except (ValueError, OSError, tk.TclError) as error:
                self.reader = self.tracker = None
                self.start_button.configure(state="disabled")
                self.finish_button.configure(state="disabled")
                self.status.set(f"Select again: {error}")
            self.root.deiconify()

        canvas.bind("<ButtonPress-1>", down)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", up)
        overlay.bind("<Escape>", cancel)
        overlay.focus_force()
        overlay.grab_set()

    def show_image(self, image):
        copy = image.copy()
        copy.thumbnail((280, 280))
        self.preview.image = ImageTk.PhotoImage(copy)
        self.preview.configure(image=self.preview.image, text="")

    def set_running(self, running):
        for widget in (self.select_button, self.browse_button, self.model_entry, self.learn_button, *self.tuning):
            widget.configure(state="disabled" if running else "normal")
        for widget in (self.color_box, self.orientation_box, self.result_box):
            widget.configure(state="disabled" if running else "readonly")
        self.start_button.configure(state="disabled" if running or self.reader is None or self.game_saved else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.finish_button.configure(state="normal" if not running and self.tracker is not None and self.searches
                                     and not self.game_saved and self.learn.get() else "disabled")

    def start(self):
        if self.reader is None or (self.worker and self.worker.is_alive()):
            return
        try:
            simulations, tolerance, delay = self.simulations.get(), self.tolerance.get(), self.delay.get()
            if not 1 <= simulations <= 10000 or not 0.08 <= tolerance <= 0.3 or not 0.1 <= delay <= 1:
                raise ValueError("Check the search count, vision tolerance, and click delay ranges")
            checkpoint = Path(self.model_path.get()).resolve()
            if not checkpoint.is_file():
                raise ValueError("Choose an existing trained .pt model")
            if self.learn.get() and (checkpoint.name not in ("model.pt", "latest.pt") or
                                    not (checkpoint.parent / "latest.pt").is_file()):
                raise ValueError("For learning, choose the run's model.pt with its latest.pt beside it, or turn learning off")
            if self.searches and self.game_checkpoint and self.game_checkpoint.parent != checkpoint.parent:
                raise ValueError("Select a new starting board before changing the training run")
        except (ValueError, tk.TclError) as error:
            self.status.set(str(error))
            return
        self.reader.tolerance = tolerance
        self.game_checkpoint = checkpoint
        self.stop_signal.clear()
        self.set_running(True)
        self.status.set("Loading your AI. Click the selected game window to let it play. F8 stops.")
        self.worker = threading.Thread(target=self.play_loop, args=(checkpoint, simulations, delay, self.learn.get()), daemon=True)
        self.worker.start()

    def finish_game(self, result=None):
        if self.game_saved:
            return
        if self.tracker is None or self.game_checkpoint is None:
            raise ValueError("Start and track a game before saving its result")
        directory = self.game_checkpoint.parent
        if self.game_checkpoint.name not in ("model.pt", "latest.pt") or not (directory / "latest.pt").is_file():
            raise ValueError("Learning needs the run's model.pt and full latest.pt checkpoint")
        save_screen_game(directory, self.tracker.board, self.searches, result)
        self.game_saved = True
        self.messages.put(("learn", directory))

    def manual_finish(self):
        try:
            result = {"White won": "1-0", "Black won": "0-1", "Draw": "1/2-1/2"}.get(self.result.get())
            if result is None:
                raise ValueError("Choose White won, Black won, or Draw")
            self.finish_game(result)
            self.status.set("Game saved for learning. Select a new starting board for the next game.")
            self.set_running(False)
        except (ValueError, OSError) as error:
            self.status.set(str(error))

    def stop(self):
        self.stop_signal.set()
        self.status.set("Stopping…")

    def play_loop(self, checkpoint, simulations, delay, learn=False):
        try:
            torch.set_num_threads(2)
            model, _ = load_model(checkpoint, "cpu")
            previous = None
            uncertain_since = None
            pending_since = time.monotonic() if self.tracker.pending else None
            while not self.windows.stopped(self.stop_signal):
                if self.windows.foreground() != self.target:
                    self.messages.put(("status", "Click the selected game window to continue. F8 stops."))
                    previous = None
                    self.stop_signal.wait(0.25)
                    continue
                try:
                    image = ImageGrab.grab(bbox=self.area.bbox, all_screens=True, include_layered_windows=True)
                    observed = self.reader.read(image)
                    if observed != previous:
                        previous = observed
                        self.stop_signal.wait(0.25)
                        continue
                    self.tracker.observe(observed)
                    uncertain_since = None
                except UncertainBoard as error:
                    uncertain_since = uncertain_since or time.monotonic()
                    if time.monotonic() - uncertain_since > 5:
                        raise UncertainBoard(f"Paused: {error}. Clear overlays or recalibrate for a new game.")
                    self.messages.put(("status", "Waiting for a clear, stable board…"))
                    self.stop_signal.wait(0.25)
                    continue
                board = self.tracker.board
                self.messages.put(("frame", image, board.fen()))
                if self.tracker.pending is not None:
                    if pending_since and time.monotonic() - pending_since > 6:
                        raise UncertainBoard("Move was not confirmed. Check the board or promotion menu, then resume.")
                    self.messages.put(("status", "Waiting for the move to appear on the board…"))
                elif board.is_game_over():
                    if learn:
                        self.finish_game()
                    self.messages.put(("status", f"Game finished: {board.result()}. Select a new starting board to play again."))
                    return
                elif board.turn != self.tracker.color:
                    self.messages.put(("status", "Waiting for the opponent. F8 stops."))
                    pending_since = None
                else:
                    self.messages.put(("status", "Your AI is choosing a move… F8 stops."))
                    move, search = select_move(model, board, simulations, stop=self.stop_signal)
                    if self.windows.stopped(self.stop_signal):
                        break
                    fresh = ImageGrab.grab(bbox=self.area.bbox, all_screens=True, include_layered_windows=True)
                    if self.reader.read(fresh) != board_labels(board):
                        raise UncertainBoard("The board changed while thinking. Resume after it settles.")
                    moves, policy = search.policy(1)
                    self.searches[len(board.move_stack)] = (move.uci(),
                        torch.tensor([move_index(candidate, board.turn) for candidate in moves]), torch.from_numpy(policy))
                    if move.promotion:
                        self.tracker.pending = move
                        self.messages.put(("status", f"Promotion: play {move.uci()} and choose {chess.piece_name(move.promotion)} yourself, then Resume."))
                        return
                    source = self.area.center(move.from_square, self.reader.white_bottom)
                    target = self.area.center(move.to_square, self.reader.white_bottom)
                    # Check both squares before the first click; no blind re-clicks after failure.
                    if any(self.windows.window_at(point) != self.target for point in (source, target)):
                        raise InterruptedError("Another window covers the selected board")
                    self.windows.click(source, self.target, self.stop_signal)
                    if self.stop_signal.wait(delay) or self.windows.stopped(self.stop_signal):
                        raise InterruptedError("Stopped before the destination click; clear any selected piece before resuming")
                    self.windows.click(target, self.target, self.stop_signal)
                    self.tracker.pending = move
                    pending_since = time.monotonic()
                    self.messages.put(("status", f"Played {board.san(move)}. Waiting for confirmation…"))
                    previous = None
                self.stop_signal.wait(0.25)
            self.messages.put(("status", "Stopped. Resume to continue this game, or select a new board."))
        except Exception as error:
            self.messages.put(("status", str(error)))
        finally:
            self.stop_signal.set()
            self.messages.put(("done",))

    def poll(self):
        if self.worker and self.worker.is_alive() and self.windows.stopped(self.stop_signal):
            self.stop_signal.set()
        try:
            while True:
                message = self.messages.get_nowait()
                if message[0] == "status":
                    self.status.set(message[1])
                elif message[0] == "frame":
                    self.show_image(message[1])
                    self.position.set(f"Tracking move {chess.Board(message[2]).fullmove_number}")
                elif message[0] == "done":
                    self.set_running(False)
                elif message[0] == "learn":
                    self.learning_queue[message[1]] = None
                    self.learning_status.set("Game saved. Learning is queued; progress appears in the dashboard.")
        except queue.Empty:
            pass
        if self.learner is not None and self.learner.poll() is not None:
            self.learning_status.set("Learning saved. The next game will load the updated model." if self.learner.returncode == 0
                                     else "Learning stopped with an error. Game data is saved; see screen-learning.log in the run folder.")
            self.learner = None
        if self.learner is None and self.learning_queue:
            directory = next(iter(self.learning_queue))
            try:
                executable = str(Path(sys.executable).with_name("python.exe")) if sys.platform == "win32" else sys.executable
                command = [executable, "-m", "chess_ai", "train", "--resume", str(directory / "latest.pt"),
                           "--external-only", "--iterations", "1"]
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                with (directory / "screen-learning.log").open("a", encoding="utf-8") as output:
                    self.learner = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                                    stdin=subprocess.DEVNULL, creationflags=flags)
                self.learning_status.set("Learning queued/running. It waits if dashboard training is active.")
            except OSError as error:
                self.learning_status.set(f"Game saved, but learning could not start: {error}")
            del self.learning_queue[directory]
        self.root.after(100, self.poll)

    def close(self):
        self.stop_signal.set()
        if self.worker and self.worker.is_alive():
            self.status.set("Stopping before closing…")
            self.root.after(100, self.close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Select a chess board anywhere on the Windows desktop")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs/main/model.pt")
    args = parser.parse_args()
    enable_dpi_awareness()
    root = tk.Tk()
    ScreenPlayer(root, args.checkpoint)
    root.mainloop()


if __name__ == "__main__":
    main()
