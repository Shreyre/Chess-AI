# Chess AI

A local chess AI that starts with random neural-network weights and learns by
playing itself. It uses **AlphaZero-style reinforcement learning**: a policy/value
network plus Monte Carlo Tree Search (MCTS). Pure self-play starts without a
pretrained model or hand-written piece-value reward. Optional **hybrid training**
adds offline Stockfish-labelled examples; the trained network still plays on its own.

Built for this computer's **NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)**. CPU mode also
works. The screen player can also learn from completed games in your chosen
chess application or website.

## Start / continue training

### Watch live in your browser

```powershell
.\chess.ps1 dashboard
```

Open **http://127.0.0.1:8765**. Choose an iteration count and click **Resume
training**. The board updates during self-play; select another parallel game or
flip its orientation. Saved totals, outcomes, and loss charts update after each
checkpoint. **Stop after this iteration** lets the current iteration finish and
save. Closing the page or dashboard leaves training running.

The dashboard can also watch training launched from the command line. It binds
only to this computer, needs no account or extra dependencies, and uses a small
`live.json` snapshot in the run folder. Its control-launched training output is
saved in `dashboard-training.log`. Use `--run-dir runs/experiment --port 8766` to
watch a different run. Old checkpoints work unchanged; live board data appears
when training runs with this version.

### Train from PowerShell

Dependencies are installed in `.venv`. From PowerShell in this folder:

```powershell
.\train.ps1 -Iterations 100
```

This starts `runs/main` or resumes its checkpoint. `Iterations` means **additional
iterations**, each containing self-play games followed by gradient updates. The
first run uses random weights. Subsequent runs preserve what it learned.

**Press Ctrl+C once to stop.** Resume returns to the last completed iteration,
including its replay buffer, optimizer and random generator states. Work since
that checkpoint is discarded. Checkpoints are replaced atomically. A run lock
prevents two trainers writing the same files. After a hard process kill, the
dashboard enables Resume and training automatically recovers the abandoned lock
after checking that its owner has exited. The persistent `.training.guard` file
holds an operating-system lock while training runs; leave it in place.

Full control:

```powershell
.\chess.ps1 train --run-dir runs/experiment --iterations 20 --games 16 --parallel-games 8 --simulations 64 --batch-size 128 --train-steps 100 --device cuda
.\chess.ps1 train --resume runs/experiment/latest.pt --iterations 20
```

Resume keeps saved settings unless you explicitly override them. Network width
and block count cannot change on resume. The original seed and saved random state
are also retained. Use another run folder for a new model.
Eight concurrent games batch neural inference on the GPU; `--threads 2` limits
PyTorch CPU threads. `--device auto` chooses CUDA when available.

### Hybrid training: teacher examples plus self-play

Use an official [Stockfish release](https://github.com/official-stockfish/Stockfish/releases)
as an offline teacher. No additional Python package is needed. Keep downloaded
engine binaries and generated datasets under `runs/` (ignored by Git).

```powershell
# Sample up to 1,024 games and add tactical/endgame practice positions.
.\chess.ps1 teach --engine runs/tools/stockfish-19/stockfish/stockfish-windows-x86-64-universal.exe --pgn runs/main/selfplay.pgn --fens docs/practice-positions.fen --samples 1024 --nodes 20000 --output runs/teacher-v1.pt

# Stop the active trainer after its checkpoint before resuming with new settings.
# Short teacher-only warmup; the existing network, optimizer and replay survive.
.\chess.ps1 train --resume runs/main/latest.pt --teacher-data runs/teacher-v1.pt --teacher-only --iterations 1 --train-steps 200 --learning-rate 0.0003

# Keep 500 concurrent games; mix 25% teacher examples with self-play batches.
.\chess.ps1 train --resume runs/main/latest.pt --iterations 20 --games 500 --parallel-games 500 --simulations 64 --opening-plies 4 --train-steps 100 --teacher-fraction 0.25

# Held-out move agreement; run on both old and new checkpoints at the same budget.
.\chess.ps1 assess --checkpoint runs/main/model.pt --teacher-data runs/teacher-v1.pt --simulations 64 --output runs/teacher-assessment.json
# Paired openings with colors swapped; use an even game count.
.\chess.ps1 evaluate --checkpoint runs/main/model.pt --opponent runs/hybrid-baseline/model.pt --games 20 --simulations 64 --opening-plies 4 --output runs/hybrid-match.json
```

Teacher labels include all legal moves, a policy over the teacher's top three,
and a win/draw/loss-based value from the side-to-move perspective. Roughly 10% of
distinct position groups are held out; reflected and color-swapped equivalents
stay in the same partition. Teacher targets are approximate engine analysis,
not guaranteed optimal moves. Generation refuses to overwrite an existing dataset.

The replay buffer now samples from the **entire incoming batch**, retaining up to
half its capacity for old experience when a large batch arrives. It no longer
keeps only the last games. `--train-steps` is a minimum for self-play: updates scale
to the number of new positions and the space left after teacher examples. For
50,000 new positions, batch size 128 and a 25% teacher mix, this means at least
521 updates. Teacher-only and screen-only learning keep their bounded step counts.

`--opening-plies 4` adds four random legal opening half-moves before searched
self-play; these exploratory moves are not used as policy targets. There is no
penalty for correct defensive repetition. Teacher-only updates increase the
iteration/update counters but do not fabricate played games.

The teacher dataset path, fraction and opening settings persist in checkpoints,
including dashboard resumes. Screen-game learning preserves these settings but
does not mix teacher examples into its short update. Set `--teacher-fraction 0`
to stop mixing teacher examples. Regenerate a new dataset from newer games as the
model changes and pass its path with `--teacher-data`; datasets are not refreshed
automatically. The starter dataset is small, so held-out checks and longer matches
matter more than low training loss. A short match is not an Elo measurement.

### Training files

| File in the run folder | Purpose |
| --- | --- |
| `latest.pt` | Full resumable checkpoint; save after every iteration |
| `model.pt` | Small current model for playing and analysis |
| `initial.pt` | Original random network for comparison |
| `model-000025.pt`, etc. | Model snapshots every 25 iterations |
| `metrics.jsonl` | Losses, outcomes, truncations, time, cumulative games and updates |
| `selfplay.pgn` | Self-play games, readable by chess applications |
| `screen-games/*.pt` | Completed external games, search targets, results and PGN text |
| `screen-learning.log` | Automatic learning progress from screen games |

Read the last progress record:

```powershell
Get-Content runs/main/metrics.jsonl -Tail 1
```

`value_positions` counts positions with known final results. Games reaching
`--max-plies` (default 512 half-moves) are marked **truncated**, not draws. Their
search policies can train the policy head, but they do not supply value targets.
If most games are truncated, increase the move limit and inspect the PGNs.
Early play can be poor and repetitive; decreasing training loss does not prove
increasing playing strength.

## Play or inspect a position

Terminal chess accepts SAN (`Nf3`, `O-O`) or UCI (`g1f3`, `e7e8q`) moves:

```powershell
.\chess.ps1 play --checkpoint runs/main/model.pt --color white
.\chess.ps1 analyze --checkpoint runs/main/model.pt --simulations 256
.\chess.ps1 analyze --checkpoint runs/main/model.pt --fen "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
```

The reported value is an uncalibrated outcome estimate from the side to move,
between -1 and +1. It is not a centipawn score or rating.

## Select a board anywhere on the Windows desktop

Run the standalone controller from Command Prompt:

```cmd
cd /d "C:\Users\patel\OneDrive\Documents\Chess AI" && screen-player.cmd
```

You can also launch `.venv/Scripts/chess-ai-screen.exe`. This is a separate
Windows window; the dashboard is not required.

**Continue after restarting:** the screen player saves its calibration, move
history, pending move, and learning targets when you stop or close it. On reopening,
click **Reposition board…**, select the current board, then **Start / resume**.
It verifies the visible position before enabling play; saved window coordinates
are not reused. The local save is `runs/screen-session.pt`.

**Join an existing game without a save:** click **Continue current game…**, paste
the full current FEN, then select the board precisely. This bypasses starting-board
grid alignment and calibrates the pieces currently visible. FEN supplies the turn,
castling rights, en passant, and move counters that a screenshot cannot determine.
Earlier repetition history and learning targets are unavailable. A previously
unseen piece type (for example, a later promotion) may need manual handling;
calibrating at the starting position provides templates for every piece.

1. Open a **new game in the standard starting position** in your app or browser.
2. In the controller, choose the AI's side and which color is at the bottom.
3. Click **Select board area**, then drag precisely around the 64 squares.
   Small selection errors snap to the grid. Exclude player names and clocks.
4. Check the captured preview. Click **Start / resume**, then click the game
   window so it is in front. The controller stays on top; keep it clear of the
   selected board.
5. The AI reads visible pieces, waits for the opponent, and clicks its own moves.
   **F8** or **Stop now** stops it. Start/resume continues the tracked game.
6. For a new game, moved/resized board, different piece theme, or changed board
   orientation, select and calibrate the starting board again.

For Chess.com, use **Play vs Computer**. Its
[Fair Play guidance](https://support.chess.com/en/articles/8568369-what-do-i-need-to-know-about-fair-play-on-chess-com)
allows engine assistance in computer/bot games and prohibits it against other
users. The controller does not access your account, credentials, or browser DOM.

The visual reader learns piece templates from the selected starting board. It
checks legal continuations and waits for stable frames before clicking. It pauses
on ambiguous readings, unconfirmed clicks, or unexpected positions. It never
automatically re-clicks an unconfirmed move. If a click was missed, clear any
selected piece and press **Start / resume**: it checks whether the move already
happened, and allows one new attempt only if the board is still unchanged.
Promotions automatically select the model's chosen piece from a recognized menu.
CPU inference uses the selected saved model.

Small central move dots are ignored. After Undo, the player can return to an exact
earlier position in its tracked history and discards learning targets for the
undone moves. Use Stop before taking back moves, then Start / resume.

If the browser or board moves or resizes, stop the player and use **Reposition
board…** to select its new bounds. This keeps the calibrated pieces, move history,
and confirmed learning records. The selected board must match the tracked game.
The player moves the pointer outside the board before reading it, keeping cursor
highlights out of the captured squares when space is available in the game window.

### Learn while playing

**Learn from completed games** is enabled by default. The player records its MCTS
search decisions while playing. Once it recognizes checkmate or an automatic
draw, it saves the confirmed AI positions with the game's result and starts up
to 10 learning updates. These updates mix the new game with its existing replay
buffer, then save both `latest.pt` and `model.pt` in the same run used by the
dashboard. The screen player checks `latest.pt` before each AI move and loads any
saved update, including during a game. Its model label shows the loaded iteration.
Each move search uses one fixed set of weights. Selecting `model.pt` also follows
the run's `latest.pt`; standalone or historical checkpoints use the selected file.

For resignations, timeouts, agreed draws or a final board covered by a pop-up,
press **Stop**, choose **White won**, **Black won** or **Draw**, then click
**Finish & learn**. Only confirmed moves are included. Pausing or closing an
unfinished game does not assign a result or train from it.

Choose the run's `model.pt` (or `latest.pt`) with its full `latest.pt` checkpoint
beside it. Turn learning off when playing a standalone/historical model.
Completed game files remain on disk; checkpoint receipts prevent importing the
same file twice. Dashboard training imports pending screen games before its
learning phase. If another trainer is active, automatic screen learning waits
for the run lock, then loads the latest checkpoint. Progress appears on the
dashboard and in `screen-learning.log`; a launched learner continues if you
close the player. Games saved before a learner starts remain available to the
next dashboard training session.

To retry queued learning from Command Prompt:

```cmd
.venv\Scripts\python.exe -m chess_ai train --resume runs/main/latest.pt --external-only --iterations 1
```

Learning updates the weights; it does not guarantee stronger play after every game.

### Screen-player limits and controls

- Standard, flat 2D boards only. Textured/animated pieces, large overlays, check
  glows, or very small boards can require disabling visual effects.
  Turn off move animations when possible. It cannot read covered or minimized boards.
- Select the board at its current size. Windows DPI awareness and virtual-desktop
  coordinates support selection on multiple monitors, including negative offsets.
- **Vision tolerance** adjusts how much graphics variation is accepted. Start at
  0.18; a larger value can accept worse matches, so prefer a clear board and precise
  selection. **Click delay** gives the app time between source/destination clicks.
- **Promotions are automatic** for menus with four square-sized piece choices
  along the destination file, extending inward from the promotion square (as on
  Chess.com). All four choices must match the calibrated pieces; menu order and
  board orientation can vary. Unrecognized menus pause: select the requested
  piece manually, then resume. Turn off the game's automatic queen promotion
  to let the model choose a rook, bishop, or knight.
- If F8 is pressed between the two clicks, clear the selected piece in the game
  before resuming. The controller only sends input while the original target
  window is in front and the click points still belong to it.
- This is a screen adapter with conservative recognition, not a guarantee that
  every Windows app or chess theme is compatible. A live end-to-end mouse session
  requires selecting your actual board. Offline checks cover Chess.com's current
  piece images, highlights, orientations, coordinate mapping, legal tracking,
  rapid opponent replies, cancellation, and input bounds.

Install the screen dependency on another computer with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[screen]"
```

Screen capture uses [Pillow ImageGrab](https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html)
and input uses the Windows [SendInput API](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput).
Captures stay in this process and are not uploaded.

## Connect using UCI

For a desktop application accepting UCI engines, use `engine.cmd`. It loads
`runs/main/model.pt`. Applications requiring an executable can instead use:

- **Executable:** the full path to `.venv/Scripts/chess-ai.exe`
- **Arguments:** `uci --checkpoint "FULL PATH TO runs/main/model.pt"`

The installed executable works from any working directory. UCI supports
`uci`, `isready`, `ucinewgame`, `position startpos/fen ... moves ...`, `go nodes`,
`go movetime`, clocks/increments, `go infinite`, `stop`, and `quit`.
`Simulations` sets the default node budget; `ModelPath` loads a different model.
The engine continues reading commands while it searches. Time limits are checked
between neural evaluations; very short limits can overshoot by one evaluation.
`depth`/`mate` limits are ignored with an info message; pondering and `searchmoves`
are not implemented. Restart the engine or set `ModelPath` to load newer weights.

UCI provides an alternative to screen recognition when an app accepts engine
executables directly.

## Measure progress

```powershell
.\chess.ps1 evaluate --checkpoint runs/main/model.pt --opponent random --games 20 --output runs/main/vs-random.json
.\chess.ps1 evaluate --checkpoint runs/main/model.pt --opponent runs/main/initial.pt --games 20 --output runs/main/vs-initial.json
```

Evaluation alternates colors and repeats each random opening with colors swapped.
Use an even game count and the same seed/search budget for comparisons. Wins,
losses, draws, and unfinished games are reported separately. The completed-game
score excludes unfinished games and can be misleading when many games truncate.
These small matches do not establish an Elo rating. Training does not guarantee
monotonic improvement; compare saved snapshots over larger matches.

## Install on another computer

Python 3.10+ is required. On Windows with a compatible NVIDIA GPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CPU-only installation use the PyTorch CPU index instead:
`https://download.pytorch.org/whl/cpu`. On Linux, use `.venv/bin/python` and call
`python -m chess_ai ...` instead of the PowerShell launchers. See the
[official PyTorch installer](https://pytorch.org/get-started/locally/) for hardware
compatibility. This implementation uses only PyTorch, NumPy, python-chess and the
Python standard library.

## Verify

Initial local verification on 15 September 2026:

- Six automated checks passed; package dependency check passed.
- CUDA training completed on the RTX 4060: 8 games, 3,643 positions and 100
  optimizer updates in 114.65 seconds. Outcomes: one White win, two draws and
  five unfinished games. The network has 242,635 parameters.
- GPU resume restored the optimizer and replay and completed two further
  updates in the separate `runs/resume-check` verification folder.
- The PowerShell launcher, engine batch launcher, and installed executable
  were exercised; the executable also worked outside the project folder.
- A two-game smoke evaluation against random moves completed with two draws
  at eight simulations per move. Results and PGNs are in
  `runs/main/smoke-evaluation.json` and `runs/main/smoke-evaluation.pgn`.

The included `runs/main` checkpoint uses 8 games per iteration and 32 simulations
per move. `train.ps1` resumes those settings. This short run establishes working
training and persistence, not playing strength.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Checks cover move encoding (including promotions, castling and en passant),
repetition, terminal outcomes, search value signs and forced mate, actual network
updates, exclusion of unfinished value targets, exact CPU checkpoint continuation,
CLI resume, and real UCI subprocess interactions including stop/time controls.

## Approach and practical limits

MCTS uses the network to guide a search. Search visit counts become move targets;
finished game results become value targets. Training minimizes legal-move policy
cross-entropy plus outcome mean squared error. PUCT exploration, root Dirichlet
noise, and early-game sampling vary self-play. This follows the core approach in
the [AlphaZero paper](https://arxiv.org/abs/1712.01815), scaled down for a laptop.
It is my choice over basic DQN for this project because chess benefits from
explicit adversarial search and learning both move preferences and outcomes.

The compact residual network has 20 input planes and 73 move planes. Inputs
include both sides' pieces, castling rights, en passant, halfmove clock, and two
repetition flags. Search retains full board history for rule correctness; the
network does not see full history. Training and play enforce automatic draws;
optional threefold/50-move draw claims are not modeled as moves. The
[python-chess library](https://python-chess.readthedocs.io/en/stable/) handles rules.

This is a working learning system, not a claim of a strong trained engine.
From-scratch chess learning has sparse rewards and requires substantial compute.
The laptop implementation rebuilds searches each move and batches across games;
it has no distributed workers or DeepMind-scale training. A short verified run
proves that the pipeline works, not that the model has learned strong chess.
