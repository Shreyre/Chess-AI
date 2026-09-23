# Larger Stockfish corpus

`--teacher-data` accepts either an existing `.pt` dataset or a directory of
datasets. A directory loads one sorted shard per learning iteration, using the
saved iteration modulo the current file count. New atomic `.pt` files become
available on subsequent iterations. Temporary files are ignored. The original
file-based behavior is unchanged.

This keeps memory bounded by one shard while retaining a much larger collection
on disk. The dashboard's teacher-position count is the current shard size, not
the total corpus size. Automatic refresh adds another shard to the directory;
it does not replace the corpus with a single dataset. Resume preserves the
directory and the existing teacher/self-play mix.

The local expansion job is `runs/expand-stockfish.py`, targeting 200,000 training
positions in batches of at most 5,000, at 20,000 Stockfish nodes per position.
It reuses distinct archived labels and generates further labels from self-play,
with a stable separate held-out partition. Engine strength limiting is disabled.
This count and search budget are not an Elo measurement.

Progress is in `runs/main/stockfish-expansion.json` and
`runs/main/stockfish-expansion.log`. Completed batches are under
`runs/main/stockfish-corpus/`. Creating `runs/main/.stop-stockfish-expansion`
stops generation after preserving its partial batch; delete that marker before
restarting the job. Restarting scans completed batches to avoid repeating saved
positions. Do not run two expansion jobs at the same time.

For an assessment, pass an individual shard containing held-out positions.
One shard's agreement is only a diagnostic for that subset.
