"""Reject training updates that do not improve a fixed neural puzzle score."""
from copy import deepcopy
import hashlib
from pathlib import Path

from .puzzles import assess_puzzles, load_puzzles


class PuzzleGate:
    def __init__(self, config, live):
        self.config, self.live = config, live
        self.path = Path(config['puzzle_data'])
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.rows = load_puzzles(self.path)
        self.check_file()
        previous = config.get('puzzle_gate')
        if previous and (previous.get('puzzle_sha256') != self.digest or
                         previous.get('version') != 1 or previous.get('simulations') != 64 or
                         previous.get('puzzles') != len(self.rows) or
                         type(previous.get('accepted_solved')) is not int or
                         not 0 <= previous['accepted_solved'] <= len(self.rows)):
            raise ValueError('Puzzle gate benchmark/settings changed; use a new run for a new baseline')

    def check_file(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.digest:
            raise ValueError('Puzzle gate file changed during training')

    def score(self, model):
        self.check_file()
        self.live.update(force=True, phase='evaluating', maintenance='Checking neural puzzle accuracy',
                         maintenance_progress=0)
        # Fixed CPU search keeps the score comparable across training devices.
        evaluated = deepcopy(model).cpu().eval()
        try:
            report = assess_puzzles(evaluated, self.rows, simulations=64,
                progress=lambda done, total: self.live.update(maintenance_progress=f'{done}/{total}'))
            self.check_file()
            return report['solved']
        finally:
            del evaluated

    def initialize(self, model, iteration):
        if not self.config.get('puzzle_gate'):
            solved = self.score(model)
            self.config['puzzle_gate'] = dict(version=1, puzzle_sha256=self.digest,
                puzzles=len(self.rows), simulations=64, device='cpu',
                accepted_solved=solved, accepted_rate=solved / len(self.rows),
                accepted_iteration=iteration, candidate_solved=solved,
                candidate_iteration=iteration, status='baseline')
        self.publish()

    def publish(self):
        self.live.update(force=True, puzzle_gate=dict(self.config['puzzle_gate']),
                         maintenance=None, maintenance_progress=0)

    def check(self, model, optimizer, before, iteration):
        previous = self.config['puzzle_gate']
        result = dict(previous, candidate_iteration=iteration, candidate_solved=None,
                      status='rejected', error=None)
        try:
            result['candidate_solved'] = self.score(model)
            if result['candidate_solved'] > previous['accepted_solved']:
                result.update(status='accepted', accepted_solved=result['candidate_solved'],
                    accepted_rate=result['candidate_solved'] / len(self.rows),
                    accepted_iteration=iteration)
        except (OSError, ValueError, RuntimeError) as error:
            result.update(status='error', error=str(error))
        if result['status'] != 'accepted':
            model.load_state_dict(before[0])
            optimizer.load_state_dict(before[1])
        model.eval()
        self.config['puzzle_gate'] = result
        self.publish()
        return dict(result)
