"""S2-only safety, provenance and non-invasive GPU occupancy monitoring."""
import json
import os
from pathlib import Path
import threading
import time

import s1_common as accepted
from s1_common import (PROJECT, SHAPE, METRICS, NETWORK_ATTEMPTS, json_write,
                       file_hash, tensor_hash, tensor_stats, gpu_snapshot,
                       ensure_no_other_process)

BATCH = PROJECT / 'runs/s2_20261016'
WARMUP_PROMPT = accepted.PROMPT
WARMUP_SEED = accepted.SEED


def configure():
    # Reuse S1's tested offline/path guards, but place all locks in this batch.
    accepted.BATCH = BATCH
    accepted.configure()


def gpu_admission(index, min_free_gib=120):
    return accepted.gpu_admission(index, min_free_gib)


def load_frozen():
    execution = json.loads((BATCH / 'execution.json').read_text())
    for rel, expected in execution['input_sha256'].items():
        assert file_hash(PROJECT / rel) == expected, 'Frozen input changed: ' + rel
    return execution, json.loads((BATCH / 'baseline.json').read_text())


class OccupancyMonitor:
    """Fail closed if another PID appears; never signal any external PID.

    nvidia-smi and progress writes run in a CPU thread, not in the timed loop.
    Polling is cooperative, not a GPU reservation or an atomic exclusivity claim.
    """
    def __init__(self, admission, output, state):
        self.uuid = admission['uuid']
        self.output, self.state = output, state
        self.stop = threading.Event()
        self.error = None
        self.observations = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                snap = ensure_no_other_process(self.uuid)
                selected = next(g for g in snap['gpus'] if g[1] == self.uuid)
                self.observations.append({'unix_time': time.time(), 'gpu': selected,
                    'processes': [p for p in snap['processes'] if p[0] == self.uuid]})
                json_write(self.output / 'progress.json', dict(self.state,
                    unix_time=time.time(), pid=os.getpid(), gpu=selected))
            except BaseException as exc:
                self.error = repr(exc)
                return
            self.stop.wait(10)

    def __enter__(self):
        ensure_no_other_process(self.uuid)
        self.thread.start()
        return self

    def check(self):
        if self.error is not None:
            raise RuntimeError('Occupancy/progress monitor failed; own worker stops: ' + self.error)

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=35)
        json_write(self.output / 'occupancy.json', {'observations': self.observations,
                                                  'error': self.error})


def complete_generations(execution, verify_video=True):
    result = []
    for pair in execution['pairs']:
        candidates = sorted((BATCH / 'pairs' / pair['pair_id']).glob('attempt*/pair_complete.json'))
        assert len(candidates) == 1, 'Require exactly one accepted pair: ' + pair['pair_id']
        complete = json.loads(candidates[0].read_text())
        assert complete['status'] == 'passed'
        pair_rows = []
        for arm in ['T50', 'S4']:
            path = candidates[0].parent / arm / 'result.json'
            row = json.loads(path.read_text())
            assert row['status'] == 'passed' and row['pair_id'] == pair['pair_id']
            assert row['prompt'] == pair['prompt'] and row['seed'] == pair['seed']
            assert row['arm'] == arm and row['warmup'] is False
            assert row['forward_count'] == (100 if arm == 'T50' else 4)
            video = path.parent / 'video.mp4'
            if verify_video:
                assert file_hash(video) == row['video']['sha256']
            row.update(video_path=str(video), result_path=str(path))
            pair_rows.append(row)
        for field in ['initial_latent_sha256', 'positive_context_sha256', 'gpu_uuid', 'worker_id']:
            assert pair_rows[0][field] == pair_rows[1][field], (pair['pair_id'], field)
        result.extend(pair_rows)
    assert len(result) == 96
    return result
