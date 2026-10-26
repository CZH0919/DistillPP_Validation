"""Explicitly authorized H2 recovery. Only scheduling/state changes; frozen workers stay intact."""
import argparse
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import hya_h2_common as common
import hya_h2_controller as controller

ROOT, RUN = common.ROOT, common.RUN
RECOVERY = RUN/'resume_20261019'
GENERATIONS = {'gen_b014', 'gen_b028', 'gen_b030'}
EVALUATIONS = {'eval_b014_multiple_objects', 'eval_b028_background_consistency',
               'eval_b028_scene', 'eval_b030_multiple_objects'}
GPU_PREFERENCE = [3, 5, 6, 7, 1, 4]


def candidates(snapshot, allowed=GPU_PREFERENCE):
    free = common.free_indices(snapshot)
    return [i for i in allowed if i in free and i not in (0, 2)]


def recovery_state(previous, stamp):
    assert previous['status'] == 'needs_attention'
    assert {tid for tid, t in previous['tasks'].items() if t['status'] == 'paused'} == GENERATIONS
    assert {tid for tid, t in previous['tasks'].items() if t['status'] == 'pending'} == EVALUATIONS
    out = copy.deepcopy(previous)
    for tid in GENERATIONS:
        task = out['tasks'][tid]
        assert len(task['attempts']) == 2 and all(a['returncode'] == 75 for a in task['attempts'])
        task.setdefault('recovery_history', []).append({'authorized_unix': stamp,
            'previous_status': task['status'], 'previous_conflicts': task['conflicts'],
            'previous_error': task.get('error'), 'reason': 'User explicitly authorized resuming missing H2 work while avoiding GPU 0 and 2'})
        task['status'] = 'pending'; task['conflicts'] = 0; task['not_before'] = 0
        task.pop('error', None)
        assert task['attempts'] == previous['tasks'][tid]['attempts']
    for tid in set(previous['tasks'])-GENERATIONS:
        assert out['tasks'][tid] == previous['tasks'][tid]
    out['status'] = 'resume_authorized'
    out['recovery_policy'] = str(RECOVERY/'policy.json')
    return out


def preserved_inventory():
    paths = set(); pairs = sorted((RUN/'pairs').glob('*.json'))
    scores = sorted((RUN/'scores').glob('*.json'))
    assert len(pairs) == 58 and sum(len(common.read(p)['rows']) for p in scores) == 208
    for path in pairs:
        record = common.read(path); assert record['status'] == 'passed'; paths.add(path)
        for arm in ['T50', 'S5']:
            rec = record['records'][arm]
            for field in ['video_path', 'archive_path']:
                video = Path(rec[field]); assert video.is_relative_to(ROOT) and video.resolve() == video
                assert common.digest(video) == rec['video']['sha256'], 'Existing completed video failed SHA check'
                paths.add(video)
    paths.update(scores)
    for kind in ['generation', 'evaluation']:
        paths.update((RUN/kind).glob('*/attempt*/failure.json'))
    return {str(p.relative_to(ROOT)): {'sha256': common.digest(p), 'bytes': p.stat().st_size} for p in sorted(paths)}


def self_test():
    manifest, protocol = common.check_frozen()
    previous = common.read(RUN/'controller.json')
    assert not controller.process_alive(previous['controller_pid'], previous['controller_start_ticks'])
    updated = recovery_state(previous, 0)
    assert sum(t['status'] == 'passed' for t in updated['tasks'].values()) == 81
    assert {k for k, v in updated['tasks'].items() if v['status'] == 'pending'} == GENERATIONS | EVALUATIONS
    fake = {'gpus': [[str(i), 'g'+str(i), '0', '143771', '0'] for i in range(8)],
            'processes': [['g5', '999999', '0']]}
    assert candidates(fake) == [3, 6, 7, 1, 4]
    definitions = controller.task_definitions(manifest, protocol)
    blocks = {b['id']: b for b in manifest['blocks']}
    ready = [t for t in definitions if t['id'] in GENERATIONS]
    ready.sort(key=lambda t: (blocks[t['block']]['ordinal'], t['shard']))
    free = [3, 5, 6]; initial = free.copy(); assigned = {}
    for task in ready:
        preferred = initial[task['shard'] % len(initial)]
        gpu = preferred if preferred in free else free[0]
        free.remove(gpu); assigned[task['id']] = gpu
    assert assigned == {'gen_b028': 3, 'gen_b014': 5, 'gen_b030': 6}
    print(json.dumps({'status': 'CPU_recovery_test_passed', 'unchanged_passed_tasks': 81,
                      'generation_attempts_next': 3, 'assignment_if_idle': assigned,
                      'excluded_devices': [0, 2], 'frozen_model_and_metric_code_unchanged': True}), flush=True)


def launch():
    self_test()
    assert not RECOVERY.exists(), 'Recovery already prepared; inspect its receipt instead of duplicating it'
    state_path = RUN/'controller.json'
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_bytes = state_path.read_bytes(); previous = json.loads(previous_bytes)
        for task in previous['tasks'].values():
            for attempt in task['attempts']:
                assert not controller.process_alive(attempt['pid'], attempt['start_ticks']), 'Recorded worker still alive'
        snapshots = []
        for _ in range(2):
            snapshots.append(common.gpu_snapshot()); assert len(candidates(snapshots[-1])) >= 3
            time.sleep(2)
        assert common.host_available_gib() > 736
        protected = preserved_inventory()
        final_snapshot = common.gpu_snapshot(); selected = candidates(final_snapshot)[:3]
        assert len(selected) == 3 and common.shutil.disk_usage(ROOT).free > 50*2**30
        assert state_path.read_bytes() == previous_bytes, 'Controller state changed during verification'
        RECOVERY.mkdir()
        with (RECOVERY/'controller_before.json').open('xb') as f:
            f.write(previous_bytes); f.flush(); os.fsync(f.fileno())
        common.write_json(RECOVERY/'preserved_files.json', protected)
        policy = {'recovery_id': 'hya_h2_resume_20261019', 'authorized_unix': time.time(),
                  'authorization': 'User requested resume of remaining six pairs and sixteen scores, avoiding GPU 0/2, detached background execution',
                  'initial_gpu_indices': selected, 'allowed_gpu_preference': GPU_PREFERENCE, 'excluded_gpu_indices': [0, 2],
                  'maximum_workers': 3, 'allowed_tasks': sorted(GENERATIONS | EVALUATIONS),
                  'remaining_pairs': 6, 'remaining_videos': 12, 'remaining_scores': 16,
                  'already_committed_pairs': 58, 'already_committed_scores': 208,
                  'snapshot_before_launch': final_snapshot, 'admission_snapshots': snapshots,
                  'original_manifest_sha256': common.digest(RUN/'manifest.json'),
                  'original_state_sha256': common.digest(RECOVERY/'controller_before.json'),
                  'preserved_files_sha256': common.digest(RECOVERY/'preserved_files.json'),
                  'resume_script_sha256': common.digest(Path(__file__)),
                  'new_consecutive_conflict_limit': 2, 'preserve_all_attempt_history': True, 'S3_authorized': False}
        common.write_json(RECOVERY/'policy.json', policy)
        common.write_json(state_path, recovery_state(previous, policy['authorized_unix']), replace=True)
    # Release the project controller lock before the detached controller acquires it.
    with (RUN/'controller.log').open('a') as log:
        proc = subprocess.Popen([common.GEN_PY, '-B', str(Path(__file__)), '--controller'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    time.sleep(4)
    assert proc.poll() is None, 'Recovery controller exited: inspect controller.log and preserved state'
    receipt = {'controller_pid': proc.pid, 'start_ticks': controller.process_ticks(proc.pid),
               'session_id': os.getsid(proc.pid), 'launched_unix': time.time(), 'initial_gpu_indices': selected,
               'preserved_file_count': len(protected), 'previous_completed_pairs': 58, 'previous_completed_scores': 208}
    assert receipt['session_id'] == proc.pid
    common.write_json(RECOVERY/'launch_receipt.json', receipt)
    print('H2_RECOVERY_STARTED', json.dumps(receipt), flush=True)


def run_controller():
    policy = common.read(RECOVERY/'policy.json')
    assert common.digest(Path(__file__)) == policy['resume_script_sha256']
    assert common.digest(RUN/'manifest.json') == policy['original_manifest_sha256']
    assert common.digest(RECOVERY/'preserved_files.json') == policy['preserved_files_sha256']
    assert policy['excluded_gpu_indices'] == [0, 2] and policy['S3_authorized'] is False
    # Keep frozen generation, evaluation, statistics and original manifest bytes unchanged.
    # Only the controller's in-memory dispatch limits/order are overridden by this recorded recovery policy.
    original_check = controller.check_frozen
    original_start = controller.start_command
    def runtime_manifest():
        original, protocol = original_check()
        updated = copy.deepcopy(original)
        updated['initial_gpu_indices'] = policy['initial_gpu_indices']
        updated['maximum_workers'] = policy['maximum_workers']
        return updated, protocol
    def guarded_start(task, gpu, attempt, log):
        assert task['id'] in policy['allowed_tasks'] and gpu in policy['allowed_gpu_preference'] and gpu not in (0, 2)
        return original_start(task, gpu, attempt, log)
    controller.check_frozen = runtime_manifest
    controller.free_indices = lambda snapshot: candidates(snapshot, policy['allowed_gpu_preference'])
    controller.start_command = guarded_start
    print('H2_RECOVERY_POLICY', json.dumps({'gpu_preference': policy['allowed_gpu_preference'],
          'excluded': [0, 2], 'workers': 3, 'tasks': policy['allowed_tasks']}), flush=True)
    controller.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test', action='store_true'); modes.add_argument('--launch', action='store_true')
    modes.add_argument('--controller', action='store_true'); args = parser.parse_args()
    if args.self_test: self_test()
    elif args.launch: launch()
    else: run_controller()
