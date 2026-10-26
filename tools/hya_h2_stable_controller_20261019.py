"""Live controller handoff: keep current workers; dispatch new work only to GPU 3/6."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import hya_h2_common as c
import hya_h2_controller as ctl
import hya_h2_resume_parallel_20261019 as parallel

FOLDER = c.RUN/'resume_stable_20261019'
STABLE = [3, 6]


def launch():
    c.check_frozen(); assert not FOLDER.exists()
    s = c.read(c.RUN/'controller.json')
    old_pid, old_ticks = s['controller_pid'], s['controller_start_ticks']
    assert ctl.process_alive(old_pid, old_ticks)
    active = {tid: t['attempts'][-1] for tid, t in s['tasks'].items() if t['status'] == 'running'}
    assert any(a['gpu'] in STABLE and ctl.process_alive(a['pid'], a['start_ticks']) for a in active.values())
    FOLDER.mkdir(); c.write_json(FOLDER/'controller_before_handoff.json', s)
    # Stop only the supervisor. Detached numerical workers remain running and are adopted by birth identity.
    assert ctl.process_alive(old_pid, old_ticks); os.kill(old_pid, signal.SIGTERM)
    deadline = time.monotonic()+20
    while ctl.process_alive(old_pid, old_ticks) and time.monotonic() < deadline: time.sleep(.25)
    assert not ctl.process_alive(old_pid, old_ticks)
    s = c.read(c.RUN/'controller.json')
    kept = []
    for tid, task in s['tasks'].items():
        if task['status'] == 'running':
            a = task['attempts'][-1]
            kept.append({'task': tid, 'pid': a['pid'], 'start_ticks': a['start_ticks'], 'gpu': a['gpu'],
                         'alive_at_handoff': ctl.process_alive(a['pid'], a['start_ticks'])})
        if tid.startswith('gen_r_') and task['status'] in ['pending', 'paused', 'running']:
            task.setdefault('recovery_history', []).append({'time_unix': time.time(),
                'previous_status': task['status'], 'previous_conflicts': task['conflicts'], 'previous_error': task.get('error'),
                'reason': 'User accepted lower concurrency and requested non-conflicting devices'})
            if task['status'] == 'paused': task['status'] = 'pending'
            task['conflicts'] = 0
    s['status'] = 'stable_handoff_authorized'; s['updated_unix'] = time.time()
    c.write_json(c.RUN/'controller.json', s, replace=True)
    policy = {'authorized_unix': time.time(), 'new_dispatch_gpu_indices': STABLE,
              'authorization': 'User requests completion on non-conflicting GPUs and accepts lower concurrency',
              'keep_running_workers': kept, 'do_not_stop_numerical_workers': True,
              'script_sha256': c.digest(Path(__file__)), 'original_manifest_sha256': c.digest(c.RUN/'manifest.json'),
              'S3_authorized': False, 'existing_inflight_on_other_cards': 'allow safe completion; never dispatch replacements there'}
    c.write_json(FOLDER/'policy.json', policy)
    with (c.RUN/'controller.log').open('a') as log:
        proc = subprocess.Popen([c.GEN_PY, '-B', str(Path(__file__)), '--controller'], cwd=c.ROOT,
             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
             env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    time.sleep(4); assert proc.poll() is None
    receipt = {'controller_pid': proc.pid, 'start_ticks': ctl.process_ticks(proc.pid),
               'session_id': os.getsid(proc.pid), 'new_dispatch_gpu_indices': STABLE, 'preserved_worker_identities': kept,
               'launched_unix': time.time()}
    assert receipt['session_id'] == proc.pid
    c.write_json(FOLDER/'launch_receipt.json', receipt)
    print('H2_STABLE_HANDOFF_STARTED', json.dumps(receipt), flush=True)


def run_controller():
    policy = c.read(FOLDER/'policy.json')
    assert c.digest(Path(__file__)) == policy['script_sha256']
    assert c.digest(c.RUN/'manifest.json') == policy['original_manifest_sha256']
    assert policy['new_dispatch_gpu_indices'] == STABLE and policy['S3_authorized'] is False
    # Parallel dispatch reads this module variable. Existing worker code and scientific protocol are unchanged.
    parallel.PREFERENCE = STABLE
    print('H2_STABLE_DISPATCH_POLICY', {'new_tasks_only_on': STABLE, 'existing_workers_adopted': True}, flush=True)
    parallel.run_controller()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch', action='store_true'); group.add_argument('--controller', action='store_true')
    args = parser.parse_args()
    if args.launch: launch()
    else: run_controller()
