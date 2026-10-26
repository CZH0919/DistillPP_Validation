"""Detached, restartable four-GPU H2 supervisor; stops after H2 report."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from hya_h2_common import (ROOT, RUN, GEN_PY, EVAL_PY, ResourceConflict, read, write_json,
                          check_frozen, gpu_snapshot, free_indices, host_available_gib)


def task_definitions(manifest, protocol):
    by_id = {p['id']: p for p in protocol['prompts']}
    tasks = []
    for b in manifest['blocks']:
        tasks.append({'id': 'gen_'+b['id'], 'kind': 'generation', 'block': b['id'], 'shard': b['shard']})
        metrics = list(dict.fromkeys(m for pid in b['prompt_ids'] for m in by_id[pid]['metrics']))
        tasks.extend({'id': 'eval_'+b['id']+'_'+m, 'kind': 'evaluation', 'block': b['id'], 'metric': m, 'shard': b['shard']} for m in metrics)
    return tasks


def task_complete(task, blocks):
    if task['kind'] == 'generation':
        return all((RUN/'pairs'/(pid+'.json')).exists() for pid in blocks[task['block']]['prompt_ids'])
    return (RUN/'scores'/(task['block']+'_'+task['metric']+'.json')).exists()


def dependent_ready(task, blocks):
    return task['kind'] == 'generation' or all((RUN/'pairs'/(pid+'.json')).exists() for pid in blocks[task['block']]['prompt_ids'])


def process_alive(pid, start_ticks):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')')+2:].split()
        return fields[0] != 'Z' and fields[19] == str(start_ticks)
    except FileNotFoundError:
        return False


def process_ticks(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


def start_command(task, gpu, attempt, log_path):
    python = GEN_PY if task['kind'] == 'generation' else EVAL_PY
    script = 'hya_h2_generate.py' if task['kind'] == 'generation' else 'hya_h2_evaluate.py'
    command = [python, '-B', str(ROOT/'tools'/script), '--block', task['block'], '--gpu', str(gpu), '--attempt', str(attempt)]
    if task['kind'] == 'evaluation': command += ['--metric', task['metric']]
    with log_path.open('x') as log:
        p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                             start_new_session=True, close_fds=True)
    return p, {'pid': p.pid, 'start_ticks': process_ticks(p.pid), 'gpu': gpu, 'attempt': attempt,
               'started_unix': time.time(), 'command': command, 'log_path': str(log_path)}


def worker_out(task, attempt):
    name = task['block'] if task['kind'] == 'generation' else task['block']+'_'+task['metric']
    return RUN/task['kind']/name/f'attempt{attempt:02d}'


def run():
    manifest, protocol = check_frozen()
    assert manifest['authorization']['H2'] is True and manifest['authorization']['S3'] is False
    blocks = {b['id']: b for b in manifest['blocks']}
    definitions = task_definitions(manifest, protocol)
    state_path = RUN/'controller.json'
    previous = read(state_path) if state_path.exists() else {}
    states = previous.get('tasks', {})
    for task in definitions:
        states.setdefault(task['id'], {'status': 'pending', 'attempts': [], 'conflicts': 0, 'not_before': 0})
    started = previous.get('started_unix', time.time())
    workers = {}
    for task in definitions:
        s = states[task['id']]
        if s['status'] == 'running':
            last = s['attempts'][-1]
            if process_alive(last['pid'], last['start_ticks']):
                workers[task['id']] = None  # adopt only our recorded PID and birth identity
            elif task_complete(task, blocks): s['status'] = 'passed'
            else: s['status'] = 'paused'; s['error'] = 'Worker disappeared; inspect recorded attempt before resume'
    def save(status='running', **extra):
        pairs = len(list((RUN/'pairs').glob('*.json')))
        score_values = sum(len(read(p)['rows']) for p in (RUN/'scores').glob('*.json'))
        write_json(state_path, {'status': status, 'controller_pid': os.getpid(), 'controller_start_ticks': process_ticks(os.getpid()),
                   'controller_session_id': os.getsid(0), 'started_unix': started, 'updated_unix': time.time(),
                   'completed_pairs': pairs, 'total_pairs': 64, 'committed_videos': pairs*2, 'total_videos': 128,
                   'completed_scores': score_values, 'total_scores': 224, 'tasks': states, **extra}, replace=True)
    try:
        while True:
            for task in definitions:
                tid = task['id']; s = states[tid]
                if tid not in workers: continue
                rec = s['attempts'][-1]; proc = workers[tid]
                alive = proc.poll() is None if proc is not None else process_alive(rec['pid'], rec['start_ticks'])
                timeout = 10800 if task['kind'] == 'generation' else 2400
                if alive and time.time()-rec['started_unix'] > timeout:
                    # Never signal a stale/reused PID or an unrecorded task.
                    assert process_alive(rec['pid'], rec['start_ticks'])
                    os.kill(rec['pid'], signal.SIGTERM)
                    rec['timeout_signal_unix'] = time.time()
                    s['error'] = 'Bounded worker timeout; only own worker signaled'
                    if proc is not None:
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            if process_alive(rec['pid'], rec['start_ticks']): os.kill(rec['pid'], signal.SIGKILL)
                            proc.wait(timeout=15)
                    alive = process_alive(rec['pid'], rec['start_ticks'])
                if alive: continue
                rc = proc.returncode if proc is not None else None
                rec.update(returncode=rc, finished_unix=time.time(), wall_seconds=time.time()-rec['started_unix'])
                out = worker_out(task, rec['attempt'])
                failure = read(out/'failure.json') if (out/'failure.json').exists() else {}
                complete = task_complete(task, blocks)
                if complete and (rc == 0 or (rc is None and (out/'result.json').exists())):
                    s['status'] = 'passed'; s['conflicts'] = 0
                elif complete and failure.get('status') == 'conflict':
                    # Fully committed valid data remains usable; conflict costs remain visible.
                    s['status'] = 'passed'; rec['postcommit_conflict'] = failure
                elif rc == 75 or failure.get('status') == 'conflict':
                    s['conflicts'] += 1; s['not_before'] = time.time()+300
                    s['status'] = 'pending' if s['conflicts'] < 2 else 'paused'
                    s['error'] = failure.get('error', 'resource conflict')
                else:
                    s['status'] = 'paused'; s['error'] = failure.get('error', 'Worker failed; no automatic algorithm/retry change')
                del workers[tid]
            # Outputs are the durable source of truth; never rerun successful pairs/scores.
            for task in definitions:
                s = states[task['id']]
                if s['status'] == 'pending' and task_complete(task, blocks): s['status'] = 'passed'
            if all(s['status'] == 'passed' for s in states.values()):
                save('finalizing')
                log_path = RUN/'finalize.log'
                with log_path.open('a') as log:
                    rc = subprocess.call([GEN_PY, '-B', str(ROOT/'tools/hya_h2_finalize.py')], cwd=ROOT,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
                if rc: save('finalization_failed', returncode=rc)
                else: save('completed', s3_started=False)
                return
            occupied = {states[tid]['attempts'][-1]['gpu'] for tid in workers}
            try:
                snap = gpu_snapshot(); free = [i for i in free_indices(snap) if i not in occupied]
            except (ResourceConflict, subprocess.TimeoutExpired) as exc:
                save('waiting_resource_query', query_error=repr(exc)); time.sleep(30); continue
            ready = [t for t in definitions if states[t['id']]['status'] == 'pending'
                     and time.time() >= states[t['id']]['not_before'] and dependent_ready(t, blocks)]
            # Drain ready evaluations promptly, while other cards continue generation.
            ready.sort(key=lambda t: (0 if t['kind'] == 'evaluation' else 1, blocks[t['block']]['ordinal'], t['shard']))
            for task in ready:
                if len(workers) >= manifest['maximum_workers'] or not free: break
                # Conservative budget for staged CPU reconstruction; actual free cards only.
                if host_available_gib() < 256+160*(len(workers)+1): break
                preferred = manifest['initial_gpu_indices'][task['shard'] % len(manifest['initial_gpu_indices'])]
                gpu = preferred if preferred in free else free[0]; free.remove(gpu)
                s = states[task['id']]; attempt = len(s['attempts'])+1
                log_path = RUN/'logs'/(task['id']+f'_attempt{attempt:02d}.log')
                proc, rec = start_command(task, gpu, attempt, log_path)
                s['attempts'].append(rec); s['status'] = 'running'; workers[task['id']] = proc
                print('H2_DISPATCH', task['id'], 'gpu', gpu, 'pid', proc.pid, flush=True)
                save()
            if not workers and not ready and any(s['status'] == 'paused' for s in states.values()):
                # If pending retries only wait on cooldown, do not call the run terminal yet.
                retry_pending = any(s['status'] == 'pending' and dependent_ready(t, blocks)
                                    for t in definitions for s in [states[t['id']]])
                if not retry_pending:
                    save('needs_attention'); return
            save('running' if workers else 'waiting_for_idle_resources', gpu_snapshot=snap)
            time.sleep(15)
    except BaseException as exc:
        # Child workers are detached. Preserve their identities so a controller restart can adopt them safely.
        save('controller_failed', error=repr(exc), traceback=traceback.format_exc())
        raise


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()


if __name__ == '__main__':
    main()
