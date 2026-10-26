"""One-pair recovery dispatch, up to six admitted GPUs; frozen numerical code unchanged."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import hya_h2_common as c
import hya_h2_controller as ctl

ROOT, RUN = c.ROOT, c.RUN
RECOVERY = RUN/'resume_20261019_sixgpu'
AGGREGATES = {'gen_b014', 'gen_b028', 'gen_b030'}
EVALUATIONS = {'eval_b014_multiple_objects', 'eval_b028_background_consistency',
               'eval_b028_scene', 'eval_b030_multiple_objects'}
PREFERENCE = [3, 4, 5, 6, 7, 1]
ORIGINAL_DEFINITIONS = ctl.task_definitions


def virtual_blocks(original):
    result = []
    for b in original['blocks']:
        if 'gen_'+b['id'] not in AGGREGATES: continue
        for j, pid in enumerate(b['prompt_ids']):
            shard = len(result)
            # Preserve the original block's T/S order parity while assigning each pair a separate slot.
            ordinal = b['ordinal']+(b['shard']-shard) % 2
            result.append({'id': 'r_'+b['id']+'_'+str(j), 'shard': shard, 'ordinal': ordinal,
                           'prompt_ids': [pid], 'parent_block': b['id']})
    return result


def build_runtime_manifest(policy):
    original, protocol = c.check_frozen()
    assert c.digest(RUN/'manifest.json') == policy['original_manifest_sha256']
    assert c.digest(Path(__file__)) == policy['script_sha256']
    result = copy.deepcopy(original)
    result['blocks'] += policy['virtual_blocks']
    result['maximum_workers'] = 6
    result['initial_gpu_indices'] = PREFERENCE
    return result, protocol


def dispatch_definitions(manifest, protocol, policy):
    base = copy.deepcopy(manifest)
    base['blocks'] = [b for b in base['blocks'] if not b['id'].startswith('r_')]
    tasks = ORIGINAL_DEFINITIONS(base, protocol)
    tasks += [{'id': 'gen_'+b['id'], 'kind': 'generation', 'block': b['id'], 'shard': b['shard']}
              for b in policy['virtual_blocks']]
    return tasks


def self_test():
    original, protocol = c.check_frozen(); state = c.read(RUN/'controller.json')
    assert state['status'] == 'six_gpu_reshard_preparing'
    assert not ctl.process_alive(state['controller_pid'], state['controller_start_ticks'])
    blocks = virtual_blocks(original)
    ids = [b['prompt_ids'][0] for b in blocks]
    assert len(ids) == len(set(ids)) == 6
    assert all(not (RUN/'pairs'/(pid+'.json')).exists() for pid in ids)
    original_by_id = {b['id']: b for b in original['blocks']}
    for b in blocks:
        parent = original_by_id[b['parent_block']]
        assert (b['ordinal']+b['shard']) % 2 == (parent['ordinal']+parent['shard']) % 2
    policy = {'virtual_blocks': blocks}
    definitions = dispatch_definitions(original, protocol, policy)
    assert len(definitions) == len({t['id'] for t in definitions}) == 94
    assert sum(t['kind'] == 'evaluation' for t in definitions) == 56
    assert sum(t['status'] == 'passed' for t in state['tasks'].values()) == 81
    assert all(state['tasks'][tid]['status'] == 'pending' for tid in AGGREGATES | EVALUATIONS)
    assert not set(PREFERENCE) & {0, 2} and len(PREFERENCE) == 6
    print('H2_PARALLEL_CPU_TEST_PASSED', {'pairs': ids, 'unchanged_passed_tasks': 81,
          'maximum_concurrency': 6, 'excluded_gpus': [0, 2]}, flush=True)


def launch():
    self_test(); assert not (RECOVERY/'policy.json').exists()
    # Reuse the complete hash inventory made before the first recovery; never overwrite accepted data.
    protected = c.read(RUN/'resume_20261019/preserved_files.json')
    for rel, record in protected.items():
        p = ROOT/rel
        assert p.stat().st_size == record['bytes'] and c.digest(p) == record['sha256'], rel
    original, protocol = c.check_frozen(); state = c.read(RUN/'controller.json')
    previous_hash = c.digest(RUN/'controller.json')
    snap = c.gpu_snapshot(); free = [i for i in PREFERENCE if i in c.free_indices(snap)]
    assert len(free) >= 3 and c.host_available_gib() > 736, 'Not enough admitted resources; no launch'
    assert not (ROOT/'Result/HYA_H2_20261019/COMPLETE.json').exists()
    blocks = virtual_blocks(original)
    policy = {'recovery_id': 'hya_h2_resume_parallel_20261019', 'created_unix': time.time(),
              'authorization': 'User explicitly requested more idle GPUs to finish the remaining six pairs sooner',
              'allowed_gpu_preference': PREFERENCE, 'excluded_gpu_indices': [0, 2],
              'maximum_workers': 6, 'virtual_blocks': blocks, 'already_passed_tasks': 81,
              'already_committed_pairs': 58, 'already_committed_scores': 208,
              'initial_free_devices': free, 'initial_snapshot': snap,
              'original_manifest_sha256': c.digest(RUN/'manifest.json'),
              'script_sha256': c.digest(Path(__file__)),
              'preserved_files_manifest_sha256': c.digest(RUN/'resume_20261019/preserved_files.json'),
              'S3_authorized': False}
    c.write_json(RECOVERY/'policy.json', policy)
    c.write_json(RECOVERY/'controller_after_stop.json', state)
    for b in blocks:
        state['tasks']['gen_'+b['id']] = {'status': 'pending', 'attempts': [], 'conflicts': 0, 'not_before': 0}
    state['status'] = 'parallel_recovery_authorized'; state['updated_unix'] = time.time()
    assert c.digest(RUN/'controller.json') == previous_hash
    c.write_json(RUN/'controller.json', state, replace=True)
    with (RUN/'controller.log').open('a') as log:
        proc = subprocess.Popen([c.GEN_PY, '-B', str(Path(__file__)), '--controller'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    time.sleep(4); assert proc.poll() is None
    receipt = {'controller_pid': proc.pid, 'start_ticks': ctl.process_ticks(proc.pid),
               'session_id': os.getsid(proc.pid), 'launched_unix': time.time(),
               'initial_free_devices': free, 'maximum_workers': 6, 'preserved_files_verified': len(protected)}
    assert receipt['session_id'] == proc.pid
    c.write_json(RECOVERY/'launch_receipt.json', receipt)
    print('H2_PARALLEL_RECOVERY_STARTED', receipt, flush=True)


def run_controller():
    policy = c.read(RECOVERY/'policy.json')
    build_runtime_manifest(policy)
    assert policy['S3_authorized'] is False and policy['excluded_gpu_indices'] == [0, 2]
    ctl.check_frozen = lambda: build_runtime_manifest(policy)
    ctl.task_definitions = lambda manifest, protocol: dispatch_definitions(manifest, protocol, policy)
    original_ready, original_start = ctl.dependent_ready, ctl.start_command
    virtual_ids = {'gen_'+b['id'] for b in policy['virtual_blocks']}
    ctl.dependent_ready = lambda task, blocks: False if task['id'] in AGGREGATES else original_ready(task, blocks)
    ctl.free_indices = lambda snapshot: [i for i in PREFERENCE if i in c.free_indices(snapshot)]
    def start(task, gpu, attempt, log_path):
        assert gpu in PREFERENCE and task['id'] in virtual_ids | EVALUATIONS
        if task['id'] not in virtual_ids: return original_start(task, gpu, attempt, log_path)
        command = [c.GEN_PY, '-B', str(Path(__file__)), '--worker', '--block', task['block'],
                   '--gpu', str(gpu), '--attempt', str(attempt)]
        with log_path.open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                         OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        return proc, {'pid': proc.pid, 'start_ticks': ctl.process_ticks(proc.pid), 'gpu': gpu, 'attempt': attempt,
                       'started_unix': time.time(), 'command': command, 'log_path': str(log_path)}
    ctl.start_command = start
    print('H2_PARALLEL_POLICY', {'preference': PREFERENCE, 'maximum_workers': 6, 'one_pair_per_worker': True}, flush=True)
    ctl.main()


def worker(args):
    policy = c.read(RECOVERY/'policy.json')
    assert args.gpu in PREFERENCE and args.block in {b['id'] for b in policy['virtual_blocks']}
    import hya_h2_generate as generation
    generation.check_frozen = lambda: build_runtime_manifest(policy)
    sys.argv = ['hya_h2_generate.py', '--block', args.block, '--gpu', str(args.gpu), '--attempt', str(args.attempt)]
    generation.main()


def status(as_json):
    policy = c.read(RECOVERY/'policy.json')
    ctl.task_definitions = lambda manifest, protocol: dispatch_definitions(manifest, protocol, policy)
    import hya_h2_status
    sys.argv = ['hya_h2_status.py']+(['--json'] if as_json else [])
    hya_h2_status.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ['self-test', 'launch', 'controller', 'worker', 'status']: modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--block'); parser.add_argument('--gpu', type=int); parser.add_argument('--attempt', type=int)
    parser.add_argument('--json', action='store_true'); args = parser.parse_args()
    if args.self_test: self_test()
    elif args.launch: launch()
    elif args.controller: run_controller()
    elif args.worker: worker(args)
    else: status(args.json)
