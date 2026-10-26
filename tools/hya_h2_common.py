"""H2 bookkeeping and admission. No CUDA initialization on import."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from hya_common import ROOT, digest, write_json

RUN = ROOT / 'runs/hya_h2_20261019'
ARCHIVE = ROOT / 'Evaluation_Videos/hya_h2_20261019'
RESULT = ROOT / 'Result/HYA_H2_20261019'
GEN_PY = '/data/chenjiayu/miniconda3/envs/changzihe/bin/python'
EVAL_PY = '/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python'
ARMS = ['T50', 'S5']


class ResourceConflict(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def check_frozen():
    manifest = read(RUN / 'manifest.json')
    for rel, expected in manifest['frozen_files'].items():
        assert digest(ROOT / rel) == expected, 'Frozen file changed: ' + rel
    verification = RUN/'resource_verification.json'
    assert digest(verification) == manifest['resource_verification_sha256']
    for asset in read(verification)['assets']:
        stat = Path(asset['path']).stat()
        assert stat.st_size == asset['bytes'] and stat.st_mtime_ns == asset['mtime_ns'], 'Resource changed since full hash verification: '+asset['path']
    return manifest, read(RUN / 'protocol.json')


def make_blocks(prompts, shards=4, block_size=2):
    """Deterministic library-balanced shards; every prompt occurs exactly once."""
    buckets = [[] for _ in range(shards)]
    offset = 0
    for library in dict.fromkeys(p['library'] for p in prompts):
        group = [p for p in prompts if p['library'] == library]
        for i, p in enumerate(group):
            buckets[(offset+i) % shards].append(p['id'])
        offset = (offset+len(group)) % shards
    blocks = []
    for shard, ids in enumerate(buckets):
        for i in range(0, len(ids), block_size):
            blocks.append({'id': f'b{len(blocks):03d}', 'shard': shard,
                           'ordinal': i//block_size, 'prompt_ids': ids[i:i+block_size]})
    return blocks


def gpu_snapshot(index=None):
    command = ['nvidia-smi'] + ([] if index is None else ['-i', str(index)])
    def query(kind, fields):
        r = subprocess.run(command + ['--query-'+kind+'='+fields, '--format=csv,noheader,nounits'],
                           text=True, capture_output=True, timeout=15)
        if r.returncode:
            raise ResourceConflict('GPU query failed: '+r.stderr)
        return [[v.strip() for v in row.split(',')] for row in r.stdout.splitlines() if row.strip()]
    gpus = query('gpu', 'index,uuid,memory.used,memory.total,utilization.gpu')
    processes = query('compute-apps', 'gpu_uuid,pid,used_gpu_memory')
    return {'gpus': gpus, 'processes': processes, 'time_unix': time.time()}


def free_indices(snapshot):
    occupied = {p[0] for p in snapshot['processes']}
    return [int(g[0]) for g in snapshot['gpus'] if g[1] not in occupied
            and float(g[2]) <= 32 and float(g[4]) == 0 and (float(g[3])-float(g[2]))/1024 >= 100]


def host_available_gib():
    return next(int(x.split()[1])/2**20 for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))


def ensure_exclusive(index):
    info = gpu_snapshot(index)
    other = [p for p in info['processes'] if int(p[1]) != os.getpid()]
    if other:
        raise ResourceConflict('Unrecognized process on GPU '+str(index)+': '+str(other))
    if shutil.disk_usage(ROOT).free < 50*2**30:
        raise ResourceConflict('Shared disk reserve below 50 GiB')
    return info


@contextlib.contextmanager
def admission(index):
    assert index in range(8) and RUN.resolve() == RUN and ROOT.resolve() == ROOT
    path = ROOT / 'runs' / f'.hya_gpu_{index}.lock'
    assert path.resolve() == path
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceConflict('Own project GPU lock busy') from exc
        for i in range(2):
            snap = gpu_snapshot(index)
            if index not in free_indices(snap):
                raise ResourceConflict('GPU no longer idle: '+str(snap))
            if i == 0:
                time.sleep(2)
        if host_available_gib() < 160 or shutil.disk_usage(ROOT).free < 50*2**30:
            raise ResourceConflict('Host RAM/disk admission failed')
        gpu = snap['gpus'][0]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu[1]
        yield {'index': index, 'uuid': gpu[1], 'pid': os.getpid(), 'snapshot': snap}


@contextlib.contextmanager
def loading_slot():
    """Serialize large CPU model reconstruction and initial weight loading only."""
    with (RUN/'model_loading.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if host_available_gib() < 160:
            raise ResourceConflict('Insufficient host RAM before model load')
        yield


@contextlib.contextmanager
def task_claim(task_id):
    assert task_id.replace('_', '').isalnum()
    path = RUN/'claims'/(task_id+'.lock'); path.parent.mkdir(exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceConflict('Task already claimed: '+task_id) from exc
        yield


def commit_copy(source, target, expected):
    source, target = Path(source), Path(target)
    assert target.resolve() == target and target.is_relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert digest(target) == expected
        return
    temp = target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        shutil.copyfile(source, temp)
        assert digest(temp) == expected
        os.link(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def commit_pair(prompt_id, records):
    assert set(records) == set(ARMS)
    a, b = [records[x] for x in ARMS]
    for key in ['prompt_id', 'seed', 'protocol_sha256', 'actual_initial_latent_sha256']:
        assert a[key] == b[key], key
    assert a['inputs']['hashes'] == b['inputs']['hashes']
    assert a['admission']['uuid'] == b['admission']['uuid']
    assert a['video']['sha256'] != b['video']['sha256'], 'Identical final T/S video; investigate'
    pointer = RUN/'pairs'/(prompt_id+'.json')
    if pointer.exists():
        previous = read(pointer)
        assert all(previous['records'][a]['video']['sha256'] == records[a]['video']['sha256'] for a in ARMS)
        return
    for arm, rec in records.items():
        assert rec['status'] == 'passed' and digest(rec['video_path']) == rec['video']['sha256']
        target = ARCHIVE/arm/(prompt_id+'.mp4')
        if target.exists() and digest(target) != rec['video']['sha256']:
            # Interrupted pair commit: preserve its orphaned copy rather than overwriting/deleting evidence.
            old_hash = digest(target)
            preserved = ARCHIVE/'_superseded'/(prompt_id+'_'+arm+'_'+old_hash+'.mp4')
            commit_copy(target, preserved, old_hash)
            assert not pointer.exists() and digest(target) == old_hash and target.resolve() == target
            target.unlink()
        commit_copy(rec['video_path'], target, rec['video']['sha256'])
        rec['archive_path'] = str(target)
    write_json(pointer, {'status': 'passed', 'prompt_id': prompt_id,
               'committed_unix': time.time(), 'records': records})


def progress(path, stage, **fields):
    obj = {'stage': stage, 'time_unix': time.time(), 'pid': os.getpid(), **fields}
    write_json(path, obj, replace=True)
    print('H2_PROGRESS', json.dumps(obj, ensure_ascii=False), flush=True)
