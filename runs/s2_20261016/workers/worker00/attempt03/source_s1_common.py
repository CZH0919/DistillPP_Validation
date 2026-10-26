"""Project-local safety and provenance helpers for S1 only."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

PROJECT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
BATCH = PROJECT / 'runs/s1_20261016'
PROMPT = 'A blue toy boat drifts across a garden pond, leaving gentle ripples in soft daylight. Static camera.'
SEED = 2026101601
SHAPE = (16, 21, 60, 104)
METRICS = ['imaging_quality', 'subject_consistency', 'motion_smoothness',
           'dynamic_degree', 'aesthetic_quality', 'overall_consistency']
NETWORK_ATTEMPTS = []


def json_write(path, data):
    path = Path(path)
    assert path.is_relative_to(PROJECT) and path.resolve() == path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.writing')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure():
    assert PROJECT.resolve() == PROJECT and PROJECT.is_dir()
    assert BATCH.resolve() == BATCH
    for key, value in {
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'MAX_JOBS': '4',
        'DTYPE': 'BF16', 'SENSITIVE_LAYER_DTYPE': 'None',
        'PROFILING_DEBUG_LEVEL': '0', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'VBENCH_CACHE_DIR': str(PROJECT / 'resources/evaluation'),
        'HF_HOME': str(PROJECT / 'resources/software_cache/hf_s1'),
        'TORCH_HOME': str(PROJECT / 'resources/software_cache/torch_s1'),
        'XDG_CACHE_HOME': str(PROJECT / 'resources/software_cache/xdg'),
        'TRITON_CACHE_DIR': str(PROJECT / 'resources/software_cache/triton_s1'),
    }.items():
        os.environ[key] = value
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            NETWORK_ATTEMPTS.append(str(address))
            raise RuntimeError('S1 offline guard rejected network connection: ' + str(address))
        return original_connect(sock, address)
    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            NETWORK_ATTEMPTS.append(str(address))
            raise RuntimeError('S1 offline guard rejected network connection: ' + str(address))
        return original_connect_ex(sock, address)
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def gpu_snapshot():
    def query(fields, kind):
        text = subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + fields,
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
        return [list(map(str.strip, line.split(','))) for line in text.splitlines() if line.strip()]
    return {'gpus': query('index,uuid,memory.used,memory.total,utilization.gpu', 'gpu'),
            'processes': query('gpu_uuid,pid,used_gpu_memory', 'compute-apps')}


def ensure_no_other_process(uuid):
    snapshot = gpu_snapshot()
    other = [row for row in snapshot['processes'] if row[0] == uuid and int(row[1]) != os.getpid()]
    if other:
        raise RuntimeError('Another process appeared on selected GPU; stopping only this S1 process: ' + str(other))
    return snapshot


@contextlib.contextmanager
def gpu_admission(index, min_free_gib=100):
    BATCH.mkdir(parents=True, exist_ok=True)
    lock_path = BATCH / ('gpu_%d.lock' % index)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last = None
        for attempt in range(2):
            last = gpu_snapshot()
            selected = [g for g in last['gpus'] if int(g[0]) == index]
            if len(selected) != 1:
                raise RuntimeError('Requested physical GPU not found')
            g = selected[0]
            if any(row[0] == g[1] for row in last['processes']) or float(g[2]) > 32 or float(g[4]) != 0:
                raise RuntimeError('Selected GPU is occupied; no model loaded: ' + str(g))
            if (float(g[3]) - float(g[2])) / 1024 < min_free_gib:
                raise RuntimeError('Insufficient GPU memory for admitted configuration')
            if attempt == 0:
                time.sleep(2)
        if shutil.disk_usage(PROJECT).free < 20 * 2**30:
            raise RuntimeError('Less than 20 GiB disk headroom')
        mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        if mem['MemAvailable'] < 80 * 2**20:
            raise RuntimeError('Less than 80 GiB available system RAM')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = g[1]
        yield {'physical_index': index, 'uuid': g[1], 'snapshot': last,
               'pid': os.getpid(), 'lock_scope': 'own project only; not a cluster reservation'}


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_stats(tensor):
    import torch
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError('Non-finite tensor')
    t = tensor.detach().float()
    return {'shape': list(t.shape), 'dtype': str(tensor.dtype), 'mean': t.mean().item(),
            'std': t.std().item(), 'min': t.min().item(), 'max': t.max().item()}
