"""Freeze authorized H2 run after CPU tests and full resource verification; no GPU compute."""
import ast
import importlib.metadata
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from hya_h2_common import (ROOT, RUN, GEN_PY, EVAL_PY, read, digest, write_json,
                          make_blocks, gpu_snapshot, free_indices, host_available_gib)


def main():
    assert read(RUN/'selftest.json')['status'] == 'passed'
    assert not (RUN/'manifest.json').exists(), 'Run already frozen; do not recreate'
    assert not (RUN/'controller.json').exists()
    protocol_path = ROOT/'Project_Plan/HUNYUAN_ACCVIDEO_PROTOCOL.json'; protocol = read(protocol_path)
    assert protocol['protocol_id'] == 'HYA_720P_QUALIFICATION_V2_20261019'
    assert protocol['counts']['videos_total'] == 128 and protocol['counts']['metric_values_total'] == 224
    assert protocol['generation']['seeds'] == [2026101901]
    assert read(ROOT/'Result/HYA_H1_20261019/H1_ACCEPTANCE.json')['status'] == 'passed'
    assets = []; started = time.time()
    for a in read(ROOT/'Project_Plan/HUNYUAN_ACCVIDEO_RESOURCES.json')['assets']:
        path = (Path('/data/chenjiayu/models') if a['target_root'] == 'models' else ROOT)/a['path']
        before = path.stat(); assert before.st_size == a['bytes']
        actual = digest(path); after = path.stat()
        if 'sha256' in a:
            assert actual == a['sha256'], str(path)
        else:
            assert a['bytes'] < 2**20 and 'git_blob_sha1' in a
            data = path.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == a['git_blob_sha1'], str(path)
        assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns, str(path)
        assets.append({'path': str(path), 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': actual})
    write_json(RUN/'resource_verification.json', {'status': 'passed', 'asset_count': len(assets),
               'elapsed_seconds': time.time()-started, 'assets': assets}, replace=True)
    environments = {}
    code = "import importlib.metadata as m,json; d={x.metadata['Name'].lower():x.version for x in m.distributions()}; print(json.dumps({p:d.get(p) for p in ['torch','transformers','diffusers','decord','numpy']}))"
    for name, exe in [('generation', GEN_PY), ('evaluation', EVAL_PY)]:
        result = subprocess.run([exe, '-B', '-c', code], capture_output=True, text=True, timeout=30, env=None)
        assert result.returncode == 0, result.stderr
        environments[name] = json.loads(result.stdout)
    assert environments['generation']['transformers'] == '4.51.3'
    assert environments['generation']['diffusers'] == '0.35.1'
    assert environments['evaluation']['transformers'] == '4.37.2'
    snap = gpu_snapshot(); free = free_indices(snap)
    assert len(free) >= 1, 'No idle GPU; preparation stops without launch'
    n = min(4, len(free))
    while n > 0 and host_available_gib() < 256+160*n: n -= 1
    assert n > 0 and shutil.disk_usage(ROOT).free > 70*2**30
    for name in ['logs', 'generation', 'evaluation', 'pairs', 'scores', 'inputs', 'claims', 'frozen_sources']:
        (RUN/name).mkdir(exist_ok=True)
    paths = list((ROOT/'tools').glob('hya_h2_*.py'))
    paths += [ROOT/'tools'/x for x in ['hya_common.py', 'hya_generate_smoke.py', 'hya_evaluate_smoke.py']]
    paths += [ROOT/'Project_Plan'/x for x in ['HUNYUAN_ACCVIDEO_PROTOCOL.json', 'HYA_H2_EXECUTION_POLICY.json',
                                            'HYA_RUNTIME_CONTRACT.json', 'HUNYUAN_ACCVIDEO_RESOURCES.json']]
    # Capture exact metric and native implementation sources, without downloading or modifying their trees.
    for directory in [ROOT/'src/VBench/vbench', ROOT/'src/CLIP/clip', ROOT/'src/DINO',
                      ROOT/'Project_Plan/hya_sources/AccVideo_reference']:
        for suffix in ['*.py', '*.yaml', '*.yml', '*.json']:
            paths += list(directory.rglob(suffix))
    frozen = {}
    for p in sorted(set(paths)):
        assert p.is_file() and p.is_relative_to(ROOT)
        rel = str(p.relative_to(ROOT)); frozen[rel] = digest(p)
        dest = RUN/'frozen_sources'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest); assert digest(dest) == frozen[rel]
    shutil.copyfile(protocol_path, RUN/'protocol.json')
    frozen[str((RUN/'protocol.json').relative_to(ROOT))] = digest(RUN/'protocol.json')
    manifest = {'run_id': 'hya_h2_20261019', 'stage': 'H2', 'created_unix': time.time(),
                'protocol_id': protocol['protocol_id'], 'protocol_sha256': digest(protocol_path),
                'authorization': {'H2': True, 'S3': False, 'source': 'User explicitly requested formal H2 background multi-GPU generation and evaluation in this turn'},
                'blocks': make_blocks(protocol['prompts'], n), 'initial_gpu_indices': free[:n],
                'maximum_workers': n, 'admission_snapshot': snap, 'host_available_gib': host_available_gib(),
                'disk_free_gib': shutil.disk_usage(ROOT).free/2**30, 'environments': environments,
                'frozen_files': frozen, 'resource_verification_sha256': digest(RUN/'resource_verification.json'),
                'cpu_selftest_sha256': digest(RUN/'selftest.json')}
    write_json(RUN/'manifest.json', manifest)
    print(json.dumps({'status': 'prepared', 'assets': len(assets), 'initial_gpu_indices': free[:n],
                      'frozen_sources': len(frozen), 'blocks': len(manifest['blocks']), 'environments': environments}), flush=True)


if __name__ == '__main__':
    main()
