"""H1 additions to the owner's evaluation environment; no model/GPU work."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

ROOT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
RUN = ROOT / 'runs/hya_h1_20261019/dependencies'
CACHE = ROOT / 'resources/software_cache/hya_h1'
PY = Path('/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python')
COMMIT = '1e3e13bbf607b54f62205c4c33922521822fb298'


def norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def installed():
    return {norm(d.metadata['Name']): d.version for d in metadata.distributions()}


def write_new(path, value):
    if not isinstance(value, str):
        value = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != value:
            raise RuntimeError('Existing record differs: ' + str(path))
    else:
        path.write_text(value)


def run(name, command, timeout):
    log = RUN / (name + '.log')
    if log.exists():
        raise RuntimeError('Preserve prior attempt; inspect before retry: ' + str(log))
    start = time.time()
    with log.open('x') as f:
        print('RUN', name, flush=True)
        proc = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=os.environ)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
            raise
    (RUN / (name + '_status.json')).write_text(json.dumps({'command': command, 'returncode': rc, 'seconds': time.time()-start}, indent=2)+'\n')
    print(log.read_text(errors='replace')[-6000:], flush=True)
    if rc:
        raise RuntimeError(name + ' failed with exit code ' + str(rc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['resolve', 'install', 'build'])
    args = parser.parse_args()
    assert ROOT.resolve() == ROOT and Path(sys.executable).absolute() == PY
    for path in [RUN, CACHE]:
        assert path.resolve() == path
        path.mkdir(parents=True, exist_ok=True)
    for key, value in {'CUDA_VISIBLE_DEVICES': '', 'CUDA_HOME': '/usr/local/cuda-12.8',
                       'TORCH_CUDA_ARCH_LIST': '9.0', 'FORCE_CUDA': '1', 'MAX_JOBS': '4',
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1',
                       'PIP_CACHE_DIR': str(CACHE/'pip'), 'TMPDIR': str(CACHE/'tmp')}.items():
        os.environ[key] = value
    (CACHE/'tmp').mkdir(exist_ok=True)
    os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ['PATH']
    before = RUN / 'before.json'
    constraints = RUN / 'constraints_before.txt'
    if args.phase == 'resolve':
        packages = installed()
        write_new(before, packages)
        write_new(constraints, ''.join(f'{k}=={v}\n' for k, v in sorted(packages.items())))
        run('resolve', [str(PY), '-m', 'pip', 'install', '--dry-run', '--report', str(RUN/'pip_resolution.json'),
                        '--no-build-isolation', '--disable-pip-version-check', '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',
                        '--timeout', '30', '--retries', '2', '-c', str(constraints), '-r', str(ROOT/'Project_Plan/requirements_hya_evaluation.txt')], 900)
        report = json.loads((RUN/'pip_resolution.json').read_text())
        lines = []
        for item in report['install']:
            name, version = norm(item['metadata']['name']), item['metadata']['version']
            assert name not in packages, 'An existing dependency would be changed: ' + name
            info = item['download_info']
            digest = info['archive_info']['hashes']['sha256']
            assert info['url'].startswith('https://')
            lines.append(f'{name} @ {info["url"]} --hash=sha256:{digest}\n')
        write_new(RUN/'resolved_additions.txt', ''.join(lines))
        print('ADDITIONS_ONLY_RESOLUTION', len(lines), flush=True)
    elif args.phase == 'install':
        previous = json.loads(before.read_text())
        assert installed() == previous, 'Environment changed after resolution'
        run('install', [str(PY), '-m', 'pip', 'install', '--require-hashes', '--no-deps', '--no-build-isolation',
                        '--disable-pip-version-check', '--timeout', '30', '--retries', '2', '-r', str(RUN/'resolved_additions.txt')], 1200)
        after = installed()
        assert all(after.get(k) == v for k, v in previous.items()), 'Existing package changed'
        write_new(RUN/'after_additions.json', after)
        print('EXISTING_PACKAGES_UNCHANGED', len(previous), flush=True)
    else:
        reference = ROOT/'Project_Plan/hya_sources'
        info = json.loads((reference/'detectron2_archive.json').read_text())
        archive = reference/info['path']
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == info['sha256']
        source = CACHE/('detectron2-' + COMMIT)
        assert not source.exists(), 'Inspect prior source/build before retry'
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            extract_members, skipped_document_links = [], []
            for item in members:
                dest = CACHE/item.name
                assert dest.resolve().is_relative_to(source) and not item.islnk()
                if item.issym():
                    # Upstream has four documentation aliases; no build code needs them.
                    assert dest.is_relative_to(source/'docs')
                    assert (dest.parent/item.linkname).resolve().is_relative_to(source)
                    skipped_document_links.append({'path': item.name, 'target': item.linkname})
                    continue
                assert item.isfile() or item.isdir()
                extract_members.append(item)
            tar.extractall(CACHE, members=extract_members)
            write_new(RUN/'source_extraction.json', {'source_commit': COMMIT, 'skipped_document_links': skipped_document_links})
        wheels = CACHE/'wheels'; wheels.mkdir(exist_ok=True)
        run('detectron2_build', [str(PY), '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-index',
                               '--disable-pip-version-check', '--wheel-dir', str(wheels), str(source)], 1800)
        candidates = list(wheels.glob('detectron2-*.whl')); assert len(candidates) == 1
        wheel = candidates[0]
        write_new(RUN/'detectron2_wheel.json', {'path': str(wheel), 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'source_commit': COMMIT})
        run('detectron2_install', [str(PY), '-m', 'pip', 'install', '--no-deps', '--no-index', str(wheel)], 120)
        old = json.loads(before.read_text()); after = installed()
        assert all(after.get(k) == v for k,v in old.items())
        write_new(RUN/'after_all.json', after)
        print('DEPENDENCY_INSTALL_COMPLETE; GPU/runtime acceptance still required', flush=True)


if __name__ == '__main__':
    main()
