#!/usr/bin/env python3
"""Reuse shared generation models read-only; download only missing evaluation assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
MODELS = Path('/data/chenjiayu/models')
HASH_PROGRESS_MIN_BYTES = 64 * 1024 * 1024
HASH_PROGRESS_INTERVAL = 5.0


def digest(path, git_blob=False):
    total = path.stat().st_size
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(('blob %d\0' % total).encode())
    start = time.monotonic()
    state = {'done': 0, 'last_change': start}
    stop = threading.Event()
    monitor = None

    def show_progress():
        # Separate heartbeat remains visible even while a file read is waiting.
        while not stop.wait(HASH_PROGRESS_INTERVAL):
            now = time.monotonic()
            done = state['done']
            elapsed = max(now - start, 0.001)
            print('HASH_PROGRESS: %s | %.1f/%.1f MiB (%.1f%%) | %.1f MiB/s | elapsed=%.0fs no_progress=%.0fs' %
                  (path.name, done / 2**20, total / 2**20,
                   100 * done / max(total, 1), done / 2**20 / elapsed,
                   elapsed, now - state['last_change']), flush=True)

    if total >= HASH_PROGRESS_MIN_BYTES:
        print('HASH_START: %s | %.1f MiB | checking local file, no network download' %
              (path, total / 2**20), flush=True)
        monitor = threading.Thread(target=show_progress, daemon=True)
        monitor.start()
    try:
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
                state['done'] += len(block)
                state['last_change'] = time.monotonic()
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=1)
    if monitor is not None:
        print('HASH_READ_DONE: %s | %.1f MiB in %.1fs' %
              (path.name, state['done'] / 2**20, time.monotonic() - start), flush=True)
    return h.hexdigest()


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        raise RuntimeError('Size/missing-file mismatch: ' + str(path))
    key = 'sha256' if 'sha256' in item else 'git_blob_sha1'
    actual = digest(path, git_blob=(key == 'git_blob_sha1'))
    if actual != item[key]:
        raise RuntimeError('Checksum mismatch; preserved, not overwritten: ' + str(path))
    if 'official_git_blob_sha1' in item:
        if digest(path, git_blob=True) != item['official_git_blob_sha1']:
            raise RuntimeError('Official Git blob mismatch: ' + str(path))
    if item['bytes'] >= HASH_PROGRESS_MIN_BYTES:
        print('HASH_OK:', path.name, flush=True)
    return actual


def checked_path(item):
    root = MODELS if item['target_root'] == 'models' else PROJECT
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Invalid manifest path')
    target = root / relative
    # Parent symlinks are not allowed to redirect writes into existing resources.
    if target.parent.resolve() != target.parent or not target.parent.is_relative_to(root):
        raise RuntimeError('Parent redirects outside the designated literal path: ' + str(target))
    if target.is_symlink():
        allowed = item.get('reuse_source')
        if not allowed or target.resolve() != Path(allowed).resolve():
            raise RuntimeError('Unexpected destination symlink: ' + str(target))
    return target


def one(item, endpoint, verify_only):
    target = checked_path(item)
    if target.exists() or target.is_symlink():
        value = verify(target, item)
        print('VERIFIED existing:', target, flush=True)
        return {'id': item['id'], 'path': str(target), 'status': 'verified_existing', 'digest': value}
    # Hard boundary independent of manifest flags: shared models are NEVER
    # created, downloaded, linked, converted, repaired or overwritten here.
    if item['target_root'] == 'models' or item['group'] == 'models':
        raise RuntimeError('REUSE_ONLY model missing; inspect existing resources, do not download: ' + str(target))
    if verify_only:
        raise RuntimeError('Missing: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.changzihe.part')
    if partial.is_symlink() or (partial.exists() and not stat.S_ISREG(partial.stat().st_mode)):
        raise RuntimeError('Unsafe partial file: ' + str(partial))
    if partial.exists() and partial.stat().st_size > item['bytes']:
        raise RuntimeError('Oversized partial preserved: ' + str(partial))
    url = (endpoint + '/' + quote(item['repo_id'], safe='/') + '/resolve/'
           + item['revision'] + '/' + quote(item['repo_file'], safe='/'))
    print('DOWNLOAD:', item['id'], '->', target, flush=True)
    # Each bounded attempt resumes from the current partial size. No automatic
    # origin change, credentials, TLS bypass, package installation or GPU calls.
    for attempt in range(1, 7):
        if partial.exists() and partial.stat().st_size == item['bytes']:
            break
        cmd = ['curl', '--fail', '--location', '--show-error', '--progress-bar',
               '--proto', '=https', '--proto-redir', '=https',
               '--connect-timeout', '20', '--max-time', '1800',
               '--speed-limit', '1024', '--speed-time', '90',
               '--continue-at', '-', '--output', str(partial), url]
        rc = subprocess.run(cmd, check=False).returncode
        if rc in (-2, 130):
            raise KeyboardInterrupt
        if rc == 0:
            break
        print('curl_exit=%d attempt=%d/6; partial retained' % (rc, attempt), flush=True)
        if attempt == 6:
            raise RuntimeError('Download interrupted; rerun this command to resume: ' + item['id'])
        time.sleep(3)
    value = verify(partial, item)
    # Atomic no-clobber publication on the same filesystem.
    os.link(partial, target)
    partial.unlink()
    return {'id': item['id'], 'path': str(target), 'status': 'downloaded_verified', 'digest': value}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=['all', 'models', 'evaluation'], default='all')
    p.add_argument('--endpoint', choices=['https://hf-mirror.com', 'https://huggingface.co'], default='https://hf-mirror.com')
    p.add_argument('--plan-only', action='store_true', help='Print manifest only; no filesystem mutation/network/hash reads')
    p.add_argument('--verify-only', action='store_true', help='Read and hash target files; no downloads or repairs')
    p.add_argument('--inventory-only', action='store_true', help='Check existence and sizes only; NOT a full checksum verification')
    args = p.parse_args()
    manifest = json.loads((HERE / 'RESOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 2:
        raise RuntimeError('Obsolete resource manifest; use the audited reuse-only version 2')
    for item in manifest['assets']:
        if item['group'] == 'models' and (item['target_root'] != 'models' or item.get('reuse_only') is not True):
            raise RuntimeError('All generation model assets must be explicitly reuse-only')
        if item['group'] == 'evaluation' and (item['target_root'] != 'project' or not item['path'].startswith('resources/evaluation/')):
            raise RuntimeError('Evaluation target must stay in project resources/evaluation')
    items = [i for i in manifest['assets'] if args.group == 'all' or i['group'] == args.group]
    if args.plan_only:
        for i in items:
            root = MODELS if i['target_root'] == 'models' else PROJECT
            print('%12d  %-12s %s' % (i['bytes'], 'REUSE_ONLY' if i.get('reuse_only') else 'EVALUATION', root / i['path']))
        print('Existing generation bytes to verify: %.3f GiB; evaluation maximum download: %.3f GiB; generation download: 0' %
              (sum(i['bytes'] for i in items if i.get('reuse_only')) / 2**30,
               sum(i['bytes'] for i in items if not i.get('reuse_only')) / 2**30))
        return 0
    if sys.platform != 'linux' or HERE != PROJECT / 'Project_Plan':
        raise RuntimeError('Run this file from the designated H200 project, not Windows or another project')
    for root in [PROJECT, MODELS]:
        if not root.is_dir() or root.resolve() != root:
            raise RuntimeError('Root missing or redirected: ' + str(root))
    downloadable = [i for i in items if i['group'] == 'evaluation' and not checked_path(i).exists()]
    if downloadable and not args.verify_only and not args.inventory_only:
        if shutil.which('curl') is None:
            raise RuntimeError('curl missing; no automatic installation')
        need = sum(i['bytes'] for i in downloadable) + 10 * 2**30
        if shutil.disk_usage(PROJECT).free < need:
            raise RuntimeError('Insufficient space for missing evaluation assets plus 10 GiB reserve')
    import fcntl
    with (HERE / '.resource_download.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        error = None
        interrupted = False
        active_asset = None
        try:
            for item in items:
                active_asset = item['id']
                if args.inventory_only:
                    path = checked_path(item)
                    if not path.is_file() or path.stat().st_size != item['bytes']:
                        raise RuntimeError('Inventory size/missing-file mismatch: ' + str(path))
                    records.append({'id': item['id'], 'path': str(path), 'bytes': item['bytes'], 'status': 'size_only_not_checksum_verified'})
                    print('SIZE_ONLY:', path, flush=True)
                else:
                    records.append(one(item, args.endpoint, args.verify_only))
                active_asset = None
        except KeyboardInterrupt:
            interrupted = True
            error = 'Interrupted by user; completed files and download partials preserved. Rerun the same command; interrupted hashes restart from the beginning.'
        except Exception as exc:
            error = str(exc)
        outdir = PROJECT / 'Project_Progress' / 'Stage_Progress'
        if outdir.resolve() != outdir:
            raise RuntimeError('Progress directory redirected')
        outdir.mkdir(parents=True, exist_ok=True)
        report = outdir / ('resource_check_%s_%d.json' % (time.strftime('%Y%m%d_%H%M%S'), os.getpid()))
        with report.open('x', encoding='utf-8') as f:
            json.dump({'ok': error is None, 'endpoint': args.endpoint, 'verify_only': args.verify_only,
                       'inventory_only': args.inventory_only, 'manifest_version': 2,
                       'records': records, 'error': error, 'interrupted': interrupted,
                       'active_asset': active_asset}, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print('Report:', report)
        if interrupted:
            print('INTERRUPTED:', error, file=sys.stderr)
            return 130
        if error:
            raise RuntimeError(error)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('INTERRUPTED: stopped before a resource report could be written; files preserved.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('ERROR:', str(exc), file=sys.stderr)
        sys.exit(1)
