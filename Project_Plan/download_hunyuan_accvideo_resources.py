#!/usr/bin/env python3
"""Manually invoked, missing-only resource downloads for the frozen HYA plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
MODELS = Path('/data/chenjiayu/models')
MANIFEST = HERE / 'HUNYUAN_ACCVIDEO_RESOURCES.json'
PART_SUFFIX = '.changzihe_hya.part'
ACC_FILES = {'accvideo-t2v-5-steps/diffusion_pytorch_model.pt', 'LICENSE', 'Notice', 'README.md'}


def check_manifest(manifest):
    if manifest.get('manifest_id') != 'HYA_720P_RESOURCES_V1_20261019':
        raise RuntimeError('Unexpected manifest identity')
    seen = set()
    for item in manifest['assets']:
        path = PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in item['path']:
            raise RuntimeError('Invalid manifest path')
        key = (item['target_root'], str(path))
        if key in seen:
            raise RuntimeError('Duplicate target in manifest')
        seen.add(key)
        if item['action'] not in ('reuse_only', 'download_if_missing'):
            raise RuntimeError('Invalid action')
        if item['target_root'] == 'models':
            allowed_new = str(path).startswith('AccVideo/') and str(path)[9:] in ACC_FILES
            allowed_reuse = str(path).startswith('HunyuanVideo/') and item['action'] == 'reuse_only'
            if not (allowed_new or allowed_reuse):
                raise RuntimeError('Shared model target outside exact allowlist')
            if allowed_new and item['repo_id'] != 'aejion/AccVideo':
                raise RuntimeError('Unexpected AccVideo repository')
        elif item['target_root'] == 'project':
            if not str(path).startswith('resources/evaluation/'):
                raise RuntimeError('Evaluation assets must stay in the project evaluation directory')
        else:
            raise RuntimeError('Unknown target root')
        if not re.fullmatch(r'[0-9a-f]{40}', item['revision']):
            raise RuntimeError('Revision must be an immutable commit')
        hash_key = 'sha256' if item.get('sha256') else 'git_blob_sha1'
        if not re.fullmatch(r'[0-9a-f]{64}' if hash_key == 'sha256' else r'[0-9a-f]{40}', item.get(hash_key, '')):
            raise RuntimeError('Missing valid digest')
        if item['bytes'] <= 0:
            raise RuntimeError('Invalid expected size')


def checked_path(item):
    root = MODELS if item['target_root'] == 'models' else PROJECT
    path = root.joinpath(*PurePosixPath(item['path']).parts)
    if path.parent.resolve() != path.parent or not path.parent.is_relative_to(root):
        raise RuntimeError('Redirected target parent: ' + str(path))
    if path.is_symlink():
        raise RuntimeError('Symlink target; inspect and reuse explicitly: ' + str(path))
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        raise RuntimeError('Target is not a regular file: ' + str(path))
    return path


def verify(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes']:
        raise RuntimeError('Missing/size mismatch; preserved: ' + str(path))
    before = path.stat()
    use_git = not item.get('sha256')
    h = hashlib.sha1() if use_git else hashlib.sha256()
    if use_git:
        h.update(('blob %d\0' % before.st_size).encode())
    done, last = 0, time.monotonic()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
            done += len(chunk)
            if time.monotonic() - last > 5:
                print('HASH %.1f%% %s' % (100 * done / before.st_size, path.name), flush=True)
                last = time.monotonic()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('File changed during verification: ' + str(path))
    expected = item['git_blob_sha1'] if use_git else item['sha256']
    if h.hexdigest() != expected:
        raise RuntimeError('Hash mismatch; NOT overwritten/deleted: ' + str(path))
    print('VERIFIED', path, flush=True)
    return h.hexdigest()


def inspect_new_acc_directory(items):
    # No broad scans. A newly added alternative AccVideo folder needs a reuse audit.
    for child in MODELS.iterdir():
        if 'accvideo' in child.name.lower() and child.name != 'AccVideo':
            raise RuntimeError('Possible reusable AccVideo directory added since audit: ' + str(child))
    root = MODELS / 'AccVideo'
    if not root.exists():
        return
    if root.is_symlink() or root.resolve() != root:
        raise RuntimeError('AccVideo path is redirected')
    allowed = {str(PurePosixPath(i['path']).relative_to('AccVideo')) for i in items if i['target_root'] == 'models' and i['path'].startswith('AccVideo/')}
    checked = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        checked += len(dirs) + len(files)
        if checked > 500:
            raise RuntimeError('AccVideo inventory exceeds bound; inspect before downloading')
        for name in dirs + files:
            if (Path(current) / name).is_symlink():
                raise RuntimeError('Unexpected symlink in AccVideo; inspect before writing')
        for name in files:
            rel = (Path(current) / name).relative_to(root).as_posix()
            if rel in allowed or rel.endswith(PART_SUFFIX) and rel[:-len(PART_SUFFIX)] in allowed:
                continue
            if name.lower().endswith(('.pt', '.pth', '.bin', '.safetensors', '.ckpt')):
                raise RuntimeError('Additional possibly reusable weight exists; no duplicate download: ' + rel)


def publish(partial, target):
    # Same-filesystem atomic create; EEXIST never replaces another user's file.
    if target.exists() or target.is_symlink():
        raise FileExistsError('Target appeared during download; both files preserved: ' + str(target))
    os.link(partial, target)
    partial.unlink()


def download_one(item, endpoint):
    target = checked_path(item)
    if target.exists():
        verify(target, item)
        return
    if item['action'] == 'reuse_only':
        raise RuntimeError('REUSE_ONLY missing; no replacement download: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    checked_path(item)
    partial = target.with_name(target.name + PART_SUFFIX)
    if partial.is_symlink() or partial.exists() and (not partial.is_file() or partial.stat().st_size > item['bytes']):
        raise RuntimeError('Unsafe/oversized partial preserved: ' + str(partial))
    url = endpoint + '/' + quote(item['repo_id'], safe='/') + '/resolve/' + item['revision'] + '/' + quote(item['repo_file'], safe='/')
    print('DOWNLOAD', item['id'], '->', target, flush=True)
    for attempt in range(1, 7):
        if partial.exists() and partial.stat().st_size == item['bytes']:
            break
        result = subprocess.run(['curl', '--fail', '--location', '--show-error', '--progress-bar',
                                 '--proto', '=https', '--proto-redir', '=https',
                                 '--connect-timeout', '20', '--max-time', '1800',
                                 '--speed-limit', '1024', '--speed-time', '90',
                                 '--continue-at', '-', '--output', str(partial), url], check=False)
        if result.returncode in (-2, 130):
            raise KeyboardInterrupt
        if result.returncode == 0:
            break
        print('curl_exit=%d attempt=%d/6; partial retained' % (result.returncode, attempt), flush=True)
        if attempt == 6:
            raise RuntimeError('Download failed; rerun to resume; no automatic source change')
        time.sleep(3)
    verify(partial, item)
    checked_path(item)
    publish(partial, target)
    print('PUBLISHED', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--download', action='store_true', help='Explicit user action: download missing allowlisted assets')
    mode.add_argument('--verify-only', action='store_true', help='Full hashes, no writes/network')
    mode.add_argument('--inventory-only', action='store_true', help='Existence and sizes, not full verification')
    mode.add_argument('--plan-only', action='store_true', help='Show frozen list, portable and read-only')
    parser.add_argument('--group', choices=['all', 'student', 'evaluation'], default='all')
    parser.add_argument('--endpoint', choices=['https://hf-mirror.com', 'https://huggingface.co'], default='https://hf-mirror.com')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    check_manifest(manifest)
    items = manifest['assets']
    selected = [i for i in items if args.group == 'all' or args.group == 'student' and i['target_root'] == 'models' or args.group == 'evaluation' and i['target_root'] == 'project']
    if args.plan_only:
        for item in selected:
            print('%12d %-19s %s/%s' % (item['bytes'], item['action'], item['target_root'], item['path']))
        print('Frozen missing-only upper bound: %.3f GiB' % (sum(i['bytes'] for i in selected if i['action'] != 'reuse_only') / 2**30))
        return 0
    if sys.platform != 'linux' or HERE != PROJECT / 'Project_Plan':
        raise RuntimeError('Use the designated server project, not another checkout')
    for root in (PROJECT, MODELS):
        if not root.is_dir() or root.resolve() != root:
            raise RuntimeError('Root missing or redirected: ' + str(root))
    inspect_new_acc_directory(items)
    missing = []
    for item in selected:
        path = checked_path(item)
        if path.exists():
            if path.stat().st_size != item['bytes']:
                raise RuntimeError('Existing size mismatch; no overwrite: ' + str(path))
            print('PRESENT_SIZE_ONLY', path, flush=True)
        else:
            print('MISSING', item['action'], path, flush=True)
            missing.append(item)
    needed = sum(i['bytes'] for i in missing if i['action'] != 'reuse_only')
    print('Incremental full missing files %.3f GiB; free %.3f GiB' % (needed / 2**30, shutil.disk_usage(PROJECT).free / 2**30), flush=True)
    if args.inventory_only:
        return 0
    # All pre-existing selected resources are validated BEFORE the first download.
    for item in selected:
        path = checked_path(item)
        if path.exists():
            verify(path, item)
        elif args.verify_only or item['action'] == 'reuse_only':
            raise RuntimeError('Required file missing; no fallback: ' + str(path))
    if args.verify_only:
        print('ALL_SELECTED_ASSETS_VERIFIED; runtime acceptance is separate', flush=True)
        return 0
    if shutil.which('curl') is None:
        raise RuntimeError('curl missing; no automatic installation')
    reserve = manifest['minimum_reserve_gib'] * 2**30
    if shutil.disk_usage(PROJECT).free < needed + reserve or shutil.disk_usage(MODELS).free < needed + reserve:
        raise RuntimeError('Insufficient free space for missing files plus 50 GiB reserve')
    import fcntl
    lock = PROJECT / 'Project_Plan/.hya_resource_download.lock'
    if lock.is_symlink():
        raise RuntimeError('Unsafe lock symlink')
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inspect_new_acc_directory(items)
        for item in missing:
            download_one(item, args.endpoint)
    print('DOWNLOADS_VERIFIED. No environments, inference or evaluation were started.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Partial files preserved; rerun the same command to resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
