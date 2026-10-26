#!/usr/bin/env python3
"""Preview by default; --apply removes only this incident's six unchanged entries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

PROJECT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
TARGET = Path('/data/chenjiayu/models/Wan2.1-T2V-14B')
ALLOWED = {
    'LICENSE.txt', 'README.md', 'Wan2.1_VAE.pth', 'config.json',
    'diffusion_pytorch_model-00001-of-00006.safetensors',
    'diffusion_pytorch_model-00002-of-00006.safetensors.changzihe.part',
}
LINK_TARGET = '/data/chenjiayu/models/Wan2.1-T2V-14B-StepDistill-CfgDistill/Wan2.1_VAE.pth'


def check_entry(entry, fd):
    name = entry['name']
    s = os.stat(name, dir_fd=fd, follow_symlinks=False)
    actual = (s.st_size, s.st_ino, s.st_dev, s.st_nlink, s.st_uid, s.st_mtime_ns)
    expected = tuple(entry[k] for k in ['bytes', 'inode', 'device', 'nlink', 'uid', 'mtime_ns'])
    if actual != expected or s.st_nlink != 1 or s.st_uid != os.getuid():
        raise RuntimeError('Object changed; nothing further removed: ' + name)
    if name == 'Wan2.1_VAE.pth':
        if not stat.S_ISLNK(s.st_mode) or os.readlink(name, dir_fd=fd) != LINK_TARGET:
            raise RuntimeError('Unexpected VAE link; stop')
    elif not stat.S_ISREG(s.st_mode):
        raise RuntimeError('Not an ordinary file: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(__file__).resolve().parent != PROJECT / 'Project_Plan':
        raise RuntimeError('Run only in the designated H200 project')
    for root in [PROJECT, PROJECT / 'Project_Progress/Stage_Progress', TARGET.parent]:
        if root.resolve() != root or not root.is_dir():
            raise RuntimeError('Root redirected or missing: ' + str(root))
    inventory = PROJECT / 'Project_Progress/Stage_Progress/resource_recovery_inventory.json'
    report = json.loads(inventory.read_text())
    if report['root'] != str(TARGET) or {e['name'] for e in report['entries']} != ALLOWED:
        raise RuntimeError('Cleanup inventory outside the exact six-entry scope')
    receipt = PROJECT / 'Project_Progress/Stage_Progress/resource_check_20261016_084354_1346101.json'
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != report['download_receipt_sha256']:
        raise RuntimeError('Original download receipt changed; review first')
    if not TARGET.exists() and not TARGET.is_symlink():
        print('Target is already absent. No action.')
        return 0
    if TARGET.resolve() != TARGET or TARGET.is_symlink():
        raise RuntimeError('Cleanup target is redirected')
    lock_path = PROJECT / 'Project_Plan/.resource_download.lock'
    if lock_path.resolve() != lock_path:
        raise RuntimeError('Download lock redirected')
    with lock_path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = os.open(TARGET, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            root_stat = os.fstat(fd)
            if (root_stat.st_ino, root_stat.st_dev) != (report['root_inode'], report['root_device']):
                raise RuntimeError('Target directory replaced; stop')
            if set(os.listdir(fd)) != ALLOWED:
                raise RuntimeError('Directory has extra/missing entries; stop without deleting')
            for entry in report['entries']:
                check_entry(entry, fd)
            regular = [str(TARGET / e['name']) for e in report['entries'] if e['kind'] == 'file']
            # This host's fuser does not accept a standalone '--'; all arguments
            # are fixed absolute paths, so none can be interpreted as options.
            busy = subprocess.run(['/usr/bin/fuser', *regular], capture_output=True, text=True, timeout=15)
            if busy.returncode != 1 or busy.stdout.strip() or busy.stderr.strip():
                raise RuntimeError('Files may be open or fuser failed; stop: ' + busy.stdout + busy.stderr)
            for e in report['entries']:
                print(('UNLINK_ONLY' if e['kind'] == 'symlink' else 'DELETE_FILE'), e['bytes'], TARGET / e['name'])
            allocated = sum(e['allocated_bytes'] for e in report['entries'])
            print('Reclaimable allocation: %.3f GiB. Original VAE target is preserved.' % (allocated / 2**30))
            if not args.apply:
                print('PREVIEW_ONLY: no files removed. Use --apply to execute this exact list.')
                return 0
            log = PROJECT / ('Project_Progress/Stage_Progress/duplicate_cleanup_%s_%d.jsonl' % (time.strftime('%Y%m%d_%H%M%S'), os.getpid()))
            with log.open('x', encoding='utf-8') as journal:
                journal.write(json.dumps({'event': 'validated', 'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(), 'target': str(TARGET)}) + '\n')
                journal.flush()
                for entry in report['entries']:
                    check_entry(entry, fd)
                    os.unlink(entry['name'], dir_fd=fd)  # Never follows the VAE symlink.
                    journal.write(json.dumps({'event': 'unlinked', 'name': entry['name']}) + '\n')
                    journal.flush()
                now = TARGET.lstat()
                if (now.st_ino, now.st_dev) != (root_stat.st_ino, root_stat.st_dev):
                    raise RuntimeError('Directory identity changed; do not remove it')
                os.rmdir(TARGET)  # Fails if anything remains; no recursive delete.
                journal.write(json.dumps({'event': 'completed', 'directory_removed': True}) + '\n')
            print('CLEANUP_COMPLETE:', log)
        finally:
            os.close(fd)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('STOP:', str(exc), file=sys.stderr)
        raise SystemExit(1)
