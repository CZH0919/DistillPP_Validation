#!/usr/bin/env python3
"""Unpack small pinned generation/evaluation sources; no network or model execution."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--group', choices=['generation', 'evaluation'], required=True)
args = parser.parse_args()
if HERE != PROJECT / 'Project_Plan' or PROJECT.resolve() != PROJECT:
    raise RuntimeError('Run in the designated remote project')
source_root = PROJECT / 'src'
if source_root.resolve() != source_root:
    raise RuntimeError('Source directory redirects elsewhere')
source_root.mkdir(exist_ok=True)

for info in json.loads((HERE / 'SOURCE_ARCHIVES.json').read_text()):
    if info['group'] != args.group:
        continue
    archive = HERE / info['archive']
    data = archive.read_bytes()
    if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
        raise RuntimeError('Source archive mismatch: ' + str(archive))
    target = source_root / info['name']
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            parts = PurePosixPath(m.name).parts
            if not (m.isfile() or m.isdir()) or not parts or parts[0] != info['name'] or '..' in parts or m.name.startswith('/'):
                raise RuntimeError('Unexpected archive member: ' + m.name)
        members = [m for m in members if m.isfile()]
        if target.exists():
            if target.is_symlink():
                raise RuntimeError('Source target is a symlink')
            for m in members:
                path = source_root / m.name
                if path.resolve() != path or not path.is_file() or path.read_bytes() != tf.extractfile(m).read():
                    raise RuntimeError('Existing source differs; not overwritten: ' + str(path))
            print('Source verified:', target)
        else:
            staging = source_root / ('.' + info['name'] + '.prepare')
            staging.mkdir()  # A prior interrupted staging directory needs review.
            for m in members:
                path = staging / m.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    f.write(tf.extractfile(m).read())
            (staging / info['name']).rename(target)
            staging.rmdir()
            print('Source prepared:', target)

if args.group == 'generation':
    print('Pinned generation source trees ready; no models loaded.')
    raise SystemExit(0)

bpe = PROJECT / 'resources/evaluation/ViCLIP/bpe_simple_vocab_16e6.txt.gz'
expected = '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
if hashlib.sha256(bpe.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Download and verify evaluation BPE first')
dest = source_root / 'CLIP/clip/bpe_simple_vocab_16e6.txt.gz'
if dest.exists():
    if dest.resolve() != dest or hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Existing CLIP vocabulary differs')
else:
    with dest.open('xb') as f, bpe.open('rb') as src:
        shutil.copyfileobj(src, f)
print('Pinned source trees and vocabulary ready; no models loaded.')
