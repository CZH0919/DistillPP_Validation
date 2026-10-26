"""Pack only S2 code provenance, small JSON/CSV/Markdown results and logs.

Never include videos, latent tensors, weights, environments or caches.
"""
import hashlib
import json
from pathlib import Path
import tarfile

PROJECT=Path('/data/chenjiayu/changzihe/DistillPP_Validation')
BATCH=PROJECT/'runs/s2_20261016'
OUT=PROJECT/'Result/S2_20261016'
assert json.loads((BATCH/'evaluation_controller.json').read_text())['phase']=='completed'
selected=[]
for f in sorted(BATCH.rglob('*')):
    if f.is_file() and f.suffix in {'.json','.py','.log'}:
        assert f.resolve()==f and f.stat().st_size < 8*2**20
        selected.append(f)
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.suffix in {'.json','.csv','.md'}:selected.append(f)
assert len(selected)==len(set(selected))
assert sum(p.stat().st_size for p in selected)<32*2**20
assert all(p.suffix not in {'.pt','.pth','.mp4','.safetensors'} for p in selected)
manifest={'files':[{'path':str(p.relative_to(PROJECT)),'bytes':p.stat().st_size,
    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected],
    'excluded':'All videos, latent tensors, model/evaluation weights, environments and caches'}
dest=OUT/'core_evidence.tar.gz'
with tarfile.open(dest,'x:gz') as archive:
    for p in selected:archive.add(p,arcname=str(p.relative_to(PROJECT)),recursive=False)
(OUT/'core_evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(dest),'files':len(selected),'uncompressed_bytes':sum(p.stat().st_size for p in selected),
    'archive_bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))
