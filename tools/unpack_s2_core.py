"""Verify the explicit small-file manifest before extracting S2 core evidence."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

ROOT=Path(__file__).resolve().parents[1]
TRANSFER=ROOT.parent/'tmp/s2_sync'
archive=TRANSFER/'core_evidence.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='344407b1cb7e37ac463f16fdde6b691073fc798f550088f1d4bf19b1eae2a0e6'
manifest=json.loads((TRANSFER/'core_evidence_manifest.json').read_text(encoding='utf-8'))
expected={r['path']:r for r in manifest['files']}
assert len(expected)==len(manifest['files'])==605
assert sum(r['bytes'] for r in expected.values())<32*2**20
with tarfile.open(archive,'r:gz') as handle:
    members=handle.getmembers()
    assert len(members)==len({m.name for m in members})==len(expected)
    assert {m.name for m in members}==set(expected)
    for member in members:
        name=member.name
        assert member.isfile() and not PurePosixPath(name).is_absolute()
        assert '..' not in PurePosixPath(name).parts
        assert name.startswith(('runs/s2_20261016/','Result/S2_20261016/'))
        assert PurePosixPath(name).suffix in {'.json','.csv','.md','.py','.log'}
        assert member.size==expected[name]['bytes'] and member.size<8*2**20
        dest=(ROOT/Path(name)).resolve();assert dest.is_relative_to(ROOT)
        blob=handle.extractfile(member).read()
        assert hashlib.sha256(blob).hexdigest()==expected[name]['sha256'],name
        if dest.exists():assert dest.read_bytes()==blob,name
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as target:target.write(blob)
dest=ROOT/'Result/S2_20261016/core_evidence_manifest.json'
blob=(TRANSFER/'core_evidence_manifest.json').read_bytes()
if dest.exists():assert dest.read_bytes()==blob
else:dest.write_bytes(blob)
print('S2_CORE_VERIFIED_AND_EXTRACTED',len(expected),'files; no videos or weights')
