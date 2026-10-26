"""Read a bounded allowlist of S2-R core text artifacts for local synchronization."""
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/chenjiayu/changzihe/DistillPP_Validation')
BATCH=ROOT/'runs/s2r_20261017'
assert ROOT.resolve()==ROOT
assert json.loads((BATCH/'finalizer_status.json').read_text())['status']=='complete'
names=['statistics.json','generation_records.json','video_index.json','video_index.csv','semantic_token_metadata.json',
    'metrics_per_video.csv','metric_summary.csv','paired_metrics.csv','prompt_means.csv','timing_per_video.csv',
    'paired_timing.csv','timing_by_gpu.csv','load_and_warmup.csv','evaluation_timing.csv','failures.json','REPORT.md']
paths=[ROOT/'Result/S2R_20261017'/n for n in names]
paths += [BATCH/n for n in ['protocol.json','execution.json','preflight.json','token_acceptance.json',
    'analysis_freeze.json','first_pair_acceptance.json','runtime_policy.json','scheduler_handoff.json','supervisor_identity.json','supervisor_status.json','finalizer_identity.json','finalizer_status.json']]
for directory in sorted((BATCH/'workers').glob('*/attempt*')):
    for name in ('result.json','partial_result.json','failure.json'):
        if (directory/name).exists():paths.append(directory/name)
assert len(paths)<=100
records=[];total=0
for p in paths:
    assert p.resolve()==p and p.is_relative_to(ROOT) and p.suffix in ('.json','.csv','.md')
    assert p.stat().st_size<=8*2**20
    data=p.read_bytes();data.decode('utf-8');total+=len(data)
    records.append({'path':str(p.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
        'data':base64.b64encode(data).decode()})
assert total<=32*2**20
print(json.dumps({'status':'complete_core_only','total_bytes':total,'files':records}))
