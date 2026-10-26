"""Read only compact metadata of this S2-R batch; no GPU initialization."""
import json
from pathlib import Path
import time

B=Path('/data/chenjiayu/changzihe/DistillPP_Validation/runs/s2r_20261017')
def read(p): return json.loads(p.read_text())
status=read(B/'supervisor_status.json') if (B/'supervisor_status.json').exists() else {}
progress=[]
for job in status.get('active',[]):
    if job['job'].startswith('generation:'):
        root=B/'workers'/('worker%02d'%int(job['job'].split(':')[1]))
    else:root=B/'evaluation'/job['job'].split(':')[1]
    p=root/('attempt%02d'%job['attempt'])/'progress.json'
    progress.append(dict(job,progress=read(p) if p.exists() else None))
failures=[{'path':str(p.relative_to(B)),'error':read(p).get('error')} for p in sorted(B.glob('**/failure.json'))]
print(json.dumps({'status':status.get('status'),'unix_time':time.time(),'controller_update':status.get('unix_time'),
    'accepted_pairs':len(list((B/'pairs').glob('*/attempt*/pair_complete.json'))),
    'successful_video_results_including_unpaired':len(list((B/'pairs').glob('*/attempt*/*/result.json'))),
    'completed_metrics':status.get('completed_metrics',[]),'active':progress,'failures':failures,
    'paused_jobs':status.get('paused_jobs',[]),'history':status.get('history',[])}))
