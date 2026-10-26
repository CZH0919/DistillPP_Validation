"""Bounded read-only status for this S2 batch; no process control."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--wait',type=int,default=0)
args=parser.parse_args()
assert 0<=args.wait<=50
if args.wait:time.sleep(args.wait)
b=Path('/data/chenjiayu/changzihe/DistillPP_Validation/runs/s2_20261016')
out={'time':time.time(),'complete_pairs':len(list((b/'pairs').glob('*/attempt*/pair_complete.json'))),
     'generated_video_records':len(list((b/'pairs').glob('*/attempt*/*/result.json'))),
     'disk_free_gib':round(shutil.disk_usage(b).free/2**30,2),'workers':[], 'evaluation':[]}
for root in sorted((b/'workers').glob('worker*')):
    w=sorted(root.glob('attempt*'))[-1]
    row={'worker':root.name,'attempt':w.name}
    f=w/'failure.json'
    if f.exists():
        r=json.loads(f.read_text());row.update(status='failed',error=r['error'])
    elif (w/'result.json').exists():row['status']='passed'
    elif (w/'progress.json').exists():
        r=json.loads((w/'progress.json').read_text())
        row.update({k:r.get(k) for k in ['phase','pair_id','arm','step','warmup','pid']})
        row['age_s']=round(time.time()-r['unix_time'],1)
    else:row['status']='starting'
    out['workers'].append(row)
for root in sorted((b/'evaluation').glob('*')):
    attempts=sorted(root.glob('attempt*'))
    if not attempts:continue
    w=attempts[-1];row={'metric':root.name,'attempt':w.name}
    if (w/'failure.json').exists():row.update(status='failed',error=json.loads((w/'failure.json').read_text())['error'])
    elif (w/'result.json').exists():row.update(status='passed',n=96)
    elif (w/'progress.json').exists():
        r=json.loads((w/'progress.json').read_text());row.update(phase=r.get('phase'),completed=r.get('completed_videos'))
    out['evaluation'].append(row)
out['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
print(json.dumps(out,ensure_ascii=False))
