"""Read own small progress records; no GPU queries and no computation."""
import json
from pathlib import Path
import time

b=Path('/data/chenjiayu/changzihe/DistillPP_Validation/runs/s2_20261016')
controller_path=b/'evaluation_controller.json'
controller=json.loads(controller_path.read_text()) if controller_path.exists() else {}
out={'time':time.time(), 'pairs':len(list((b/'pairs').glob('*/attempt*/pair_complete.json'))),
     'videos':len(list((b/'pairs').glob('*/attempt*/*/result.json'))), 'workers':[],'evaluation':[]}
for root in sorted((b/'workers').glob('worker*')):
    w=sorted(root.glob('attempt*'))[-1]
    r=json.loads((w/'progress.json').read_text()) if (w/'progress.json').exists() else {}
    row={'id':root.name,'pair':r.get('pair_id'),'arm':r.get('arm'),'step':r.get('step'),'phase':r.get('phase'),
         'age_s':round(time.time()-r.get('unix_time',time.time()),1)}
    if (w/'failure.json').exists():row['error']=json.loads((w/'failure.json').read_text())['error']
    if (w/'result.json').exists():row['phase']='passed'
    out['workers'].append(row)
for root in sorted((b/'evaluation').glob('*')):
    if controller.get('phase') in ['evaluation_retry','evaluation_resume'] and root.name in controller.get('pending',[]):
        out['evaluation'].append({'metric':root.name,'status':'pending_retry','n':0})
        continue
    w=sorted(root.glob('attempt*'))[-1]
    if (w/'result.json').exists():row={'metric':root.name,'status':'passed','n':96}
    elif (w/'failure.json').exists():row={'metric':root.name,'error':json.loads((w/'failure.json').read_text())['error']}
    else:
        r=json.loads((w/'progress.json').read_text()) if (w/'progress.json').exists() else {}
        row={'metric':root.name,'n':r.get('completed_videos',0),'phase':r.get('phase')}
    out['evaluation'].append(row)
f=b/'evaluation_controller.json'
if f.exists():
    r=json.loads(f.read_text());out['controller']=r['phase']
    if 'error' in r:out['controller_error']=r['error']
print(json.dumps(out))
