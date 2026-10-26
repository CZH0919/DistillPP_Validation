"""Own-project bounded S2-R generation/evaluation supervisor; never starts S3."""
import fcntl
import json
import os
import subprocess
import sys
import time
from s2r_common import PROJECT,BATCH,gpu_snapshot,json_write,load_frozen,METRICS

GEN='/data/chenjiayu/miniconda3/envs/changzihe/bin/python'
EVAL='/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python'


def main():
    assert json.loads((BATCH/'preflight.json').read_text())['status']=='passed'
    execution,_=load_frozen()
    lock=(BATCH/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={}; history=[]
    logs=BATCH/'launch_logs';logs.mkdir(exist_ok=True)
    json_write(BATCH/'supervisor_identity.json',{'pid':os.getpid(),'unix_start':time.time(),'cwd':str(PROJECT)})
    while True:
        for key,job in list(active.items()):
            rc=job['process'].poll()
            if rc is not None:
                job['log'].close()
                history.append({k:v for k,v in job.items() if k not in ('process','log') } | {'exit_code':rc,'unix_end':time.time()})
                del active[key]
        accepted={p.parent.parent.name for p in (BATCH/'pairs').glob('*/attempt*/pair_complete.json')}
        completed_metrics={m for m in METRICS if list((BATCH/'evaluation'/m).glob('attempt*/result.json'))}
        jobs=[]
        for w in range(execution['worker_count']):
            expected={p['pair_id'] for p in execution['pairs'] if p['worker_id']==w}
            if expected-accepted: jobs.append(('generation',str(w)))
        if len(accepted)==96:
            jobs.extend(('evaluation',m) for m in METRICS if m not in completed_metrics)
        if not jobs and not active:
            json_write(BATCH/'supervisor_status.json',{'status':'compute_complete','accepted_pairs':96,
                'completed_metrics':sorted(completed_metrics),'history':history,'unix_time':time.time(),'s3_started':False})
            return
        try:
            snap=gpu_snapshot()
        except Exception as exc:
            json_write(BATCH/'supervisor_status.json',{'status':'gpu_query_unavailable_no_new_launch',
                'error':repr(exc),'accepted_pairs':len(accepted),'completed_metrics':sorted(completed_metrics),
                'active':[{k:v for k,v in j.items() if k not in ('process','log')} for j in active.values()],
                'history':history,'unix_time':time.time(),'s3_started':False})
            time.sleep(20)
            continue
        used={j['gpu'] for j in active.values()}
        free=[int(g[0]) for g in snap['gpus'] if int(g[0]) not in used and float(g[2])<=32 and float(g[4])==0
            and not any(p[0]==g[1] for p in snap['processes'])]
        paused=[]
        for kind,identifier in jobs:
            key=kind+':'+identifier
            if key in active: continue
            jobroot=(BATCH/'workers'/('worker%02d'%int(identifier))) if kind=='generation' else BATCH/'evaluation'/identifier
            attempts=sorted(jobroot.glob('attempt*'))
            failures=[p for p in attempts if (p/'failure.json').exists()]
            # Retry transient occupancy/query failures; stop this job on a science/runtime error.
            reasons=[json.loads((p/'failure.json').read_text()).get('error','') for p in failures]
            transient=lambda e:any(x in e for x in ('Another process appeared','Selected GPU is occupied','TimeoutExpired','Occupancy/progress monitor failed'))
            if len(failures)>=4 or any(not transient(e) for e in reasons):
                paused.append({'job':key,'errors':reasons}); continue
            if len(active)>=3 or not free: continue
            gpu=free.pop(0)
            attempt=max([int(p.name.removeprefix('attempt')) for p in attempts]+[0])+1
            name=key.replace(':','_')+'_attempt%02d'%attempt
            log=(logs/(name+'.log')).open('x')
            tool='run_s2r_worker.py' if kind=='generation' else 'evaluate_s2r.py'
            command=[GEN if kind=='generation' else EVAL,'-B','-u',str(PROJECT/'tools'/tool),
                '--gpu',str(gpu),'--attempt',str(attempt),'--worker' if kind=='generation' else '--metric',identifier]
            child=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]={'process':child,'log':log,'job':key,'pid':child.pid,'gpu':gpu,'attempt':attempt,'unix_start':time.time()}
            print('S2R_LAUNCHED',key,child.pid,'GPU',gpu,'attempt',attempt,flush=True)
            # Stagger model reads; each child still independently verifies vacancy twice.
            time.sleep(5)
        json_write(BATCH/'supervisor_status.json',{'status':'running' if active else 'waiting_or_paused',
            'accepted_pairs':len(accepted),'completed_metrics':sorted(completed_metrics),
            'active':[{k:v for k,v in j.items() if k not in ('process','log')} for j in active.values()],
            'paused_jobs':paused,'history':history,'unix_time':time.time(),'s3_started':False})
        if paused and not active:
            print('S2R_ATTENTION_REQUIRED',json.dumps(paused),flush=True)
            return
        time.sleep(20)


if __name__=='__main__': main()
