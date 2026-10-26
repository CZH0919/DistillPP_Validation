"""Operational S2 recovery on physically idle GPUs 6/5; at most two scorers."""
import json
import os
import subprocess
import time
import traceback

from s2_common import PROJECT,BATCH,METRICS,gpu_snapshot,json_write,file_hash


def main():
    freeze=json.loads((BATCH/'analysis_freeze.json').read_text())
    for rel,h in freeze['source_sha256'].items():assert file_hash(PROJECT/rel)==h
    assert len(list((BATCH/'pairs').glob('*/attempt*/pair_complete.json')))==48
    completed=[m for m in METRICS if list((BATCH/'evaluation'/m).glob('attempt*/result.json'))]
    pending=[m for m in ['dynamic_degree','motion_smoothness','aesthetic_quality','overall_consistency','imaging_quality','subject_consistency'] if m not in completed]
    active={};last_launch=0;first=None
    state={'phase':'evaluation_resume','started':time.time(),'pid':os.getpid(),'max_parallel':2,
        'pending':pending,'completed':completed,'reason':'GPU 0-4 and 7 occupied; resume only validated missing chunks on idle GPUs 6/5.',
        'source_sha256':{name:file_hash(PROJECT/'tools'/name) for name in ['resume_s2_evaluation.py','resume_s2_metric.py']}}
    json_write(BATCH/'evaluation_resume_plan.json',state)
    try:
        while pending or active:
            assert time.time()-state['started']<2*3600,'Bounded resume exceeded 2 hours'
            for metric,(proc,log,gpu,attempt,started) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close()
                json_write(BATCH/('exit_eval_%s_attempt%02d.json'%(metric,attempt)),
                    {'metric':metric,'gpu':gpu,'attempt':attempt,'timeout_pid':proc.pid,'exit_code':rc,'elapsed_seconds':time.time()-started})
                assert rc==0,'Resumed metric failed: '+metric+' exit='+str(rc)
                completed.append(metric);del active[metric]
            healthy=first is None or first in completed
            if not healthy and first in active:
                attempt=active[first][3];f=BATCH/'evaluation'/first/('attempt%02d'%attempt)/'progress.json'
                if f.exists():
                    r=json.loads(f.read_text());healthy=r.get('completed_videos',0)>r.get('reused_videos',0)
            if pending and len(active)<2 and time.time()-last_launch>=30 and healthy:
                snap=gpu_snapshot();reserved={v[2] for v in active.values()}
                available=[]
                for gpu in [6,5]:
                    g=next(g for g in snap['gpus'] if int(g[0])==gpu)
                    if gpu not in reserved and float(g[2])<=32 and float(g[4])==0 and not any(p[0]==g[1] for p in snap['processes']):available.append(gpu)
                if available:
                    gpu=available[0];metric=pending.pop(0)
                    attempt=1+max(int(p.name.replace('attempt','')) for p in (BATCH/'evaluation'/metric).glob('attempt*'))
                    log=(BATCH/'logs'/('evaluation_%s_attempt%02d.log'%(metric,attempt))).open('xb')
                    cmd=['timeout','--signal=TERM','--kill-after=30s','60m',
                        '/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python','-B',str(PROJECT/'tools/resume_s2_metric.py'),
                        '--metric',metric,'--gpu',str(gpu),'--attempt',str(attempt)]
                    proc=subprocess.Popen(cmd,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    last_launch=time.time();active[metric]=(proc,log,gpu,attempt,last_launch)
                    if first is None:first=metric
                    json_write(BATCH/('launch_eval_%s_attempt%02d.json'%(metric,attempt)),
                        {'metric':metric,'gpu':gpu,'attempt':attempt,'timeout_pid':proc.pid,'command':cmd,'time':last_launch})
                    print('EVAL_RESUME_LAUNCHED',metric,gpu,attempt,proc.pid,flush=True)
            state.update(pending=pending,completed=completed,running={m:{'gpu':v[2],'attempt':v[3],'timeout_pid':v[0].pid} for m,v in active.items()})
            json_write(BATCH/'evaluation_controller.json',state)
            if pending or active:time.sleep(10)
        state['phase']='finalizing';json_write(BATCH/'evaluation_controller.json',state)
        with (BATCH/'logs/finalize.log').open('xb') as log:
            r=subprocess.run(['/data/chenjiayu/miniconda3/envs/changzihe/bin/python','-B',str(PROJECT/'tools/finalize_s2.py')],
                cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert r.returncode==0,'Finalizer exit='+str(r.returncode)
        state.update(phase='completed',finished=time.time(),next_stage_started=False)
        json_write(BATCH/'evaluation_controller.json',state)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
        json_write(BATCH/'evaluation_controller.json',state)
        raise


if __name__=='__main__':main()
