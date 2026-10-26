"""S2 operational recovery only: at most two evaluators, stagger CUDA startup.

The scoring and statistical scripts remain byte-for-byte frozen. Skip GPU 0
after two observed transient PID collisions; all selected GPUs still require
vacancy checks. No downloads, environment changes, generation or S3.
"""
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
    pending=list(METRICS);active={};completed=[];last_launch=0;first_metric=None
    state={'phase':'evaluation_retry','max_parallel':2,'attempt':2,'pending':pending,
        'completed':completed,'pid':os.getpid(),'started':time.time(),
        'reason':'Six concurrent CUDA/model starts coincided with nvidia-smi 15-second query timeouts; zero scoring rows were committed.',
        'unchanged':'evaluate_s2.py, all videos, model weights, preprocessing, normalization and statistics',
        'source_sha256':file_hash(PROJECT/'tools/retry_s2_evaluation.py')}
    json_write(BATCH/'evaluation_retry_plan.json',state)
    try:
        while pending or active:
            assert time.time()-state['started']<2*3600,'Bounded evaluation retry exceeded 2 hours'
            for metric,(proc,log,gpu,started) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close()
                json_write(BATCH/('exit_eval_'+metric+'_attempt02.json'),
                    {'metric':metric,'gpu':gpu,'attempt':2,'timeout_pid':proc.pid,'exit_code':rc,'elapsed_seconds':time.time()-started})
                assert rc==0,'Evaluation retry failed: '+metric+' exit='+str(rc)
                completed.append(metric);del active[metric]
            # Establish one healthy, completed six-video chunk before the first
            # overlap. Thereafter never initialize two CUDA contexts together.
            first_healthy=(first_metric is None or first_metric in completed or
                bool(list((BATCH/'evaluation'/first_metric/'attempt02').glob('M*.json'))))
            if pending and len(active)<2 and time.time()-last_launch>=30 and first_healthy:
                snap=gpu_snapshot();reserved={v[2] for v in active.values()}
                available=[int(g[0]) for g in snap['gpus'] if int(g[0]) in range(1,7) and int(g[0]) not in reserved
                    and float(g[2])<=32 and float(g[4])==0 and not any(p[0]==g[1] for p in snap['processes'])]
                if available:
                    gpu=available[0];metric=pending.pop(0)
                    assert not (BATCH/'evaluation'/metric/'attempt02').exists()
                    log=(BATCH/'logs'/('evaluation_'+metric+'_attempt02.log')).open('xb')
                    cmd=['timeout','--signal=TERM','--kill-after=30s','60m',
                        '/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python','-B',
                        str(PROJECT/'tools/evaluate_s2.py'),'--metric',metric,'--gpu',str(gpu),'--attempt','2']
                    proc=subprocess.Popen(cmd,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    last_launch=time.time();active[metric]=(proc,log,gpu,last_launch)
                    if first_metric is None:first_metric=metric
                    json_write(BATCH/('launch_eval_'+metric+'_attempt02.json'),
                        {'metric':metric,'gpu':gpu,'attempt':2,'timeout_pid':proc.pid,'command':cmd,'time':last_launch})
                    print('EVAL_RETRY_LAUNCHED',metric,gpu,proc.pid,flush=True)
            state.update(pending=pending,completed=completed,
                running={m:{'gpu':v[2],'timeout_pid':v[0].pid} for m,v in active.items()})
            json_write(BATCH/'evaluation_controller.json',state)
            if pending or active:time.sleep(10)
        state['phase']='finalizing';json_write(BATCH/'evaluation_controller.json',state)
        with (BATCH/'logs/finalize.log').open('xb') as log:
            result=subprocess.run(['/data/chenjiayu/miniconda3/envs/changzihe/bin/python','-B',
                str(PROJECT/'tools/finalize_s2.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0,'Finalizer exit='+str(result.returncode)
        state.update(phase='completed',finished=time.time(),next_stage_started=False)
        json_write(BATCH/'evaluation_controller.json',state)
        print('S2_COMPLETE_NO_S3',flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
        json_write(BATCH/'evaluation_controller.json',state)
        raise


if __name__=='__main__':main()
