"""One bounded S2 pipeline continuation: wait for generation, evaluate, finalize.

No task scheduling/recurrence, installation, downloads, process termination or S3.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from s2_common import PROJECT,BATCH,METRICS,gpu_snapshot,json_write,file_hash


def main():
    start=time.time()
    state={'phase':'waiting_for_generation','started':start,'pid':os.getpid()}
    path=BATCH/'evaluation_controller.json'
    processes={};completed=[]
    pending=list(METRICS)
    try:
        # A failure requires explicit diagnosis/retry, never silently skip a pair.
        while len(list((BATCH/'pairs').glob('*/attempt*/pair_complete.json')))<48:
            assert time.time()-start<3*3600, 'Generation wait exceeded 3 hours'
            for root in (BATCH/'workers').glob('worker*'):
                latest=sorted(root.glob('attempt*'))[-1]
                assert not (latest/'failure.json').exists(), 'Worker failed; continuation pauses: '+str(latest)
            json_write(path,state)
            time.sleep(20)
        state.update(phase='evaluation',generation_complete=time.time())
        freeze=json.loads((BATCH/'analysis_freeze.json').read_text())
        for rel,h in freeze['source_sha256'].items():assert file_hash(PROJECT/rel)==h
        while pending or processes:
            assert time.time()-state['generation_complete']<90*60, 'Evaluation budget exceeded'
            for metric,(proc,log,gpu) in list(processes.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close()
                assert rc==0, 'Evaluation failed: '+metric+' exit='+str(rc)
                completed.append(metric);del processes[metric]
            if pending:
                snapshot=gpu_snapshot()
                reserved={v[2] for v in processes.values()}
                free=[int(g[0]) for g in snapshot['gpus'] if int(g[0]) not in reserved and
                    float(g[2])<=32 and float(g[4])==0 and
                    not any(p[0]==g[1] for p in snapshot['processes'])]
                for gpu in free:
                    if not pending:break
                    metric=pending.pop(0)
                    # Each evaluator independently checks vacancy twice before CUDA.
                    log=(BATCH/'logs'/('evaluation_'+metric+'_attempt01.log')).open('xb')
                    cmd=['timeout','--signal=TERM','--kill-after=30s','60m',
                        '/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python','-B',
                        str(PROJECT/'tools/evaluate_s2.py'),'--metric',metric,'--gpu',str(gpu)]
                    proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT)
                    processes[metric]=(proc,log,gpu)
                    json_write(BATCH/('launch_eval_'+metric+'_attempt01.json'),
                        {'metric':metric,'gpu':gpu,'timeout_pid':proc.pid,'command':cmd,'time':time.time()})
                    print('EVAL_LAUNCHED',metric,gpu,proc.pid,flush=True)
            state.update(pending=pending,completed=completed,running={m:{'timeout_pid':v[0].pid,'gpu':v[2]} for m,v in processes.items()})
            json_write(path,state)
            if processes or pending:time.sleep(15)
        state.update(phase='finalizing');json_write(path,state)
        log=(BATCH/'logs/finalize.log').open('xb')
        try:
            result=subprocess.run(['/data/chenjiayu/miniconda3/envs/changzihe/bin/python','-B',
                str(PROJECT/'tools/finalize_s2.py')],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        finally:log.close()
        assert result.returncode==0, 'Finalizer exit='+str(result.returncode)
        state.update(phase='completed',finished=time.time(),next_stage_started=False)
        json_write(path,state)
        print('S2_PIPELINE_COMPLETED_NO_S3',flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
        json_write(path,state)
        # Running independent metric workers may finish; do not kill them or
        # any other process, and never create duplicate retries here.
        raise


if __name__=='__main__':main()
