"""Finish authorized S2-R validation/archiving once all GPU jobs have exited."""
import fcntl
import json
import os
import subprocess
import time
from s2r_common import PROJECT,BATCH,json_write

def main():
    assert BATCH.resolve()==BATCH
    lock=(BATCH/'finalizer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    json_write(BATCH/'finalizer_identity.json',{'pid':os.getpid(),'unix_start':time.time(),'cwd':str(PROJECT)})
    while True:
        p=BATCH/'supervisor_status.json'
        status=json.loads(p.read_text()) if p.exists() else {}
        if status.get('status')=='compute_complete':break
        if status.get('paused_jobs') and not status.get('active'):
            json_write(BATCH/'finalizer_status.json',{'status':'needs_attention_before_finalization',
                'paused_jobs':status['paused_jobs'],'unix_time':time.time(),'s3_started':False})
            return
        time.sleep(30)
    assert status['accepted_pairs']==96 and len(status['completed_metrics'])==6
    json_write(BATCH/'finalizer_status.json',{'status':'validating_archiving_analyzing','unix_time':time.time(),'s3_started':False})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    with (BATCH/'finalization.log').open('x') as log:
        p=subprocess.run(['/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python','-B',str(PROJECT/'tools/finalize_s2r.py')],
            cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    result={'status':'complete' if p.returncode==0 else 'finalization_failed','exit_code':p.returncode,
        'unix_time':time.time(),'report_path':str(PROJECT/'Result/S2R_20261017/REPORT.md'),'s3_started':False}
    if p.returncode==0:
        report=json.loads((PROJECT/'Result/S2R_20261017/statistics.json').read_text())
        result['qualification_ready']=report['qualification_ready']
        result['integrity']=report['integrity']
    json_write(BATCH/'finalizer_status.json',result)

if __name__=='__main__':main()
