"""Wait for this H1 pair, then score five dimensions serially on GPU 6."""
import json
import os
import subprocess
import time
from hya_common import ROOT,RUN,METRICS,write_json

PY='/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python'

def main():
    deadline=time.monotonic()+14400
    while not (RUN/'generation_controller_done.json').exists():
        status=json.loads((RUN/'generation_controller.json').read_text())
        if status['status']=='failed':
            raise RuntimeError('Generation failed; evaluation was not started')
        if time.monotonic()>deadline:raise RuntimeError('Bounded H1 generation wait expired')
        time.sleep(15)
    assert json.loads((RUN/'generation_controller_done.json').read_text())['status']=='passed'
    assert json.loads((RUN/'eval_cpu_preflight/attempt02/result.json').read_text())['status']=='passed'
    results=[]
    for metric in METRICS:
        command=[PY,'-B',str(ROOT/'tools/hya_evaluate_smoke.py'),'--metric',metric,'--attempt','1']
        log=RUN/('eval_'+metric+'_attempt01.log');assert not log.exists()
        start=time.time()
        with log.open('x') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
            write_json(RUN/'evaluation_controller.json',{'status':'running','metric':metric,'pid':proc.pid,'start_unix':start},replace=True)
            try:rc=proc.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait(timeout=30);raise
        results.append({'metric':metric,'returncode':rc,'wall_seconds':time.time()-start})
        if rc:
            write_json(RUN/'evaluation_controller.json',{'status':'failed','results':results},replace=True)
            raise SystemExit(rc)
    write_json(RUN/'evaluation_controller_done.json',{'status':'passed','results':results,'scope':'H1 two-video acceptance; 10 scores, not formal qualification'})
    write_json(RUN/'evaluation_controller.json',{'status':'passed','results':results},replace=True)

if __name__=='__main__':
    main()
