"""Detached H1-only serial launcher. Never launches the formal qualification panel."""
import json
import os
import subprocess
import sys
import time
from hya_common import ROOT,RUN,write_json

PY='/data/chenjiayu/miniconda3/envs/changzihe/bin/python'

def main():
    assert ROOT.resolve()==ROOT and RUN.resolve()==RUN
    assert not (RUN/'generation_controller_done.json').exists()
    results=[]
    for arm in ['S5','T50']:
        log=RUN/(arm+'_attempt01.log')
        assert not log.exists(), 'Inspect prior log before starting another attempt'
        command=[PY,'-B',str(ROOT/'tools/hya_generate_smoke.py'),'--arm',arm,'--attempt','1']
        start=time.time()
        with log.open('x') as f:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
            write_json(RUN/'generation_controller.json',{'status':'running','controller_pid':os.getpid(),
                       'arm':arm,'worker_pid':p.pid,'start_unix':start,'command':command},replace=True)
            try:
                rc=p.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                p.terminate();p.wait(timeout=45);raise
        result={'arm':arm,'returncode':rc,'wall_seconds':time.time()-start}
        results.append(result)
        if rc:
            write_json(RUN/'generation_controller.json',{'status':'failed','results':results},replace=True)
            raise SystemExit(rc)
        assert json.loads((RUN/'generation'/arm/'attempt01/result.json').read_text())['status']=='passed'
    write_json(RUN/'generation_controller_done.json',{'status':'passed','results':results,'scope':'H1 only; two videos'})
    write_json(RUN/'generation_controller.json',{'status':'passed','results':results},replace=True)

if __name__=='__main__':
    main()
