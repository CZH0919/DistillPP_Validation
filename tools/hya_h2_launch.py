"""Start the authorized H2 supervisor detached from SSH/terminal; idempotent if alive."""
import os
from pathlib import Path
import subprocess
import time

from hya_h2_common import ROOT, RUN, GEN_PY, read, write_json, check_frozen
from hya_h2_controller import process_alive, process_ticks


def main():
    manifest, _ = check_frozen()
    assert manifest['authorization']['H2'] and not manifest['authorization']['S3']
    if (RUN/'controller.json').exists():
        old = read(RUN/'controller.json')
        if process_alive(old['controller_pid'], old['controller_start_ticks']):
            print('H2_ALREADY_RUNNING', old['controller_pid']); return
        assert old['status'] not in ['completed', 'needs_attention', 'finalization_failed'], 'Inspect terminal state before explicit recovery'
    with (RUN/'controller.log').open('a') as log:
        child = subprocess.Popen([GEN_PY, '-B', str(ROOT/'tools/hya_h2_controller.py')], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                                          OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    time.sleep(3)
    assert child.poll() is None, 'Controller exited during startup; inspect controller.log'
    receipt = {'controller_pid': child.pid, 'start_ticks': process_ticks(child.pid),
               'session_id': os.getsid(child.pid), 'launcher_pid': os.getpid(),
               'stdin': 'DEVNULL', 'stdout_stderr': str(RUN/'controller.log'),
               'start_new_session': True, 'launched_unix': time.time()}
    assert receipt['session_id'] == child.pid
    write_json(RUN/('launch_'+str(child.pid)+'.json'), receipt)
    print('H2_DETACHED_STARTED', receipt, flush=True)


if __name__ == '__main__':
    main()
