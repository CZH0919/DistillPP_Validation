"""Detached controller for the reviewed correction protocol; select currently idle GPUs."""
import fcntl
import os
import signal
import subprocess
import time
import traceback

import hya_h2_common as h2
import hya_h3_timing_common as c


def select_idle_gpus(snapshot, active_gpus, cooldowns, conflicts, now):
    available = [gpu for gpu in h2.free_indices(snapshot)
                 if gpu not in active_gpus and now >= cooldowns.get(str(gpu), 0)]
    return sorted(available, key=lambda gpu: (conflicts.get(str(gpu), 0), gpu))


def launch_controller():
    c.check_frozen()
    if (c.RUN/"controller.json").exists():
        old = c.read(c.RUN/"controller.json")
        assert not c.alive(old["pid"], old["birth"]), "Controller is already alive"
        assert old["status"] != "completed", "This batch is complete"
        assert not any(c.alive(a["pid"], a["birth"]) for t in old["tasks"].values()
                       for a in t["attempts"][-1:]), "A previous worker is still alive"
    with (c.RUN/"controller.log").open("ab",buffering=0) as log:
        p=subprocess.Popen([c.GEN_PY,"-B",str(c.ROOT/"tools/hya_h3_timing_controller.py")],
            cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"))
    receipt={"pid":p.pid,"birth":c.ticks(p.pid),"launched_unix":time.time(),"detached":True}
    c.write_json(c.RUN/"launch_receipt.json",receipt,replace=True)
    print(receipt,flush=True)


def main():
    _,p=c.check_frozen()
    definitions=[{"id":"gen_"+pid,"kind":"gen","key":pid} for pid in p["prompt_ids"]]
    definitions += [{"id":"eval_"+m,"kind":"eval","key":m} for m in p["metrics"]]
    state={d["id"]:{"status":"pending","attempts":[],"not_before":0} for d in definitions}
    cooldowns, conflicts = {}, {}
    if (c.RUN/"controller.json").exists():
        old = c.read(c.RUN/"controller.json")
        state = old["tasks"]
        cooldowns, conflicts = old.get("gpu_cooldowns", {}), old.get("gpu_conflicts", {})
        for task in state.values():
            if task["status"] != "passed":
                task.update(status="pending", not_before=0)
                task.pop("ordinary_failure", None)
    start=time.time();workers={}
    def save(status,**extra):
        c.write_json(c.RUN/"controller.json",{"status":status,"pid":os.getpid(),"birth":c.ticks(os.getpid()),
            "session_id":os.getsid(0),"started_unix":start,"updated_unix":time.time(),"tasks":state,
            "gpu_cooldowns":cooldowns,"gpu_conflicts":conflicts,
            **extra},replace=True)
    def complete(d):
        return c.generation_complete(p,d["key"]) if d["kind"]=="gen" else (c.RUN/"scores"/(d["key"]+".json")).exists()
    def ready(d):
        if d["kind"]=="gen":
            return True
        return all(c.generation_complete(p,pid) for pid in p["prompt_ids"])
    for d in definitions:
        if complete(d):
            state[d["id"]]["status"]="passed"
    save("starting")
    try:
        while True:
            for tid,(proc,d,rec) in list(workers.items()):
                elapsed=time.time()-rec["started_unix"]
                if proc.poll() is None and elapsed>(10800 if d["kind"]=="gen" else 7200):
                    assert c.alive(proc.pid,rec["birth"])
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        if c.alive(proc.pid,rec["birth"]):proc.kill()
                        proc.wait(timeout=10)
                    rec["own_worker_timeout"]=True
                if proc.poll() is None:continue
                rec.update(returncode=proc.returncode,finished_unix=time.time(),wall_seconds=elapsed)
                out=c.RUN/("generation" if d["kind"]=="gen" else "evaluation")/d["key"]/f"attempt{rec['attempt']:02d}"
                failure=c.read(out/"failure.json") if (out/"failure.json").exists() else {}
                if complete(d) and (proc.returncode==0 or failure.get("status")=="conflict"):
                    state[tid]["status"]="passed"
                elif proc.returncode==75:
                    # Preserve committed arms; avoid repeatedly loading on a contested card.
                    key = str(rec["gpu"])
                    conflicts[key] = conflicts.get(key, 0) + 1
                    cooldowns[key] = time.time() + (600 if conflicts[key] == 1 else 1800)
                    state[tid]["status"]="pending"
                    state[tid]["not_before"]=time.time()+30
                    state[tid]["error"]=failure.get("error","resource conflict")
                else:
                    state[tid]["status"]="paused"
                    state[tid]["error"]=failure.get("error","worker failed; inspect before continuing")
                    state[tid]["ordinary_failure"]=True
                del workers[tid]
            if all(t["status"]=="passed" for t in state.values()):
                save("finalizing")
                with (c.RUN/"finalize.log").open("ab",buffering=0) as log:
                    rc=subprocess.call([c.GEN_PY,"-B",str(c.ROOT/"tools/hya_h3_timing_finalize.py")],
                        cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"),
                        timeout=300)
                save("completed" if rc==0 else "finalization_failed",finalize_returncode=rc)
                return
            ordinary=any(t.get("ordinary_failure") for t in state.values())
            if ordinary and not workers:
                save("needs_attention");return
            expired = time.time()-start > p["max_controller_seconds"]
            if expired and not workers:
                save("needs_attention",reason="Configured controller window ended");return
            try:
                snap=h2.gpu_snapshot()
                free=select_idle_gpus(snap,{v[2]["gpu"] for v in workers.values()},
                                      cooldowns,conflicts,time.time())
            except (h2.ResourceConflict,subprocess.TimeoutExpired) as exc:
                save("waiting_resource_query",error=repr(exc));time.sleep(30);continue
            if not ordinary and not expired:
                for d in definitions:
                    s=state[d["id"]]
                    if len(workers)>=p["max_workers"] or not free:break
                    if s["status"]!="pending" or time.time()<s["not_before"] or not ready(d):continue
                    if h2.host_available_gib()<256+160*(len(workers)+1):break
                    gpu=free.pop(0)
                    attempt=len(s["attempts"])+1
                    logpath=c.RUN/"logs"/f"{d['id']}_attempt{attempt:02d}.log"
                    python=c.GEN_PY if d["kind"]=="gen" else c.EVAL_PY
                    command=[python,"-B",str(c.ROOT/"tools/hya_h3_timing_worker.py"),
                             "--prompt" if d["kind"]=="gen" else "--metric",d["key"],
                             "--gpu",str(gpu),"--attempt",str(attempt)]
                    with logpath.open("ab",buffering=0) as log:
                        proc=subprocess.Popen(command,cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True,close_fds=True,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"))
                    rec={"pid":proc.pid,"birth":c.ticks(proc.pid),"gpu":gpu,"attempt":attempt,
                         "started_unix":time.time(),"command":command,"log_path":str(logpath)}
                    s["attempts"].append(rec);s["status"]="running";workers[d["id"]]=(proc,d,rec)
                    print("H3_DISPATCH",d["id"],"GPU",gpu,"PID",proc.pid,flush=True)
            if not workers and any(t["status"]=="paused" for t in state.values()):
                if not any(s["status"]=="pending" and ready(d) for d in definitions for s in [state[d["id"]]]):
                    save("needs_attention");return
            save("running" if workers else "waiting_for_idle_resources",gpu_snapshot=snap)
            time.sleep(15)
    except BaseException as exc:
        save("controller_failed",error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=="__main__":
    import sys
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    if "--launch" in sys.argv:launch_controller()
    else:
        with (c.RUN/"controller.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
