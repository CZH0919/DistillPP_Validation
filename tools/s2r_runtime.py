"""Runtime-only resource policy and safe adoption of existing S2-R workers."""
import json
import os
from pathlib import Path
import time


def process_info(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'uid':p.stat().st_uid,'state':stat[0],'start_ticks':int(stat[19]),
            'cwd':str((p/'cwd').resolve(strict=True)),
            'args':[(x.decode()) for x in (p/'cmdline').read_bytes().split(b'\0') if x]}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


class AdoptedProcess:
    def __init__(self,pid,start_ticks,jobroot):
        self.pid=pid;self.start_ticks=start_ticks;self.jobroot=jobroot
    def poll(self):
        p=process_info(self.pid)
        if p and p['start_ticks']==self.start_ticks and p['state']!='Z':return None
        if (self.jobroot/'result.json').exists():return 0
        if (self.jobroot/'failure.json').exists():return 1
        return 255  # Unexplained disappearance is retained in controller history.


def recover_jobs(project,batch):
    p=batch/'supervisor_status.json'
    previous=json.loads(p.read_text()) if p.exists() else {}
    old_active={r['job']:r for r in previous.get('active',[])}
    active={};history=previous.get('history',[])
    allowed={str(project/'tools/run_s2r_worker.py'):'generation',str(project/'tools/evaluate_s2r.py'):'evaluation'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid() or (proc/'cwd').resolve(strict=True)!=project:continue
        except (OSError,RuntimeError):continue
        info=process_info(int(proc.name))
        if not info or info['state']=='Z':continue
        args=info['args'];matches=[s for s in allowed if s in args]
        if not matches:continue
        assert len(matches)==1
        kind=allowed[matches[0]]
        value=lambda flag:args[args.index(flag)+1]
        identifier=value('--worker' if kind=='generation' else '--metric')
        gpu=int(value('--gpu'));attempt=int(value('--attempt'));key=kind+':'+identifier
        assert key not in active,'Multiple live processes for '+key
        jobroot=(batch/'workers'/('worker%02d'%int(identifier)) if kind=='generation' else batch/'evaluation'/identifier)/('attempt%02d'%attempt)
        old=old_active.get(key,{})
        started=old.get('unix_start') if old.get('pid')==info['pid'] else (jobroot.stat().st_mtime if jobroot.exists() else time.time())
        active[key]={'process':AdoptedProcess(info['pid'],info['start_ticks'],jobroot),'log':None,
            'job':key,'pid':info['pid'],'gpu':gpu,'attempt':attempt,'unix_start':started,
            'adopted':True,'start_ticks':info['start_ticks']}
    return active,history


def eligible_gpus(snapshot,used,allowed,idle_since,conflicts,now,stable_seconds,cooldown_seconds,deprioritize):
    free=[]
    for g in snapshot['gpus']:
        index=int(g[0]);idle=(index in allowed and index not in used and float(g[2])<=32 and float(g[4])==0
            and not any(p[0]==g[1] for p in snapshot['processes']))
        if not idle:
            idle_since.pop(index,None);continue
        first=idle_since.setdefault(index,now)
        if now-first<stable_seconds:continue
        if now-conflicts.get(index,{}).get('last',float('-inf'))<cooldown_seconds:continue
        free.append(index)
    return sorted(free,key=lambda i:(i in deprioritize,conflicts.get(i,{}).get('count',0),i))


class RuntimePolicy:
    def __init__(self,batch):self.batch=batch;self.idle_since={};self.data={};self.conflicts={}
    def candidates(self,snapshot,active,history):
        q=json.loads((self.batch/'runtime_policy.json').read_text())
        assert q['stage']=='S2_R_ONLY' and isinstance(q['max_concurrent'],int) and 1<=q['max_concurrent']<=3
        allowed=q['allowed_gpu_indexes'];assert allowed and len(allowed)==len(set(allowed))
        assert all(isinstance(x,int) and 0<=x<=7 for x in allowed)
        assert q['stable_idle_seconds']>=60 and q['conflict_cooldown_seconds']>=900
        self.data=q;conflicts={}
        for p in list((self.batch/'workers').glob('*/attempt*/failure.json'))+list((self.batch/'evaluation').glob('*/attempt*/failure.json')):
            r=json.loads(p.read_text());error=r.get('error','')
            if 'Another process appeared' not in error and 'Selected GPU is occupied' not in error:continue
            gpu=r.get('admission',{}).get('physical_index')
            if gpu is None:continue
            c=conflicts.setdefault(gpu,{'count':0,'last':0})
            c['count']+=1;c['last']=max(c['last'],p.stat().st_mtime)
        self.conflicts=conflicts
        return eligible_gpus(snapshot,{j['gpu'] for j in active.values()},allowed,self.idle_since,conflicts,
            time.time(),q['stable_idle_seconds'],q['conflict_cooldown_seconds'],q['deprioritize_gpu_indexes'])
    @property
    def max_concurrent(self):return self.data['max_concurrent']


def self_test():
    snapshot={'gpus':[[str(i),'uuid'+str(i),'0','143771','0'] for i in range(8)],'processes':[]}
    idle={};conflicts={0:{'count':2,'last':900},1:{'count':2,'last':900}}
    args=(snapshot,set(),[6],idle,conflicts)
    assert eligible_gpus(*args,1000,60,900,[0,1])==[]
    assert eligible_gpus(*args,1060,60,900,[0,1])==[6]
    assert eligible_gpus(snapshot,{6},[6],idle,conflicts,1061,60,900,[0,1])==[]
    assert 6 not in idle
    idle={i:0 for i in range(4)}
    assert eligible_gpus(snapshot,set(),[0,1,2,3],idle,conflicts,1000,60,900,[0,1])==[2,3]
    assert eligible_gpus(snapshot,set(),[0,1,2,3],idle,conflicts,2000,60,900,[0,1])==[2,3,0,1]
    snapshot['processes']=[['uuid2','123','1']]
    assert eligible_gpus(snapshot,set(),[2,3],idle,{},2001,60,900,[])==[3]
    print('S2R_RUNTIME_POLICY_TESTS_PASS')


if __name__=='__main__':self_test()
