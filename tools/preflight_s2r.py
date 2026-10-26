"""Bounded CPU-only S2-R resource/contract check and immutable batch setup."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from s2r_common import PROJECT,BATCH,validate_protocol,file_hash,json_write,gpu_snapshot,load_frozen

assert PROJECT.resolve()==PROJECT and BATCH.resolve()==BATCH
q=json.loads((PROJECT/'Project_Plan/QUALIFICATION_V2.json').read_text())
validate_protocol(q)
execution=json.loads((PROJECT/'Project_Plan/S2R_EXECUTION.json').read_text())
for rel,digest in execution['input_sha256'].items(): assert file_hash(PROJECT/rel)==digest,rel
old=json.loads((PROJECT/'Project_Progress/Stage_Progress/S1_PREFLIGHT.json').read_text())
for row in old['generation_files']:
    s=Path(row['path']).stat()
    assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),row['path']
manifest=json.loads((PROJECT/'Project_Plan/RESOURCE_MANIFEST.json').read_text())
evaluated=[]
for row in manifest['assets']:
    if row['group']!='evaluation': continue
    p=PROJECT/row['path']
    assert p.resolve()==p and p.stat().st_size==row['bytes']
    digest=file_hash(p); assert digest==row['sha256'],row['id']
    evaluated.append({'id':row['id'],'path':str(p),'sha256':digest,'bytes':p.stat().st_size})
review=json.loads((PROJECT/'Project_Progress/Stage_Progress/BASELINE_AND_PROMPT_REVIEW_20261017.json').read_text())
source={r:d for r,d in review['source_sha256'].items() if r.startswith('src/')}
for rel,digest in source.items(): assert file_hash(PROJECT/rel)==digest,rel
mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')}
assert mem['MemAvailable']>300*2**20 and shutil.disk_usage(PROJECT).free>30*2**30
envs={}
code="import importlib.metadata as m,json; names=['torch','numpy','transformers','decord','opencv-python','opencv-python-headless','imageio']; installed={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}; print(json.dumps({n:installed.get(n) for n in names}))"
for name in ('changzihe','changzihe_eval'):
    p=subprocess.run(['/data/chenjiayu/miniconda3/envs/'+name+'/bin/python','-B','-c',code],capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
    envs[name]=json.loads(p.stdout)
    assert all(envs[name][n] for n in ('torch','numpy','decord','imageio'))
BATCH.mkdir(exist_ok=True)
for src,dest in [('Project_Plan/QUALIFICATION_V2.json','protocol.json'),('Project_Plan/S2R_EXECUTION.json','execution.json')]:
    target=BATCH/dest
    if target.exists(): assert file_hash(target)==file_hash(PROJECT/src)
    else:
        with target.open('xb') as out: out.write((PROJECT/src).read_bytes())
load_frozen()
report={'status':'passed','unix_time':time.time(),'generation_files_unchanged':len(old['generation_files']),
    'generation_evidence':'Previously complete hashes plus unchanged size/mtime; no full weight reread',
    'evaluation_resources_rehashed':evaluated,'native_source_sha256':source,'dependencies':envs,
    'gpu_snapshot':gpu_snapshot(),'mem_available_gib':mem['MemAvailable']/2**20,
    'disk_free_gib':shutil.disk_usage(PROJECT).free/2**30,'cuda_initialized':False,
    'active_panel_sha256':q['screen']['active_panel_sha256'],'execution_sha256':file_hash(BATCH/'execution.json'),
    'sampling_core':'run_s2r_worker.generate AST equals accepted S2 core except progress label; no algorithm changes',
    'formal_generation_started':False}
json_write(BATCH/'preflight.json',report)
print(json.dumps(report))
