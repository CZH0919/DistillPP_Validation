"""Independently reconstruct S2-R statistics from synchronized CSVs, CPU only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from s2r_statistics import analyze

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Result/S2R_20261017'

def rows(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    q=json.loads((ROOT/'runs/s2r_20261017/protocol.json').read_text(encoding='utf-8'))
    remote=json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))
    assert remote['integrity']['videos']==192 and remote['integrity']['valid_assigned_metric_values']==576
    metric_rows=rows('metrics_per_video.csv');timing_rows=rows('timing_per_video.csv')
    assert len(metric_rows)==576 and len(timing_rows)==192
    scores={(r['pair_id'],r['arm'],r['metric']):r for r in metric_rows}
    times={(r['pair_id'],r['arm']):r for r in timing_rows}
    assert len(scores)==576 and len(times)==192
    arrays={m:np.empty((16,3,2),dtype=np.float64) for ms in q['metric_domains'].values() for m in ms}
    timings={d:np.empty((16,3,2),dtype=np.float64) for d in q['metric_domains']}
    expected_keys=set()
    for d,ids in q['screen']['active_prompt_ids_by_domain'].items():
        for i,pid in enumerate(ids):
            for s,seed in enumerate(q['screen']['seed_slots']):
                pair=pid+'_seed'+str(seed)
                for a,arm in enumerate(['T50','S4']):
                    t=times[pair,arm];assert t['domain']==d
                    timings[d][i,s,a]=float(t['denoise_plus_vae_seconds'])
                    for metric in q['metric_domains'][d]:
                        key=(pair,arm,metric);expected_keys.add(key)
                        r=scores[key];assert r['domain']==d and r['prompt_id']==pid and int(r['seed'])==seed
                        arrays[metric][i,s,a]=float(r['normalized_score'])
    assert expected_keys==set(scores)
    computed,prompt_rows=analyze(arrays,timings,q['screen']['active_prompt_ids_by_domain'],q['score_policy']['screen_viclip_truncated_ids'])
    errors=[];diffs=[]
    def compare(a,b,path):
        if isinstance(a,dict):
            for key,value in a.items():
                if key not in b:errors.append(path+'.'+key+' missing')
                else:compare(value,b[key],path+'.'+key)
        elif isinstance(a,list):
            if len(a)!=len(b):errors.append(path+' length')
            else:
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'['+str(i)+']')
        elif isinstance(a,float):
            delta=abs(a-b);diffs.append(delta)
            if not math.isfinite(delta) or delta>1e-10:errors.append(path+' numerical mismatch '+str(delta))
        elif a!=b:errors.append(path+' value mismatch')
    compare(computed,remote,'statistics')
    saved_prompt=rows('prompt_means.csv')
    lookup={(r['prompt_id'],r['metric']):r for r in saved_prompt}
    assert len(lookup)==96
    for row in prompt_rows:
        for key in ('T50_mean','S4_mean','T50_minus_S4'):
            compare(row[key],float(lookup[row['prompt_id'],row['metric']][key]),row['prompt_id']+'.'+key)
    receipt={'status':'passed' if not errors else 'failed','local_python':sys.version.split()[0],
        'local_numpy':np.__version__,'remote_numpy':remote['numpy_version'],
        'reconstructed_videos':192,'reconstructed_metric_values':576,
        'compared_float_count':len(diffs),'max_absolute_difference':max(diffs,default=0),
        'tolerance':1e-10,'qualification_same':computed['statistical_teacher_advantage']==remote['statistical_teacher_advantage'],
        'errors':errors,'sources':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
            ['statistics.json','metrics_per_video.csv','timing_per_video.csv','prompt_means.csv']}}
    content=json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path=OUT/'LOCAL_REPRODUCTION.json'
    if path.exists():assert path.read_text(encoding='utf-8')==content
    else:path.write_text(content,encoding='utf-8',newline='\n')
    print(json.dumps(receipt,ensure_ascii=False))
    assert not errors, 'Local statistical reproduction failed'

if __name__=='__main__':main()
