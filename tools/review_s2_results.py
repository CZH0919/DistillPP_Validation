"""CPU-only result cross-check and readable supplementary tables; no new inference."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

import numpy as np
from s2_statistics import METRICS,analyze

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Result/S2_20261016'
BATCH=ROOT/'runs/s2_20261016'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def csv_write(name,rows):
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    path=OUT/name
    if path.exists():assert path.read_text(encoding='utf-8')==text.getvalue()
    else:path.write_text(text.getvalue(),encoding='utf-8',newline='\n')


def main():
    original=read(OUT/'statistics.json');panel=read(BATCH/'panel.json')
    with (OUT/'metrics_per_video.csv').open(encoding='utf-8',newline='') as f:metrics=list(csv.DictReader(f))
    with (OUT/'timing_per_video.csv').open(encoding='utf-8',newline='') as f:times=list(csv.DictReader(f))
    lookup={(r['prompt_id'],int(r['seed']),r['arm'],r['metric']):float(r['normalized_score']) for r in metrics}
    t_lookup={(r['prompt_id'],int(r['seed']),r['arm']):float(r['denoise_plus_vae_seconds']) for r in times}
    assert len(lookup)==576 and len(t_lookup)==96
    strata=list(dict.fromkeys(p['stratum'] for p in panel['prompts']))
    scores=np.empty((4,4,3,2,6));timings=np.empty((4,4,3,2));wide=[]
    for h,stratum in enumerate(strata):
        for i,prompt in enumerate(p for p in panel['prompts'] if p['stratum']==stratum):
            for s,seed in enumerate(panel['seed_slots']):
                for a,arm in enumerate(['T50','S4']):
                    row={'prompt_id':prompt['id'],'stratum':stratum,'prompt':prompt['prompt'],'seed':seed,'arm':arm}
                    for k,metric in enumerate(METRICS):
                        scores[h,i,s,a,k]=lookup[prompt['id'],seed,arm,metric]
                        row[metric]=float(scores[h,i,s,a,k])
                    timings[h,i,s,a]=t_lookup[prompt['id'],seed,arm]
                    row['denoise_plus_vae_seconds']=float(timings[h,i,s,a]);wide.append(row)
    recomputed,_=analyze(scores,timings)
    errors=[]
    for old,new in zip(original['metric_summaries'],recomputed['metric_summaries']):
        assert old['metric']==new['metric']
        for key in ['T50_mean','S4_mean','T50_minus_S4','ci_low','ci_high']:
            errors.append(abs(old[key]-new[key]))
        for arm in ['T50','S4']:
            mean=statistics.fmean(float(r['normalized_score']) for r in metrics if r['metric']==old['metric'] and r['arm']==arm)
            assert abs(mean-old[arm+'_mean'])<1e-12
    assert max(errors)<1e-12
    assert original['passing_targets_in_priority_order']==recomputed['passing_targets_in_priority_order']
    assert np.max(np.abs(np.array(original['timing']['speed_ratio_95ci'])-recomputed['timing']['speed_ratio_95ci']))<1e-9
    csv_write('metrics_wide.csv',wide)
    rows=[];warmups=[]
    for path in sorted((BATCH/'workers').glob('worker*/attempt*')):
        result=path/'result.json';failure=path/'failure.json'
        if not result.exists() and not failure.exists():continue
        r=read(result if result.exists() else failure)
        launch=read(BATCH/('launch_'+path.parent.name+'_'+path.name+'.json'))
        warm=[read(f) for f in sorted((path/'warmup').glob('*/result.json'))]
        warmups.extend(warm)
        first=min(warm,key=lambda w:w['unix_end']) if warm else None
        rows.append({'worker':path.parent.name,'attempt':path.name,'status':r['status'],
            'gpu_uuid':r.get('gpu_uuid',r.get('admission',{}).get('uuid')),
            'completed_formal_pairs':len(r.get('completed_pairs',[])),
            'common_load_seconds':r.get('common_load_seconds'),
            'teacher_load_seconds':r.get('teacher_load_seconds'),
            'student_load_seconds':r.get('student_load_seconds'),
            'first_warmup_arm':first['arm'] if first else None,
            'launch_to_first_decoded_warmup_seconds':first['unix_end']-launch['time'] if first else None,
            'launch_to_both_warmups_completed_seconds':max(w['unix_end'] for w in warm)-launch['time'] if len(warm)==2 else None,
            'T50_warmup_seconds':next((w['denoise_plus_vae_seconds'] for w in warm if w['arm']=='T50'),None),
            'S4_warmup_seconds':next((w['denoise_plus_vae_seconds'] for w in warm if w['arm']=='S4'),None),
            'error':r.get('error','')})
    assert sum(r['completed_formal_pairs'] for r in rows)==48
    csv_write('worker_attempts_and_cold_start.csv',rows)
    warm_hashes={}
    for arm in ['T50','S4']:
        ref=ROOT/'runs/s1_20261016'/('teacher/attempt01/result.json' if arm=='T50' else 'student/attempt02/result.json')
        ref=read(ref)
        records=[r for r in warmups if r['arm']==arm]
        for r in records:
            for key in ['initial_latent_sha256','positive_context_sha256','final_latent_sha256']:
                assert r[key]==ref[key]
        warm_hashes[arm]={'completed_warmups':len(records),'all_match_S1':True,'final_latent_sha256':ref['final_latent_sha256']}
    positive={};initial={}
    for row in times:
        positive.setdefault(row['prompt_id'],set()).add(row['positive_context_sha256'])
        initial.setdefault(row['seed'],set()).add(row['initial_latent_sha256'])
    assert len(positive)==16 and len(initial)==3
    assert all(len(x)==1 for x in [*positive.values(),*initial.values()])
    audit={'status':'passed','original_numpy':original['numpy_version'],'local_numpy':np.__version__,
        'recomputed_from':'576-row metrics CSV and 96-row timing CSV',
        'max_absolute_score_or_CI_error':max(errors),'qualification_decision_identical':True,
        'completed_formal_pairs_across_attempts':48,'worker_attempts':len(rows),
        'warmup_equivalence':warm_hashes,'positive_condition_hash_consistent_per_prompt':16,
        'initial_noise_hash_consistent_per_seed':3,
        'student_initial_cast_metadata':'CPU-derived expected BF16 cast/hash/error outside timing; actual native runtime state dtype asserted each step. S1 records the actual GPU cast, and all completed S2 warmup endpoints match S1.'}
    path=OUT/'analysis_crosscheck.json'
    path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
