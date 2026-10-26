"""Frozen S2-R domain-specific paired prompt-cluster statistics; CPU only."""
import numpy as np

PRIORITY=['imaging_quality','subject_consistency','overall_consistency','aesthetic_quality','motion_smoothness']
DOMAINS={'motion':['subject_consistency','motion_smoothness','dynamic_degree'],
         'visual_semantic':['imaging_quality','aesthetic_quality','overall_consistency']}


def summarize_metric(metric,x,draws,domain):
    x=np.asarray(x,dtype=np.float64)
    assert x.ndim==3 and x.shape[1:]==(3,2) and np.isfinite(x).all()
    if metric=='dynamic_degree': assert np.isin(x,[0,1]).all()
    d=x[:,:,0]-x[:,:,1]
    prompt_d=d.mean(axis=1)
    boot=prompt_d[draws].mean(axis=1)
    quality=metric!='dynamic_degree'
    ci=np.quantile(boot,[.005,.995] if quality else [.025,.975],method='linear')
    return {'metric':metric,'domain':domain,'prompt_count':len(x),'seeds_per_prompt':3,
        'paired_video_count':len(x)*3,'T50_mean':float(x[:,:,0].mean()),'S4_mean':float(x[:,:,1].mean()),
        'T50_minus_S4':float(prompt_d.mean()),'ci_level':.99 if quality else .95,
        'ci_low':float(ci[0]),'ci_high':float(ci[1]),
        'teacher_advantage_candidate':bool(ci[0]>0) if quality else None,
        'student_advantage_nominal_family_interval':bool(ci[1]<0) if quality else None,
        'paired_video_teacher_higher':int((d>0).sum()),'paired_video_student_higher':int((d<0).sum()),
        'paired_video_equal':int((d==0).sum()),'paired_prompt_teacher_higher':int((prompt_d>0).sum()),
        'paired_prompt_student_higher':int((prompt_d<0).sum()),'paired_prompt_equal':int((prompt_d==0).sum()),
        'T50_min':float(x[:,:,0].min()),'T50_max':float(x[:,:,0].max()),
        'S4_min':float(x[:,:,1].min()),'S4_max':float(x[:,:,1].max()),
        'T50_exact_zero_count':int((x[:,:,0]==0).sum()),'T50_exact_one_count':int((x[:,:,0]==1).sum()),
        'S4_exact_zero_count':int((x[:,:,1]==0).sum()),'S4_exact_one_count':int((x[:,:,1]==1).sum())}


def summarize_timing(x,boot):
    flat=np.asarray(x,dtype=np.float64).reshape(-1,2)
    assert (flat>0).all() and np.isfinite(flat).all()
    ratio=np.median(boot[:,:,0],axis=1)/np.median(boot[:,:,1],axis=1)
    result={'speed_ratio_medians':float(np.median(flat[:,0])/np.median(flat[:,1])),
        'speed_ratio_95ci':np.quantile(ratio,[.025,.975],method='linear').tolist(),
        'paired_ratio_median':float(np.median(flat[:,0]/flat[:,1])),'per_arm':{}}
    for j,arm in enumerate(('T50','S4')):
        t=flat[:,j];quartiles=np.quantile(t,[.25,.5,.75],method='linear')
        result['per_arm'][arm]={'n':len(t),'median':float(quartiles[1]),'q1':float(quartiles[0]),
            'q3':float(quartiles[2]),'iqr':float(quartiles[2]-quartiles[0]),'mean':float(t.mean()),
            'min':float(t.min()),'max':float(t.max()),'total_seconds':float(t.sum())}
    return result


def analyze(metric_arrays,timing_arrays,prompt_ids,flagged_ids=('VB0724',)):
    assert set(metric_arrays)=={m for v in DOMAINS.values() for m in v}
    assert set(timing_arrays)==set(prompt_ids)==set(DOMAINS)
    rng=np.random.Generator(np.random.PCG64(20261017))
    # Every draw samples prompts within each domain; all seeds and arms stay together.
    draws=rng.integers(0,16,size=(10000,2,16))
    summaries=[]; prompt_rows=[]; timing={}; boot_time=[]; all_time=[]
    for j,(domain,metrics) in enumerate(DOMAINS.items()):
        assert len(prompt_ids[domain])==len(set(prompt_ids[domain]))==16
        for metric in metrics:
            x=np.asarray(metric_arrays[metric],dtype=np.float64)
            assert x.shape==(16,3,2)
            summaries.append(summarize_metric(metric,x,draws[:,j],domain))
            for i,pid in enumerate(prompt_ids[domain]):
                prompt_rows.append({'domain':domain,'prompt_id':pid,'metric':metric,
                    'T50_mean':float(x[i,:,0].mean()),'S4_mean':float(x[i,:,1].mean()),
                    'T50_minus_S4':float((x[i,:,0]-x[i,:,1]).mean()),'paired_seeds':3})
        t=np.asarray(timing_arrays[domain],dtype=np.float64)
        assert t.shape==(16,3,2)
        sampled=t[draws[:,j]].reshape(10000,48,2)
        timing[domain]=summarize_timing(t,sampled)
        boot_time.append(sampled);all_time.append(t.reshape(48,2))
    timing['all_domains_equal_weight']=summarize_timing(np.concatenate(all_time),np.concatenate(boot_time,axis=1))
    dynamic_x=np.asarray(metric_arrays['dynamic_degree'],dtype=np.float64)
    td=dynamic_x[:,:,0].astype(bool);sd=dynamic_x[:,:,1].astype(bool)
    dynamic={'domain':'motion','n_per_arm':48,'T50_dynamic_count':int(td.sum()),'S4_dynamic_count':int(sd.sum()),
        'both_dynamic':int((td&sd).sum()),'both_static':int((~td&~sd).sum()),
        'T50_dynamic_S4_static':int((td&~sd).sum()),'T50_static_S4_dynamic':int((~td&sd).sum()),
        'decision_threshold':None,'higher_is_better':False,
        'scope':'Motion domain only; does not establish motion preservation in visual_semantic domain'}
    passing=[m for m in PRIORITY if next(r for r in summaries if r['metric']==m)['teacher_advantage_candidate']]
    keep=[i for i,pid in enumerate(prompt_ids['visual_semantic']) if pid not in flagged_ids]
    sensitivity=None
    if len(keep)<16:
        assert len(keep)==15
        sub=np.asarray(metric_arrays['overall_consistency'],dtype=np.float64)[keep]
        subdraw=np.random.Generator(np.random.PCG64(20261017)).integers(0,len(keep),size=(10000,len(keep)))
        sensitivity=summarize_metric('overall_consistency',sub,subdraw,'visual_semantic_nontruncated')
        sensitivity.pop('teacher_advantage_candidate');sensitivity.pop('student_advantage_nominal_family_interval')
        sensitivity.update(excluded_prompt_ids=list(flagged_ids),role='Predeclared sensitivity only; not a replacement qualification test')
    return {'metric_summaries':summaries,'dynamic':dynamic,'timing_by_domain':timing,
        'semantic_nontruncated_sensitivity':sensitivity,'target_priority':PRIORITY,
        'passing_targets_in_priority_order':passing,'suggested_target':passing[0] if passing else None,
        'statistical_teacher_advantage':bool(passing),'s3_started':False,
        'bootstrap':{'cluster':'prompt within designated domain','prompts_per_domain':16,
            'seeds_per_prompt':3,'replicates':10000,'rng':'PCG64','seed':20261017,
            'quantile':'linear','dtype':'float64','quality_family_size':5,'quality_ci_level':.99,
            'dynamic_ci_level':.95,'speed_ci_level':.95,
            'interpretation':'Nominal Bonferroni family control using small-sample percentile bootstrap; conditional on fixed seeds; no exact coverage or official leaderboard claim.'}},prompt_rows


def self_test():
    ids={d:[d+str(i) for i in range(16)] for d in DOMAINS};ids['visual_semantic'][14]='VB0724'
    arrays={m:np.tile([.6,.5],(16,3,1)) for ms in DOMAINS.values() for m in ms}
    arrays['dynamic_degree']=np.tile([1.,0.],(16,3,1))
    times={d:np.tile([10.,1.],(16,3,1)) for d in DOMAINS}
    r,p=analyze(arrays,times,ids)
    assert r['passing_targets_in_priority_order']==PRIORITY and len(p)==96
    assert r['dynamic']['T50_dynamic_S4_static']==48
    assert r['timing_by_domain']['all_domains_equal_weight']['speed_ratio_95ci']==[10.,10.]
    assert r['semantic_nontruncated_sensitivity']['paired_video_count']==45
    # Strongly different seeds whose mean is zero must never be resampled independently.
    arrays['imaging_quality'][:,:,0]=np.array([.1,.5,.9]);arrays['imaging_quality'][:,:,1]=.5
    r,_=analyze(arrays,times,ids)
    row=next(x for x in r['metric_summaries'] if x['metric']=='imaging_quality')
    assert abs(row['ci_low'])<1e-12 and abs(row['ci_high'])<1e-12
    assert not row['teacher_advantage_candidate']
    # A metric's inference must not depend on the other domain's values.
    before={x['metric']:x for x in r['metric_summaries']}
    for m in DOMAINS['motion']:
        if m!='dynamic_degree': arrays[m]=np.tile([.2,.8],(16,3,1))
    after,_=analyze(arrays,times,ids)
    for row in after['metric_summaries']:
        if row['domain']=='visual_semantic': assert row==before[row['metric']]
    assert next(x for x in after['metric_summaries'] if x['metric']=='subject_consistency')['ci_high']<0
    print('S2R_STATISTICS_SELF_TEST_PASS')


if __name__=='__main__': self_test()
