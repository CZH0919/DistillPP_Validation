"""Predefined, paired, stratified prompt-cluster analysis (CPU only)."""
import numpy as np

METRICS = ['imaging_quality','subject_consistency','motion_smoothness',
           'dynamic_degree','aesthetic_quality','overall_consistency']
PRIORITY = ['imaging_quality','subject_consistency','overall_consistency',
            'aesthetic_quality','motion_smoothness']


def analyze(scores, timings):
    """Shapes: scores [4 strata,4 prompts,3 seeds,2 arms,6 metrics]; times [...,2].

    Arms are always T50 then S4. One bootstrap draw retains every seed and
    both methods of each selected prompt. The same draw is used for all metrics.
    """
    scores=np.asarray(scores,dtype=np.float64)
    timings=np.asarray(timings,dtype=np.float64)
    assert scores.shape==(4,4,3,2,6) and timings.shape==(4,4,3,2)
    assert np.isfinite(scores).all() and np.isfinite(timings).all() and (timings>0).all()
    assert np.isin(scores[...,3],[0,1]).all()
    rng=np.random.Generator(np.random.PCG64(20261016))
    draws=rng.integers(0,4,size=(10000,4,4))
    sampled=scores[np.arange(4)[None,:,None],draws]
    boot_means=sampled.mean(axis=(1,2,3))
    boot_delta=boot_means[:,0,:]-boot_means[:,1,:]
    delta=scores[:,:,:,0,:]-scores[:,:,:,1,:]
    means=scores.mean(axis=(0,1,2))
    summaries=[]
    for k,metric in enumerate(METRICS):
        quality=metric!='dynamic_degree'
        q=[.005,.995] if quality else [.025,.975]
        ci=np.quantile(boot_delta[:,k],q,method='linear')
        d=delta[...,k]
        summaries.append({'metric':metric,'T50_mean':float(means[0,k]),'S4_mean':float(means[1,k]),
            'T50_minus_S4':float(d.mean()),'ci_level':.99 if quality else .95,
            'ci_low':float(ci[0]),'ci_high':float(ci[1]),
            'teacher_advantage_candidate':bool(ci[0]>0) if quality else None,
            'student_advantage_nominal_family_interval':bool(ci[1]<0) if quality else None,
            'paired_video_teacher_higher':int((d>0).sum()),
            'paired_video_student_higher':int((d<0).sum()),'paired_video_equal':int((d==0).sum()),
            'paired_prompt_teacher_higher':int((d.mean(axis=2)>0).sum()),
            'T50_min':float(scores[:,:,:,0,k].min()),'T50_max':float(scores[:,:,:,0,k].max()),
            'S4_min':float(scores[:,:,:,1,k].min()),'S4_max':float(scores[:,:,:,1,k].max())})
    td=scores[:,:,:,0,3].astype(bool);sd=scores[:,:,:,1,3].astype(bool)
    dynamic={'T50_dynamic_count':int(td.sum()),'S4_dynamic_count':int(sd.sum()),'n_per_arm':48,
        'both_dynamic':int((td&sd).sum()),'both_static':int((~td&~sd).sum()),
        'T50_dynamic_S4_static':int((td&~sd).sum()),'T50_static_S4_dynamic':int((~td&sd).sum()),
        'decision_threshold':None,'interpretation':'Movement indicator, not higher-is-better quality; researcher decides tradeoffs.'}
    boot_times=timings[np.arange(4)[None,:,None],draws].reshape(10000,48,2)
    boot_speed=np.median(boot_times[:,:,0],axis=1)/np.median(boot_times[:,:,1],axis=1)
    speed_ci=np.quantile(boot_speed,[.025,.975],method='linear')
    flat=timings.reshape(48,2)
    timing={'primary':'synchronized denoise + VAE seconds, hot; one GPU per method',
        'speed_ratio_medians':float(np.median(flat[:,0])/np.median(flat[:,1])),
        'speed_ratio_95ci':speed_ci.tolist(),
        'paired_ratio_median':float(np.median(flat[:,0]/flat[:,1])),
        'per_arm':{}}
    for k,arm in enumerate(['T50','S4']):
        x=flat[:,k];quartiles=np.quantile(x,[.25,.5,.75],method='linear')
        timing['per_arm'][arm]={'n':48,'median':float(quartiles[1]),'q1':float(quartiles[0]),
            'q3':float(quartiles[2]),'iqr':float(quartiles[2]-quartiles[0]),'mean':float(x.mean()),
            'min':float(x.min()),'max':float(x.max()),'total_seconds':float(x.sum())}
    passing=[m for m in PRIORITY if next(r for r in summaries if r['metric']==m)['teacher_advantage_candidate']]
    return {'metric_summaries':summaries,'dynamic':dynamic,'timing':timing,
        'target_priority':PRIORITY,'passing_targets_in_priority_order':passing,
        'suggested_target':passing[0] if passing else None,
        'statistical_teacher_advantage':bool(passing),'next_stage_started':False,
        'bootstrap':{'cluster':'prompt','strata':4,'prompts_per_stratum':4,'seeds_per_prompt':3,
            'replicates':10000,'rng':'PCG64','seed':20261016,'quantile':'linear','dtype':'float64',
            'quality_family_size':5,'quality_ci_level':.99,'dynamic_ci_level':.95,
            'interpretation':'Nominal Bonferroni family control with small-sample percentile-bootstrap approximation; no exact coverage claim.'}}, delta.mean(axis=2)


def self_test():
    scores=np.zeros((4,4,3,2,6),dtype=np.float64)
    scores[...,0,:]=.6;scores[...,1,:]=.5
    scores[...,3]=0
    times=np.ones((4,4,3,2));times[...,0]=10
    report,d=analyze(scores,times)
    assert len(report['passing_targets_in_priority_order'])==5
    assert report['suggested_target']=='imaging_quality'
    assert report['timing']['speed_ratio_95ci']==[10.,10.]
    assert d.shape==(4,4,6)
    assert abs(report['metric_summaries'][0]['ci_low']-.1)<1e-12
    # Seeds may differ strongly but must stay together within their prompt.
    # Exact zero prompt-mean effects must yield zero-width paired CIs.
    scores[...,0,0]=np.array([.1,.5,.9]);scores[...,1,0]=.5
    report,_=analyze(scores,times)
    row=report['metric_summaries'][0]
    assert abs(row['ci_low'])<1e-12 and abs(row['ci_high'])<1e-12
    assert not row['teacher_advantage_candidate']
    scores[...,0,3]=1
    report,_=analyze(scores,times)
    assert report['dynamic']['T50_dynamic_S4_static']==48
    assert report['metric_summaries'][3]['teacher_advantage_candidate'] is None
    # Reverse sign: T cannot be selected even if every individual CI is negative.
    scores[...,0,:]=.4;scores[...,1,:]=.7;scores[...,3]=0
    report,_=analyze(scores,times)
    assert not report['statistical_teacher_advantage'] and report['suggested_target'] is None
    print('S2_STATISTICS_SELF_TEST_PASS')


if __name__=='__main__':self_test()
