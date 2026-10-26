"""Resume committed, validated prompt chunks using the frozen scoring function.

Only complete chunks written AFTER the original occupancy checks are reused.
The original evaluate_s2.load_metric, weights, inputs and normalization remain
unchanged. Never replace a score according to its value.
"""
import argparse
import json
import math
import shutil
import sys
import time
import traceback

from s2_common import (PROJECT,BATCH,METRICS,configure,load_frozen,complete_generations,
    gpu_admission,OccupancyMonitor,json_write,file_hash,ensure_no_other_process,NETWORK_ATTEMPTS)
from evaluate_s2 import load_metric


def validate(chunk,expected,metric):
    rows=chunk['rows'];assert len(rows)==6
    by_path={r['video_path']:r for r in rows};assert len(by_path)==6
    assert set(by_path)=={g['video_path'] for g in expected}
    for g in expected:
        r=by_path[g['video_path']]
        for key in ['pair_id','prompt_id','stratum','prompt','seed','arm']:assert r[key]==g[key]
        assert r['metric']==metric and r['video_sha256']==g['video']['sha256']
        value=r['source_returned_score'];assert math.isfinite(value)
        assert r['normalized_score']==(value/100 if metric=='imaging_quality' else value)
        assert r['raw_score']==(value*10 if metric=='aesthetic_quality' else value)
        n=r['normalized_score'];assert (-1<=n<=1) if metric=='overall_consistency' else (0<=n<=1)
        if metric=='dynamic_degree':assert value in [0,1]
    assert abs(float(chunk['aggregate'])-sum(r['normalized_score'] for r in rows)/6)<1e-6


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',choices=METRICS,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args();configure()
    execution,_=load_frozen();generation=complete_generations(execution)
    root=BATCH/'evaluation'/args.metric
    assert not list(root.glob('attempt*/result.json')), 'Metric already complete'
    original_hash=file_hash(PROJECT/'tools/evaluate_s2.py')
    freeze=json.loads((BATCH/'analysis_freeze.json').read_text())
    assert original_hash==freeze['source_sha256']['tools/evaluate_s2.py']
    groups={pid:sorted([r for r in generation if r['prompt_id']==pid],key=lambda r:(r['seed'],r['arm']))
            for pid in sorted({r['prompt_id'] for r in generation})}
    reused={}
    for pid,expected in groups.items():
        candidates=sorted(root.glob('attempt*/'+pid+'.json'))
        for f in candidates:
            header_path=f.parent/'failure.json'
            assert header_path.exists(), 'Do not import from a running attempt'
            header=json.loads(header_path.read_text())
            assert header['source_sha256']==original_hash and header['network_attempts']==[]
            chunk=json.loads(f.read_text());validate(chunk,expected,args.metric)
            if pid in reused:assert reused[pid][1]['rows']==chunk['rows']
            else:reused[pid]=(f,chunk)
    output=root/('attempt%02d'%args.attempt);output.mkdir(parents=True,exist_ok=False)
    for name in ['evaluate_s2.py','resume_s2_metric.py']:
        shutil.copyfile(PROJECT/'tools'/name,output/('source_'+name))
    report={'status':'starting','metric':args.metric,'attempt':args.attempt,'rows':[],
        'source_sha256':original_hash,'runner_sha256':file_hash(PROJECT/'tools/resume_s2_metric.py'),
        'source_commit':'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490',
        'normalization':'MUSIQ raw/100 once; LAION source already /10; others unchanged',
        'reuse_policy':'Only complete six-video chunks previously committed after occupancy checks; no outcome-based selection.',
        'reused_chunks':[]}
    try:
        for pid,(path,chunk) in reused.items():
            provenance={'path':str(path),'sha256':file_hash(path),'prompt_id':pid}
            json_write(output/(pid+'.json'),dict(chunk,reused_from=provenance))
            report['rows'].extend(chunk['rows']);report['reused_chunks'].append(provenance)
        missing=[pid for pid in groups if pid not in reused]
        if missing:
            sys.path[:0]=[str(PROJECT/'src'/x) for x in ['CLIP','VBench','DINO']]
            with gpu_admission(args.gpu,min_free_gib=20) as admission:
                report['admission']=admission
                import torch
                import cv2
                torch.set_num_threads(4);torch.set_num_interop_threads(4)
                torch.set_grad_enabled(False)
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                torch.backends.cudnn.benchmark=False;torch.manual_seed(20261016);cv2.setNumThreads(1)
                state={'phase':'loading_metric','metric':args.metric,
                    'completed_videos':len(report['rows']),'reused_videos':len(report['rows'])}
                with OccupancyMonitor(admission,output,state) as monitor:
                    start=time.perf_counter();score=load_metric(args.metric,torch);torch.cuda.synchronize()
                    report['model_load_seconds']=time.perf_counter()-start
                    for pid in missing:
                        monitor.check();ensure_no_other_process(admission['uuid'])
                        expected=groups[pid];videos=[r['video_path'] for r in expected]
                        state.update(phase='scoring',prompt_id=pid)
                        start=time.perf_counter()
                        with torch.no_grad():aggregate,values=score(videos,expected[0]['prompt'])
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        assert len(values)==6 and {r['video_path'] for r in values}==set(videos)
                        rows=[]
                        for g in expected:
                            value=float(next(r['video_results'] for r in values if r['video_path']==g['video_path']))
                            r={k:g[k] for k in ['pair_id','prompt_id','stratum','prompt','seed','arm','video_path']}
                            r.update(metric=args.metric,raw_score=value*10 if args.metric=='aesthetic_quality' else value,
                                normalized_score=value/100 if args.metric=='imaging_quality' else value,
                                source_returned_score=value,video_sha256=g['video']['sha256'])
                            rows.append(r)
                        chunk={'rows':rows,'aggregate':float(aggregate),'scoring_seconds':elapsed}
                        validate(chunk,expected,args.metric)
                        monitor.check();ensure_no_other_process(admission['uuid'])
                        json_write(output/(pid+'.json'),chunk)
                        report['rows'].extend(rows);state['completed_videos']=len(report['rows'])
                        print('S2_RESUMED_METRIC_PROGRESS',args.metric,pid,len(report['rows']),flush=True)
                    monitor.check();ensure_no_other_process(admission['uuid'])
                    report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                    state['phase']='completed'
        assert len(report['rows'])==96 and not NETWORK_ATTEMPTS
        report.update(status='passed',network_attempts=NETWORK_ATTEMPTS)
        json_write(output/'result.json',report)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),network_attempts=NETWORK_ATTEMPTS)
        json_write(output/'failure.json',report)
        raise


if __name__=='__main__':main()
