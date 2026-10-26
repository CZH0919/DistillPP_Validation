"""S2-R: score only the three assigned dimensions; resume committed chunks."""
import argparse
import json
import math
import shutil
import sys
import time
import traceback

from s2r_common import (PROJECT,BATCH,METRICS,configure,load_frozen,complete_generations,
    gpu_admission,OccupancyMonitor,json_write,file_hash,tensor_hash,ensure_no_other_process,NETWORK_ATTEMPTS)
from evaluate_s2 import load_metric as accepted_load_metric


def load_metric(metric, torch):
    if metric != 'overall_consistency':
        return accepted_load_metric(metric, torch), None
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
    from vbench.third_party.ViCLIP.viclip import ViCLIP
    from vbench.overall_consistency import overall_consistency
    weights=PROJECT/'resources/evaluation/ViCLIP'
    tokenizer=SimpleTokenizer(str(weights/'bpe_simple_vocab_16e6.txt.gz'))
    model=ViCLIP(tokenizer=tokenizer,pretrain=str(weights/'ViClip-InternVid-10M-FLT.pth'))
    model.eval().requires_grad_(False).to('cuda')
    assert model.max_txt_l==32
    def metadata(prompt):
        token_ids=model.text_encoder._tokenizer.encode(prompt)
        assert token_ids==tokenizer.encode(prompt)
        effective=model.text_encoder.tokenize(prompt,context_length=32,truncate=True)
        return {'original_text':prompt,'tokens_with_special_before_truncation':len(token_ids)+2,
            'truncated':len(token_ids)+2>32,'context_length':32,
            'effective_token_ids':effective.cpu().tolist()[0],
            'effective_token_sha256':tensor_hash(effective),
            'policy':'Official native truncate=True; original task text preserved'}
    def score(videos,prompt):
        return overall_consistency(model,[{'prompt':prompt,'video_list':videos}],tokenizer,'cuda')
    return score, metadata


def validate_chunk(chunk, expected, metric, source_hash):
    assert chunk['source_sha256']==source_hash and len(chunk['rows'])==6
    rows={r['video_path']:r for r in chunk['rows']}
    assert set(rows)=={r['video_path'] for r in expected}
    for g in expected:
        r=rows[g['video_path']]
        for key in ('pair_id','prompt_id','domain','eval_prompt','seed','arm'):
            assert r[key]==g[key]
        assert r['metric']==metric and metric in g['primary_metrics']
        assert r['video_sha256']==g['video']['sha256']
        v=r['source_returned_score']; n=r['normalized_score']
        assert math.isfinite(v)
        assert n==(v/100 if metric=='imaging_quality' else v)
        assert r['raw_score']==(v*10 if metric=='aesthetic_quality' else v)
        assert (-1<=n<=1) if metric=='overall_consistency' else (0<=n<=1)
        if metric=='dynamic_degree': assert v in (0,1)
    assert abs(chunk['aggregate']-sum(r['normalized_score'] for r in rows.values())/6)<1e-6
    if metric=='overall_consistency':
        meta=chunk['text_metadata']
        assert meta['original_text']==expected[0]['eval_prompt']
        assert meta['tokens_with_special_before_truncation']==expected[0]['token_counts']['viclip_eval_tokens_with_special']
        assert meta['truncated']==expected[0]['token_counts']['viclip_exceeds_32']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',choices=METRICS,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args(); configure()
    execution,_=load_frozen()
    generation=[r for r in complete_generations(execution) if args.metric in r['primary_metrics']]
    assert len(generation)==96 and len({r['prompt_id'] for r in generation})==16
    sys.path[:0]=[str(PROJECT/'src'/x) for x in ['CLIP','VBench','DINO']]
    metric_root=BATCH/'evaluation'/args.metric
    assert not list(metric_root.glob('attempt*/result.json')), 'Metric already complete'
    output=metric_root/('attempt%02d'%args.attempt)
    output.mkdir(parents=True,exist_ok=False)
    source_hash=file_hash(PROJECT/'tools/evaluate_s2r.py')
    for name in ('evaluate_s2r.py','s2r_common.py','evaluate_s2.py','s1_common.py','s2_common.py'):
        shutil.copyfile(PROJECT/'tools'/name,output/('source_'+name))
    report={'status':'starting','metric':args.metric,'attempt':args.attempt,'rows':[],
        'source_sha256':source_hash,'chunks':[],
        'normalization':'MUSIQ raw/100 once; LAION source already /10; others unchanged',
        'source_commit':'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'}
    pending=[]
    for pid in sorted({r['prompt_id'] for r in generation}):
        rows=sorted([r for r in generation if r['prompt_id']==pid],key=lambda r:(r['seed'],r['arm']))
        assert len(rows)==6 and len({r['eval_prompt'] for r in rows})==1
        candidates=list(metric_root.glob('attempt*/'+pid+'.json'))
        assert len(candidates)<=1, 'Duplicate committed chunk'
        if candidates:
            chunk=json.loads(candidates[0].read_text())
            validate_chunk(chunk,rows,args.metric,source_hash)
            report['rows'].extend(chunk['rows'])
            report['chunks'].append({'path':str(candidates[0]),'reused':True})
        else: pending.append((pid,rows))
    try:
        if pending:
            with gpu_admission(args.gpu,min_free_gib=20) as admission:
                report['admission']=admission
                import torch
                import cv2
                torch.set_num_threads(4); torch.set_num_interop_threads(4)
                torch.set_grad_enabled(False)
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                torch.backends.cudnn.benchmark=False
                torch.manual_seed(20261017); cv2.setNumThreads(1)
                state={'phase':'loading_metric','metric':args.metric,'completed_videos':len(report['rows'])}
                with OccupancyMonitor(admission,output,state) as monitor:
                    start=time.perf_counter()
                    score,text_meta=load_metric(args.metric,torch)
                    torch.cuda.synchronize()
                    report['model_load_seconds']=time.perf_counter()-start
                    for pid,rows in pending:
                        monitor.check(); ensure_no_other_process(admission['uuid'])
                        videos=[r['video_path'] for r in rows]
                        metadata=text_meta(rows[0]['eval_prompt']) if text_meta else None
                        state.update(phase='scoring',prompt_id=pid)
                        torch.cuda.synchronize(); start=time.perf_counter()
                        with torch.no_grad(): aggregate,values=score(videos,rows[0]['eval_prompt'])
                        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                        assert len(values)==6 and {v['video_path'] for v in values}==set(videos)
                        chunk=[]
                        for row in rows:
                            v=float(next(x['video_results'] for x in values if x['video_path']==row['video_path']))
                            chunk.append({k:row[k] for k in ('pair_id','prompt_id','domain','eval_prompt','seed','arm','video_path')})
                            chunk[-1].update(metric=args.metric,raw_score=v*10 if args.metric=='aesthetic_quality' else v,
                                normalized_score=v/100 if args.metric=='imaging_quality' else v,
                                source_returned_score=v,video_sha256=row['video']['sha256'])
                        commit={'rows':chunk,'aggregate':float(aggregate),'scoring_seconds':elapsed,
                            'source_sha256':source_hash,'text_metadata':metadata}
                        validate_chunk(commit,rows,args.metric,source_hash)
                        monitor.check(); ensure_no_other_process(admission['uuid'])
                        assert not NETWORK_ATTEMPTS
                        target=output/(pid+'.json'); json_write(target,commit)
                        report['rows'].extend(chunk); report['chunks'].append({'path':str(target),'reused':False})
                        state['completed_videos']=len(report['rows'])
                        print('S2R_EVAL_PROGRESS',args.metric,pid,len(report['rows']),flush=True)
                    report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                    state.update(phase='completed')
        assert len(report['rows'])==96 and not NETWORK_ATTEMPTS
        report.update(status='passed',network_attempts=NETWORK_ATTEMPTS)
        json_write(output/'result.json',report)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),network_attempts=NETWORK_ATTEMPTS)
        json_write(output/'failure.json',report)
        raise


if __name__=='__main__': main()
