"""Fixed S1-accepted VBench functions, one loaded model per dimension, S2 only."""
import argparse
import json
import math
import shutil
import sys
import time
import traceback

from s2_common import (PROJECT, BATCH, METRICS, configure, load_frozen,
    gpu_admission, OccupancyMonitor, complete_generations, json_write,
    file_hash, ensure_no_other_process, NETWORK_ATTEMPTS)


def load_metric(metric, torch):
    weights = PROJECT/'resources/evaluation'
    device = 'cuda'
    if metric == 'subject_consistency':
        import vision_transformer
        from vbench.subject_consistency import subject_consistency
        model = vision_transformer.vit_base(patch_size=16, num_classes=0)
        model.load_state_dict(torch.load(weights/'dino_model/dino_vitbase16_pretrain.pth', map_location='cpu', weights_only=True), strict=True)
        model.eval().requires_grad_(False).to(device)
        return lambda videos,prompt: subject_consistency(model,videos,device,read_frame=False)
    if metric == 'imaging_quality':
        from pyiqa.archs.musiq_arch import MUSIQ
        from vbench.imaging_quality import technical_quality
        model=MUSIQ(pretrained_model_path=str(weights/'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'))
        model.eval().requires_grad_(False).to(device)
        return lambda videos,prompt: technical_quality(model,videos,device,imaging_quality_preprocessing_mode='longer')
    if metric == 'motion_smoothness':
        from vbench.motion_smoothness import MotionSmoothness,motion_smoothness
        model=MotionSmoothness(str(PROJECT/'src/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'), str(weights/'amt_model/amt-s.pth'), device)
        return lambda videos,prompt: motion_smoothness(model,videos)
    if metric == 'dynamic_degree':
        from easydict import EasyDict
        from vbench.dynamic_degree import DynamicDegree,dynamic_degree
        model=DynamicDegree(EasyDict(model=str(weights/'raft_model/models/raft-things.pth'),small=False,mixed_precision=False,alternate_corr=False),device)
        return lambda videos,prompt: dynamic_degree(model,videos)
    if metric == 'aesthetic_quality':
        import clip
        from vbench.aesthetic_quality import get_aesthetic_model,laion_aesthetic
        model,_=clip.load(str(weights/'clip_model/ViT-L-14.pt'),device=device,jit=False)
        model.eval().requires_grad_(False)
        head=get_aesthetic_model(str(weights/'aesthetic_model/emb_reader')).to(device).eval()
        return lambda videos,prompt: laion_aesthetic(head,model,videos,device)
    if metric == 'overall_consistency':
        from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
        from vbench.third_party.ViCLIP.viclip import ViCLIP
        from vbench.overall_consistency import overall_consistency
        tokenizer=SimpleTokenizer(str(weights/'ViCLIP/bpe_simple_vocab_16e6.txt.gz'))
        model=ViCLIP(tokenizer=tokenizer,pretrain=str(weights/'ViCLIP/ViClip-InternVid-10M-FLT.pth'))
        model.eval().requires_grad_(False).to(device)
        def score(videos,prompt):
            assert len(tokenizer.encode(prompt))+2 <= 32
            model.text_encoder.tokenize(prompt,context_length=32,truncate=False)
            return overall_consistency(model,[{'prompt':prompt,'video_list':videos}],tokenizer,device)
        return score
    raise ValueError(metric)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',choices=METRICS,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    configure()
    execution,settings=load_frozen()
    generation=complete_generations(execution)
    sys.path[:0]=[str(PROJECT/'src'/x) for x in ['CLIP','VBench','DINO']]
    output=BATCH/'evaluation'/args.metric/('attempt%02d'%args.attempt)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(PROJECT/'tools/evaluate_s2.py',output/'source_evaluate_s2.py')
    report={'status':'starting','metric':args.metric,'attempt':args.attempt,'rows':[],
            'source_sha256':file_hash(PROJECT/'tools/evaluate_s2.py'),
            'normalization':'MUSIQ raw/100 once; LAION source already /10; others unchanged',
            'source_commit':'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'}
    try:
        with gpu_admission(args.gpu,min_free_gib=20) as admission:
            report['admission']=admission
            import torch
            import cv2
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            torch.manual_seed(20261016)
            cv2.setNumThreads(1)
            state={'phase':'loading_metric','metric':args.metric,'completed_videos':0}
            with OccupancyMonitor(admission,output,state) as monitor:
                start=time.perf_counter()
                score=load_metric(args.metric,torch)
                torch.cuda.synchronize()
                report['model_load_seconds']=time.perf_counter()-start
                for prompt_id in sorted({r['prompt_id'] for r in generation}):
                    monitor.check()
                    ensure_no_other_process(admission['uuid'])
                    rows=[r for r in generation if r['prompt_id']==prompt_id]
                    # Fixed prompt + paired seed order; exactly six videos per chunk.
                    rows.sort(key=lambda r:(r['seed'],r['arm']))
                    assert len(rows)==6 and len({r['prompt'] for r in rows})==1
                    videos=[r['video_path'] for r in rows]
                    state.update(phase='scoring',prompt_id=prompt_id)
                    start=time.perf_counter()
                    with torch.no_grad():aggregate,values=score(videos,rows[0]['prompt'])
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    assert len(values)==6 and {v['video_path'] for v in values}==set(videos)
                    chunk=[]
                    for row in rows:
                        value=float(next(v['video_results'] for v in values if v['video_path']==row['video_path']))
                        normalized=value/100 if args.metric=='imaging_quality' else value
                        raw=value*10 if args.metric=='aesthetic_quality' else value
                        assert math.isfinite(value)
                        assert (-1<=normalized<=1) if args.metric=='overall_consistency' else (0<=normalized<=1)
                        if args.metric=='dynamic_degree':assert value in [0,1]
                        chunk.append({k:row[k] for k in ['pair_id','prompt_id','stratum','prompt','seed','arm','video_path']})
                        chunk[-1].update(metric=args.metric,raw_score=raw,normalized_score=normalized,
                            source_returned_score=value,video_sha256=row['video']['sha256'])
                    assert abs(float(aggregate)-sum(v['normalized_score'] for v in chunk)/6)<1e-6
                    monitor.check()
                    ensure_no_other_process(admission['uuid'])
                    json_write(output/(prompt_id+'.json'),{'rows':chunk,'aggregate':float(aggregate),'scoring_seconds':elapsed})
                    report['rows'].extend(chunk)
                    state['completed_videos']=len(report['rows'])
                    print('S2_EVAL_PROGRESS',args.metric,prompt_id,len(report['rows']),flush=True)
                assert len(report['rows'])==96 and not NETWORK_ATTEMPTS
                report.update(status='passed',network_attempts=NETWORK_ATTEMPTS,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                json_write(output/'result.json',report)
                state.update(phase='completed')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),network_attempts=NETWORK_ATTEMPTS)
        json_write(output/'failure.json',report)
        raise


if __name__=='__main__':main()
