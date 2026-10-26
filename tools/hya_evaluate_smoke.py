"""Offline five-metric H1 acceptance on the two independent smoke videos."""
import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from hya_common import ROOT,RUN,PROMPT,METRICS,NETWORK_ATTEMPTS,configure,gpu_admission,ensure_exclusive,write_json,digest

AUDIT=[]


def protect_forward(module):
    last=[0.0]
    def check(_module,_inputs):
        now=time.monotonic()
        if now-last[0]>=5:
            ensure_exclusive();last[0]=now
    module.register_forward_pre_hook(check)
    return module


def prepare_imports():
    sys.path[:0]=[str(ROOT/'src'/p) for p in ['CLIP','VBench','DINO']]
    import torch
    from transformers import BertTokenizer
    bert=ROOT/'resources/evaluation/bert-base-uncased'
    original_tokenizer=BertTokenizer.from_pretrained.__func__
    def tokenizer(cls,name,*a,**kw):
        if str(name)=='bert-base-uncased':
            AUDIT.append({'adapter':'BertTokenizer local path','original':str(name),'path':str(bert)})
            name=str(bert);kw['local_files_only']=True
        return original_tokenizer(cls,name,*a,**kw)
    BertTokenizer.from_pretrained=classmethod(tokenizer)
    original_load=torch.load
    allow={str((ROOT/'resources/evaluation'/p).resolve()) for p in ['caption_model/tag2text_swin_14m.pth','grit_model/grit_b_densecap_objectdet.pth']}
    def load(f,*a,**kw):
        name=getattr(f,'name',f)
        if isinstance(name,(str,os.PathLike)) and str(Path(name).resolve()) in allow and 'weights_only' not in kw:
            # Original VBench checkpoints are full author checkpoints, already hash verified.
            kw['weights_only']=False
            AUDIT.append({'adapter':'trusted allowlisted legacy checkpoint loader','path':str(Path(name).resolve())})
        return original_load(f,*a,**kw)
    torch.load=load
    return torch


def tag_model(device):
    tag=importlib.import_module('vbench.third_party.tag2Text.tag2text')
    original=tag.BertModel.from_pretrained.__func__
    def local_model(cls,name,*a,**kw):
        if str(name)=='bert-base-uncased':
            name=str(ROOT/'resources/evaluation/bert-base-uncased');kw['local_files_only']=True
            AUDIT.append({'adapter':'Tag2Text BERT local path','path':name})
        return original(cls,name,*a,**kw)
    tag.BertModel.from_pretrained=classmethod(local_model)
    load=tag.load_checkpoint_swinbase
    def checked(*a,**kw):
        model,msg=load(*a,**kw)
        missing=list(msg.missing_keys);unexpected=list(msg.unexpected_keys)
        allowed=lambda k:k.endswith(('relative_position_index','attn_mask','position_ids'))
        assert not [k for k in missing if not allowed(k)],'Missing learned Tag2Text weights: '+str(missing)
        AUDIT.append({'tag2text_missing_nonlearned_buffers':missing,'tag2text_unexpected':unexpected})
        return model,msg
    tag.load_checkpoint_swinbase=checked
    model=tag.tag2text_caption(pretrained=str(ROOT/'resources/evaluation/caption_model/tag2text_swin_14m.pth'),image_size=384,vit='swin_b')
    model.eval().requires_grad_(False).to(device)
    if device.type=='cuda':
        protect_forward(model.visual_encoder);protect_forward(model.text_decoder)
    return model


def grit_model(device):
    from detectron2.checkpoint import DetectionCheckpointer
    original=DetectionCheckpointer._load_model
    def load(self,*a,**kw):
        checkpoint=a[0] if a else kw['checkpoint']
        supplied_keys=set(checkpoint['model'])
        incompatible=original(self,*a,**kw)
        missing=list(incompatible.missing_keys);unexpected=list(incompatible.unexpected_keys)
        bad_shapes=getattr(incompatible,'incorrect_shapes',[])
        shared_aliases=[]
        state=self.model.state_dict(keep_vars=True)
        def allowed(k):
            if k in ['pixel_mean','pixel_std'] or k.endswith('position_ids'):
                return True
            if k.startswith('roi_heads.text_decoder_det.textual.'):
                paired=k.replace('roi_heads.text_decoder_det.','roi_heads.text_decoder.',1)
                # VBench constructs both decoder wrappers around the SAME textual module.
                # Accept the absent alias only when its exact shared Parameter was loaded.
                assert paired in supplied_keys and paired not in missing
                assert state[k] is state[paired], 'GRiT alias is not shared: '+k
                shared_aliases.append(k)
                return True
            return False
        assert not bad_shapes and not [k for k in missing if not allowed(k)],'Incomplete GRiT weights: '+str(incompatible)
        AUDIT.append({'grit_missing':missing,'grit_verified_shared_aliases':shared_aliases,
                      'grit_unexpected':unexpected,'grit_incorrect_shapes':bad_shapes})
        return incompatible
    DetectionCheckpointer._load_model=load
    from vbench.third_party.grit_model import DenseCaptioning
    model=DenseCaptioning(device)
    model.initialize_model_det(model_weight=str(ROOT/'resources/evaluation/grit_model/grit_b_densecap_objectdet.pth'))
    if device.type=='cuda':protect_forward(model.demo.predictor.model)
    return model


def evaluate(metric, videos, device, torch):
    weights=ROOT/'resources/evaluation'
    if metric=='subject_consistency':
        import vision_transformer
        from vbench.subject_consistency import subject_consistency
        model=vision_transformer.vit_base(patch_size=16,num_classes=0)
        model.load_state_dict(torch.load(weights/'dino_model/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
        protect_forward(model)
        return subject_consistency(model.eval().requires_grad_(False).to(device),videos,device,read_frame=False)
    if metric=='dynamic_degree':
        from easydict import EasyDict
        from vbench.dynamic_degree import DynamicDegree,dynamic_degree
        model=DynamicDegree(EasyDict(model=str(weights/'raft_model/models/raft-things.pth'),small=False,mixed_precision=False,alternate_corr=False),device)
        protect_forward(model.model)
        return dynamic_degree(model,videos)
    if metric=='background_consistency':
        import clip
        from vbench.background_consistency import background_consistency
        model,preprocess=clip.load(str(weights/'clip_model/ViT-B-32.pt'),device=device,jit=False)
        protect_forward(model.visual)
        return background_consistency(model.eval().requires_grad_(False),preprocess,videos,device,False)
    from vbench.utils import get_frame_indices
    AUDIT.append({'metric':metric,'sampled_frame_indices':[int(x) for x in get_frame_indices(16,129,sample='middle')]})
    if metric=='scene':
        module=importlib.import_module('vbench.scene');model=tag_model(device)
        original=module.get_caption
        def caption(*a,**kw):
            result=original(*a,**kw);AUDIT.append({'scene_frame_captions':result});return result
        module.get_caption=caption
        return module.scene(model,[{'prompt':PROMPT,'video_list':videos,'auxiliary_info':{'scene':{'scene':'garden'}}}],device)
    if metric=='multiple_objects':
        module=importlib.import_module('vbench.multiple_objects');model=grit_model(device)
        original=module.get_dect_from_grit
        def detect(*a,**kw):
            result=original(*a,**kw);AUDIT.append({'object_frame_labels':[sorted(x) for x in result]});return result
        module.get_dect_from_grit=detect
        return module.multiple_objects(model,[{'prompt':PROMPT,'video_list':videos,'auxiliary_info':{'object':'dog and cat'}}],device)
    raise ValueError(metric)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',choices=METRICS)
    parser.add_argument('--cpu-load-check',action='store_true')
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args();configure()
    if args.cpu_load_check:os.environ['CUDA_VISIBLE_DEVICES']=''
    out=RUN/('eval_cpu_preflight' if args.cpu_load_check else 'evaluation/'+str(args.metric))/f'attempt{args.attempt:02d}'
    assert out.resolve()==out;out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'tools/hya_evaluate_smoke.py',out/'source_hya_evaluate_smoke.py')
    report={'scope':'H1_ACCEPTANCE_NOT_QUALIFICATION','metric':args.metric,'status':'starting',
            'source_sha256':digest(out/'source_hya_evaluate_smoke.py')}
    try:
        if args.cpu_load_check:
            torch=prepare_imports();torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
            import detectron2._C as extension
            report['detectron2_compiler']=extension.get_compiler_version();report['detectron2_cuda']=extension.get_cuda_version()
            for name,loader in [('scene',tag_model),('multiple_objects',grit_model)]:
                start=time.perf_counter();obj=loader(torch.device('cpu'))
                report[name+'_cpu_load_seconds']=time.perf_counter()-start;del obj;gc.collect()
            assert not torch.cuda.is_initialized()
        else:
            assert args.metric
            records={}
            for arm in ['T50','S5']:
                candidates=sorted((RUN/'generation'/arm).glob('attempt*/result.json'))
                assert candidates
                record=json.loads(candidates[-1].read_text());assert record['status']=='passed'
                video=candidates[-1].parent/'H1_smoke.mp4';assert digest(video)==record['video']['sha256']
                records[arm]={'record':record,'video':str(video)}
            assert records['T50']['record']['actual_initial_latent_sha256']==records['S5']['record']['actual_initial_latent_sha256']
            videos=[records[a]['video'] for a in ['T50','S5']]
            report['vbench_source_sha256']={f:digest(ROOT/'src/VBench'/f) for f in
                ['vbench/'+args.metric+'.py','vbench/utils.py']}
            with gpu_admission(min_free_gib=40) as admission:
                report['admission']=admission
                torch=prepare_imports();torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
                torch.manual_seed(20261019)
                import cv2
                cv2.setNumThreads(1)
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                aggregate,rows=evaluate(args.metric,videos,torch.device('cuda'),torch)
                torch.cuda.synchronize();report['load_scoring_seconds']=time.perf_counter()-start
                assert len(rows)==2 and {r['video_path'] for r in rows}==set(videos)
                assert all(math.isfinite(float(r['video_results'])) and 0<=float(r['video_results'])<=1 for r in rows)
                assert abs(float(aggregate)-sum(float(r['video_results']) for r in rows)/2)<1e-6
                report.update(aggregate=float(aggregate),videos=rows,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                ensure_exclusive()
        report.update(status='passed',audit=AUDIT,network_attempts=NETWORK_ATTEMPTS)
        assert not NETWORK_ATTEMPTS
        write_json(out/'result.json',report);print('H1_EVALUATION_PASS',args.metric or 'cpu_load',flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),audit=AUDIT,network_attempts=NETWORK_ATTEMPTS)
        write_json(out/'failure.json',report);raise


if __name__=='__main__':
    main()
