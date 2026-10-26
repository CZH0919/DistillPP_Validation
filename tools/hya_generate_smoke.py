"""One non-formal 720p HYA H1 video, always on exclusive physical GPU 6."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from hya_common import (ROOT,RUN,MODEL,PROMPT,SEED,SHAPE,NETWORK_ATTEMPTS,configure,gpu_admission,
                        ensure_exclusive,write_json,digest,tensor_digest,native_modules,load_native,
                        prepare_inputs,scheduler_contract)


def export_video(video, path, fps=24):
    import imageio_ffmpeg
    assert video.shape==(1,3,129,720,1280)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error','-n',
         '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-',
         '-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i in range(129):
            frame=((video[0,:,i].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).contiguous().numpy()
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        error=proc.stderr.read().decode(errors='replace'); rc=proc.wait(timeout=180)
        assert rc==0, error
    except BaseException:
        if proc.poll() is None:
            proc.terminate();proc.wait(timeout=20)
        raise
    return {'codec':'libx264','crf':18,'preset':'medium','pixel_format':'yuv420p','fps':24}


def check_video(path):
    import cv2
    import decord
    import numpy as np
    cv2.setNumThreads(1)
    reader=decord.VideoReader(str(path),ctx=decord.cpu(0),num_threads=1)
    assert len(reader)==129 and abs(reader.get_avg_fps()-24)<1e-6
    selected=reader.get_batch([0,64,128]).asnumpy()
    assert selected.shape==(3,720,1280,3) and np.isfinite(selected).all()
    assert float(selected.std())>1.0,'Degenerate constant smoke video'
    cap=cv2.VideoCapture(str(path));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(720,1280,3)
        count+=1
    cap.release();assert count==129
    return {'sha256':digest(path),'bytes':path.stat().st_size,'frames_decord':len(reader),'frames_opencv':count,
            'width':1280,'height':720,'fps':24,'sample_pixel_std':float(selected.std())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['T50','S5'],required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args();configure()
    assert args.attempt>=1
    out=RUN/'generation'/args.arm/f'attempt{args.attempt:02d}'
    out.mkdir(parents=True,exist_ok=False)
    for f in ['hya_common.py','hya_generate_smoke.py']:
        shutil.copyfile(ROOT/'tools'/f,out/('source_'+f))
    report={'arm':args.arm,'status':'starting','scope':'H1_ACCEPTANCE_ONLY_NOT_QUALIFICATION',
            'prompt':PROMPT,'seed':SEED,'protocol_sha256':digest(ROOT/'Project_Plan/HUNYUAN_ACCVIDEO_PROTOCOL.json'),
            'source_hashes':{f:digest(out/('source_'+f)) for f in ['hya_common.py','hya_generate_smoke.py']}}
    def progress(stage,**kwargs):
        write_json(out/'progress.json',{'stage':stage,'arm':args.arm,'time_unix':time.time(),**kwargs},replace=True)
        print('H1_PROGRESS',stage,json.dumps(kwargs),flush=True)
    try:
        assert json.loads((RUN/'resource_verification.json').read_text())['all_assets_verified']
        with gpu_admission() as admission:
            report['admission']=admission
            import torch
            torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
            torch.manual_seed(SEED)
            report['torch']=torch.__version__;report['cuda']=torch.version.cuda
            report['scheduler_contract']=scheduler_contract()
            progress('shared_inputs')
            inputs,input_info=prepare_inputs();report['inputs']=input_info
            ensure_exclusive();progress('native_model_load')
            start=time.perf_counter();model,mapping=load_native(args.arm)
            write_json(out/'weight_mapping.json',mapping)
            report['weight_mapping']={k:v for k,v in mapping.items() if k!='mapping'}
            ensure_exclusive();model.to('cuda',dtype=torch.bfloat16);torch.cuda.synchronize()
            report['model_load_seconds']=time.perf_counter()-start
            context=inputs['context'].to('cuda');mask=inputs['mask'].to('cuda');pooled=inputs['pooled'].to('cuda')
            noise=inputs['noise'].to('cuda',dtype=context.dtype)
            report['actual_initial_latent_sha256']=tensor_digest(noise)
            report['actual_initial_latent_dtype']=str(noise.dtype)
            assert noise.shape==SHAPE and noise.dtype==torch.float16
            freqs=[x.to('cuda') for x in model.get_rotary_pos_embed((33,45,80))]
            guidance=torch.tensor([6.0],device='cuda',dtype=torch.bfloat16)*1000
            _,_,scheduler_cls=native_modules();steps=50 if args.arm=='T50' else 5
            scheduler=scheduler_cls(shift=7,reverse=True,solver='euler');scheduler.set_timesteps(steps,'cuda')
            def forward(z,t):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    return model(z,t.repeat(1),text_states=context,text_mask=mask,text_states_2=pooled,
                                 freqs_cos=freqs[0],freqs_sin=freqs[1],guidance=guidance,return_dict=True)['x']
            ensure_exclusive();progress('full_shape_warmup')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            warm=forward(noise.clone(),scheduler.timesteps[0]);torch.cuda.synchronize()
            assert torch.isfinite(warm).all()
            report['warmup_seconds']=time.perf_counter()-start
            warm_hash=tensor_digest(warm);del warm
            latent=noise.clone();records=[];denoise=0
            for i,t in enumerate(scheduler.timesteps):
                ensure_exclusive()
                torch.cuda.synchronize();start=time.perf_counter()
                pred=forward(latent,t)
                latent=scheduler.step(pred,t,latent,return_dict=False)[0]
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;denoise+=elapsed
                assert latent.dtype==torch.float32 and torch.isfinite(latent).all() and torch.isfinite(pred).all()
                if i==0:
                    report['first_prediction_sha256']=tensor_digest(pred)
                    report['warmup_repeat_exact']=report['first_prediction_sha256']==warm_hash
                    assert report['warmup_repeat_exact'],'Repeated first step differs; inspect before accepting'
                records.append({'step':i,'timestep':float(t),'sigma':float(scheduler.sigmas[i]),'next_sigma':float(scheduler.sigmas[i+1]),
                                'seconds':elapsed,'prediction_dtype':str(pred.dtype),'state_dtype':str(latent.dtype)})
                del pred
                progress('denoising',completed=i+1,total=steps,step_seconds=elapsed)
            report['denoising_seconds']=denoise;report['steps']=records
            report['formal_path_dit_calls']=len(records);report['extra_warmup_dit_calls']=1
            report['final_latent_sha256']=tensor_digest(latent)
            report['denoise_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            report['denoise_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            latent_cpu=latent.cpu();del model,latent,noise,context,mask,pooled,freqs;gc.collect();torch.cuda.empty_cache()
            ensure_exclusive();progress('vae_load_decode')
            from diffusers import AutoencoderKLHunyuanVideo
            start=time.perf_counter()
            vae=AutoencoderKLHunyuanVideo.from_pretrained(str(MODEL/'vae'),torch_dtype=torch.float16,local_files_only=True)
            vae.eval().requires_grad_(False).to('cuda');vae.enable_tiling();torch.cuda.synchronize()
            report['vae_load_seconds']=time.perf_counter()-start
            report['vae_tiling']={k:v for k,v in vars(vae).items() if k.startswith('tile_') and isinstance(v,(int,float,bool))}
            z=latent_cpu.to('cuda')/vae.config.scaling_factor
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):
                decoded=vae.decode(z,return_dict=False)[0]
            torch.cuda.synchronize();report['vae_decode_seconds']=time.perf_counter()-start
            assert decoded.shape==(1,3,129,720,1280) and torch.isfinite(decoded).all()
            report['decode_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            decoded=decoded.cpu();del vae,z;gc.collect();torch.cuda.empty_cache()
            ensure_exclusive();progress('export_and_validate')
            start=time.perf_counter();video=out/'H1_smoke.mp4'
            report['encoding']=export_video(decoded,video);report['export_seconds']=time.perf_counter()-start
            del decoded
            report['video']=check_video(video)
            report['precision']={'dit':'BF16 weights/autocast','text_encoders':'FP16 native defaults','initial_latent':'FP16 after shared FP32 source cast','persistent_state':'FP32 after step','vae':'FP16 autocast, FP32 latent divided by scaling_factor=0.476986'}
            report['attention']='unmodified native AccVideo full bidirectional FlashAttention 2; no caching, quantization or sparse attention'
            report['hot_denoise_decode_seconds']=report['denoising_seconds']+report['vae_decode_seconds']
            report['network_attempts']=NETWORK_ATTEMPTS;assert not NETWORK_ATTEMPTS
            ensure_exclusive();report['status']='passed'
            write_json(out/'result.json',report);progress('passed',video=str(video))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),network_attempts=NETWORK_ATTEMPTS)
        write_json(out/'failure.json',report)
        raise


if __name__=='__main__':
    main()
