"""HYA H1 safety, native Hunyuan loading and shared input helpers."""
import contextlib
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import types
import uuid

ROOT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
RUN = ROOT / 'runs/hya_h1_20261019'
MODEL = Path('/data/chenjiayu/models/HunyuanVideo')
STUDENT = Path('/data/chenjiayu/models/AccVideo/accvideo-t2v-5-steps/diffusion_pytorch_model.pt')
REFERENCE = ROOT / 'Project_Plan/hya_sources/AccVideo_reference'
PROMPT = 'A brown dog and a white cat walk together along a stone path in a sunlit garden. A steady wide shot keeps both animals visible as they move past green plants.'
SEED = 2026101900
SHAPE = (1, 16, 33, 90, 160)
METRICS = ['subject_consistency', 'dynamic_degree', 'background_consistency', 'scene', 'multiple_objects']
NETWORK_ATTEMPTS = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj, replace=False):
    path = Path(path)
    assert path.is_relative_to(ROOT) and path.resolve() == path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        if replace:
            assert not path.is_symlink()
            os.replace(temp, path)
        else:
            os.link(temp, path); temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()


def configure():
    assert ROOT.resolve() == ROOT and RUN.resolve() == RUN
    for k,v in {'CUDA_DEVICE_ORDER':'PCI_BUS_ID', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                'HF_HUB_DISABLE_TELEMETRY':'1', 'TOKENIZERS_PARALLELISM':'false', 'OMP_NUM_THREADS':'4',
                'MKL_NUM_THREADS':'4', 'PYTHONDONTWRITEBYTECODE':'1', 'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
                'HF_HOME':str(ROOT/'resources/software_cache/hya_h1/hf'),
                'TORCH_HOME':str(ROOT/'resources/software_cache/hya_h1/torch'),
                'XDG_CACHE_HOME':str(ROOT/'resources/software_cache/hya_h1/xdg'),
                'VBENCH_CACHE_DIR':str(ROOT/'resources/evaluation')}.items():
        os.environ[k]=v
    original, original_ex = socket.socket.connect, socket.socket.connect_ex
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            NETWORK_ATTEMPTS.append(str(address))
            raise RuntimeError('HYA offline guard rejected network: '+str(address))
        return original(sock,address)
    def connect_ex(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):
            NETWORK_ATTEMPTS.append(str(address))
            raise RuntimeError('HYA offline guard rejected network: '+str(address))
        return original_ex(sock,address)
    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def gpu_snapshot():
    def query(kind, fields):
        p = subprocess.run(['nvidia-smi','-i','6','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],
                           text=True,capture_output=True,timeout=15)
        if p.returncode:
            raise RuntimeError('GPU 6 query failed: '+p.stderr)
        return [[s.strip() for s in line.split(',')] for line in p.stdout.splitlines() if line.strip()]
    return {'gpu':query('gpu','index,uuid,memory.used,memory.total,utilization.gpu')[0],
            'processes':query('compute-apps','pid,used_gpu_memory')}


def ensure_exclusive():
    info=gpu_snapshot()
    other=[p for p in info['processes'] if int(p[0])!=os.getpid()]
    if other:
        raise RuntimeError('Another process on GPU 6; stopping only this H1 process: '+str(other))
    return info


@contextlib.contextmanager
def gpu_admission(min_free_gib=100):
    RUN.mkdir(parents=True,exist_ok=True)
    lock=RUN/'gpu6.lock'
    assert not lock.is_symlink()
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i in range(2):
            info=gpu_snapshot(); g=info['gpu']
            assert int(g[0])==6
            if info['processes'] or float(g[2])>32 or float(g[4])!=0:
                raise RuntimeError('GPU 6 occupied; no fallback to another GPU: '+str(info))
            if (float(g[3])-float(g[2]))/1024 < min_free_gib:
                raise RuntimeError('Insufficient free GPU 6 memory')
            if i==0: time.sleep(2)
        available=next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        assert available > 160*2**20 and shutil.disk_usage(ROOT).free>50*2**30
        os.environ['CUDA_VISIBLE_DEVICES']=g[1]
        yield {'physical_gpu':6,'uuid':g[1],'pid':os.getpid(),'initial_snapshot':info,
               'lock_scope':'this project only; not a cluster reservation'}


def tensor_digest(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def native_modules():
    manifest=json.loads((ROOT/'Project_Plan/hya_sources/AccVideo_reference_manifest.json').read_text())
    for row in manifest['files']:
        assert digest(ROOT/'Project_Plan/hya_sources'/row['path'])==row['sha256']
    if 'hya_native' not in sys.modules:
        package=types.ModuleType('hya_native');package.__path__=[str(REFERENCE/'models/hunyuan')]
        sys.modules['hya_native']=package
    module=importlib.import_module('hya_native.modules.models')
    scheduler_path=REFERENCE/'models/hunyuan/diffusion/schedulers/scheduling_flow_match_discrete.py'
    spec=importlib.util.spec_from_file_location('hya_native_flow',scheduler_path)
    scheduler=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scheduler;spec.loader.exec_module(scheduler)
    return module.HYVideoDiffusionTransformer, module.HUNYUAN_VIDEO_CONFIG['HYVideo-T/2-cfgdistill'], scheduler.FlowMatchDiscreteScheduler


def native_skeleton():
    import torch
    cls, settings, _=native_modules()
    args=types.SimpleNamespace(text_states_dim=4096,text_states_dim_2=768,rope_theta=256)
    with torch.device('meta'):
        model=cls(args,in_channels=16,out_channels=16,dtype=torch.bfloat16,device='meta',**settings)
    return model


def inverse_teacher_state(native_template):
    """Invert the pinned official Diffusers mapping, verify every source tensor bitwise."""
    import torch
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import convert_hunyuan_video_transformer_to_diffusers as convert
    index=json.loads((MODEL/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
    source={}
    for filename in sorted(set(index['weight_map'].values())):
        part=load_file(str(MODEL/'transformer'/filename),device='cpu')
        assert not (source.keys() & part.keys())
        source.update(part)
    assert set(source)==set(index['weight_map'])
    native,used,mapping={},set(),[]
    for key,template in native_template.items():
        # The upstream converter is per-tensor: rename, split dim 0, and one scale/shift swap.
        pieces=convert({key:template})
        keys=list(pieces)
        assert not (used & set(keys)); used.update(keys)
        assert all(tuple(source[k].shape)==tuple(pieces[k].shape) for k in keys)
        if len(keys)>1:
            value=torch.cat([source[k] for k in keys],dim=0)
        else:
            value=source[keys[0]]
            if 'final_layer.adaLN_modulation.1' in key:
                scale,shift=value.chunk(2,dim=0);value=torch.cat([shift,scale],dim=0)
        assert value.shape==template.shape and value.dtype==torch.bfloat16
        roundtrip=convert({key:value})
        assert set(roundtrip)==set(keys)
        assert all(torch.equal(roundtrip[k],source[k]) for k in keys), 'Lossy mapping: '+key
        native[key]=value
        mapping.append({'native_key':key,'diffusers_keys':keys,'shape':list(value.shape),'bitwise_roundtrip':True})
    assert used==set(source)
    return native,{'native_tensors':len(native),'source_tensors':len(source),'all_tensors_bitwise_roundtrip':True,'mapping':mapping}


def load_native(arm):
    import torch
    model=native_skeleton()
    if arm=='T50':
        state,proof=inverse_teacher_state(model.state_dict())
    else:
        state=torch.load(STUDENT,map_location='cpu',weights_only=True,mmap=True)
        proof={'native_tensors':len(state),'loader':'original author PT; weights_only=True, mmap=True'}
    result=model.load_state_dict(state,strict=True,assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval().requires_grad_(False)
    return model,proof


def scheduler_contract():
    import torch
    _,_,cls=native_modules()
    records={}
    for n in [5,50]:
        scheduler=cls(shift=7,reverse=True,solver='euler');scheduler.set_timesteps(n,'cpu')
        sigma=scheduler.sigmas.clone(); expected=7*torch.linspace(1,0,n+1)/(1+6*torch.linspace(1,0,n+1))
        assert torch.equal(sigma,expected)
        sample=torch.linspace(-1,1,23).half(); max_error=0
        for i,t in enumerate(scheduler.timesteps):
            velocity=torch.full_like(sample,0.125,dtype=torch.bfloat16)
            calculated=sample.float()+velocity.float()*(sigma[i+1]-sigma[i])
            actual=scheduler.step(velocity,t,sample,return_dict=False)[0]
            max_error=max(max_error,(calculated-actual).abs().max().item())
            assert actual.dtype==torch.float32 and torch.equal(actual,calculated)
            sample=actual
        records[str(n)]={'sigmas':sigma.tolist(),'timesteps':scheduler.timesteps.tolist(),'max_euler_error':max_error,'persistent_state':'float32 after every native Euler step'}
    return records


def prepare_inputs():
    import torch
    path=RUN/'inputs.pt'; metadata_path=RUN/'inputs.json'
    if path.exists():
        assert not path.is_symlink() and metadata_path.exists()
        info=json.loads(metadata_path.read_text());assert digest(path)==info['file_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert info['prompt']==PROMPT and info['seed']==SEED
        return payload,info
    native_modules()
    text_module=importlib.import_module('hya_native.text_encoder')
    constants=importlib.import_module('hya_native.constants')
    start=time.perf_counter()
    enc=text_module.TextEncoder('llm',max_length=351,text_encoder_precision='fp16',
        text_encoder_path=str(MODEL/'text_encoder'),tokenizer_path=str(MODEL/'tokenizer'),
        prompt_template=constants.PROMPT_TEMPLATE['dit-llm-encode'],
        prompt_template_video=constants.PROMPT_TEMPLATE['dit-llm-encode-video'],hidden_state_skip_layer=2,
        apply_final_norm=False,device='cuda')
    tokens=enc.text2tokens(PROMPT,data_type='video')
    full=enc.tokenizer(constants.PROMPT_TEMPLATE['dit-llm-encode-video']['template'].format(PROMPT),truncation=False)['input_ids']
    assert len(full)<=351
    torch.cuda.synchronize(); encode_start=time.perf_counter()
    emb=enc.encode(tokens,data_type='video')
    torch.cuda.synchronize(); llama_seconds=time.perf_counter()-encode_start
    payload={'context':emb.hidden_state.cpu(),'mask':emb.attention_mask.cpu(),'llama_input_ids':tokens['input_ids']}
    del enc,emb;torch.cuda.empty_cache()
    enc2=text_module.TextEncoder('clipL',max_length=77,text_encoder_precision='fp16',
        text_encoder_path=str(MODEL/'text_encoder_2'),tokenizer_path=str(MODEL/'tokenizer_2'),device='cuda')
    tokens2=enc2.text2tokens(PROMPT,data_type='video')
    count2=len(enc2.tokenizer(PROMPT,truncation=False)['input_ids']);assert count2<=77
    torch.cuda.synchronize();encode_start=time.perf_counter()
    emb2=enc2.encode(tokens2,data_type='video');torch.cuda.synchronize()
    clip_seconds=time.perf_counter()-encode_start
    payload['pooled']=emb2.hidden_state.cpu();payload['clip_input_ids']=tokens2['input_ids']
    del enc2,emb2;torch.cuda.empty_cache()
    payload['noise']=torch.randn(SHAPE,generator=torch.Generator('cpu').manual_seed(SEED),dtype=torch.float32)
    for k,v in payload.items(): assert bool(torch.isfinite(v).all())
    protocol=json.loads((ROOT/'Project_Plan/HUNYUAN_ACCVIDEO_PROTOCOL.json').read_text())
    assert PROMPT not in {x['generation_prompt'] for x in protocol['prompts']}
    info={'prompt':PROMPT,'seed':SEED,'scope':'H1 smoke only, outside formal panel',
          'tokens':{'llama_with_template':len(full),'llama_crop_start':95,'llama_max_with_template':351,'clip_with_special':count2},
          'hashes':{k:tensor_digest(v) for k,v in payload.items()},'dtypes':{k:str(v.dtype) for k,v in payload.items()},
          'text_encoder_load_encode_seconds':time.perf_counter()-start,'llama_encode_seconds':llama_seconds,'clip_encode_seconds':clip_seconds,
          'initial_latent_policy':'same CPU FP32 source explicitly supplied; cast to native prompt dtype FP16 before first step; subsequent native Euler state FP32'}
    assert payload['context'].shape==(1,256,4096) and payload['pooled'].shape==(1,768)
    with path.open('xb') as f:torch.save(payload,f)
    info['file_sha256']=digest(path);write_json(metadata_path,info)
    return payload,info
