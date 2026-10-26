"""Bounded H2 block: native H1 math, two paired prompts, one admitted GPU."""
import argparse
import gc
import importlib
import json
import os
import sys
import time
import traceback
import uuid

import hya_common as h1
from hya_generate_smoke import export_video, check_video
from hya_h2_common import (ROOT, RUN, ARMS, ResourceConflict, read, check_frozen, admission,
                          ensure_exclusive, loading_slot, task_claim, digest, write_json, commit_pair, progress)


def prepare_inputs(row, seed, gpu):
    import torch
    pointer = RUN/'inputs'/(row['id']+'.json')
    if pointer.exists():
        info = read(pointer)
        assert info['prompt'] == row['generation_prompt'] and info['seed'] == seed
        assert digest(info['tensor_file']) == info['file_sha256']
        return torch.load(info['tensor_file'], map_location='cpu', weights_only=True), info
    h1.native_modules()
    text_module = importlib.import_module('hya_native.text_encoder')
    constants = importlib.import_module('hya_native.constants')
    prompt = row['generation_prompt']
    start = time.perf_counter()
    ensure_exclusive(gpu)
    with loading_slot():
        enc = text_module.TextEncoder('llm', max_length=351, text_encoder_precision='fp16',
            text_encoder_path=str(h1.MODEL/'text_encoder'), tokenizer_path=str(h1.MODEL/'tokenizer'),
            prompt_template=constants.PROMPT_TEMPLATE['dit-llm-encode'],
            prompt_template_video=constants.PROMPT_TEMPLATE['dit-llm-encode-video'], hidden_state_skip_layer=2,
            apply_final_norm=False, device='cuda')
    tokens = enc.text2tokens(prompt, data_type='video')
    full = enc.tokenizer(constants.PROMPT_TEMPLATE['dit-llm-encode-video']['template'].format(prompt), truncation=False)['input_ids']
    assert len(full) <= 351
    ensure_exclusive(gpu); torch.cuda.synchronize(); timer = time.perf_counter()
    emb = enc.encode(tokens, data_type='video'); torch.cuda.synchronize()
    llama_seconds = time.perf_counter()-timer
    payload = {'context': emb.hidden_state.cpu(), 'mask': emb.attention_mask.cpu(), 'llama_input_ids': tokens['input_ids']}
    del enc, emb; gc.collect(); torch.cuda.empty_cache()
    ensure_exclusive(gpu)
    with loading_slot():
        enc = text_module.TextEncoder('clipL', max_length=77, text_encoder_precision='fp16',
            text_encoder_path=str(h1.MODEL/'text_encoder_2'), tokenizer_path=str(h1.MODEL/'tokenizer_2'), device='cuda')
    tokens = enc.text2tokens(prompt, data_type='video')
    clip_count = len(enc.tokenizer(prompt, truncation=False)['input_ids']); assert clip_count <= 77
    torch.cuda.synchronize(); timer = time.perf_counter()
    emb = enc.encode(tokens, data_type='video'); torch.cuda.synchronize()
    clip_seconds = time.perf_counter()-timer
    payload.update(pooled=emb.hidden_state.cpu(), clip_input_ids=tokens['input_ids'])
    del enc, emb; gc.collect(); torch.cuda.empty_cache()
    payload['noise'] = torch.randn(h1.SHAPE, generator=torch.Generator('cpu').manual_seed(seed), dtype=torch.float32)
    assert all(torch.isfinite(v).all() for v in payload.values())
    assert payload['context'].shape == (1, 256, 4096) and payload['pooled'].shape == (1, 768)
    info = {'prompt_id': row['id'], 'prompt': prompt, 'seed': seed,
            'tokens': {'llama_with_template': len(full), 'clip_with_special': clip_count},
            'hashes': {k: h1.tensor_digest(v) for k, v in payload.items()},
            'dtypes': {k: str(v.dtype) for k, v in payload.items()},
            'text_encoder_load_encode_seconds': time.perf_counter()-start,
            'llama_encode_seconds': llama_seconds, 'clip_encode_seconds': clip_seconds}
    pointer.parent.mkdir(exist_ok=True)
    tensor_file = pointer.with_name(row['id']+'.'+uuid.uuid4().hex+'.pt')
    with tensor_file.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    info.update(tensor_file=str(tensor_file), file_sha256=digest(tensor_file))
    write_json(pointer, info)
    return payload, info


def generate(row, arm, model, vae, inputs, input_info, admission_info, out, warmup, log_progress):
    import torch
    gpu = admission_info['index']; steps = 50 if arm == 'T50' else 5
    report = {'status': 'starting', 'arm': arm, 'prompt_id': row['id'], 'seed': input_info['seed'],
              'prompt': row['generation_prompt'], 'inputs': input_info, 'admission': admission_info,
              'protocol_sha256': digest(RUN/'protocol.json'), 'started_unix': time.time()}
    ensure_exclusive(gpu); log_progress('model_to_gpu', prompt_id=row['id'], arm=arm)
    timer = time.perf_counter(); model.to('cuda', dtype=torch.bfloat16); torch.cuda.synchronize()
    report['model_to_gpu_seconds'] = time.perf_counter()-timer
    context = inputs['context'].to('cuda'); mask = inputs['mask'].to('cuda'); pooled = inputs['pooled'].to('cuda')
    noise = inputs['noise'].to('cuda', dtype=context.dtype)
    assert noise.shape == h1.SHAPE and noise.dtype == torch.float16
    report['actual_initial_latent_sha256'] = h1.tensor_digest(noise)
    report['actual_initial_latent_dtype'] = str(noise.dtype)
    freqs = [x.to('cuda') for x in model.get_rotary_pos_embed((33, 45, 80))]
    guidance = torch.tensor([6.0], device='cuda', dtype=torch.bfloat16)*1000
    _, _, scheduler_cls = h1.native_modules()
    scheduler = scheduler_cls(shift=7, reverse=True, solver='euler'); scheduler.set_timesteps(steps, 'cuda')
    def forward(z, t):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(z, t.repeat(1), text_states=context, text_mask=mask, text_states_2=pooled,
                         freqs_cos=freqs[0], freqs_sin=freqs[1], guidance=guidance, return_dict=True)['x']
    warm_hash = None
    report['warmup_seconds'] = 0.0
    if warmup:
        ensure_exclusive(gpu); log_progress('warmup', prompt_id=row['id'], arm=arm)
        torch.cuda.synchronize(); timer = time.perf_counter()
        warm = forward(noise.clone(), scheduler.timesteps[0]); torch.cuda.synchronize()
        report['warmup_seconds'] = time.perf_counter()-timer
        assert torch.isfinite(warm).all(); warm_hash = h1.tensor_digest(warm); del warm
    torch.cuda.reset_peak_memory_stats()
    latent = noise.clone(); records = []; denoise = 0.0
    for i, t in enumerate(scheduler.timesteps):
        ensure_exclusive(gpu); torch.cuda.synchronize(); timer = time.perf_counter()
        pred = forward(latent, t); latent = scheduler.step(pred, t, latent, return_dict=False)[0]
        torch.cuda.synchronize(); elapsed = time.perf_counter()-timer; denoise += elapsed
        assert latent.dtype == torch.float32 and torch.isfinite(latent).all() and torch.isfinite(pred).all()
        if i == 0:
            report['first_prediction_sha256'] = h1.tensor_digest(pred)
            if warm_hash is not None:
                assert report['first_prediction_sha256'] == warm_hash
                report['warmup_repeat_exact'] = True
        records.append({'step': i, 'timestep': float(t), 'sigma': float(scheduler.sigmas[i]),
                        'next_sigma': float(scheduler.sigmas[i+1]), 'seconds': elapsed,
                        'prediction_dtype': str(pred.dtype), 'state_dtype': str(latent.dtype)})
        del pred
        log_progress('denoising', prompt_id=row['id'], arm=arm, completed=i+1, total=steps, step_seconds=elapsed)
    report.update(denoising_seconds=denoise, steps=records, formal_path_dit_calls=len(records),
                  extra_warmup_dit_calls=int(warmup), final_latent_sha256=h1.tensor_digest(latent),
                  denoise_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  denoise_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    latent_cpu = latent.cpu(); del latent, noise, context, mask, pooled, freqs, guidance
    timer = time.perf_counter(); model.to('cpu'); torch.cuda.synchronize()
    report['model_offload_seconds'] = time.perf_counter()-timer
    gc.collect(); torch.cuda.empty_cache(); ensure_exclusive(gpu)
    log_progress('vae_decode', prompt_id=row['id'], arm=arm)
    timer = time.perf_counter(); vae.to('cuda'); torch.cuda.synchronize()
    report['vae_to_gpu_seconds'] = time.perf_counter()-timer
    z = latent_cpu.to('cuda')/vae.config.scaling_factor
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); timer = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16):
        decoded = vae.decode(z, return_dict=False)[0]
    torch.cuda.synchronize(); report['vae_decode_seconds'] = time.perf_counter()-timer
    assert decoded.shape == (1, 3, 129, 720, 1280) and torch.isfinite(decoded).all()
    report['decode_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    decoded = decoded.cpu(); del z, latent_cpu
    timer = time.perf_counter(); vae.to('cpu'); torch.cuda.synchronize()
    report['vae_offload_seconds'] = time.perf_counter()-timer
    gc.collect(); torch.cuda.empty_cache(); ensure_exclusive(gpu)
    log_progress('export', prompt_id=row['id'], arm=arm)
    video = out/arm/(row['id']+'.mp4'); video.parent.mkdir(exist_ok=True)
    timer = time.perf_counter(); report['encoding'] = export_video(decoded, video)
    report['export_seconds'] = time.perf_counter()-timer; del decoded
    report['video'] = check_video(video); report['video_path'] = str(video)
    report['hot_denoise_decode_seconds'] = denoise+report['vae_decode_seconds']
    report['video_wall_seconds'] = time.time()-report['started_unix']
    report['network_attempts'] = h1.NETWORK_ATTEMPTS; assert not h1.NETWORK_ATTEMPTS
    ensure_exclusive(gpu); report['status'] = 'passed'; report['finished_unix'] = time.time()
    write_json(video.with_suffix('.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--block', required=True)
    parser.add_argument('--gpu', type=int, required=True); parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args(); manifest, protocol = check_frozen(); h1.configure()
    block = next(x for x in manifest['blocks'] if x['id'] == args.block)
    rows = [p for pid in block['prompt_ids'] for p in protocol['prompts'] if p['id'] == pid and not (RUN/'pairs'/(pid+'.json')).exists()]
    out = RUN/'generation'/args.block/f'attempt{args.attempt:02d}'; out.mkdir(parents=True, exist_ok=False)
    log = lambda stage, **kw: progress(out/'progress.json', stage, block=args.block, **kw)
    try:
        with task_claim('gen_'+args.block), admission(args.gpu) as admitted:
            import torch
            torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
            seed = protocol['generation']['seeds'][0]; torch.manual_seed(seed)
            contract = h1.scheduler_contract(); assert contract == read(ROOT/'Project_Plan/HYA_RUNTIME_CONTRACT.json')['schedule']
            inputs = {}
            for row in rows:
                log('shared_inputs', prompt_id=row['id']); inputs[row['id']] = prepare_inputs(row, seed, args.gpu)
            from diffusers import AutoencoderKLHunyuanVideo
            with loading_slot():
                vae = AutoencoderKLHunyuanVideo.from_pretrained(str(h1.MODEL/'vae'), torch_dtype=torch.float16, local_files_only=True)
            vae.eval().requires_grad_(False); vae.enable_tiling()
            runtime = read(ROOT/'Project_Plan/HYA_RUNTIME_CONTRACT.json')
            assert all(getattr(vae, k) == v for k, v in runtime['vae_tiling'].items())
            order = ['S5', 'T50'] if (block['ordinal']+block['shard']) % 2 == 0 else ARMS
            paired = {p['id']: {} for p in rows}; loads = {}
            for arm in order:
                ensure_exclusive(args.gpu); log('native_model_load', arm=arm)
                timer = time.perf_counter()
                with loading_slot():
                    model, proof = h1.load_native(arm)
                loads[arm] = time.perf_counter()-timer
                write_json(out/(arm+'_weight_mapping.json'), proof)
                for i, row in enumerate(rows):
                    payload, info = inputs[row['id']]
                    rec = generate(row, arm, model, vae, payload, info, admitted, out, i == 0, log)
                    paired[row['id']][arm] = rec
                    if len(paired[row['id']]) == 2:
                        commit_pair(row['id'], paired[row['id']]); log('pair_committed', prompt_id=row['id'])
                del model; gc.collect(); torch.cuda.empty_cache()
            write_json(out/'result.json', {'status': 'passed', 'block': args.block, 'prompt_ids': block['prompt_ids'],
                       'admission': admitted, 'arm_order': order, 'model_cpu_load_seconds': loads, 'finished_unix': time.time()})
            log('passed')
    except BaseException as exc:
        write_json(out/'failure.json', {'status': 'conflict' if isinstance(exc, ResourceConflict) else 'failed',
                   'error': repr(exc), 'traceback': traceback.format_exc(), 'time_unix': time.time()})
        raise SystemExit(75 if isinstance(exc, ResourceConflict) else 1)


if __name__ == '__main__':
    main()
