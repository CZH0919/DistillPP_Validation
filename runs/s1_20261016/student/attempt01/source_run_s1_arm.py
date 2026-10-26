"""One fixed, non-qualification S1 video. No S2 or teacher-correction entry point."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

from s1_common import (PROJECT, BATCH, PROMPT, SEED, SHAPE, configure, gpu_admission,
                       ensure_no_other_process, tensor_hash, tensor_stats, json_write, file_hash, NETWORK_ATTEMPTS)


def load_teacher(settings, torch):
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from wan.modules.model import WanModel
    config = json.loads(Path(settings['common_components']['native_config']).read_text())
    with init_empty_weights():
        model = WanModel.from_config(config)
    expected = model.state_dict()
    assert all(t.device.type == 'meta' for t in expected.values())
    rules = json.loads((PROJECT / 'Project_Plan/WAN_KEY_RENAME_RULES.json').read_text())
    assert rules['special_keys_remap'] == {}
    mapping = {}
    for key in expected:
        renamed = key
        for old, new in rules['rename_pairs']:
            renamed = renamed.replace(old, new)
        assert renamed not in mapping
        mapping[renamed] = key
    root = Path(settings['teacher']['checkpoint_path'])
    index = json.loads((root / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    assert set(index) == set(mapping)
    loaded = set()
    for shard in sorted(set(index.values())):
        assert Path(shard).name == shard
        state = {}
        with safe_open(root / shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                native = mapping[key]
                assert index[key] == shard and native not in loaded
                tensor = f.get_tensor(key)
                assert tuple(tensor.shape) == tuple(expected[native].shape)
                # Native Wan keeps FP32 weights and uses BF16 autocast around
                # forward; time embeddings and output head explicitly run FP32.
                assert tensor.dtype == torch.float32
                state[native] = tensor.to(device='cuda:0')
                loaded.add(native)
        result = model.load_state_dict(state, strict=False, assign=True)
        assert not result.unexpected_keys
        del state
        print('TEACHER_SHARD_LOADED', shard, len(loaded), flush=True)
    assert loaded == set(expected)
    assert all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters())
    assert model.freqs.device.type != 'meta'
    return model.eval().requires_grad_(False), {'keys': len(loaded), 'missing': 0, 'unexpected': 0,
             'parameter_storage': 'float32 native', 'forward_autocast': 'bfloat16 native',
             'fp32_operations': 'native time embeddings, norm/modulation and output head', 'on_disk_conversion': False}


def load_student(settings, torch):
    import lightx2v.common.ops
    from lightx2v.utils.set_config import build_startup_config
    from lightx2v.models.networks.wan.distill_model import WanDistillModel
    from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
    config = build_startup_config({
        'task': 't2v', 'model_cls': 'wan2.1', 'model_path': settings['student']['model_path'],
        'infer_steps': 4, 'distill_method': 'dmd2', 'enable_cfg': False,
        'size': [480, 832], 'num_frames': 81, 'fps': 16, 'sample_shift': 5,
        'denoising_step_list': [1000, 750, 500, 250],
        'self_attn_1_type': 'flash_attn2', 'cross_attn_1_type': 'flash_attn2',
        'rms_norm_type': 'torch', 'layer_norm_type': 'torch', 'modulate_type': 'torch',
        'rope_type': 'torch_complex_rope', 'double_precision_rope': True,
        'cpu_offload': False, 'parallel': False, 'feature_caching': 'NoCaching',
        'dit_quantized': False, 'lazy_load': False, 'warmup': False,
    })
    assert not config.get('dit_original_ckpt') and not config.get('transformer_model_path')
    class AuditedWanDistillModel(WanDistillModel):
        def _load_ckpt(self, unified_dtype, sensitive_layer):
            state = super()._load_ckpt(unified_dtype, sensitive_layer)
            assert len(state) == 1095
            assert all(v.dtype == torch.bfloat16 and v.device.type == 'cuda' for v in state.values())
            self.s1_load_audit = {'keys': len(state), 'parameter_storage': 'bfloat16',
                                  'loader': 'native WanDistillModel original PT branch'}
            return state
    model = AuditedWanDistillModel(settings['student']['model_path'], config, torch.device('cuda:0'))
    attention_types = []
    for block in model.transformer_weights.blocks:
        for phase, name in [(0, 'self_attn_1'), (1, 'cross_attn_1')]:
            backend = type(getattr(block.compute_phases[phase], name)).__name__
            assert backend == 'FlashAttn2Weight', backend
            attention_types.append(backend)
    model.s1_load_audit['attention_instances'] = len(attention_types)
    model.s1_load_audit['attention_backend'] = 'FlashAttn2Weight'
    scheduler = WanStepDistillScheduler(config)
    model.set_scheduler(scheduler)
    return model, scheduler, dict(config)


def export_video(video, out, torch):
    import imageio.v2 as imageio
    import decord
    import cv2
    assert list(video.shape) == [3, 81, 480, 832]
    frames = ((video.float().clamp(-1, 1).cpu().permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).numpy()
    assert not out.exists()
    with imageio.get_writer(str(out), fps=16, codec='libx264', quality=None, pixelformat='yuv420p',
             macro_block_size=1, ffmpeg_params=['-crf', '18', '-preset', 'medium', '-threads', '4']) as writer:
        for frame in frames:
            writer.append_data(frame)
    vr = decord.VideoReader(str(out), ctx=decord.cpu(0), num_threads=1)
    decoded = vr.get_batch(list(range(len(vr)))).asnumpy()
    assert decoded.shape == (81, 480, 832, 3) and abs(vr.get_avg_fps() - 16) < 1e-6
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(out)); count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (480, 832, 3)
        count += 1
    cap.release()
    assert count == 81
    return {'sha256': file_hash(out), 'bytes': out.stat().st_size, 'frames': count,
            'size': [480, 832], 'fps': float(vr.get_avg_fps()),
            'decord_all_frames_decoded': True, 'opencv_all_frames_decoded': True,
            'pixel_std': float(decoded.std()), 'decoded_first_frame_sha256': __import__('hashlib').sha256(decoded[0].tobytes()).hexdigest()}


def run(args, output, admission, report):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(PROJECT / 'src/Wan2.1'))
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from wan.modules import attention as native_attention
    assert native_attention.FLASH_ATTN_2_AVAILABLE
    # Force the selected, installed FA2 backend for native Wan as planned.
    native_attention.FLASH_ATTN_3_AVAILABLE = False
    settings = json.loads((PROJECT / 'Project_Plan/BASELINE_SETTINGS.json').read_text())
    assert settings['size'] == [480, 832] and settings['num_frames'] == 81
    assert PROMPT not in [p['prompt'] for p in json.loads((PROJECT / 'Project_Plan/PROMPTS_QUALIFICATION.json').read_text())['prompts']]
    report.update(settings=settings, s1_prompt=PROMPT, seed=SEED, torch=torch.__version__,
                  device_name=torch.cuda.get_device_name(0), scope='S1_ONLY_NOT_QUALIFICATION',
                  common_vae='native WanVAE FP32 weights and decode for both arms',
                  timing_kind='S1 first full generation; diagnostics included; not a warmed benchmark')
    report['script_sha256'] = file_hash(Path(__file__))
    report['attention_backend'] = 'flash_attn2'
    report['native_fa3_enabled'] = native_attention.FLASH_ATTN_3_AVAILABLE
    common = BATCH / 'common'; common.mkdir(exist_ok=True)
    zpath = common / 'initial_latent.pt'
    if zpath.exists():
        initial = torch.load(zpath, map_location='cpu', weights_only=True)
    else:
        initial = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(SEED), dtype=torch.float32)
        with zpath.open('xb') as f:
            torch.save(initial, f)
    assert tuple(initial.shape) == SHAPE and initial.dtype == torch.float32
    expected_initial = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(SEED), dtype=torch.float32)
    assert torch.equal(initial, expected_initial)
    report['initial_latent_sha256'] = tensor_hash(initial)
    report['initial_latent'] = tensor_stats(initial)
    components = settings['common_components']
    start = time.perf_counter()
    encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device('cuda:0'),
              checkpoint_path=components['t5'], tokenizer_path=components['tokenizer'])
    vae = WanVAE(vae_pth=components['vae'], dtype=torch.float32, device='cuda:0')
    assert all(p.dtype == torch.float32 for p in vae.model.parameters())
    torch.cuda.synchronize()
    report['common_load_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    context = encoder([PROMPT], torch.device('cuda:0'))
    context_null = encoder([settings['negative_prompt']], torch.device('cuda:0')) if args.arm == 'teacher' else None
    torch.cuda.synchronize()
    report['text_seconds'] = time.perf_counter() - start
    report['positive_context_sha256'] = tensor_hash(context[0])
    report['positive_context'] = tensor_stats(context[0])
    report['common_components_resident'] = True
    start = time.perf_counter()
    if args.arm == 'teacher':
        model, report['load_acceptance'] = load_teacher(settings, torch)
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(50, device='cuda:0', shift=5)
        latents = initial.to('cuda:0')
        forward_count = 0
        def count_forward(module, inputs, output):
            nonlocal forward_count
            forward_count += 1
        model.register_forward_hook(count_forward)
    else:
        model, scheduler, config = load_student(settings, torch)
        report['load_acceptance'] = model.s1_load_audit
        report['resolved_student_config'] = config
        scheduler.prepare(seed=SEED, latent_shape=SHAPE)
        scheduler.latents = initial.to('cuda:0')
        assert torch.allclose(scheduler.sigmas, torch.tensor(settings['student']['expected_sigmas']), atol=1e-6, rtol=0)
        padded = torch.stack([torch.cat([u, u.new_zeros(512-u.size(0), u.size(1))]) for u in context])
        inputs = {'text_encoder_output': {'context': padded, 'context_null': None},
                  'image_encoder_output': {'clip_encoder_out': None, 'vae_encoder_out': None}}
        forward_count = 0
        original_forward = model._infer_cond_uncond
        def counted_forward(*call_args, **kwargs):
            nonlocal forward_count
            forward_count += 1
            return original_forward(*call_args, **kwargs)
        model._infer_cond_uncond = counted_forward
    torch.cuda.synchronize()
    report['dit_load_seconds'] = time.perf_counter() - start
    report['timesteps'] = scheduler.timesteps.cpu().tolist()
    report['sigmas'] = scheduler.sigmas.cpu().tolist()
    report['resident_allocated_gib'] = torch.cuda.memory_allocated() / 2**30
    report['latent_at_entry_sha256'] = tensor_hash(latents if args.arm == 'teacher' else scheduler.latents)
    assert report['latent_at_entry_sha256'] == report['initial_latent_sha256']
    json_write(output / 'state.json', dict(report, status='loaded'))
    ensure_no_other_process(admission['uuid'])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    step_records = []
    # Retain autocast weight cache across all teacher steps, matching the native
    # sampling loop. The student runs its own explicit precision implementation.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.arm == 'teacher'):
        for step, t in enumerate(scheduler.timesteps):
            step_start = time.perf_counter()
            if args.arm == 'teacher':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    timestep = torch.stack([t])
                    cond = model([latents], t=timestep, context=context, seq_len=32760)[0]
                    uncond = model([latents], t=timestep, context=context_null, seq_len=32760)[0]
                    prediction = uncond + 5 * (cond - uncond)
                    latents = scheduler.step(prediction.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
            else:
                scheduler.step_pre(step_index=step)
                model.infer(inputs)
                scheduler.step_post()
                latents = scheduler.latents
            assert latents.dtype == torch.float32
            torch.cuda.synchronize()
            row = {'step': step+1, 'timestep': float(t), 'seconds': time.perf_counter()-step_start,
                   'latent': tensor_stats(latents)}
            step_records.append(row)
            print('S1_STEP', args.arm, row['step'], 'seconds=%.3f' % row['seconds'], flush=True)
            if step % 5 == 0:
                ensure_no_other_process(admission['uuid'])
            json_write(output / 'step_progress.json', {'arm': args.arm, 'steps': step_records})
    torch.cuda.synchronize()
    report['denoise_seconds'] = time.perf_counter() - start
    report['forward_count'] = forward_count
    assert forward_count == (100 if args.arm == 'teacher' else 4)
    report['steps'] = step_records
    report['final_latent'] = tensor_stats(latents)
    report['final_latent_sha256'] = tensor_hash(latents)
    # Same original WanVAE and FP32 input for both methods; no tiling or upsampling.
    start = time.perf_counter()
    video = vae.decode([latents.float()])[0]
    torch.cuda.synchronize()
    report['vae_seconds'] = time.perf_counter() - start
    report['denoise_plus_vae_seconds'] = report['denoise_seconds'] + report['vae_seconds']
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    report['output_tensor'] = tensor_stats(video)
    start = time.perf_counter()
    report['video'] = export_video(video, output / 's1_video.mp4', torch)
    report['export_and_decode_check_seconds'] = time.perf_counter() - start
    report['network_attempts'] = NETWORK_ATTEMPTS
    assert not NETWORK_ATTEMPTS
    report['status'] = 'passed'
    json_write(output / 'result.json', report)
    print('S1_ARM_PASS', args.arm, output / 'result.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['teacher', 'student'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    configure()
    assert json.loads((PROJECT / 'Project_Progress/Stage_Progress/S1_PREFLIGHT.json').read_text())['status'] == 'passed'
    output = BATCH / args.arm / ('attempt%02d' % args.attempt)
    output.mkdir(parents=True, exist_ok=False)
    report = {'arm': args.arm, 'attempt': args.attempt, 'status': 'starting'}
    try:
        with gpu_admission(args.gpu) as admission:
            report['admission'] = admission
            json_write(output / 'state.json', report)
            run(args, output, admission, report)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), network_attempts=NETWORK_ATTEMPTS)
        json_write(output / 'failure.json', report)
        raise


if __name__ == '__main__':
    main()
