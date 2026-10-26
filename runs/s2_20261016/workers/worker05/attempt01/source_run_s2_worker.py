"""S2 paired T50/S4 generation only. No correction or S3 entry point."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from s2_common import (PROJECT, BATCH, SHAPE, WARMUP_PROMPT, WARMUP_SEED, configure,
    gpu_admission, OccupancyMonitor, load_frozen, json_write, file_hash, tensor_hash,
    tensor_stats, ensure_no_other_process, NETWORK_ATTEMPTS)
from run_s1_arm import load_teacher, load_student, export_video


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
    from wan.modules import attention
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    assert attention.FLASH_ATTN_2_AVAILABLE
    attention.FLASH_ATTN_3_AVAILABLE = False
    execution, settings = load_frozen()
    assigned = [p for p in execution['pairs'] if p['worker_id'] == args.worker]
    if args.pair_ids:
        assigned = [p for p in assigned if p['pair_id'] in args.pair_ids.split(',')]
    assigned = [p for p in assigned if not list((BATCH/'pairs'/p['pair_id']).glob('attempt*/pair_complete.json'))]
    assert assigned, 'No remaining pairs; do not load models'
    report.update(gpu_uuid=admission['uuid'], worker_id=args.worker, assigned=assigned,
        torch=torch.__version__, gpu_name=torch.cuda.get_device_name(0),
        timing='CUDA synchronized wall time; denoise + VAE; no per-step tensor diagnostics or I/O',
        memory_scope='Both DiTs, T5 and VAE resident on the same single GPU; no offload',
        diagnostics='10-second CPU occupancy/progress thread; small overhead is included, not subtracted')
    state = {'phase': 'loading', 'worker_id': args.worker, 'pair_id': None, 'arm': None, 'step': 0}
    with OccupancyMonitor(admission, output, state) as monitor:
        start = time.perf_counter()
        cp = settings['common_components']
        encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device('cuda:0'),
                    checkpoint_path=cp['t5'], tokenizer_path=cp['tokenizer'])
        vae = WanVAE(vae_pth=cp['vae'], dtype=torch.float32, device='cuda:0')
        assert all(p.dtype == torch.float32 for p in vae.model.parameters())
        torch.cuda.synchronize()
        report['common_load_seconds'] = time.perf_counter() - start
        monitor.check()
        start = time.perf_counter()
        teacher, report['teacher_load_audit'] = load_teacher(settings, torch)
        torch.cuda.synchronize()
        report['teacher_load_seconds'] = time.perf_counter() - start
        monitor.check()
        start = time.perf_counter()
        student, ss, config = load_student(settings, torch)
        torch.cuda.synchronize()
        report['student_load_seconds'] = time.perf_counter() - start
        report['student_load_audit'] = student.s1_load_audit
        report['student_config'] = config
        report['resident_allocated_gib'] = torch.cuda.memory_allocated()/2**30
        counts = {'T50': 0, 'S4': 0}
        def teacher_count(module, inputs, result):
            counts['T50'] += 1
        teacher.register_forward_hook(teacher_count)
        native_student_forward = student._infer_cond_uncond
        def student_count(*a, **kw):
            counts['S4'] += 1
            return native_student_forward(*a, **kw)
        student._infer_cond_uncond = student_count

        def generate(arm, pair, context, negative, initial, destination, warmup=False):
            state.update(phase='prepare', pair_id=pair['pair_id'], arm=arm, step=0, warmup=warmup)
            ensure_no_other_process(admission['uuid'])
            monitor.check()
            destination.mkdir(parents=True, exist_ok=False)
            record = {**pair, 'arm': arm, 'warmup': warmup, 'status': 'started',
                'worker_id': args.worker, 'gpu_uuid': admission['uuid'], 'pid': admission['pid'],
                'initial_latent_sha256': tensor_hash(initial),
                'positive_context_sha256': tensor_hash(context[0]),
                'unix_start': time.time(), 'gpu_count': 1, 'offload': False}
            json_write(destination/'started.json', record)
            torch.cuda.synchronize()
            setup = time.perf_counter()
            if arm == 'T50':
                scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
                scheduler.set_timesteps(50, device='cuda:0', shift=5)
                latents = initial.to('cuda:0')
            else:
                ss.generator = None
                ss.prepare(seed=pair['seed'], latent_shape=SHAPE)
                ss.latents = initial.to('cuda:0')
                scheduler = ss
                assert torch.allclose(ss.sigmas, torch.tensor(settings['student']['expected_sigmas']), atol=1e-6, rtol=0)
                padded = torch.stack([torch.cat([u, u.new_zeros(512-u.size(0), u.size(1))]) for u in context])
                inputs = {'text_encoder_output': {'context': padded, 'context_null': None},
                    'image_encoder_output': {'clip_encoder_out': None, 'vae_encoder_out': None}}
                latents = ss.latents
            torch.cuda.synchronize()
            record['scheduler_setup_and_initial_transfer_seconds'] = time.perf_counter() - setup
            record['timesteps'] = scheduler.timesteps.cpu().tolist()
            record['sigmas'] = scheduler.sigmas.cpu().tolist()
            assert tensor_hash(latents) == record['initial_latent_sha256']
            if arm == 'S4':
                # Computed outside the timer; actual runtime dtype is asserted below.
                cast = initial.to(torch.bfloat16)
                record['student_initial_cast_sha256'] = tensor_hash(cast)
                record['student_initial_cast_max_abs_error'] = float((cast.float()-initial).abs().max())
                del cast
            counts[arm] = 0
            torch.cuda.reset_peak_memory_stats()
            state['phase'] = 'denoising'
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=arm == 'T50'):
                for step, t in enumerate(scheduler.timesteps):
                    monitor.check()
                    state['step'] = step + 1
                    if arm == 'T50':
                        timestep = torch.stack([t])
                        cond = teacher([latents], t=timestep, context=context, seq_len=32760)[0]
                        uncond = teacher([latents], t=timestep, context=negative, seq_len=32760)[0]
                        prediction = uncond + 5 * (cond - uncond)
                        latents = scheduler.step(prediction.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
                    else:
                        ss.step_pre(step_index=step)
                        assert ss.latents.dtype == torch.bfloat16
                        student.infer(inputs)
                        ss.step_post()
                        latents = ss.latents
            torch.cuda.synchronize()
            record['denoise_seconds'] = time.perf_counter() - start
            assert counts[arm] == (100 if arm == 'T50' else 4)
            assert latents.dtype == (torch.float32 if arm == 'T50' else torch.bfloat16)
            record['forward_count'] = counts[arm]
            monitor.check()
            state['phase'] = 'vae'
            # No latent statistics or disk I/O between the timed GPU phases.
            torch.cuda.synchronize()
            start = time.perf_counter()
            video = vae.decode([latents.float()])[0]
            torch.cuda.synchronize()
            record['vae_seconds'] = time.perf_counter() - start
            record['denoise_plus_vae_seconds'] = record['denoise_seconds'] + record['vae_seconds']
            record['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            record['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
            monitor.check()
            record['final_latent'] = tensor_stats(latents)
            record['final_latent_sha256'] = tensor_hash(latents)
            record['output_tensor'] = tensor_stats(video)
            if not warmup:
                state['phase'] = 'export_decode_validation'
                start = time.perf_counter()
                record['video'] = export_video(video, destination/'video.mp4', torch)
                record['export_and_decode_check_seconds'] = time.perf_counter() - start
            record['unix_end'] = time.time()
            monitor.check()
            ensure_no_other_process(admission['uuid'])
            assert not NETWORK_ATTEMPTS
            record.update(status='passed', network_attempts=NETWORK_ATTEMPTS)
            json_write(destination/'result.json', record)
            print('S2_VIDEO_PASS', pair['pair_id'], arm, 'warmup='+str(warmup),
                  'seconds=%.3f' % record['denoise_plus_vae_seconds'], flush=True)
            # S1 reused scheduler references are cleared only after the output is saved.
            if arm == 'S4':
                ss.latents = None
                ss.noise_pred = None
            return record

        warm_pair = {'pair_id': 'warmup', 'prompt_id': 'warmup', 'stratum': 'nonformal',
                     'prompt': WARMUP_PROMPT, 'seed': WARMUP_SEED}
        assert WARMUP_PROMPT not in [p['prompt'] for p in execution['pairs']]
        warm_context = encoder([WARMUP_PROMPT], torch.device('cuda:0'))
        negative = encoder([settings['negative_prompt']], torch.device('cuda:0'))
        warm_initial = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(WARMUP_SEED), dtype=torch.float32)
        report['warmup'] = {}
        for arm in (['T50','S4'] if args.worker % 2 == 0 else ['S4','T50']):
            report['warmup'][arm] = generate(arm, warm_pair, warm_context, negative, warm_initial, output/'warmup'/arm, True)
        del warm_context, warm_initial
        report['completed_pairs'] = []
        for pair in assigned:
            monitor.check()
            pair_out = BATCH/'pairs'/pair['pair_id']/('attempt%02d' % args.attempt)
            pair_out.mkdir(parents=True, exist_ok=False)
            state.update(phase='text', pair_id=pair['pair_id'])
            torch.cuda.synchronize()
            start = time.perf_counter()
            context = encoder([pair['prompt']], torch.device('cuda:0'))
            torch.cuda.synchronize()
            text_seconds = time.perf_counter() - start
            initial = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(pair['seed']), dtype=torch.float32)
            with (pair_out/'initial_latent.pt').open('xb') as handle:
                torch.save(initial, handle)
            records = {}
            for arm in pair['order']:
                records[arm] = generate(arm, pair, context, negative, initial, pair_out/arm)
            assert records['T50']['initial_latent_sha256'] == records['S4']['initial_latent_sha256']
            assert records['T50']['positive_context_sha256'] == records['S4']['positive_context_sha256']
            monitor.check()
            json_write(pair_out/'pair_complete.json', {'status':'passed', **pair,
                'shared_positive_text_encoding_seconds': text_seconds,
                'negative_text_policy': 'one fixed negative embedding cached per worker; T50 only',
                'results': {a:str(pair_out/a/'result.json') for a in records}})
            report['completed_pairs'].append(pair['pair_id'])
            json_write(output/'partial_result.json', report)
            del context, initial
        report.update(status='passed', network_attempts=NETWORK_ATTEMPTS)
        state.update(phase='completed', pair_id=None, arm=None)
        json_write(output/'result.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--pair-ids', default='')
    args = parser.parse_args()
    configure()
    output = BATCH/'workers'/('worker%02d' % args.worker)/('attempt%02d' % args.attempt)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'starting', 'worker_id': args.worker, 'attempt': args.attempt}
    for name in ['run_s2_worker.py', 's2_common.py', 'run_s1_arm.py', 's1_common.py']:
        shutil.copyfile(PROJECT/'tools'/name, output/('source_'+name))
    try:
        with gpu_admission(args.gpu) as admission:
            report['admission'] = admission
            json_write(output/'started.json', report)
            run(args, output, admission, report)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        json_write(output/'failure.json', report)
        raise


if __name__ == '__main__':
    main()
