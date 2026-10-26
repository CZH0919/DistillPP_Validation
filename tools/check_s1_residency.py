"""Bounded co-residency acceptance: one T CFG query, one S query, one VAE decode.

No teacher correction, no full trajectory and no extra video is exported.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from s1_common import (PROJECT, BATCH, PROMPT, SEED, SHAPE, configure, gpu_admission,
                       ensure_no_other_process, json_write, tensor_stats, NETWORK_ATTEMPTS)
from run_s1_arm import load_teacher, load_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    configure()
    output = BATCH / 'co_residency'
    output.mkdir(exist_ok=False)
    report = {'status': 'starting', 'scope': 'S1 bounded memory probe; no correction algorithm'}
    try:
        with gpu_admission(args.gpu, min_free_gib=120) as admission:
            report['admission'] = admission
            import torch
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            sys.path.insert(0, str(PROJECT / 'src/Wan2.1'))
            from wan.modules.t5 import T5EncoderModel
            from wan.modules.vae import WanVAE
            from wan.modules import attention
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
            attention.FLASH_ATTN_3_AVAILABLE = False
            settings = json.loads((PROJECT / 'Project_Plan/BASELINE_SETTINGS.json').read_text())
            cp = settings['common_components']
            encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device('cuda:0'),
                       checkpoint_path=cp['t5'], tokenizer_path=cp['tokenizer'])
            vae = WanVAE(vae_pth=cp['vae'], dtype=torch.float32, device='cuda:0')
            context = encoder([PROMPT], torch.device('cuda:0'))
            negative = encoder([settings['negative_prompt']], torch.device('cuda:0'))
            teacher, report['teacher_load'] = load_teacher(settings, torch)
            student, scheduler, _ = load_student(settings, torch)
            report['student_load'] = student.s1_load_audit
            torch.cuda.synchronize()
            report['both_models_t5_vae_resident_gib'] = torch.cuda.memory_allocated() / 2**30
            print('BOTH_RESIDENT_GIB', report['both_models_t5_vae_resident_gib'], flush=True)
            ensure_no_other_process(admission['uuid'])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial = torch.load(BATCH/'common/initial_latent.pt', map_location='cpu', weights_only=True).to('cuda:0')
            ts = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            ts.set_timesteps(50, device='cuda:0', shift=5)
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cond = teacher([initial], t=ts.timesteps[:1], context=context, seq_len=32760)[0]
                uncond = teacher([initial], t=ts.timesteps[:1], context=negative, seq_len=32760)[0]
                teacher_prediction = uncond + 5 * (cond - uncond)
            torch.cuda.synchronize()
            report['teacher_query_seconds'] = time.perf_counter() - start
            report['teacher_prediction'] = tensor_stats(teacher_prediction)
            scheduler.prepare(seed=SEED, latent_shape=SHAPE)
            scheduler.latents = initial.clone()
            padded = torch.stack([torch.cat([u, u.new_zeros(512-u.size(0), u.size(1))]) for u in context])
            inputs = {'text_encoder_output': {'context': padded, 'context_null': None},
                      'image_encoder_output': {'clip_encoder_out': None, 'vae_encoder_out': None}}
            scheduler.step_pre(step_index=0)
            start = time.perf_counter()
            student.infer(inputs)
            scheduler.step_post()
            torch.cuda.synchronize()
            report['student_query_seconds'] = time.perf_counter() - start
            report['student_latent'] = tensor_stats(scheduler.latents)
            start = time.perf_counter()
            decoded = vae.decode([scheduler.latents.float()])[0]
            torch.cuda.synchronize()
            report['vae_decode_seconds'] = time.perf_counter() - start
            report['decoded_shape'] = list(decoded.shape)
            assert report['decoded_shape'] == [3, 81, 480, 832] and bool(torch.isfinite(decoded).all())
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            report['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
            report['device_total_gib'] = torch.cuda.get_device_properties(0).total_memory/2**30
            report['headroom_vs_peak_reserved_gib'] = report['device_total_gib'] - report['peak_reserved_gib']
            report['network_attempts'] = NETWORK_ATTEMPTS
            assert not NETWORK_ATTEMPTS
            report.update(status='passed', teacher_branch_predictions=2, student_predictions=1,
                          correction_applied=False, extra_videos_exported=0,
                          limitation='One fixed-shape co-residency probe; not acceptance of a future correction algorithm')
            ensure_no_other_process(admission['uuid'])
            json_write(output/'result.json', report)
            print('S1_CO_RESIDENCY_PASS', json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        json_write(output/'failure.json', report)
        raise


if __name__ == '__main__':
    main()
