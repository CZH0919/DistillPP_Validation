"""One fixed VBench dimension on the two completed S1 videos; strictly offline."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

from s1_common import (PROJECT, BATCH, PROMPT, METRICS, configure, gpu_admission,
                       json_write, file_hash, NETWORK_ATTEMPTS, ensure_no_other_process)


def generation_results():
    found = {}
    for arm in ['teacher', 'student']:
        candidates = sorted((BATCH / arm).glob('attempt*/result.json'))
        assert candidates, 'No completed generation: ' + arm
        path = candidates[-1]
        row = json.loads(path.read_text())
        assert row['status'] == 'passed' and row['s1_prompt'] == PROMPT
        video = path.parent / 's1_video.mp4'
        assert file_hash(video) == row['video']['sha256']
        found[arm] = {'result_path': str(path), 'video_path': str(video), 'record': row}
    assert found['teacher']['record']['initial_latent_sha256'] == found['student']['record']['initial_latent_sha256']
    assert found['teacher']['record']['positive_context_sha256'] == found['student']['record']['positive_context_sha256']
    return found


def evaluate(metric, videos, torch):
    weights = PROJECT / 'resources/evaluation'
    device = 'cuda'
    if metric == 'subject_consistency':
        import vision_transformer
        from vbench.subject_consistency import subject_consistency
        model = vision_transformer.vit_base(patch_size=16, num_classes=0)
        model.load_state_dict(torch.load(weights / 'dino_model/dino_vitbase16_pretrain.pth', map_location='cpu', weights_only=True), strict=True)
        model.eval().requires_grad_(False).to(device)
        return subject_consistency(model, videos, device, read_frame=False)
    if metric == 'imaging_quality':
        from pyiqa.archs.musiq_arch import MUSIQ
        from vbench.imaging_quality import technical_quality
        model = MUSIQ(pretrained_model_path=str(weights / 'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'))
        model.eval().requires_grad_(False).to(device)
        return technical_quality(model, videos, device, imaging_quality_preprocessing_mode='longer')
    if metric == 'motion_smoothness':
        from vbench.motion_smoothness import MotionSmoothness, motion_smoothness
        model = MotionSmoothness(str(PROJECT / 'src/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'),
                                  str(weights / 'amt_model/amt-s.pth'), device)
        return motion_smoothness(model, videos)
    if metric == 'dynamic_degree':
        from easydict import EasyDict
        from vbench.dynamic_degree import DynamicDegree, dynamic_degree
        model = DynamicDegree(EasyDict(model=str(weights / 'raft_model/models/raft-things.pth'),
                                      small=False, mixed_precision=False, alternate_corr=False), device)
        return dynamic_degree(model, videos)
    if metric == 'aesthetic_quality':
        import clip
        from vbench.aesthetic_quality import get_aesthetic_model, laion_aesthetic
        model, _ = clip.load(str(weights / 'clip_model/ViT-L-14.pt'), device=device, jit=False)
        model.eval().requires_grad_(False)
        head = get_aesthetic_model(str(weights / 'aesthetic_model/emb_reader')).to(device).eval()
        return laion_aesthetic(head, model, videos, device)
    if metric == 'overall_consistency':
        from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
        from vbench.third_party.ViCLIP.viclip import ViCLIP
        from vbench.overall_consistency import overall_consistency
        tokenizer = SimpleTokenizer(str(weights / 'ViCLIP/bpe_simple_vocab_16e6.txt.gz'))
        assert len(tokenizer.encode(PROMPT)) + 2 <= 32, 'Never silently truncate the prompt'
        model = ViCLIP(tokenizer=tokenizer, pretrain=str(weights / 'ViCLIP/ViClip-InternVid-10M-FLT.pth'))
        model.eval().requires_grad_(False).to(device)
        # Prevent the upstream default truncate=True from concealing a bad prompt.
        model.text_encoder.tokenize(PROMPT, context_length=32, truncate=False)
        return overall_consistency(model, [{'prompt': PROMPT, 'video_list': videos}], tokenizer, device)
    raise ValueError(metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=METRICS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    configure()
    sys.path[:0] = [str(PROJECT / 'src' / p) for p in ['CLIP', 'VBench', 'DINO']]
    output = BATCH / 'evaluation' / args.metric / ('attempt%02d' % args.attempt)
    output.mkdir(parents=True, exist_ok=False)
    report = {'metric': args.metric, 'status': 'starting', 'scope': 'S1_ONLY_NOT_QUALIFICATION'}
    try:
        results = generation_results()
        videos = [results[arm]['video_path'] for arm in ['teacher', 'student']]
        with gpu_admission(args.gpu, min_free_gib=20) as admission:
            report['admission'] = admission
            import torch
            import cv2
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            cv2.setNumThreads(1)
            torch.manual_seed(20261016)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad():
                aggregate, rows = evaluate(args.metric, videos, torch)
            torch.cuda.synchronize()
            report['load_plus_scoring_seconds'] = time.perf_counter() - start
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
            assert len(rows) == 2 and {r['video_path'] for r in rows} == set(videos)
            values = []
            for arm in ['teacher', 'student']:
                row = next(r for r in rows if r['video_path'] == results[arm]['video_path'])
                value = float(row['video_results'])
                assert math.isfinite(value)
                normalized = value / 100 if args.metric == 'imaging_quality' else value
                raw = value * 10 if args.metric == 'aesthetic_quality' else value
                assert (-1 <= normalized <= 1) if args.metric == 'overall_consistency' else (0 <= normalized <= 1)
                if args.metric == 'dynamic_degree':
                    assert value in [0, 1]
                values.append({'arm': arm, 'video_path': row['video_path'], 'raw_score': raw,
                               'normalized_score': normalized, 'source_returned_score': value})
            assert abs(float(aggregate) - sum(r['normalized_score'] for r in values) / 2) < 1e-6
            report.update(status='passed', aggregate=float(aggregate), videos=values,
                          network_attempts=NETWORK_ATTEMPTS,
                          normalization='imaging raw/100; aesthetic source already /10; others unchanged',
                          scientific_comparison='Not reported: one paired smoke sample is not qualification evidence')
            assert not NETWORK_ATTEMPTS
            ensure_no_other_process(admission['uuid'])
            json_write(output / 'result.json', report)
            print('S1_METRIC_PASS', args.metric, json.dumps(values), flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), network_attempts=NETWORK_ATTEMPTS)
        json_write(output / 'failure.json', report)
        raise


if __name__ == '__main__':
    main()
