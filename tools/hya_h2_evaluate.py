"""Score one completed H2 block with the frozen official dimension mapping."""
import argparse
import importlib
import math
import time
import traceback

import hya_common as h1
import hya_evaluate_smoke as acceptance
from hya_h2_common import (ROOT, RUN, ResourceConflict, check_frozen, read, admission,
                          ensure_exclusive, task_claim, digest, write_json, progress)


def evaluation_info(row, metric, video):
    return {'prompt': row['evaluation_prompt'], 'video_list': [video], 'auxiliary_info': row['auxiliary_info'][metric]}


def evaluate(metric, rows, videos, device, torch):
    if metric not in ['scene', 'multiple_objects']:
        return acceptance.evaluate(metric, videos, device, torch)
    from vbench.utils import get_frame_indices
    acceptance.AUDIT.append({'metric': metric, 'sampled_frame_indices': [int(x) for x in get_frame_indices(16, 129, sample='middle')]})
    module = importlib.import_module('vbench.'+metric)
    # Match VBench load_dimension_info: select this dimension's nested auxiliary labels first.
    dictionaries = [evaluation_info(row, metric, video) for row, video in zip(rows, videos)]
    calls = [0]
    if metric == 'scene':
        model = acceptance.tag_model(device); original = module.get_caption
        def caption(*a, **kw):
            result = original(*a, **kw)
            acceptance.AUDIT.append({'video_path': videos[calls[0]], 'scene_frame_captions': result})
            calls[0] += 1
            return result
        module.get_caption = caption
        result = module.scene(model, dictionaries, device)
    else:
        model = acceptance.grit_model(device); original = module.get_dect_from_grit
        def detect(*a, **kw):
            result = original(*a, **kw)
            acceptance.AUDIT.append({'video_path': videos[calls[0]], 'object_frame_labels': [sorted(x) for x in result]})
            calls[0] += 1
            return result
        module.get_dect_from_grit = detect
        result = module.multiple_objects(model, dictionaries, device)
    assert calls[0] == len(videos)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--block', required=True)
    parser.add_argument('--metric', required=True, choices=h1.METRICS)
    parser.add_argument('--gpu', type=int, required=True); parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args(); manifest, protocol = check_frozen(); h1.configure()
    block = next(b for b in manifest['blocks'] if b['id'] == args.block)
    out = RUN/'evaluation'/(args.block+'_'+args.metric)/f'attempt{args.attempt:02d}'
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'starting', 'block': args.block, 'metric': args.metric,
              'protocol_sha256': digest(RUN/'protocol.json'), 'started_unix': time.time()}
    try:
        rows, videos, identities = [], [], []
        for pid in block['prompt_ids']:
            row = next(p for p in protocol['prompts'] if p['id'] == pid)
            if args.metric not in row['metrics']:
                continue
            pair = read(RUN/'pairs'/(pid+'.json')); assert pair['status'] == 'passed'
            for arm in ['T50', 'S5']:
                rec = pair['records'][arm]; path = rec['archive_path']
                assert digest(path) == rec['video']['sha256']
                rows.append(row); videos.append(path)
                identities.append({'prompt_id': pid, 'arm': arm, 'seed': rec['seed'], 'video_sha256': rec['video']['sha256']})
        assert videos
        with task_claim('eval_'+args.block+'_'+args.metric), admission(args.gpu) as admitted:
            # Reuse the accepted offline adapters and model loading checks; bind the guard to this worker's GPU.
            acceptance.ensure_exclusive = lambda: ensure_exclusive(args.gpu)
            torch = acceptance.prepare_imports()
            torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
            torch.manual_seed(20261019)
            import cv2
            cv2.setNumThreads(1)
            progress(out/'progress.json', 'scoring', block=args.block, metric=args.metric, videos=len(videos))
            torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            aggregate, scores = evaluate(args.metric, rows, videos, torch.device('cuda'), torch)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            assert len(scores) == len(videos) and {x['video_path'] for x in scores} == set(videos)
            by_path = {x['video_path']: x for x in scores}; values = []
            for video, identity in zip(videos, identities):
                raw = by_path[video]; value = float(raw['video_results'])
                assert math.isfinite(value) and 0 <= value <= 1
                if args.metric == 'dynamic_degree': assert value in (0, 1)
                values.append({**identity, 'metric': args.metric, 'score': value, 'video_path': video, 'official_row': raw})
            assert abs(float(aggregate)-sum(v['score'] for v in values)/len(values)) < 1e-6
            ensure_exclusive(args.gpu); assert not h1.NETWORK_ATTEMPTS
            report.update(status='passed', admission=admitted, rows=values, aggregate=float(aggregate),
                          load_scoring_seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          audit=acceptance.AUDIT, network_attempts=h1.NETWORK_ATTEMPTS, finished_unix=time.time())
            write_json(out/'result.json', report)
            write_json(RUN/'scores'/(args.block+'_'+args.metric+'.json'), report)
            progress(out/'progress.json', 'passed', metric=args.metric, scores=len(values))
    except BaseException as exc:
        report.update(status='conflict' if isinstance(exc, ResourceConflict) else 'failed', error=repr(exc),
                      traceback=traceback.format_exc(), audit=acceptance.AUDIT)
        write_json(out/'failure.json', report)
        raise SystemExit(75 if isinstance(exc, ResourceConflict) else 1)


if __name__ == '__main__':
    main()
