"""Validate all S1 artifacts, archive the two videos remotely, write compact results."""
import csv
import io
import json
import math
from pathlib import Path
import shutil

from s1_common import PROJECT, BATCH, PROMPT, SEED, METRICS, json_write, file_hash, gpu_snapshot


def no_clobber_text(path, text):
    assert path.resolve() == path and path.is_relative_to(PROJECT)
    if path.exists():
        assert path.read_bytes() == text.encode('utf-8'), 'Existing result differs: ' + str(path)
    else:
        with path.open('x', encoding='utf-8', newline='') as f:
            f.write(text)


def main():
    assert PROJECT.resolve() == PROJECT and BATCH.resolve() == BATCH
    preflight = json.loads((PROJECT/'Project_Progress/Stage_Progress/S1_PREFLIGHT.json').read_text())
    cpu = json.loads((PROJECT/'Project_Progress/Stage_Progress/S1_CPU_CONTRACT.json').read_text())
    assert preflight['status'] == cpu['status'] == 'passed'
    settings = json.loads((PROJECT/'Project_Plan/BASELINE_SETTINGS.json').read_text())
    arms = {}
    for arm in ['teacher', 'student']:
        p = sorted((BATCH/arm).glob('attempt*/result.json'))[-1]
        result = json.loads(p.read_text())
        assert result['status'] == 'passed' and result['s1_prompt'] == PROMPT and result['seed'] == SEED
        assert result['forward_count'] == (100 if arm == 'teacher' else 4)
        assert len(result['steps']) == (50 if arm == 'teacher' else 4)
        assert not result['network_attempts']
        video = p.parent/'s1_video.mp4'
        assert result['video']['frames'] == 81 and result['video']['fps'] == 16
        assert result['video']['decord_all_frames_decoded'] and result['video']['opencv_all_frames_decoded']
        assert file_hash(video) == result['video']['sha256']
        expected_dtype = 'torch.float32' if arm == 'teacher' else 'torch.bfloat16'
        assert all(step['latent']['dtype'] == expected_dtype for step in result['steps'])
        arms[arm] = {'path': str(p), 'result_sha256': file_hash(p), 'video_path': str(video), 'result': result}
    assert arms['teacher']['result']['initial_latent_sha256'] == arms['student']['result']['initial_latent_sha256']
    assert arms['teacher']['result']['positive_context_sha256'] == arms['student']['result']['positive_context_sha256']
    scores = []
    scoring_files = []
    for metric in METRICS:
        p = sorted((BATCH/'evaluation'/metric).glob('attempt*/result.json'))[-1]
        result = json.loads(p.read_text())
        assert result['status'] == 'passed' and len(result['videos']) == 2 and not result['network_attempts']
        for row in result['videos']:
            assert row['video_path'] == arms[row['arm']]['video_path']
            assert math.isfinite(row['raw_score']) and math.isfinite(row['normalized_score'])
            scores.append(dict(metric=metric, arm=row['arm'], raw_score=row['raw_score'], normalized_score=row['normalized_score']))
        scoring_files.append({'metric': metric, 'path': str(p), 'sha256': file_hash(p)})
    assert len(scores) == 12 and len({(r['metric'],r['arm']) for r in scores}) == 12
    residency = json.loads((BATCH/'co_residency/result.json').read_text())
    assert residency['status'] == 'passed' and not residency['correction_applied']
    assert residency['extra_videos_exported'] == 0
    archive = []
    for arm, group in [('teacher', 'T50'), ('student', 'S4')]:
        source = Path(arms[arm]['video_path'])
        dest = PROJECT/'Evaluation_Videos/s1_20261016'/group/('S1_smoke_seed%d.mp4' % SEED)
        assert dest.resolve() == dest and dest.is_relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with dest.open('xb') as out, source.open('rb') as inp:
                shutil.copyfileobj(inp, out)
        digest = file_hash(dest)
        assert digest == arms[arm]['result']['video']['sha256']
        archive.append({'arm': arm, 'path': str(dest), 'sha256': digest, 'bytes': dest.stat().st_size})
    result_dir = PROJECT/'Result/S1_20261016'
    assert result_dir.resolve() == result_dir
    result_dir.mkdir(exist_ok=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['metric','arm','raw_score','normalized_score'])
    writer.writeheader(); writer.writerows(scores)
    no_clobber_text(result_dir/'metrics.csv', stream.getvalue())
    timing = []
    for arm in ['teacher','student']:
        r = arms[arm]['result']
        timing.append({'arm': arm, 'steps': len(r['steps']), 'branch_predictions': r['forward_count'],
                       **{k:r[k] for k in ['denoise_seconds','vae_seconds','denoise_plus_vae_seconds','text_seconds',
                                           'dit_load_seconds','peak_allocated_gib','peak_reserved_gib']}})
    stream = io.StringIO(newline='');writer = csv.DictWriter(stream, fieldnames=list(timing[0]));writer.writeheader();writer.writerows(timing)
    no_clobber_text(result_dir/'timing.csv', stream.getvalue())
    report = {'status':'S1_PASSED', 'scope':'one paired smoke sample only; no qualification or correction experiment',
              'formal_qualification_started':False, 'prompt':PROMPT,'seed':SEED,'settings':settings,
              'generation':arms,'metrics':scores,'scoring_evidence':scoring_files,'timing':timing,
              'timing_limitation':'First full generations, with diagnostics and no repeated warmed measurements; no scientific speedup estimate',
              'precision_clarification':'Teacher native FP32 weights/BF16 autocast and FP32 state; student native BF16 weights/state. Same CPU FP32 source noise, before native student casting. Both use native FP32 VAE.',
              'teacher_input_metadata_note':'The raw first teacher result embeds earlier plan wording; its explicit load_acceptance and preserved executed source record actual native FP32 storage. This consolidated record uses the corrected wording; raw result is retained.',
              'cpu_contract':cpu,'co_residency':residency,'archived_videos':archive,
              'gpu_snapshot_after_work':gpu_snapshot(),
              'failures_retained':[str(p) for p in sorted(BATCH.glob('**/failure.json'))]}
    json_write(result_dir/'acceptance.json',report)
    print(json.dumps({'status':report['status'],'timing':timing,'metrics':scores,'co_residency_peak_gib':residency['peak_allocated_gib'],
                      'headroom_gib':residency['headroom_vs_peak_reserved_gib'],'result_dir':str(result_dir),'archive':archive},indent=2))


if __name__ == '__main__':
    main()
