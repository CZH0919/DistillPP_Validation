"""Read-only S2-R completion audit; standard library only, no CUDA imports."""
import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
BATCH = ROOT / 'runs/s2r_20261017'
OUT = ROOT / 'Result/S2R_20261017'


def safe(path):
    path = Path(path)
    assert path.resolve() == path and path.is_relative_to(ROOT), str(path)
    return path


def read(path):
    path = safe(path)
    assert path.stat().st_size < 8 * 2**20
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert ROOT.resolve() == ROOT
    execution = read(BATCH / 'execution.json')
    protocol = read(BATCH / 'protocol.json')
    freeze = read(BATCH / 'analysis_freeze.json')
    final = read(BATCH / 'finalizer_status.json')
    supervisor = read(BATCH / 'supervisor_status.json')
    assert final['status'] == 'complete' and final['exit_code'] == 0
    assert supervisor['status'] == 'compute_complete'
    assert not final['s3_started'] and not supervisor['s3_started']
    hashes = {**execution['input_sha256'], **freeze['source_sha256']}
    hashes.update(read(BATCH / 'preflight.json')['native_source_sha256'])
    for rel, digest in hashes.items():
        assert sha(ROOT / rel) == digest, rel
    assert sha(BATCH / 'protocol.json') == execution['input_sha256']['Project_Plan/QUALIFICATION_V2.json']
    current_doc_hashes = {rel: sha(ROOT / rel) for rel in [
        'Project_Progress/Total_Progress.md', 'Project_Plan/OFFICIAL_QUALIFICATION_V2.md',
        'Project_Plan/EXPERIMENT_PLAN.md']}
    records = read(OUT / 'generation_records.json')
    index = read(OUT / 'video_index.json')
    lookup = {(r['pair_id'], r['arm']): r for r in records}
    archive = {(r['pair_id'], r['arm']): r for r in index}
    assert len(records) == len(lookup) == len(index) == len(archive) == 192
    assert set(lookup) == set(archive)
    assert len({r['video']['sha256'] for r in records}) == 192
    assert sum(r['bytes'] for r in index) < 4 * 2**30
    expected = {(p['pair_id'], a) for p in execution['pairs'] for a in ('T50', 'S4')}
    assert set(lookup) == expected
    seen_workers = {}
    schedules = {}
    for pair in execution['pairs']:
        commits = list((BATCH / 'pairs' / pair['pair_id']).glob('attempt*/pair_complete.json'))
        assert len(commits) == 1
        assert read(commits[0])['status'] == 'passed'
        rows = [lookup[pair['pair_id'], a] for a in ('T50', 'S4')]
        for key in ('initial_latent_sha256', 'positive_context_sha256', 'gpu_uuid', 'worker_id'):
            assert rows[0][key] == rows[1][key]
        for row in rows:
            arm = row['arm']
            raw = read(Path(row['result_path']))
            assert all(row[k] == v for k, v in raw.items())
            assert row['status'] == 'passed' and not row['warmup'] and not row['network_attempts']
            for key in ('pair_id', 'prompt_id', 'prompt', 'eval_prompt', 'domain', 'seed', 'primary_metrics'):
                assert row[key] == pair[key]
            assert row['forward_count'] == (100 if arm == 'T50' else 4)
            schedule = (row['timesteps'], row['sigmas'])
            assert schedule == schedules.setdefault(arm, schedule)
            assert len(row['timesteps']) == (50 if arm == 'T50' else 4)
            assert row['final_latent']['dtype'] == ('torch.float32' if arm == 'T50' else 'torch.bfloat16')
            for field in ('final_latent', 'output_tensor'):
                assert all(math.isfinite(row[field][k]) for k in ('min', 'max', 'mean', 'std'))
                assert row[field]['std'] > 0
            v = row['video']
            assert v['frames'] == 81 and v['size'] == [480, 832] and v['fps'] == 16
            assert v['decord_all_frames_decoded'] and v['opencv_all_frames_decoded'] and v['pixel_std'] > 0
            ix = archive[row['pair_id'], arm]
            assert ix['source'] == row['video_path'] and ix['sha256'] == v['sha256']
            for key in ('source', 'archive'):
                p = safe(ix[key])
                assert p.stat().st_size == v['bytes'] and sha(p) == v['sha256']
            worker_dir = BATCH / 'workers' / ('worker%02d' % row['worker_id']) / Path(row['result_path']).parents[1].name
            seen_workers[str(worker_dir)] = row
    for directory, row in seen_workers.items():
        w = read(Path(directory) / 'result.json')
        assert w['status'] == 'passed' and len(w['completed_pairs']) == 32
        assert w['admission']['physical_index'] == 6 and w['admission']['pid'] == row['pid']
        assert w['teacher_load_audit']['keys'] == w['student_load_audit']['keys'] == 1095
        assert w['teacher_load_audit']['missing'] == w['teacher_load_audit']['unexpected'] == 0
        assert w['student_config']['infer_steps'] == 4 and w['student_config']['enable_cfg'] is False
        assert not w['student_config']['dit_quantized'] and not w['network_attempts']
        assert all(w['warmup'][a]['status'] == 'passed' for a in ('T50', 'S4'))
    with (OUT / 'metrics_per_video.csv').open(newline='', encoding='utf-8') as f:
        csv_rows = list(csv.DictReader(f))
    csv_lookup = {(r['pair_id'], r['arm'], r['metric']): r for r in csv_rows}
    assert len(csv_rows) == len(csv_lookup) == 576
    matched = set()
    metric_names = [m for ms in protocol['metric_domains'].values() for m in ms]
    for metric in metric_names:
        paths = list((BATCH / 'evaluation' / metric).glob('attempt*/result.json'))
        assert len(paths) == 1
        report = read(paths[0])
        assert report['status'] == 'passed' and len(report['rows']) == 96 and not report['network_attempts']
        assert report['admission']['physical_index'] == 6
        assert report['source_sha256'] == freeze['source_sha256']['tools/evaluate_s2r.py']
        chunks = [read(e['path']) for e in report['chunks']]
        assert len(chunks) == 16
        combined = [r for c in chunks for r in c['rows']]
        order = lambda r: (r['pair_id'], r['arm'], r['metric'])
        assert sorted(combined, key=order) == sorted(report['rows'], key=order)
        for r in combined:
            key = (r['pair_id'], r['arm'], r['metric'])
            assert key not in matched
            matched.add(key)
            g = lookup[key[:2]]
            assert r['metric'] in g['primary_metrics'] and r['video_sha256'] == g['video']['sha256']
            assert r['video_path'] == g['video_path'] and r['eval_prompt'] == g['eval_prompt']
            val = r['source_returned_score']
            assert math.isfinite(val)
            assert r['normalized_score'] == (val / 100 if metric == 'imaging_quality' else val)
            assert r['raw_score'] == (val * 10 if metric == 'aesthetic_quality' else val)
            c = csv_lookup[key]
            assert c['video_sha256'] == r['video_sha256']
            for k in ('raw_score', 'source_returned_score', 'normalized_score'):
                assert float(c[k]) == r[k]
    assert matched == set(csv_lookup) == {(r['pair_id'], r['arm'], m) for r in records for m in r['primary_metrics']}
    original = read(ROOT / 'Project_Progress/Stage_Progress/S1_PREFLIGHT.json')
    unchanged = 0
    for row in original['generation_files']:
        # Only stat the explicitly recorded, previously verified shared resources.
        s = Path(row['path']).stat()
        assert (s.st_size, s.st_mtime_ns) == (row['bytes'], row['mtime_ns'])
        unchanged += 1
    failures = list((BATCH / 'workers').glob('*/attempt*/failure.json'))
    assert len(failures) == 4
    failed_pids = {read(p)['admission']['pid'] for p in failures}
    assert not failed_pids.intersection(r['pid'] for r in records)
    tracked_pids = {h['pid'] for h in supervisor['history']}
    for name in ('supervisor_identity.json', 'finalizer_identity.json'):
        identity = read(BATCH / name)
        if 'pid' in identity:
            tracked_pids.add(identity['pid'])
    live_own = []
    for pid in tracked_pids:
        p = Path('/proc') / str(pid)
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ')
            if str(ROOT).encode() in cmd:
                live_own.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert not live_own
    print(json.dumps({'status': 'passed', 'checked_at_beijing': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'accepted_pairs': 96, 'formal_videos': 192, 'archive_videos': 192,
        'files_fully_rehashed': 384, 'source_video_bytes': sum(r['bytes'] for r in index),
        'paired_input_and_gpu_identity': True, 'decoder_acceptance_records_verified': 192,
        'decode_scope': 'Existing full dual-decoder acceptance records plus fresh exact file hashes; no repeat decoding or human rating.',
        'matched_original_score_rows': 576, 'metric_chunks': 96, 'normalization_verified': True,
        'frozen_code_and_input_files_checked': len(hashes), 'shared_generation_metadata_unchanged': unchanged,
        'successful_generation_workers': 3, 'all_formal_generation_and_evaluation_on_gpu': 6,
        'historical_failures': 4, 'failed_attempts_contributed_formal_videos': 0,
        'remaining_tracked_project_processes': live_own, 's3_started': False,
        'finalized_at_beijing': datetime.fromtimestamp(final['unix_time'], timezone(timedelta(hours=8))).isoformat(),
        'doc_sha256_before_review': current_doc_hashes}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
