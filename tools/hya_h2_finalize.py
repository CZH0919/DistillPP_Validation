"""CPU-only H2 complete-panel validation, statistics, timing and report."""
import csv
import io
import json
import os
from pathlib import Path
import time
import uuid

from hya_h2_common import ROOT, RUN, RESULT, ARMS, read, check_frozen, digest, write_json


def metric_summary(metric, pairs):
    import numpy as np
    x = np.asarray(pairs, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 2 and np.isfinite(x).all()
    delta = x[:, 0]-x[:, 1]
    draws = np.random.Generator(np.random.PCG64(20261019)).integers(0, len(x), size=(20000, len(x)))
    quality = metric != 'dynamic_degree'
    ci = np.quantile(delta[draws].mean(axis=1), [.00625, .99375] if quality else [.025, .975], method='linear')
    out = {'metric': metric, 'prompt_count': len(x), 'seeds_per_prompt': 1,
           'T50_mean': float(x[:, 0].mean()), 'S5_mean': float(x[:, 1].mean()), 'T50_minus_S5': float(delta.mean()),
           'ci_level': .9875 if quality else .95, 'ci_low': float(ci[0]), 'ci_high': float(ci[1]),
           'teacher_advantage_candidate': bool(ci[0] > 0) if quality else None,
           'teacher_higher': int((delta > 0).sum()), 'equal': int((delta == 0).sum()), 'student_higher': int((delta < 0).sum()),
           'T50_min': float(x[:, 0].min()), 'T50_max': float(x[:, 0].max()),
           'S5_min': float(x[:, 1].min()), 'S5_max': float(x[:, 1].max())}
    if not quality:
        assert np.isin(x, [0, 1]).all()
        t, s = x[:, 0].astype(bool), x[:, 1].astype(bool)
        out['dynamic_counts'] = {'T50': int(t.sum()), 'S5': int(s.sum()), 'both_dynamic': int((t&s).sum()),
                                 'both_static': int((~t&~s).sum()), 'T_dynamic_S_static': int((t&~s).sum()),
                                 'T_static_S_dynamic': int((~t&s).sum()), 'automatic_tolerance': None}
    return out


def timing_summary(pairs, libraries):
    import numpy as np
    x = np.asarray(pairs, dtype=np.float64)
    assert x.shape == (len(libraries), 2) and (x > 0).all() and np.isfinite(x).all()
    out = {}
    for i, arm in enumerate(ARMS):
        q = np.quantile(x[:, i], [.25, .5, .75], method='linear')
        out[arm] = {'median_seconds': float(q[1]), 'q1': float(q[0]), 'q3': float(q[2]),
                    'iqr': float(q[2]-q[0]), 'total_seconds': float(x[:, i].sum()), 'n': len(x)}
    rng = np.random.Generator(np.random.PCG64(20261019)); draws = []
    for library in dict.fromkeys(libraries):
        indices = np.array([i for i, x in enumerate(libraries) if x == library])
        draws.append(indices[rng.integers(0, len(indices), size=(20000, len(indices)))])
    sampled = x[np.concatenate(draws, axis=1)]
    ratio = np.median(sampled[:, :, 0], axis=1)/np.median(sampled[:, :, 1], axis=1)
    out.update(ratio_of_medians=float(np.median(x[:, 0])/np.median(x[:, 1])),
               median_paired_ratio=float(np.median(x[:, 0]/x[:, 1])),
               ratio_of_medians_95ci=np.quantile(ratio, [.025, .975], method='linear').tolist(),
               scope='synchronized denoise+VAE only; includes neither load/warmup/offload nor multi-card throughput gain')
    return out


def atomic_text(path, value):
    path = Path(path); assert path.resolve() == path and path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def csv_text(rows):
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def update_progress_marker(text):
    path = ROOT/'Project_Progress/Total_Progress.md'
    before = path.read_text(); old_sha = digest(path)
    start, end = '<!-- HYA_H2_AUTO_START -->', '<!-- HYA_H2_AUTO_END -->'
    assert before.count(start) == before.count(end) == 1
    updated = before.split(start)[0]+start+'\n'+text+'\n'+end+before.split(end)[1]
    assert digest(path) == old_sha, 'Progress changed concurrently'
    atomic_text(path, updated)


def main():
    import numpy as np
    manifest, protocol = check_frozen(); assert not (RESULT/'COMPLETE.json').exists()
    assert len(list((RUN/'pairs').glob('*.json'))) == 64
    scores = {}; raw_rows = []
    for path in sorted((RUN/'scores').glob('*.json')):
        rec = read(path); assert rec['status'] == 'passed' and rec['protocol_sha256'] == digest(RUN/'protocol.json')
        for row in rec['rows']:
            key = (row['prompt_id'], row['metric'], row['arm']); assert key not in scores
            scores[key] = row
            raw_rows.append({k: row[k] for k in ['prompt_id', 'seed', 'arm', 'metric', 'score', 'video_sha256', 'video_path']})
    expected = {(p['id'], m, a) for p in protocol['prompts'] for m in p['metrics'] for a in ARMS}
    assert set(scores) == expected and len(scores) == 224
    timing_rows, times, libraries, paired_rows, video_receipts = [], [], [], [], []
    timing_fields = ['denoising_seconds', 'vae_decode_seconds', 'hot_denoise_decode_seconds', 'warmup_seconds',
                     'model_to_gpu_seconds', 'model_offload_seconds', 'vae_to_gpu_seconds', 'vae_offload_seconds',
                     'export_seconds', 'video_wall_seconds', 'formal_path_dit_calls', 'extra_warmup_dit_calls']
    for p in protocol['prompts']:
        pair = read(RUN/'pairs'/(p['id']+'.json')); records = pair['records']
        assert records['T50']['admission']['uuid'] == records['S5']['admission']['uuid']
        assert records['T50']['actual_initial_latent_sha256'] == records['S5']['actual_initial_latent_sha256']
        assert records['T50']['inputs']['hashes'] == records['S5']['inputs']['hashes']
        for arm in ARMS:
            rec = records[arm]; assert rec['seed'] == 2026101901 and rec['protocol_sha256'] == digest(RUN/'protocol.json')
            assert rec['formal_path_dit_calls'] == (50 if arm == 'T50' else 5)
            assert rec['video']['frames_decord'] == rec['video']['frames_opencv'] == 129
            assert digest(rec['video_path']) == digest(rec['archive_path']) == rec['video']['sha256']
            for m in p['metrics']: assert scores[(p['id'], m, arm)]['video_sha256'] == rec['video']['sha256']
            timing_rows.append({'prompt_id': p['id'], 'library': p['library'], 'seed': rec['seed'], 'arm': arm,
                                'gpu_index': rec['admission']['index'], 'gpu_uuid': rec['admission']['uuid'],
                                **{k: rec[k] for k in timing_fields}})
            video_receipts.append({'prompt_id': p['id'], 'arm': arm, 'sha256': rec['video']['sha256'],
                                   'source': rec['video_path'], 'archive': rec['archive_path']})
        times.append([records[a]['hot_denoise_decode_seconds'] for a in ARMS]); libraries.append(p['library'])
        for m in p['metrics']:
            t, s = [scores[(p['id'], m, a)]['score'] for a in ARMS]
            paired_rows.append({'prompt_id': p['id'], 'library': p['library'], 'seed': 2026101901,
                                'metric': m, 'T50': t, 'S5': s, 'T50_minus_S5': t-s})
    metrics = protocol['metrics']['quality_family']+protocol['metrics']['guardrail']
    summaries = [metric_summary(m, [[scores[(p['id'], m, a)]['score'] for a in ARMS]
                                    for p in protocol['prompts'] if m in p['metrics']]) for m in metrics]
    timing = timing_summary(times, libraries); controller = read(RUN/'controller.json')
    execution = [{'task_id': tid, 'status': t['status'], **attempt}
                 for tid, t in controller['tasks'].items() for attempt in t['attempts']]
    # Preserve off-path completed video records from failed attempts as cost evidence, never select by score.
    attempt_records = []
    for path in sorted((RUN/'generation').glob('*/attempt*/*/*.json')):
        r = read(path)
        attempt_records.append({'path': str(path), 'prompt_id': r['prompt_id'], 'arm': r['arm'],
                                'hot_denoise_decode_seconds': r['hot_denoise_decode_seconds'],
                                'video_wall_seconds': r['video_wall_seconds'], 'video_sha256': r['video']['sha256']})
    candidates = [x['metric'] for x in summaries if x['teacher_advantage_candidate']]
    report = {'status': 'complete', 'stage': 'H2', 'protocol_id': protocol['protocol_id'],
              'protocol_sha256': digest(RUN/'protocol.json'), 'counts': protocol['counts'], 'metrics': summaries,
              'teacher_advantage_candidates': candidates, 'timing': timing,
              'wall_seconds_through_validation': time.time()-controller['started_unix'],
              'execution_attempts': execution, 'all_completed_video_attempts': attempt_records,
              'limitations': ['Fixed preselected prompts, one fixed seed; no cross-seed robustness evidence',
                              'Approximate bootstrap intervals, not full VBench replication',
                              'Dynamic degree is a guardrail; researcher judges tradeoffs'],
              'S3_started': False, 'researcher_confirmation_required_before_S3': True}
    RESULT.mkdir(exist_ok=True)
    atomic_text(RESULT/'raw_scores.csv', csv_text(raw_rows))
    atomic_text(RESULT/'paired_scores.csv', csv_text(paired_rows))
    atomic_text(RESULT/'timing_per_video.csv', csv_text(timing_rows))
    write_json(RESULT/'statistics.json', report, replace=True)
    write_json(RESULT/'video_archive_sha256.json', video_receipts, replace=True)
    lines = ['# S2-HYA / H2 教师资格结果', '', '128 条视频、224 个指标值及 64 个配对条件完整验收；原始与归档视频哈希一致。', '',
             '| 指标 | T50 | S5 | T−S | 配对区间 | 教师优势候选 |', '|---|---:|---:|---:|---|---|']
    for x in summaries:
        decision = '护栏，不判优' if x['metric'] == 'dynamic_degree' else ('是' if x['teacher_advantage_candidate'] else '未建立')
        lines.append(f"| {x['metric']} | {x['T50_mean']:.6f} | {x['S5_mean']:.6f} | {x['T50_minus_S5']:+.6f} | {x['ci_level']*100:g}% [{x['ci_low']:+.6f}, {x['ci_high']:+.6f}] | {decision} |")
    lines += ['', '四项质量的名义 Bonferroni 区间按固定 seed 下 prompt 配对 bootstrap 20,000 次计算；区间不含跨 seed 不确定性。',
              '目标维度候选：'+('、'.join(candidates) if candidates else '没有建立；不进入 S3。'),
              '提升幅度、其他指标变化和运动取舍由研究者决定；不宣称教师整体最优。S3 未启动。', '',
              f"热态去噪+VAE 中位数：T50 {timing['T50']['median_seconds']:.3f} 秒，S5 {timing['S5']['median_seconds']:.3f} 秒；中位数比 {timing['ratio_of_medians']:.3f}×。",
              '加载、预热、换权重/offload、导出与重试不混入该热态比值，分别保存在 timing_per_video.csv 和 statistics.json；多卡墙钟收益不算模型加速。',
              '逐样本分数见 raw_scores.csv，配对差见 paired_scores.csv，动态配对计数及失败成本见 statistics.json。', '']
    atomic_text(RESULT/'REPORT.md', '\n'.join(lines))
    atomic_text(ROOT/'Project_Progress/Stage_Progress/HYA_H2_QUALIFICATION.md', '\n'.join(lines))
    update_progress_marker('**H2 已完成：128/128 视频、224/224 分数、64/64 配对；S3 未启动。**\n\n教师优势候选：'+('、'.join(candidates) if candidates else '未建立')+'。详见 `Result/HYA_H2_20261019/REPORT.md`；最终目标与运动取舍等待研究者确认。')
    hashes = {p.name: digest(p) for p in RESULT.iterdir() if p.is_file()}
    write_json(RESULT/'COMPLETE.json', {'status': 'complete', 'finished_unix': time.time(), 'sha256': hashes, 'S3_started': False})
    print('H2_COMPLETE', json.dumps({'candidates': candidates, 'videos': 128, 'scores': 224}), flush=True)


if __name__ == '__main__':
    main()
