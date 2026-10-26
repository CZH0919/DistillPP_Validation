"""Render the 14-group experiment data as one bilingual PNG using Matplotlib.

Data are recomputed from scores.csv and timing.csv, then checked against both
summary.json and REVIEW.json. No model imports, CUDA, or remote access.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import warnings


METRICS = ('subject_consistency', 'multiple_objects', 'dynamic_degree')
VARIANTS = [f'step{step}_lambda_0p{strength}' for step in range(1, 5) for strength in (1, 3, 5)]
ARMS = ['T50', 'S5'] + VARIANTS


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(result):
    complete = read_json(result / 'COMPLETE.json')
    assert complete['status'] == 'complete'
    for name in ('scores.csv', 'timing.csv', 'summary.json'):
        assert digest(result / name) == complete['sha256'][name], name
    summary, review = [read_json(result / name) for name in ('summary.json', 'REVIEW.json')]
    scored = defaultdict(list)
    keys = set()
    with (result / 'scores.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['prompt_id'], row['group'], row['metric'])
            assert key not in keys
            keys.add(key)
            value = float(row['score'])
            assert math.isfinite(value) and 0 <= value <= 1
            scored[(row['group'], row['metric'])].append(value)
    assert len(keys) == 1120 and set(scored) == {(a, m) for a in ARMS for m in METRICS}
    times = defaultdict(list)
    timekeys = set()
    with (result / 'timing.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['prompt_id'], row['group'])
            assert key not in timekeys
            timekeys.add(key)
            value = float(row['inference_with_transfers_seconds'])
            assert math.isfinite(value) and value > 0
            times[row['group']].append(value)
    assert len(timekeys) == 672 and set(times) == set(ARMS)
    for arm in ARMS:
        assert len(times[arm]) == 48
        assert [len(scored[(arm, m)]) for m in METRICS] == [32, 16, 32]
    means = {(a, m): statistics.fmean(scored[(a, m)]) * 100 for a in ARMS for m in METRICS}
    medians = {a: statistics.median(times[a]) for a in ARMS}
    output = []
    for arm in ARMS:
        row = {'group': arm, 'metrics': {}, 'inference_seconds': medians[arm],
               'delta_seconds_vs_S5': medians[arm] - medians['S5'],
               'delta_time_percent_vs_S5': (medians[arm] / medians['S5'] - 1) * 100}
        for metric in METRICS:
            value, delta = means[(arm, metric)], means[(arm, metric)] - means[('S5', metric)]
            row['metrics'][metric] = {'value': value, 'delta_vs_S5': delta, 'n': len(scored[(arm, metric)])}
            for source in (summary, review):
                ref = next(x for x in source['metrics'] if x['metric'] == metric
                           and x['variant'] == (VARIANTS[0] if arm in ('T50', 'S5') else arm))
                expected = ref[arm] if arm in ('T50', 'S5') else ref['corrected']
                assert abs(value - expected * 100) < 1e-10
                if arm in VARIANTS:
                    assert abs(delta - ref['corrected_minus_student'] * 100) < 1e-10
        for source in (summary, review):
            assert abs(medians[arm] - source['timing_medians'][arm]['inference_with_transfers_seconds']) < 1e-10
        output.append(row)
    hashes = {name: digest(result / name) for name in ('scores.csv', 'timing.csv', 'summary.json', 'REVIEW.json')}
    return output, hashes


def main(project, output, fontfile=None):
    result = project / 'Result/HYA_H3_STEP24_V1'
    rows, source_hashes = collect(result)
    cache_root = project.parent / 'tmp/hya_h3_figure_20261025'
    cache_root.mkdir(parents=True, exist_ok=True)
    assert cache_root.resolve().is_relative_to(project.parent.resolve())
    with tempfile.TemporaryDirectory(prefix='matplotlib_', dir=cache_root) as config_dir:
        assert Path(config_dir).resolve().parent == cache_root.resolve()
        os.environ['MPLCONFIGDIR'] = config_dir
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.font_manager import FontProperties, findfont
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Bbox
        from PIL import Image

        if fontfile is None:
            candidates = [Path('C:/Windows/Fonts/msyh.ttc'),
                          Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
            fontfile = next((p for p in candidates if p.is_file()), None)
        if fontfile is None:
            for family in ('Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei'):
                try:
                    fontfile = Path(findfont(FontProperties(family=family), fallback_to_default=False))
                    break
                except ValueError:
                    pass
        if fontfile is None or not fontfile.is_file():
            raise RuntimeError('A Chinese font is required; pass --fontfile /absolute/path/to/font')
        boldfile = fontfile.with_name('msyhbd.ttc')
        normal = FontProperties(fname=str(fontfile))
        bold = FontProperties(fname=str(boldfile if boldfile.is_file() else fontfile))
        warnings.filterwarnings('error', message=r'Glyph .* missing from font')
        plt.rcParams.update({'axes.unicode_minus': False, 'text.usetex': False})
        fig = plt.figure(figsize=(20, 14.5), dpi=220, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ink, secondary = '#1C2D40', '#526171'
        increase, decrease, neutral = '#1E6091', '#AC4E28', '#66717D'
        rules = '#D9E1E8'
        boundaries = [.035, .215, .405, .595, .765, .965]
        centers = [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
        header_top, body_top, row_height = .895, .805, .0435
        body_bottom = body_top - len(rows) * row_height
        boxes = []

        def text(x, y, value, size=14, color=ink, strong=False, ha='center', bounds=None):
            artist = ax.text(x, y, value, ha=ha, va='center', fontsize=size, color=color,
                             fontproperties=bold if strong else normal, zorder=3)
            boxes.append((artist, bounds or (.025, .015, .975, .985)))
            return artist

        def delta_color(value):
            return increase if value > 1e-10 else decrease if value < -1e-10 else neutral

        text(.035, .958, '第1–4步 × 三种λ：14组实验数据汇总', 25, strong=True, ha='left')
        text(.035, .925, 'HunyuanVideo T50 → AccVideo S5   |   720p · 129帧 · 24 FPS   |   seed = 2026101901',
             12.5, color=secondary, ha='left')
        ax.add_patch(Rectangle((boundaries[0], body_top), boundaries[-1]-boundaries[0],
                               header_top-body_top, facecolor='#EAF0F6', edgecolor='none'))
        headings = [
            ('Experiment', '实验组 / 介入步骤与强度', '教师 + 学生 + 12个纠偏组'),
            ('subject_consistency', '主体一致性', '均值 · n = 32'),
            ('multiple_objects', '多物体符合度', '均值 · n = 16'),
            ('dynamic_degree', '动态比例', '运动护栏 · n = 32'),
            ('Inference time', '实测推理耗时（含传输）', '中位数 · 秒 · n = 48'),
        ]
        for i, (english, chinese, detail) in enumerate(headings):
            bounds = (boundaries[i]+.005, body_top+.004, boundaries[i+1]-.005, header_top-.004)
            text(centers[i], .869, english, 14.1, strong=True, bounds=bounds)
            text(centers[i], .843, chinese, 12.6, bounds=bounds)
            text(centers[i], .822, detail, 10.5, color=secondary, bounds=bounds)
        for index, row in enumerate(rows):
            arm = row['group']
            top = body_top - index * row_height
            bottom = top - row_height
            if arm == 'S5':
                fill = '#EAF3FC'
            elif arm == 'T50':
                fill = '#F1F4F7'
            else:
                step = int(arm[4])
                fill = '#FFFFFF' if step % 2 else '#F7F9FB'
            ax.add_patch(Rectangle((boundaries[0], bottom), boundaries[-1]-boundaries[0], row_height,
                                   facecolor=fill, edgecolor='none', zorder=0))
            if index in (0, 2, 5, 8, 11):
                ax.plot([boundaries[0], boundaries[-1]], [top, top], color='#A8B8C9', lw=1.0, zorder=1)
            else:
                ax.plot([boundaries[0], boundaries[-1]], [top, top], color=rules, lw=.45, zorder=1)
            left_bounds = (boundaries[0]+.007, bottom+.002, boundaries[1]-.007, top-.002)
            if arm in ('T50', 'S5'):
                text(centers[0], top-.0145, 'Teacher T50' if arm == 'T50' else 'Student S5',
                     15, strong=True, bounds=left_bounds)
                text(centers[0], top-.032, '原始教师 · 50步' if arm == 'T50' else '蒸馏学生 · 5步 / 比较基准',
                     10.8, color=secondary, bounds=left_bounds)
            else:
                step, strength = int(arm[4]), int(arm[-1])/10
                text(centers[0], top-row_height/2, f'第{step}步   λ = {strength:.1f}', 14.7, bounds=left_bounds)
            for col, metric in enumerate(METRICS, 1):
                value = row['metrics'][metric]['value']
                delta = row['metrics'][metric]['delta_vs_S5']
                bounds = (boundaries[col]+.005, bottom+.002, boundaries[col+1]-.005, top-.002)
                if metric == 'dynamic_degree':
                    main, change = f'{value:.3f}%', f'Δ {delta:+.3f} pp'
                else:
                    main, change = f'{value:.4f}', f'Δ {delta:+.4f}'
                text(centers[col], top-.0145, main, 15.5, strong=arm in ('T50','S5'), bounds=bounds)
                text(centers[col], top-.032, change, 11.6, color=delta_color(delta), bounds=bounds)
            bounds = (boundaries[4]+.004, bottom+.002, boundaries[5]-.004, top-.002)
            text(centers[4], top-.0145, f"{row['inference_seconds']:,.3f} s", 15.5,
                 strong=arm in ('T50','S5'), bounds=bounds)
            text(centers[4], top-.032,
                 f"Δ {row['delta_seconds_vs_S5']:+,.3f} s  ({row['delta_time_percent_vs_S5']:+.2f}%)",
                 11.2, color=delta_color(row['delta_seconds_vs_S5']), bounds=bounds)
        ax.plot([boundaries[0], boundaries[-1]], [body_bottom, body_bottom], color='#A8B8C9', lw=1)
        for x in boundaries[1:-1]:
            ax.plot([x, x], [body_bottom, header_top], color=rules, lw=.6, zorder=1)

        text(.035, .169, 'Δ = 当前组 − Student S5（由未舍入数据计算）', 11.6, strong=True, ha='left')
        text(.624, .169, '+ 数值增加', 11.4, color=increase)
        text(.744, .169, '− 数值减少', 11.4, color=decrease)
        text(.899, .169, '0 数值持平', 11.4, color=neutral)
        notes = [
            '指标口径：三项指标均为原分×100；前两项Δ单位为分，dynamic_degree的Δ单位为百分点（pp）。颜色仅表示增减，不表示优劣。',
            '时间口径：48条视频含传输推理耗时的中位数，含去噪、教师查询/纠偏、VAE及模型传输；不含加载、预热、导出与排队重试。',
            '时间括号：相对S5的耗时变化百分比；各组沿用既有prompt与seed，主体/运动32条、多物体16条；历史批次的计时采集时段不同。',
            '数据来源：HYA_H3_STEP24_V1 / scores.csv、timing.csv；与summary.json、REVIEW.json复核一致；全部正式样本保留。',
        ]
        for y, note in zip((.137, .109, .081, .053), notes):
            text(.035, y, note, 10.4, color=secondary, ha='left')

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist, bounds in boxes:
            actual = artist.get_window_extent(renderer)
            expected = Bbox.from_extents(*bounds).transformed(ax.transData)
            assert actual.x0 >= expected.x0 and actual.x1 <= expected.x1, f'Horizontal overflow: {artist.get_text()}'
            assert actual.y0 >= expected.y0 and actual.y1 <= expected.y1, f'Vertical overflow: {artist.get_text()}'
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {'Title': 'HunyuanVideo / AccVideo: 14-group objective metrics and measured inference time',
                    'SourceHashes': json.dumps(source_hashes, sort_keys=True),
                    'FigureData': json.dumps(rows, ensure_ascii=False, allow_nan=False),
                    'TimeDefinition': 'median(inference_with_transfers_seconds), 48 videos per group',
                    'DeltaDefinition': 'current group minus S5; quality points, dynamic percentage points; time seconds and relative percent'}
        fig.savefig(output, dpi=220, facecolor='white', metadata=metadata)
        plt.close(fig)
        with Image.open(output) as saved:
            assert saved.size == (4400, 3190)
            assert json.loads(saved.info['FigureData']) == rows
            saved.verify()
        print(json.dumps({'output': str(output), 'groups': len(rows), 'png_pixels': [4400, 3190],
                          'sha256': digest(output), 'bytes': output.stat().st_size,
                          'all_text_within_cells': True, 'source_hashes': source_hashes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fontfile', type=Path)
    args = parser.parse_args()
    target = args.output or args.project / 'Result/HYA_H3_STEP24_V1/Figures/HYA_H3_14_GROUPS_SUMMARY.png'
    main(args.project.resolve(), target.resolve(), args.fontfile)
