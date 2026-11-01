"""Render the 14-group experiment data as a minimal monochrome table.

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
        fig = plt.figure(figsize=(24, 12), dpi=180, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        boundaries = [.018, .149, .350, .551, .735, .982]
        header_top, body_top, body_bottom = .985, .920, .020
        row_height = (body_top - body_bottom) / len(rows)
        boxes = []

        def text(col, y, value, size=17, strong=False, bottom=0, top=1):
            artist = ax.text(boundaries[col] + .010, y, value, ha='left',
                             va='center', fontsize=size, color='black',
                             fontproperties=bold if strong else normal)
            boxes.append((artist, (boundaries[col] + .004, bottom + .002,
                                   boundaries[col + 1] - .004, top - .002)))
            return artist

        headings = [
            '组别',
            '主体一致性（较 S5）',
            '多物体符合度（较 S5）',
            '动态比例（较 S5）',
            '耗时中位数 / 秒（较 S5）',
        ]
        for col, heading in enumerate(headings):
            text(col, (header_top + body_top) / 2, heading, strong=True,
                 bottom=body_top, top=header_top)
        ax.plot([boundaries[0], boundaries[-1]], [body_top, body_top],
                color='#BBBBBB', lw=.8)

        for index, row in enumerate(rows):
            arm = row['group']
            top = body_top - index * row_height
            bottom = top - row_height
            y = (top + bottom) / 2
            baseline = arm in ('T50', 'S5')
            label = arm if baseline else f'第{int(arm[4])}步 λ={int(arm[-1])/10:.1f}'
            text(0, y, label, bottom=bottom, top=top)
            for col, metric in enumerate(METRICS, 1):
                value = row['metrics'][metric]['value']
                delta = row['metrics'][metric]['delta_vs_S5']
                if metric == 'dynamic_degree':
                    display = f'{value:.3f}%'
                    if not baseline:
                        display += f' ({delta:+.3f} pp)'
                else:
                    display = f'{value:.4f}'
                    if not baseline:
                        display += f' ({delta:+.4f})'
                text(col, y, display, bottom=bottom, top=top)
            time_display = f"{row['inference_seconds']:.3f}"
            if not baseline:
                time_display += (f" ({row['delta_seconds_vs_S5']:+.3f}; "
                                 f"{row['delta_time_percent_vs_S5']:+.2f}%)")
            text(4, y, time_display, bottom=bottom, top=top)
            ax.plot([boundaries[0], boundaries[-1]], [bottom, bottom],
                    color='#DDDDDD', lw=.55)

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
        fig.savefig(output, dpi=180, facecolor='white', metadata=metadata)
        plt.close(fig)
        with Image.open(output) as saved:
            assert saved.size == (4320, 2160)
            assert json.loads(saved.info['FigureData']) == rows
            saved.verify()
        print(json.dumps({'output': str(output), 'groups': len(rows), 'png_pixels': [4320, 2160],
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
