"""Draw the 12 HYA correction settings as a minimal two-metric scatter plot.

CPU only. Reuses the verified CSV aggregation and preserves source provenance in
PNG metadata, without rendering a title or explanatory footnotes.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import warnings

from plot_hya_h3_14_groups import collect, digest


def main(project, fontfile=None):
    result = project / 'Result/HYA_H3_STEP24_V1'
    rows, hashes = collect(result)
    groups = {row['group']: row for row in rows}
    output = result / 'Figures/HYA_H3_QUALITY_TRADEOFF.png'
    cache_root = project.parent / 'tmp/hya_h3_figure_20261025'
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tradeoff_', dir=cache_root) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.font_manager import FontProperties
        from matplotlib.text import Text
        from PIL import Image

        if fontfile is None:
            fontfile = next((p for p in [Path('C:/Windows/Fonts/msyh.ttc'),
                Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
                if p.is_file()), None)
        if fontfile is None or not fontfile.is_file():
            raise RuntimeError('A Chinese font is required; pass --fontfile')
        font_manager.fontManager.addfont(str(fontfile))
        font = FontProperties(fname=str(fontfile))
        warnings.filterwarnings('error', message=r'Glyph .* missing from font')
        plt.rcParams.update({
            'font.family': font.get_name(), 'font.size': 12,
            'axes.unicode_minus': False, 'text.color': 'black',
            'axes.labelcolor': 'black', 'xtick.color': 'black',
            'ytick.color': 'black', 'axes.edgecolor': '#555555',
            'axes.linewidth': .8, 'text.usetex': False,
        })
        fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=250, facecolor='white')
        fig.subplots_adjust(left=.105, right=.980, bottom=.120, top=.975)
        ax.set(xlim=(-.025, .265), ylim=(-3.5, 7.0))
        ax.set_xlabel('主体一致性变化（较 S5，分）', labelpad=12)
        ax.set_ylabel('多物体符合度变化（较 S5，分）', labelpad=12)
        ax.set_xticks([0, .05, .10, .15, .20, .25],
                      ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25'])
        ax.set_yticks([-2, 0, 2, 4, 6])
        ax.tick_params(length=3, pad=7)
        ax.grid(color='#E5E5E5', linewidth=.65, linestyle=':', zorder=0)
        ax.axhline(0, color='#777777', lw=.9, linestyle=(0, (3, 3)), zorder=1)
        ax.axvline(0, color='#777777', lw=.9, linestyle=(0, (3, 3)), zorder=1)

        # Preserve the supplied plot's step colors and distinct marker shapes.
        styles = [(1, '#1f77b4', 'o'), (2, '#ff7f0e', 's'),
                  (3, '#2ca02c', '^'), (4, '#d62728', 'D')]
        offsets = {
            (1, 1): (7, 10, 'left'), (1, 3): (7, 10, 'left'),
            (1, 5): (-7, 10, 'right'),
            (2, 1): (-8, 10, 'right'), (2, 3): (8, 10, 'left'),
            (2, 5): (8, 10, 'left'),
            (3, 1): (-10, -18, 'right'), (3, 3): (8, 10, 'left'),
            (3, 5): (10, -18, 'left'),
            (4, 1): (-8, 10, 'right'), (4, 3): (9, 4, 'left'),
            (4, 5): (9, 10, 'left'),
        }
        points, annotations = [], []
        for step, color, marker in styles:
            xs, ys = [], []
            for k in (1, 3, 5):
                group = f'step{step}_lambda_0p{k}'
                row = groups[group]
                x = row['metrics']['subject_consistency']['delta_vs_S5']
                y = row['metrics']['multiple_objects']['delta_vs_S5']
                xs.append(x)
                ys.append(y)
                points.append({'group': group, 'step': step, 'lambda': k / 10,
                               'subject_delta_points': x, 'objects_delta_points': y})
                dx, dy, align = offsets[(step, k)]
                annotations.append(ax.annotate(
                    f'λ={k / 10:.1f}', xy=(x, y), xytext=(dx, dy),
                    textcoords='offset points', ha=align, va='center',
                    fontsize=11.5, color='black', annotation_clip=False,
                ))
            ax.scatter(xs, ys, color=color, marker=marker, s=78,
                       edgecolors='white', linewidths=.65, zorder=3,
                       label=f'第{step}步')
        ax.legend(loc='upper right', frameon=False, handletextpad=.5,
                  labelspacing=.6, borderaxespad=.7, fontsize=12)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= width, text.get_text()
            assert box.y0 >= 0 and box.y1 <= height, text.get_text()
        boxes = [a.get_window_extent(renderer) for a in annotations]
        for i, box in enumerate(boxes):
            assert not any(box.overlaps(other) for other in boxes[i + 1:]), \
                f'Overlapping lambda label: {points[i]["group"]}'
        assert len(points) == 12
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            'SourceHashes': json.dumps(hashes, sort_keys=True),
            'FigureData': json.dumps(points, allow_nan=False),
            'Units': 'Both axes: raw metric score x 100; correction minus S5',
            'Scope': 'Subject n=32, objects n=16; seed=2026101901; observed means',
        }
        fig.savefig(output, dpi=250, facecolor='white', metadata=metadata)
        plt.close(fig)
        with Image.open(output) as saved:
            assert json.loads(saved.info['FigureData']) == points
            assert saved.size == (width, height)
            saved.verify()
        print(json.dumps({'output': str(output), 'points': len(points),
                          'pixels': [width, height], 'sha256': digest(output),
                          'source_hashes': hashes,
                          'label_bounds_and_overlap_checked': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fontfile', type=Path)
    args = parser.parse_args()
    main(args.project.resolve(), args.fontfile)
