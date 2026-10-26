"""Two CPU-only figures focused on intervention step and correction strength."""
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
    groups = {r['group']: r for r in rows}
    figure_dir = result / 'Figures'
    figure_dir.mkdir(exist_ok=True)
    cache_root = project.parent / 'tmp/hya_h3_figure_20261025'
    cache_root.mkdir(parents=True, exist_ok=True)
    assert cache_root.resolve().is_relative_to(project.parent.resolve())
    with tempfile.TemporaryDirectory(prefix='comparison_', dir=cache_root) as cache:
        assert Path(cache).resolve().parent == cache_root.resolve()
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import colors, font_manager
        from matplotlib.font_manager import FontProperties
        from matplotlib.lines import Line2D
        from matplotlib.text import Text
        import numpy as np
        from PIL import Image

        if fontfile is None:
            candidates = [Path('C:/Windows/Fonts/msyh.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
            fontfile = next((p for p in candidates if p.is_file()), None)
        if fontfile is None:
            for family in ('Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei'):
                try:
                    fontfile = Path(font_manager.findfont(FontProperties(family=family), fallback_to_default=False))
                    break
                except ValueError:
                    pass
        if fontfile is None:
            raise RuntimeError('A Chinese font is required; use --fontfile')
        regular = FontProperties(fname=str(fontfile))
        boldfile = fontfile.with_name('msyhbd.ttc')
        bold = FontProperties(fname=str(boldfile if boldfile.exists() else fontfile))
        font_manager.fontManager.addfont(str(fontfile))
        plt.rcParams.update({'font.family': regular.get_name(), 'axes.unicode_minus': False,
                             'axes.edgecolor': '#B1BCC8', 'axes.labelcolor': '#34475A',
                             'xtick.color': '#526171', 'ytick.color': '#526171', 'font.size': 11})
        warnings.filterwarnings('error', message=r'Glyph .* missing from font')
        ink, secondary = '#1C2D40', '#526171'
        blue, orange = '#1E6091', '#B45C2F'
        signed = colors.LinearSegmentedColormap.from_list('signed_changes', [orange, '#FFFFFF', blue])
        sequential = colors.LinearSegmentedColormap.from_list('time_cost', ['#F4F7FA', blue])
        saved = []

        def ft(fig, x, y, text, size=12, color=ink, strong=False, ha='left', **kw):
            return fig.text(x, y, text, fontsize=size, color=color, fontproperties=bold if strong else regular,
                            ha=ha, va='center', **kw)

        def save(fig, name, payload):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            width, height = fig.canvas.get_width_height()
            for artist in fig.findobj(Text):
                if not artist.get_visible() or not artist.get_text():
                    continue
                box = artist.get_window_extent(renderer)
                assert box.x0 >= -1 and box.x1 <= width + 1, 'Horizontal clipping: ' + artist.get_text()
                assert box.y0 >= -1 and box.y1 <= height + 1, 'Vertical clipping: ' + artist.get_text()
            path = figure_dir / name
            fig.savefig(path, dpi=220, facecolor='white', metadata={
                'SourceHashes': json.dumps(hashes, sort_keys=True),
                'FigureData': json.dumps(payload, ensure_ascii=False, allow_nan=False),
                'Scope': 'Fixed 48 prompts and one seed (2026101901); all formal samples retained',
                'Units': 'Quality: raw score x 100, delta in points; dynamic delta in percentage points',
            })
            plt.close(fig)
            with Image.open(path) as im:
                assert json.loads(im.info['FigureData']) == payload
                assert im.size == (width, height)
                im.verify()
            saved.append({'path': str(path), 'size': [width, height], 'sha256': digest(path), 'bytes': path.stat().st_size})

        # Figure 1: aligned 4x3 matrices with separate, explicit scales.
        fig = plt.figure(figsize=(18, 12.5), dpi=220, facecolor='white')
        ft(fig, .055, .955, '介入步骤 × 纠偏强度：相对学生的变化', 24, strong=True)
        ft(fig, .055, .921, 'HunyuanVideo → AccVideo  |  12个纠偏设置  |  每组48条prompt · 单seed · 720p', 12.5, color=secondary)
        panels = [
            ('subject_consistency', '主体一致性', 32, .25, '分', (.09, .56, .335, .25), (.442, .56, .013, .25)),
            ('multiple_objects', '多物体符合度', 16, 6.5, '分', (.575, .56, .335, .25), (.927, .56, .013, .25)),
            ('dynamic_degree', '动态比例 / 运动护栏', 32, 3.125, '百分点', (.09, .205, .335, .25), (.442, .205, .013, .25)),
            ('inference_time', '含传输推理耗时', 48, 45, '%', (.575, .205, .335, .25), (.927, .205, .013, .25)),
        ]
        matrices = {}
        for metric, chinese, n, limit, unit, rect, cbrect in panels:
            ax = fig.add_axes(rect)
            values = []
            seconds = []
            for step in range(1, 5):
                names = [f'step{step}_lambda_0p{k}' for k in (1, 3, 5)]
                values.append([groups[v]['delta_time_percent_vs_S5'] if metric == 'inference_time'
                               else groups[v]['metrics'][metric]['delta_vs_S5'] for v in names])
                seconds.append([groups[v]['inference_seconds'] for v in names])
            data = np.array(values)
            matrices[metric] = values
            istime = metric == 'inference_time'
            norm = colors.Normalize(0, limit) if istime else colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            cmap = sequential if istime else signed
            im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
            ax.set_xticks(range(3), ['λ = 0.1', 'λ = 0.3', 'λ = 0.5'])
            ax.set_yticks(range(4), ['第1步', '第2步', '第3步', '第4步'])
            ax.tick_params(axis='both', which='major', length=0, pad=10, labelsize=12)
            ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=2.5)
            ax.tick_params(which='minor', left=False, bottom=False)
            for spine in ax.spines.values(): spine.set_visible(False)
            for i in range(4):
                for j in range(3):
                    rgba = cmap(norm(data[i, j]))
                    luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                    color = 'white' if luminance < .55 else ink
                    label = f'{data[i,j]:+.2f}%' if istime else f'{data[i,j]:+.3f}' if metric == 'dynamic_degree' else f'{data[i,j]:+.4f}'
                    ax.text(j, i - (.12 if istime else 0), label, ha='center', va='center',
                            fontsize=16.3, fontproperties=bold, color=color)
                    if istime:
                        ax.text(j, i+.24, f'{seconds[i][j]:.3f} s', ha='center', va='center',
                                fontsize=10.8, fontproperties=regular, color=color)
            title_y = rect[1] + rect[3] + .065
            ft(fig, rect[0]-.035, title_y, 'Inference time' if istime else metric, 15.2, strong=True)
            if istime:
                sub = f'{chinese} · n={n}  |  S5 {groups["S5"]["inference_seconds"]:.3f} s'
            else:
                value = groups['S5']['metrics'][metric]['value']
                sub = f'{chinese} · n={n}  |  S5 {value:.3f}%' if metric == 'dynamic_degree' else f'{chinese} · n={n}  |  S5 {value:.4f}'
            ft(fig, rect[0]-.035, title_y-.029, sub, 11.5, color=secondary)
            cb = fig.colorbar(im, cax=fig.add_axes(cbrect))
            cb.outline.set_visible(False)
            cb.set_ticks([0, 15, 30, 45] if istime else [-limit, 0, limit])
            cb.ax.tick_params(labelsize=9.5, length=2)
            cb.ax.set_title(unit, fontsize=10, pad=8, fontproperties=regular, color=secondary)
        ft(fig, .055, .115, 'Δ = 当前组 − S5；前两项为原分×100后的分数差，dynamic_degree为百分点变化。各面板色标独立，颜色不跨指标比较。', 10.4, color=secondary)
        ft(fig, .055, .084, '时间面板：含传输推理耗时中位数及较S5增加的百分比；不含加载、预热、导出、排队与重试。颜色仅表示数值方向与大小。', 10.4, color=secondary)
        ft(fig, .055, .053, '数据：HYA_H3_STEP24_V1 / scores.csv、timing.csv。全部正式样本；单seed观察结果，未按教师优势筛选prompt。', 10.4, color=secondary)
        save(fig, 'HYA_H3_STEP_LAMBDA_EFFECTS.png', {'matrices': matrices, 'row_steps': [1,2,3,4], 'column_lambdas': [.1,.3,.5]})

        # Figure 2: three strengths across the four discrete intervention steps.
        fig = plt.figure(figsize=(19.5, 7.5), dpi=220, facecolor='white')
        ft(fig, .045, .938, '介入时机与纠偏幅度：三项指标的变化曲线', 23, strong=True)
        ft(fig, .045, .875, '纵轴：相对纯学生S5的均值变化  |  横轴：唯一一次教师介入的学生步骤', 11.8, color=secondary)
        styles = [(.1, '#236B9B', 'o', 8.5, ':'), (.3, '#B87822', 'D', 7, '-'), (.5, '#65784F', 's', 4.5, '--')]
        legend = [Line2D([0],[0],color=color,marker=marker,markersize=size,linestyle=style,
                         markerfacecolor='white' if strength<.5 else color,linewidth=1.7,label=f'λ = {strength:.1f}')
                  for strength,color,marker,size,style in styles]
        legendfont = regular.copy(); legendfont.set_size(12)
        fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.967,.925),ncol=3,frameon=False,prop=legendfont)
        specs = [
            ('subject_consistency','主体一致性 · n=32',(.075,.255,.245,.45),(-.027,.267),[-.0,.05,.10,.15,.20,.25],'变化（分）'),
            ('multiple_objects','多物体符合度 · n=16',(.395,.255,.245,.45),(-3.6,7.2),[-3,0,3,6],'变化（分）'),
            ('dynamic_degree','动态比例 / 运动护栏 · n=32',(.715,.255,.245,.45),(-3.7,.65),[-3.125,0],'变化（百分点）'),
        ]
        curves = {}
        for metric,chinese,rect,limits,ticks,unit in specs:
            ax = fig.add_axes(rect)
            curves[metric] = {}
            ax.set(xlim=(.85,4.15),ylim=limits)
            ax.set_xticks(range(1,5),['第1步','第2步','第3步','第4步'])
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=11,length=3)
            ax.grid(axis='y',color='#E1E7ED',lw=.65)
            ax.axhline(0,color='#677785',lw=1.0,linestyle=(0,(3,3)),zorder=2)
            for strength,color,marker,size,style in styles:
                values = [groups[f'step{step}_lambda_0p{int(strength*10)}']['metrics'][metric]['delta_vs_S5'] for step in range(1,5)]
                curves[metric][str(strength)] = values
                ax.plot(range(1,5),values,color=color,linestyle=style,linewidth=1.8,marker=marker,markersize=size,
                        markeredgewidth=1.4,markerfacecolor='white' if strength<.5 else color,zorder=3)
            for side in ('top','right'): ax.spines[side].set_visible(False)
            ax.set_ylabel(unit,fontproperties=regular,fontsize=11.3,labelpad=10)
            ax.set_xlabel('教师介入步骤',fontproperties=regular,fontsize=11.3,labelpad=10)
            ft(fig,rect[0]-.025,.79,metric,15,strong=True)
            ft(fig,rect[0]-.025,.746,chinese,11.6,color=secondary)
            if metric=='dynamic_degree':
                ax.text(3.15,-2.35,'λ=0.3 与 λ=0.5 的四个均值重合',ha='center',va='center',
                        fontsize=10,fontproperties=regular,color=secondary)
        ft(fig,.045,.114,'各点为完整任务库的观测均值，Δ由未舍入数据计算；连线仅辅助比较四个离散步骤。三种λ在重合位置用不同符号区分。',10.4,color=secondary)
        ft(fig,.045,.063,'固定48条prompt与单seed=2026101901；dynamic_degree不按越高越好判断。数据：HYA_H3_STEP24_V1 / scores.csv。',10.4,color=secondary)
        save(fig,'HYA_H3_STEP_LAMBDA_CURVES.png',{'steps':[1,2,3,4],'lambdas':[.1,.3,.5],'curves':curves})
        print(json.dumps({'figures': saved, 'source_hashes': hashes, 'all_plotted_means_checked': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fontfile', type=Path)
    args = parser.parse_args()
    main(args.project.resolve(), args.fontfile)
