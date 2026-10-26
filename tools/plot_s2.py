"""Render numeric S2 results only; no video viewing or human quality scores."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result-dir',required=True,type=Path)
    parser.add_argument('--overwrite',action='store_true',help='Replace this script\'s previously generated figures')
    args=parser.parse_args()
    root=args.result_dir.resolve()
    assert root.name=='S2_20261016' and root.parent.name=='Result'
    report=json.loads((root/'statistics.json').read_text(encoding='utf-8'))
    assert report['integrity']['valid_metric_values']==576
    with (root/'prompt_means.csv').open(encoding='utf-8',newline='') as f:prompts=list(csv.DictReader(f))
    with (root/'timing_per_video.csv').open(encoding='utf-8',newline='') as f:timings=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2878B5','#E59E38','#45A778','#A369AE']
    strata=list(dict.fromkeys(r['stratum'] for r in prompts))
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for ax,summary in zip(axes.flat,report['metric_summaries']):
        metric=summary['metric']
        ax.axvline(0,color='#888888',lw=1,ls='--')
        for k,stratum in enumerate(strata):
            rows=[r for r in prompts if r['metric']==metric and r['stratum']==stratum]
            assert len(rows)==4
            y=np.linspace(.26,.53,4)+k*.015
            ax.scatter([float(r['paired_difference']) for r in rows],y,s=27,c=colors[k],alpha=.8,
                       label=stratum.replace('_',' '),zorder=3)
        low,high=summary['ci_low'],summary['ci_high'];delta=summary['T50_minus_S4']
        ax.plot([low,high],[.81,.81],color='#172D45',lw=3,zorder=4)
        ax.plot([low,low],[.77,.85],color='#172D45',lw=1.5)
        ax.plot([high,high],[.77,.85],color='#172D45',lw=1.5)
        ax.scatter([delta],[.81],marker='D',s=45,c='#172D45',zorder=5)
        ax.set_title(metric.replace('_',' '),loc='left',fontweight='bold')
        ax.text(.02,.95,'T %.4f  |  S %.4f  |  difference %+.4f'%(summary['T50_mean'],summary['S4_mean'],delta),
                transform=ax.transAxes,va='top',fontsize=9)
        ax.set_ylim(.1,1.03);ax.set_yticks([])
        ax.set_xlabel('T50 - S4 (normalized score)')
        ax.grid(axis='x',alpha=.18)
        ax.text(.02,.02,'95% CI; movement, no threshold' if metric=='dynamic_degree' else '99% CI; five quality comparisons',
                transform=ax.transAxes,fontsize=8,color='#555555')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('S2 paired results | 16 prompt clusters x 3 seeds\nDots: prompt means; diamond/bar: stratified paired estimate and bootstrap CI',fontsize=14)
    for suffix in ['png','pdf']:
        dest=root/('paired_differences.'+suffix)
        assert not dest.exists() or args.overwrite, 'Refuse overwrite: '+str(dest)
        fig.savefig(dest,dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    values=[[float(r['denoise_plus_vae_seconds']) for r in timings if r['arm']==arm] for arm in ['T50','S4']]
    axes[0].boxplot(values,tick_labels=['T50','S4'],showfliers=True,widths=.45)
    axes[0].set_yscale('log');axes[0].set_ylabel('Denoise + VAE (seconds; log scale)')
    axes[0].set_ylim(22,700);axes[0].set_yticks([30,100,500],labels=['30','100','500'])
    axes[0].set_title('Warmed, one H200 per video',loc='left')
    axes[0].grid(axis='y',alpha=.2)
    ratio=report['timing']['speed_ratio_medians'];lo,hi=report['timing']['speed_ratio_95ci']
    axes[0].text(.03,.62,'Median T/S: %.2fx\n95%% cluster CI: [%.2f, %.2f]'%(ratio,lo,hi),transform=axes[0].transAxes,va='top')
    for i,arm in enumerate(['T50','S4']):
        rows=sorted([r for r in timings if r['arm']==arm],key=lambda r:float(r['unix_start']))
        x=[(float(r['unix_start'])-min(float(x['unix_start']) for x in timings))/60 for r in rows]
        # Divide by that arm's median to expose load variation without hiding S4.
        y=np.array([float(r['denoise_plus_vae_seconds']) for r in rows])/np.median(values[i])
        axes[1].scatter(x,y,s=22,label=arm,alpha=.75,c=colors[i])
    axes[1].axhline(1,color='#888888',ls='--',lw=1)
    axes[1].set_xlabel('Minutes from first formal generation')
    axes[1].set_ylabel('Time / arm median')
    axes[1].set_title('Per-video timing variation',loc='left')
    axes[1].legend(frameon=False);axes[1].grid(alpha=.18)
    for suffix in ['png','pdf']:
        dest=root/('timing.'+suffix);assert not dest.exists() or args.overwrite
        fig.savefig(dest,dpi=180)
    plt.close(fig)
    print('S2_PLOTS_SAVED',root)


if __name__=='__main__':main()
