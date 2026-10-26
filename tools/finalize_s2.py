"""Validate all S2 evidence, archive 96 videos, compute frozen paired statistics."""
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from s2_common import PROJECT,BATCH,METRICS,load_frozen,complete_generations,json_write,file_hash
from s2_statistics import analyze

OUT=PROJECT/'Result/S2_20261016'


def write_once(path,data):
    assert path.resolve()==path and path.is_relative_to(PROJECT)
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=data.encode('utf-8') if isinstance(data,str) else data
    if path.exists():
        assert path.read_bytes()==blob, 'Refuse to overwrite different final artifact: '+str(path)
    else:
        with path.open('xb') as handle:handle.write(blob)


def save_json(name,value):
    write_once(OUT/name,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_csv(name,rows,fields=None):
    assert rows
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]),extrasaction='ignore',lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    write_once(OUT/name,stream.getvalue())


def main():
    execution,settings=load_frozen()
    generation=complete_generations(execution)
    panel=json.loads((BATCH/'panel.json').read_text())
    freeze=json.loads((BATCH/'analysis_freeze.json').read_text())
    for rel,h in freeze['source_sha256'].items():assert file_hash(PROJECT/rel)==h,rel
    worker_records=[]
    for worker in range(execution['worker_count']):
        valid=list((BATCH/'workers'/('worker%02d'%worker)).glob('attempt*/result.json'))
        assert len(valid)==1
        result=json.loads(valid[0].read_text());assert result['status']=='passed'
        for arm in ['T50','S4']:
            assert result['warmup'][arm]['status']=='passed' and result['warmup'][arm]['warmup'] is True
        worker_records.append(result)
    scored=[]
    for metric in METRICS:
        valid=list((BATCH/'evaluation'/metric).glob('attempt*/result.json'))
        assert len(valid)==1, metric
        result=json.loads(valid[0].read_text())
        assert result['status']=='passed' and len(result['rows'])==96 and result['network_attempts']==[]
        assert result['source_sha256']==freeze['source_sha256']['tools/evaluate_s2.py']
        scored.extend(result['rows'])
    assert len(scored)==576
    lookup={(r['pair_id'],r['arm'],r['metric']):r for r in scored}
    assert len(lookup)==576
    gen={(r['pair_id'],r['arm']):r for r in generation}
    strata=list(dict.fromkeys(p['stratum'] for p in panel['prompts']))
    scores=np.empty((4,4,3,2,6),dtype=np.float64)
    timings=np.empty((4,4,3,2),dtype=np.float64)
    for h,stratum in enumerate(strata):
        prompts=[p for p in panel['prompts'] if p['stratum']==stratum]
        assert len(prompts)==4
        for i,prompt in enumerate(prompts):
            for s,seed in enumerate(panel['seed_slots']):
                pair_id=prompt['id']+'_seed'+str(seed)
                for a,arm in enumerate(['T50','S4']):
                    row=gen[pair_id,arm]
                    assert row['video']['frames']==81 and row['video']['size']==[480,832] and row['video']['fps']==16
                    assert row['video']['decord_all_frames_decoded'] and row['video']['opencv_all_frames_decoded']
                    assert row['video']['pixel_std']>0 and row['output_tensor']['std']>0
                    timings[h,i,s,a]=row['denoise_plus_vae_seconds']
                    for k,metric in enumerate(METRICS):
                        value=lookup[pair_id,arm,metric]
                        assert value['prompt']==prompt['prompt'] and value['seed']==seed
                        assert value['video_sha256']==row['video']['sha256']
                        assert value['video_path']==row['video_path']
                        scores[h,i,s,a,k]=value['normalized_score']
    assert len({r['video']['sha256'] for r in generation})==96, 'Unexpected duplicate encoded outputs'
    assert len({r['initial_latent_sha256'] for r in generation})==3
    report,prompt_deltas=analyze(scores,timings)
    anomalies=[]
    endpoint_counts={}
    for k,metric in enumerate(METRICS):
        values=scores[...,k]
        endpoint_counts[metric]={'zero':int((values==0).sum()),'one':int((values==1).sum()),'n':96}
        if metric!='dynamic_degree' and np.ptp(values)==0:
            anomalies.append('Constant scores across all videos: '+metric)
    report.update(status='S2_COMPLETE',batch=execution['batch_id'],numpy_version=np.__version__,
        integrity={'accepted_pairs':48,'videos':96,'valid_metric_values':576,'unique_video_hashes':96,
            'same_noise_context_gpu_within_pair':True,'both_full_decoders_passed':True,
            'all_frames_dimensions_fps_passed':True,'nonfinite_scores':0,'technical_anomalies':anomalies,
            'score_endpoint_counts':endpoint_counts},
        qualification_ready=report['statistical_teacher_advantage'] and not anomalies,
        limitations=['16 fixed prompt clusters, four per stratum; exploratory small-sample bootstrap.',
            'Conditional on this panel and three fixed seed slots; no broad population or exact-coverage claim.',
            'Native precision/samplers/deployment implementations differ; not an isolated causal effect of distillation.',
            'Motion indicator has no acceptance threshold; quality-motion tradeoffs require researcher judgment.',
            'S2 has no teacher correction; no evidence of Distillation++ transfer effectiveness.'],
        input_sha256=execution['input_sha256'],analysis_freeze=freeze)
    failures=[]
    for path in sorted(BATCH.glob('**/failure.json')):
        row=json.loads(path.read_text())
        failures.append({'path':str(path.relative_to(PROJECT)),'error':row['error'],
            'worker_id':row.get('worker_id'),'metric':row.get('metric'),'attempt':row.get('attempt')})
    report['failures']=failures
    save_json('statistics.json',report)
    scored.sort(key=lambda r:(r['prompt_id'],r['seed'],r['arm'],METRICS.index(r['metric'])))
    save_csv('metrics_per_video.csv',scored)
    save_csv('metric_summary.csv',report['metric_summaries'])
    paired=[];prompt_rows=[];stratum_rows=[]
    for h,stratum in enumerate(strata):
        prompts=[p for p in panel['prompts'] if p['stratum']==stratum]
        for i,prompt in enumerate(prompts):
            for k,metric in enumerate(METRICS):
                prompt_rows.append({'prompt_id':prompt['id'],'stratum':stratum,'prompt':prompt['prompt'],
                    'metric':metric,'T50_mean':float(scores[h,i,:,0,k].mean()),
                    'S4_mean':float(scores[h,i,:,1,k].mean()),'paired_difference':float(prompt_deltas[h,i,k])})
                for s,seed in enumerate(panel['seed_slots']):
                    paired.append({'prompt_id':prompt['id'],'stratum':stratum,'seed':seed,'metric':metric,
                        'T50':float(scores[h,i,s,0,k]),'S4':float(scores[h,i,s,1,k]),
                        'paired_difference':float(scores[h,i,s,0,k]-scores[h,i,s,1,k])})
        for k,metric in enumerate(METRICS):
            stratum_rows.append({'stratum':stratum,'metric':metric,'T50_mean':float(scores[h,:,:,0,k].mean()),
                'S4_mean':float(scores[h,:,:,1,k].mean()),'paired_difference':float(prompt_deltas[h,:,k].mean())})
    save_csv('paired_metrics.csv',paired)
    save_csv('prompt_means.csv',prompt_rows)
    save_csv('stratum_means.csv',stratum_rows)
    timing_rows=[];index=[]
    for row in sorted(generation,key=lambda r:(r['prompt_id'],r['seed'],r['arm'])):
        fields=['pair_id','prompt_id','stratum','seed','arm','worker_id','gpu_uuid','global_sequence',
            'unix_start','unix_end','denoise_seconds','vae_seconds','denoise_plus_vae_seconds',
            'scheduler_setup_and_initial_transfer_seconds','export_and_decode_check_seconds',
            'peak_allocated_gib','peak_reserved_gib','forward_count','initial_latent_sha256',
            'positive_context_sha256','final_latent_sha256']
        t={k:row[k] for k in fields};t['arm_order']='-'.join(row['order'])
        pair_record=json.loads((Path(row['result_path']).parents[1]/'pair_complete.json').read_text())
        t['shared_positive_text_encoding_seconds']=pair_record['shared_positive_text_encoding_seconds']
        timing_rows.append(t)
        source=Path(row['video_path'])
        dest=PROJECT/'Evaluation_Videos'/execution['batch_id']/row['arm']/(row['pair_id']+'.mp4')
        assert dest.resolve()==dest
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert file_hash(dest)==row['video']['sha256']
        else:
            temp=dest.with_name('.'+dest.name+'.copying')
            with temp.open('xb') as handle,source.open('rb') as origin:shutil.copyfileobj(origin,handle)
            assert file_hash(temp)==row['video']['sha256']
            assert not dest.exists();os.rename(temp,dest)
        index.append({'pair_id':row['pair_id'],'prompt_id':row['prompt_id'],'prompt':row['prompt'],
            'seed':row['seed'],'arm':row['arm'],'source':str(source),'archive':str(dest),
            'sha256':row['video']['sha256'],'bytes':row['video']['bytes'],
            'frames':81,'fps':16,'height':480,'width':832})
    assert Counter(row['arm'] for row in index)=={'T50':48,'S4':48}
    save_csv('timing_per_video.csv',timing_rows)
    save_json('video_index.json',index)
    save_csv('video_index.csv',index)
    cold=[]
    for r in worker_records:
        row={k:r[k] for k in ['worker_id','gpu_uuid','common_load_seconds','teacher_load_seconds','student_load_seconds','resident_allocated_gib']}
        row['T50_warmup_seconds']=r['warmup']['T50']['denoise_plus_vae_seconds']
        row['S4_warmup_seconds']=r['warmup']['S4']['denoise_plus_vae_seconds']
        cold.append(row)
    save_csv('load_and_warmup.csv',cold)
    by_gpu=[]
    for uuid in sorted({r['gpu_uuid'] for r in generation}):
        for arm in ['T50','S4']:
            rows=[r for r in generation if r['gpu_uuid']==uuid and r['arm']==arm]
            values=np.array([r['denoise_plus_vae_seconds'] for r in rows])
            by_gpu.append({'gpu_uuid':uuid,'arm':arm,'n':len(rows),'median_seconds':float(np.median(values)),
                'min_seconds':float(values.min()),'max_seconds':float(values.max())})
    save_csv('timing_by_gpu.csv',by_gpu)
    lines=['# S2 正式教师—学生资格实验报告','',
        '**96/96 条正式视频、576/576 个六指标分数、48/48 对数据完整；S3 未开始。**','',
        '## 资格结论','']
    if report['qualification_ready']:
        lines += ['按预设五项质量指标的 99% 配对区间判据，**本面板建立了目标维度上的教师优势证据**。通过指标按预设顺序为：'+', '.join(report['passing_targets_in_priority_order'])+'。建议提交研究者审查的首选目标为 **'+report['suggested_target']+'**。这不是全面质量更优或 Distillation++ 迁移有效性的结论。']
    else:
        lines += ['**本面板未建立可进入下一阶段的教师优势证据。**无优势证据不等于证明等效；若有技术异常，应先解决。未更换模型、指标、seed 或阈值来追求通过。']
    lines += ['', '## 六指标完整结果','',
        '分数为一次归一化后的量纲；差值始终为 T50 − S4。五质量维度采用 99% 区间，dynamic_degree 采用描述性 95% 区间。', '',
        '| 指标 | T50 均值 | S4 均值 | 配对差 | 区间 | 教师优势候选 |',
        '|---|---:|---:|---:|---|---|']
    for r in report['metric_summaries']:
        verdict='运动变化，不作质量判定' if r['metric']=='dynamic_degree' else ('是' if r['teacher_advantage_candidate'] else '否')
        lines.append('| %s | %.6f | %.6f | %+.6f | [%+.6f, %+.6f] | %s |' %
            (r['metric'],r['T50_mean'],r['S4_mean'],r['T50_minus_S4'],r['ci_low'],r['ci_high'],verdict))
    d=report['dynamic']
    lines += ['',f"动态视频：T50 {d['T50_dynamic_count']}/48，S4 {d['S4_dynamic_count']}/48；T 动/S 静 {d['T50_dynamic_S4_static']} 对，T 静/S 动 {d['T50_static_S4_dynamic']} 对，两组均动 {d['both_dynamic']} 对、均静 {d['both_static']} 对。运动比例升降不按越高越好解释，不设容许界，由研究者决定可接受程度。",'',
        '## 真实耗时与资源','',
        '| 组 | 去噪+VAE 中位数（秒） | Q1–Q3（秒） | 每视频分支预测数 |',
        '|---|---:|---|---:|']
    for arm in ['T50','S4']:
        r=report['timing']['per_arm'][arm]
        lines.append('| %s | %.3f | %.3f–%.3f | %d |'%(arm,r['median'],r['q1'],r['q3'],100 if arm=='T50' else 4))
    t=report['timing'];ci=t['speed_ratio_95ci']
    lines += ['', '教师/学生中位耗时比 **%.3f×**，配对 prompt 簇 bootstrap 95%% 区间 [%.3f, %.3f]；这是原生基线的热态实测，不是步数比例推算。'%(t['speed_ratio_medians'],ci[0],ci[1]),'',
        '每对在同一张 H200 上依冻结顺序执行，两模型、T5、VAE 共同驻留；每 worker 各组先完成一次独立完整预热。主计时包含去噪与 VAE，CUDA 同步，不含模型加载、文本编码、源噪声传输/采样器初始化和视频编码。文本条件两组复用；这些开销另表保留，不把主计时称为冷启动端到端延迟。峰值显存含两模型共同驻留，不能当作单模型最小部署需求。十秒一次 CPU 占用监控的微小开销未扣除。','',
        '## 方法与限制','',
        '四类各四个 prompt，每条三个固定 seeds；先在 prompt 内平均 seeds，再四类等权。分层有放回重采样 10,000 次，PCG64 seed=20261016、float64、线性分位数；同一 prompt 的 seeds 和两组始终一起抽取。五质量指标的逐项 99% 区间为 Bonferroni 名义校正，16 簇的小样本 percentile bootstrap 不保证精确覆盖率。', '',
        '本实验是冻结 16-prompt 面板的探索性资格筛查，不能外推所有提示词。Teacher 原生 FP32 参数/BF16 autocast/FP32 状态；Student 原生 BF16 参数和状态、更新以 FP32 计算再回写。两者采样器、CFG 与部署实现不同，差异不能全部归因于蒸馏训练。六指标是客观代理；overall_consistency 是文本语义一致性，不是综合质量总分。没有人工质量评分。','',
        '本阶段只有 T50 和 S4 基线，没有教师纠偏组；论文图像模型结论、Flow 代数推导和未来迁移假设均不构成本批实测迁移证据。所有质量下降、运动变化与收益均应并列审查，实际价值由研究者决定。','',
        '## 完整性、安全与产物','',
        '96 条视频均完成两种解码器的完整 81 帧、832×480、16 FPS 核查；共享源噪声/正向条件哈希逐对一致，96 个成片哈希唯一。评分有限、身份映射完整，技术异常列表：'+json.dumps(anomalies,ensure_ascii=False)+'。原值和归一化值均保留。', '',
        '失败记录共 %d 项，保留在原路径，未用替换 seed 或删除难样本补齐。仅有完整有效配对进入分析；相关条目见 statistics.json 的 failures。'%len(failures),'',
        '- `metrics_per_video.csv`：576 条逐视频指标，含原值、归一化值和成片哈希。',
        '- `paired_metrics.csv`、`prompt_means.csv`、`stratum_means.csv`：配对、prompt 和分层结果。',
        '- `statistics.json`、`metric_summary.csv`：区间、资格候选、动态转移及统计设置。',
        '- `timing_per_video.csv`、`timing_by_gpu.csv`、`load_and_warmup.csv`：热态、加载、预热、文本/导出等分项。',
        '- `video_index.json/csv`：原视频及按 T50/S4 分组归档路径、SHA-256。',
        '- 归档：`Evaluation_Videos/s2_20261016/T50/` 与 `S4/`，各 48 条，复制后核对哈希。','',
        '**停在 S2：等待研究者审查完整取舍并确认是否及如何进入 S3。**','']
    write_once(OUT/'REPORT.md','\n'.join(lines))
    print(json.dumps({'status':report['status'],'qualification_ready':report['qualification_ready'],
        'suggested_target':report['suggested_target'],'metrics':report['metric_summaries'],
        'timing':report['timing'],'archived':len(index)},ensure_ascii=False))


if __name__=='__main__':main()
