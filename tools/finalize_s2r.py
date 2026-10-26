"""Verify, archive and analyze S2-R only; outputs CSV/JSON/Markdown, no plots."""
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import shutil
import numpy as np
from s2r_common import PROJECT,BATCH,METRICS,load_frozen,complete_generations,file_hash
from s2r_statistics import analyze
from evaluate_s2r import validate_chunk

OUT=PROJECT/'Result/S2R_20261017'


def write_once(path,data):
    assert path.resolve()==path and path.is_relative_to(PROJECT)
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=data.encode('utf-8') if isinstance(data,str) else data
    if path.exists(): assert path.read_bytes()==blob,'Different existing final artifact: '+str(path)
    else:
        with path.open('xb') as f:f.write(blob)


def save_json(name,data):
    write_once(OUT/name,json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_csv(name,rows):
    assert rows
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows);write_once(OUT/name,stream.getvalue())


def main():
    execution,settings=load_frozen()
    protocol=json.loads((BATCH/'protocol.json').read_text())
    freeze=json.loads((BATCH/'analysis_freeze.json').read_text())
    for rel,digest in freeze['source_sha256'].items():assert file_hash(PROJECT/rel)==digest,rel
    generation=complete_generations(execution)
    gen={(r['pair_id'],r['arm']):r for r in generation}
    assert len(gen)==192 and len({r['video']['sha256'] for r in generation})==192
    assert len({r['initial_latent_sha256'] for r in generation})==3
    workers={};cold=[]
    for row in generation:
        attempt=Path(row['result_path']).parents[1].name
        key=(row['worker_id'],attempt)
        if key in workers:continue
        directory=BATCH/'workers'/('worker%02d'%row['worker_id'])/attempt
        paths=[p for p in [directory/'result.json',directory/'partial_result.json',directory/'failure.json'] if p.exists()]
        assert paths,'Missing worker provenance'
        r=json.loads(paths[0].read_text())
        assert r['admission']['pid']==row['pid'] and r['gpu_uuid']==row['gpu_uuid']
        assert r['teacher_load_audit']['keys']==r['student_load_audit']['keys']==1095
        assert r['teacher_load_audit']['missing']==r['teacher_load_audit']['unexpected']==0
        assert r['student_config']['infer_steps']==4 and r['student_config']['enable_cfg'] is False
        assert r['student_load_audit']['attention_backend']=='FlashAttn2Weight'
        assert all(r['warmup'][a]['status']=='passed' and r['warmup'][a]['warmup'] for a in ['T50','S4'])
        workers[key]=r
    # Include successful warmups of failed attempts too; do not hide discarded compute.
    for directory in sorted((BATCH/'workers').glob('*/attempt*')):
        candidates=[p for p in [directory/'result.json',directory/'partial_result.json',directory/'failure.json'] if p.exists()]
        if not candidates:continue
        r=json.loads(candidates[0].read_text())
        record={'worker_id':r['worker_id'],'attempt':r['attempt'],'status':r['status'],
            'gpu_uuid':r.get('gpu_uuid',''),'common_load_seconds':r.get('common_load_seconds'),
            'teacher_load_seconds':r.get('teacher_load_seconds'),'student_load_seconds':r.get('student_load_seconds'),
            'resident_allocated_gib':r.get('resident_allocated_gib')}
        for a in ['T50','S4']:
            record[a+'_warmup_seconds']=r.get('warmup',{}).get(a,{}).get('denoise_plus_vae_seconds')
        cold.append(record)
    scored=[];token_rows=[];eval_cost=[]
    for metric in METRICS:
        paths=list((BATCH/'evaluation'/metric).glob('attempt*/result.json'))
        assert len(paths)==1,metric
        r=json.loads(paths[0].read_text())
        assert r['status']=='passed' and len(r['rows'])==96 and not r['network_attempts']
        assert r['source_sha256']==freeze['source_sha256']['tools/evaluate_s2r.py']
        committed=[]
        for entry in r['chunks']:
            p=Path(entry['path']);assert p.is_relative_to(BATCH/'evaluation'/metric)
            chunk=json.loads(p.read_text());pid=chunk['rows'][0]['prompt_id']
            expected=sorted([g for g in generation if g['prompt_id']==pid and metric in g['primary_metrics']],key=lambda g:(g['seed'],g['arm']))
            validate_chunk(chunk,expected,metric,r['source_sha256'])
            committed.extend(chunk['rows'])
            eval_cost.append({'metric':metric,'prompt_id':pid,'videos':6,'scoring_seconds':chunk['scoring_seconds'],'chunk_path':str(p)})
            if metric=='overall_consistency':token_rows.append({'prompt_id':pid,**chunk['text_metadata']})
        sortkey=lambda x:(x['prompt_id'],x['seed'],x['arm'])
        assert sorted(committed,key=sortkey)==sorted(r['rows'],key=sortkey)
        scored.extend(r['rows'])
    assert len(scored)==576
    lookup={(r['pair_id'],r['arm'],r['metric']):r for r in scored};assert len(lookup)==576
    expected_keys={(g['pair_id'],g['arm'],m) for g in generation for m in g['primary_metrics']}
    assert set(lookup)==expected_keys,'Missing assigned metric or unexpected cross-domain score'
    ids=protocol['screen']['active_prompt_ids_by_domain'];seeds=protocol['screen']['seed_slots']
    arrays={m:np.empty((16,3,2),dtype=np.float64) for m in METRICS}
    timings={d:np.empty((16,3,2),dtype=np.float64) for d in ids}
    paired=[];timing_rows=[];paired_times=[];index=[]
    for domain,pids in ids.items():
        for i,pid in enumerate(pids):
            for s,seed in enumerate(seeds):
                pair_id=pid+'_seed'+str(seed)
                for a,arm in enumerate(['T50','S4']):
                    g=gen[pair_id,arm];v=g['video']
                    assert v['frames']==81 and v['size']==[480,832] and v['fps']==16
                    assert v['decord_all_frames_decoded'] and v['opencv_all_frames_decoded']
                    assert v['pixel_std']>0 and g['output_tensor']['std']>0 and not g['network_attempts']
                    assert g['domain']==domain
                    timings[domain][i,s,a]=g['denoise_plus_vae_seconds']
                    for metric in protocol['metric_domains'][domain]:
                        r=lookup[pair_id,arm,metric]
                        assert r['video_sha256']==v['sha256'] and r['video_path']==g['video_path']
                        arrays[metric][i,s,a]=r['normalized_score']
                for metric in protocol['metric_domains'][domain]:
                    t,st=arrays[metric][i,s]
                    paired.append({'prompt_id':pid,'domain':domain,'seed':seed,'metric':metric,
                        'T50':float(t),'S4':float(st),'T50_minus_S4':float(t-st)})
                paired_times.append({'prompt_id':pid,'domain':domain,'seed':seed,
                    'T50_seconds':float(timings[domain][i,s,0]),'S4_seconds':float(timings[domain][i,s,1]),
                    'ratio_T50_over_S4':float(timings[domain][i,s,0]/timings[domain][i,s,1])})
    report,prompt_rows=analyze(arrays,timings,ids,protocol['score_policy']['screen_viclip_truncated_ids'])
    anomalies=['Constant metric: '+m for m,x in arrays.items() if m!='dynamic_degree' and np.ptp(x)==0]
    for g in sorted(generation,key=lambda r:(r['prompt_id'],r['seed'],r['arm'])):
        fields=['pair_id','prompt_id','domain','seed','arm','worker_id','gpu_uuid','worker_sequence','unix_start','unix_end',
            'denoise_seconds','vae_seconds','denoise_plus_vae_seconds','scheduler_setup_and_initial_transfer_seconds',
            'export_and_decode_check_seconds','peak_allocated_gib','peak_reserved_gib','forward_count',
            'initial_latent_sha256','positive_context_sha256','final_latent_sha256']
        t={k:g[k] for k in fields};t['arm_order']='-'.join(g['order'])
        complete=json.loads((Path(g['result_path']).parents[1]/'pair_complete.json').read_text())
        t['shared_positive_text_encoding_seconds']=complete['shared_positive_text_encoding_seconds'];timing_rows.append(t)
        source=Path(g['video_path']);dest=PROJECT/'Evaluation_Videos'/execution['batch_id']/g['arm']/(g['pair_id']+'.mp4')
        assert source.resolve()==source and dest.resolve()==dest
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert file_hash(dest)==g['video']['sha256']
        else:
            temp=dest.with_name('.'+dest.name+'.copying')
            with source.open('rb') as inp,temp.open('xb') as out:shutil.copyfileobj(inp,out)
            assert file_hash(temp)==g['video']['sha256']
            assert not dest.exists();os.rename(temp,dest)
        assert file_hash(dest)==g['video']['sha256']
        index.append({'pair_id':g['pair_id'],'prompt_id':g['prompt_id'],'domain':g['domain'],'generation_prompt':g['prompt'],
            'eval_prompt':g['eval_prompt'],'seed':g['seed'],'arm':g['arm'],'source':str(source),'archive':str(dest),
            'sha256':g['video']['sha256'],'bytes':g['video']['bytes'],'frames':81,'fps':16,'height':480,'width':832})
    assert Counter(r['arm'] for r in index)=={'T50':96,'S4':96}
    failures=[]
    for p in sorted(BATCH.glob('**/failure.json')):
        r=json.loads(p.read_text())
        failures.append({'path':str(p.relative_to(PROJECT)),'error':r.get('error'),
            'worker_id':r.get('worker_id'),'metric':r.get('metric'),'attempt':r.get('attempt')})
    preflight=json.loads((BATCH/'preflight.json').read_text())
    original=json.loads((PROJECT/'Project_Progress/Stage_Progress/S1_PREFLIGHT.json').read_text())
    changed=[]
    for r in original['generation_files']:
        stat=Path(r['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):changed.append(r['path'])
    for rel,h in preflight['native_source_sha256'].items():assert file_hash(PROJECT/rel)==h
    if changed:anomalies.append('Shared generation files changed metadata during batch')
    report.update(status='S2_R_COMPLETE',batch=execution['batch_id'],numpy_version=np.__version__,
        qualification_ready=report['statistical_teacher_advantage'] and not anomalies,
        integrity={'accepted_pairs':96,'videos':192,'valid_assigned_metric_values':576,'metrics_per_video':3,
            'unique_video_hashes':192,'same_noise_context_gpu_within_pair':True,'full_decoders_passed':True,
            'frames_dimensions_fps_passed':True,'nonfinite_scores':0,'technical_anomalies':anomalies,
            'generation_metadata_changed':changed,'archived_videos':192},
        active_panel_sha256=protocol['screen']['active_panel_sha256'],input_sha256=execution['input_sha256'],
        analysis_freeze=freeze,failures=failures,
        limitations=['16 prompt clusters per metric, three fixed seeds; small-sample paired screening, not power guaranteed.',
            'Published prompt/metric mapping, reduced subset and 480P; not an official leaderboard reproduction.',
            'Dynamic degree measured only in motion domain and has no acceptance threshold.',
            'Native samplers, CFG and parameter/state precision differ; not an isolated causal distillation effect.',
            'VB0724 uses native ViCLIP 32-token truncation; predeclared nontruncated sensitivity does not replace primary result.',
            'No correction group or Distillation++ transfer result; S3 requires researcher review.'])
    by_gpu=[]
    for uuid in sorted({r['gpu_uuid'] for r in generation}):
        for arm in ['T50','S4']:
            x=np.array([r['denoise_plus_vae_seconds'] for r in generation if r['gpu_uuid']==uuid and r['arm']==arm])
            by_gpu.append({'gpu_uuid':uuid,'arm':arm,'n':len(x),'median_seconds':float(np.median(x)),
                'min_seconds':float(x.min()),'max_seconds':float(x.max())})
    save_json('statistics.json',report)
    save_json('generation_records.json',generation)
    save_json('video_index.json',index);save_csv('video_index.csv',index)
    save_json('semantic_token_metadata.json',sorted(token_rows,key=lambda r:r['prompt_id']))
    save_csv('metrics_per_video.csv',sorted(scored,key=lambda r:(r['prompt_id'],r['seed'],r['arm'],r['metric'])))
    save_csv('metric_summary.csv',report['metric_summaries']);save_csv('paired_metrics.csv',paired)
    save_csv('prompt_means.csv',prompt_rows);save_csv('timing_per_video.csv',timing_rows)
    save_csv('paired_timing.csv',paired_times);save_csv('timing_by_gpu.csv',by_gpu)
    save_csv('load_and_warmup.csv',cold);save_csv('evaluation_timing.csv',eval_cost)
    save_json('failures.json',failures)
    create_report(report)
    print(json.dumps({'status':report['status'],'qualification_ready':report['qualification_ready'],
        'passing_targets':report['passing_targets_in_priority_order'],'metric_summaries':report['metric_summaries'],
        'timing':report['timing_by_domain'],'archived':192},ensure_ascii=False))


def create_report(r):
    lines=['# S2-R 官方依据教师—学生资格验证报告','',
        '**192/192 条正式视频、576/576 个对应指标值、96/96 对输入完整；S3 未开始。**','',
        '## 资格结论','']
    if r['qualification_ready']:
        lines+=['本面板按预设五质量指标的逐项 99% 配对区间建立了候选教师优势。通过指标（固定优先级顺序）：'+
            '、'.join(r['passing_targets_in_priority_order'])+'；建议提交研究者审查的目标：**'+r['suggested_target']+'**。实际研究价值及其他维度取舍由研究者确认。这不证明全面质量优越或 Distillation++ 已迁移有效。']
    else:
        lines+=['**本面板未建立可进入下一阶段的教师优势证据。**区间跨零不证明等效；若有技术异常须先解决。没有依据分数更换文本、seed、权重、指标或阈值，也没有进入 S3。']
    lines+=['','## 六指标对应库结果','',
        '每库 16 条 prompt，每条三个固定 seeds。每视频只评所属库三项；每指标 48 个配对、16 个 prompt 簇。下表为一次归一化后的分数，差始终为 T50−S4。质量区间 99%，dynamic_degree 区间 95%。','',
        '| 指标 | 库 | T50 | S4 | 配对差 | 区间 | 教师优势候选 |','|---|---|---:|---:|---:|---|---|']
    for x in r['metric_summaries']:
        decision='运动护栏，不作质量判定' if x['metric']=='dynamic_degree' else ('是' if x['teacher_advantage_candidate'] else '否')
        lines.append('| %s | %s | %.6f | %.6f | %+.6f | [%+.6f, %+.6f] | %s |'%(x['metric'],x['domain'],x['T50_mean'],x['S4_mean'],x['T50_minus_S4'],x['ci_low'],x['ci_high'],decision))
    d=r['dynamic']
    lines+=['',f"主体与运动库：T50 动态 {d['T50_dynamic_count']}/48，S4 {d['S4_dynamic_count']}/48；T 动/S 静 {d['T50_dynamic_S4_static']} 对，T 静/S 动 {d['T50_static_S4_dynamic']} 对，两者均动 {d['both_dynamic']} 对、均静 {d['both_static']} 对。不把动态程度当作越高越好，不设可接受降幅；本结果不覆盖成像与语义库的运动情况。",'',
        '## 预设语义截断敏感性','']
    s=r['semantic_nontruncated_sensitivity']
    lines+=['完整主统计保留 VB0724，评分沿用官方原生 32-token 截断；原始文本、实际有效 token 和哈希见 semantic_token_metadata.json。',
        '预设排除该截断条目的敏感性子集为 15 prompts、45 对：T50 %.6f，S4 %.6f，差 %+.6f，描述性 99%% 区间 [%+.6f, %+.6f]。该子集不替代主统计，不用于另行挑选资格结论。'%(s['T50_mean'],s['S4_mean'],s['T50_minus_S4'],s['ci_low'],s['ci_high']), '',
        '## 真实耗时','',
        '| 集合 | T50 去噪+VAE 中位数/秒 | S4 中位数/秒 | 中位耗时比 T50/S4 | 95% 簇区间 |',
        '|---|---:|---:|---:|---|']
    for domain,t in r['timing_by_domain'].items():
        ci=t['speed_ratio_95ci']
        lines.append('| %s | %.3f | %.3f | %.3f× | [%.3f, %.3f] |'%(domain,t['per_arm']['T50']['median'],t['per_arm']['S4']['median'],t['speed_ratio_medians'],ci[0],ci[1]))
    lines+=['','同对 T50/S4 在同一张空闲 H200 上运行，两模型、T5、VAE 共同驻留；每 worker 使用同一冻结面板输入作完整预热，重复预热不计正式样本。组别顺序在 worker/库内均衡。主计时为 CUDA 同步的热态去噪+VAE；加载、文本编码、源噪声传输/采样器初始化、导出与预热另表记录，不称冷启动端到端延迟。峰值显存含共同驻留。Teacher 每视频 100 次分支预测，Student 4 次；比值来自实测，不按 50/4 推算。整体耗时汇总在两个库中等权，不能代替各指标对应库的质量结论。','',
        '## 统计方法与证据边界','',
        '先对同 prompt 的三个 seed 配对差取均值，再在对应库的 16 条 prompt 中等权平均。库内按 prompt 有放回抽样 10,000 次，PCG64 seed=20261017、float64、线性分位数；同一 prompt 的全部 seeds、两组随簇保留。五质量比较族用逐项 99% 双侧区间作 Bonferroni 名义校正；小样本 percentile bootstrap 不保证精确覆盖率，不将 48 个 seed 配对当作 48 个独立 prompts。','',
        '使用固定 VBench 原文对应公开 Wan 扩写，生成读扩写、语义评分读原文；两组共享实际源 latent 与正向文本条件。协议是官方 prompt/指标对应关系下的缩小配对筛查，480P、每库 16 条和三个 seeds 不等于官方完整 720P/五 seed 榜单复现。旧 S2 不混池；本轮与旧轮同时改变了面板及生成文本，不能把跨轮差异单独归因于 prompt 增强。','',
        'Teacher 保持原生 FP32 参数、BF16 autocast 与 FP32 状态；Student 保持 BF16 参数及持久状态。不同原生采样器、CFG 和部署实现属于系统比较的一部分，不能全部归因于蒸馏训练。六指标均为客观代理，overall_consistency 是语义相似度而非综合质量；不含人工评分。','',
        '本阶段仅有 T50/S4，没有纠偏组；论文结论、Flow 代数推导和后续迁移假设不能写成本批迁移有效性的实测证据。若资格通过且研究者批准，下一阶段沿用同一冻结 prompt/seed/指标配置；不自动扩样或开始 S3。','',
        '## 完整性、失败与数据入口','',
        '192 条视频全部通过双解码器的 81 帧、832×480、16 FPS 检查；每对实际源噪声、正向条件及 GPU 哈希/身份相同，成片 SHA-256 均唯一。每条仅对应三项评分，无非有限数；技术异常：'+json.dumps(r['integrity']['technical_anomalies'],ensure_ascii=False)+'。','',
        '本批失败/保护退出记录共 %d 项，见 failures.json；保留原路径与错误，不更换 seed 或删除难样本。已安全提交的完整配对/评分块按身份校验复用，未完成项按相同配置重试。未完失败尝试的全部计算成本无法由成功输出耗时完整还原，不能将成功样本总时间称为全部作业成本。'%len(r['failures']),'',
        '- metrics_per_video.csv：576 行对应指标，含原值、归一化值、路径和成片哈希。',
        '- paired_metrics.csv：288 行逐 seed 配对差；prompt_means.csv：96 行逐 prompt/指标均值。',
        '- metric_summary.csv、statistics.json：全部区间、候选目标、运动转移、敏感性及统计设置。',
        '- timing_per_video.csv、paired_timing.csv、timing_by_gpu.csv：逐视频/逐对/GPU 热态耗时及分项。',
        '- load_and_warmup.csv、evaluation_timing.csv：包含失败尝试已记录部分的加载预热、有效评分块耗时。',
        '- generation_records.json、semantic_token_metadata.json、failures.json：可追溯运行与评分元数据。',
        '- video_index.json/csv：原始视频及 Evaluation_Videos/s2r_20261017/T50、S4 的归档路径，各 96 条，复制后核对 SHA-256。','',
        '**本轮停在 S2-R，等待研究者审查，不自动进入 S3。**','']
    write_once(OUT/'REPORT.md','\n'.join(lines))


if __name__=='__main__':main()
