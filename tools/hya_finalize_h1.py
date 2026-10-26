"""Audit and archive only completed H1 smoke acceptance, never run H2 or H3."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
from hya_common import ROOT,RUN,METRICS,PROMPT,digest,write_json,gpu_snapshot


def latest_passed(directory):
    candidates=sorted(Path(directory).glob('attempt*/result.json'))
    assert candidates, 'No completed result: '+str(directory)
    path=candidates[-1];data=json.loads(path.read_text());assert data['status']=='passed'
    assert not data.get('network_attempts')
    return path,data


def main():
    assert ROOT.resolve()==ROOT
    assert json.loads((RUN/'generation_controller_done.json').read_text())['status']=='passed'
    out=ROOT/'Result/HYA_H1_20261019';assert out.resolve()==out and not out.exists()
    generation={};metrics={};video_paths={}
    for arm,n in [('T50',50),('S5',5)]:
        path,data=latest_passed(RUN/'generation'/arm)
        assert data['formal_path_dit_calls']==n and len(data['steps'])==n and data['warmup_repeat_exact']
        assert data['prompt']==PROMPT and data['admission']['physical_gpu']==6
        assert data['video']['frames_decord']==data['video']['frames_opencv']==129
        video=path.parent/'H1_smoke.mp4';assert digest(video)==data['video']['sha256']
        generation[arm]=data;video_paths[arm]=video
    assert generation['T50']['actual_initial_latent_sha256']==generation['S5']['actual_initial_latent_sha256']
    assert generation['T50']['inputs']['hashes']==generation['S5']['inputs']['hashes']
    assert len({x['video']['sha256'] for x in generation.values()})==2
    assert len({x['final_latent_sha256'] for x in generation.values()})==2
    assert generation['T50']['weight_mapping']['all_tensors_bitwise_roundtrip']
    rows=[]
    for metric in METRICS:
        path,data=latest_passed(RUN/'evaluation'/metric)
        assert data['admission']['physical_gpu']==6 and len(data['videos'])==2
        assert {x['video_path'] for x in data['videos']}=={str(p) for p in video_paths.values()}
        metrics[metric]=data
        for arm,video in video_paths.items():
            value=float(next(x for x in data['videos'] if x['video_path']==str(video))['video_results'])
            assert math.isfinite(value) and 0<=value<=1
            rows.append({'arm':arm,'metric':metric,'score':value,'video_sha256':generation[arm]['video']['sha256'],'scope':'H1 smoke only'})
    before=json.loads((RUN/'dependencies/before.json').read_text());after=json.loads((RUN/'dependencies/after_all.json').read_text())
    assert all(after.get(k)==v for k,v in before.items())
    additions={k:v for k,v in after.items() if k not in before}
    out.mkdir(parents=True,exist_ok=False)
    archives={}
    for arm,source in video_paths.items():
        archive=ROOT/'Evaluation_Videos/hya_h1_20261019'/arm/'H1_smoke.mp4'
        assert archive.resolve()==archive
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():assert digest(archive)==generation[arm]['video']['sha256']
        else:
            with source.open('rb') as src,archive.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        assert digest(archive)==generation[arm]['video']['sha256']
        archives[arm]={'path':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size}
    with (out/'smoke_scores.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'status':'passed','stage':'H1','scope':'implementation acceptance only, not teacher qualification',
             'resource_verification':json.loads((RUN/'resource_verification.json').read_text()),
             'evaluation_cpu_preflight':json.loads((RUN/'eval_cpu_preflight/attempt02/result.json').read_text()),
             'generation':generation,'evaluation':metrics,'archives':archives,
             'environment_additions':additions,'existing_eval_packages_unchanged':len(before),
             'pip_check':{env:json.loads((RUN/(env+'_pip_check.json')).read_text()) for env in ['changzihe','changzihe_eval']},
             'formal_panel_token_contract':json.loads((RUN/'formal_panel_token_contract.json').read_text()),
             'gpu6_at_finalization':gpu_snapshot(),'h2_started':False,'h3_started':False}
    write_json(out/'H1_ACCEPTANCE.json',summary)
    runtime={'status':'H1_verified_H2_not_started','parent_protocol_sha256':generation['T50']['protocol_sha256'],
             'model_implementation':'unmodified AccVideo Hunyuan native DiT and FlowMatchDiscreteScheduler',
             'native_source_commit':'a2752d2f6990a1363713f1d57907bd28e3ad365c',
             'teacher_loader':'existing Diffusers shards reconstructed in CPU memory using inverse of official converter; 1264 source tensors verified by complete round trip',
             'student_loader':'original author 856-tensor PT; strict load; no converted checkpoint saved',
             'video':{'height':720,'width':1280,'frames':129,'fps':24,'batch_size':1},
             'precision':generation['T50']['precision'],'source_noise_dtype':'float32',
             'first_actual_latent_dtype':generation['T50']['actual_initial_latent_dtype'],
             'llama':{'precision':'fp16','max_tokens_including_template':351,'crop_start':95,'hidden_state_skip_layer':2,'apply_final_norm':False},
             'clip_text':{'precision':'fp16','max_tokens':77},'attention':generation['T50']['attention'],
             'embedded_guidance_scale':6.0,'external_cfg_scale':1.0,'flow_shift':7.0,
             'allow_tf32':False,'cudnn_benchmark':False,
             'schedule':generation['T50']['scheduler_contract'],'vae_tiling':generation['T50']['vae_tiling'],
             'video_encoding':generation['T50']['encoding'],'allowed_gpu_indices':[6],
             'formal_prompt_seed_metric_panel':'unchanged HUNYUAN_ACCVIDEO_PROTOCOL.json; H1 uses a separate single prompt and seed',
             'offline':True,'formal_execution_requires_next_stage_authorization':True}
    write_json(ROOT/'Project_Plan/HYA_RUNTIME_CONTRACT.json',runtime)
    timing=['| 组别 | 去噪 / 秒 | VAE 解码 / 秒 | 两项合计 / 秒 | 额外预热 / 秒 | 去噪峰值显存 / GiB |','|---|---:|---:|---:|---:|---:|']
    for arm,d in generation.items():
        timing.append(f"| {arm} | {d['denoising_seconds']:.3f} | {d['vae_decode_seconds']:.3f} | {d['hot_denoise_decode_seconds']:.3f} | {d['warmup_seconds']:.3f} | {d['denoise_peak_allocated_gib']:.3f} |")
    scores=['| 指标 | T50 | S5 |','|---|---:|---:|']
    for metric in METRICS:
        values={x['arm']:x['score'] for x in rows if x['metric']==metric}
        scores.append(f"| {metric} | {values['T50']:.6f} | {values['S5']:.6f} |")
    projected_hours=192*sum(x['hot_denoise_decode_seconds'] for x in generation.values())/3600
    text='''# HunyuanVideo → AccVideo 720p：H1 实现与验收

**H1 通过；正式 S2-HYA/H2 教师资格实验与 S3 均未启动。**本报告仅证明当前资源、原生调用、生成与五指标链路可运行；两条独立冒烟视频不能建立教师优势或迁移效果。

## 资源与环境

40 个清单文件完整哈希校验全部通过，缺失数为 0。复用已有 HunyuanVideo、文本编码器、tokenizer、VAE、DINO、RAFT；用户新增的 AccVideo 和评测资源身份匹配。Codex 本轮没有下载任何模型或评测权重，也没有保存第二份转换权重。

在本人 changzihe_eval 增加 21 个锁定依赖及固定源码编译的 Detectron2，原有 110 个包版本全部保持不变；生成环境没有安装或降级软件。Detectron2 使用 CUDA 12.8、SM90 和最多 4 个 CPU 构建 worker。完整版本、安装记录及编译 wheel 哈希见 `runs/hya_h1_20261019/dependencies/`。

两个环境的 `pip check` 仍只返回既有 `decord 0.6.0 is not supported on this platform` 标签告警（退出码 1），不是新增依赖缺失。该告警保留；两条完整视频都已通过 Decord 和 OpenCV 的实际 129 帧解码验收。不能将 pip check 描述成退出 0。

## 原生调用与数值契约

两组使用同一固定版本的原生双向 Hunyuan DiT、FlashAttention 2 和 Flow Euler 调度器。已有教师 Diffusers 权重仅在内存重组，1264 个源张量完成转换回环相等检查；学生 856 项原生参数张量直接严格加载。教师运行文件身份对应固定社区转换版；没有额外下载腾讯原生 PT 做另一份权重比较。

H1 查明 Diffusers 默认调度器的 BF16 状态回写与原生实现不同，因此实际使用未修改的原生调度器。严格保留 native defaults：DiT BF16，两个文本编码器 FP16，给定共同 CPU FP32 源噪声先转 FP16，之后每步保留 FP32 latent；VAE FP16 autocast。嵌入 guidance=6、外部 CFG=1、shift=7，T50/S5 的采样前向数分别为 50/5，各另有一次单独计费的预热前向；各组自身的预热首步与该组冒烟首步输出完全一致；教师与学生的预测并不相同。

两组实际起始 latent、文本条件和 token 哈希一致；所有步输出有限。50/5 步时间数组和 FP32 Euler 更新通过独立代数检查。完整 64 条正式 prompt 仅做了 CPU tokenizer 检查：原生模板最长 123 tokens（上限 351），CLIP 最长 28（上限 77），未发生截断，没有生成正式面板视频。

已验收的运行配置见 `Project_Plan/HYA_RUNTIME_CONTRACT.json`，与原冻结 prompt/seed/指标面板配合使用；不得重新采用 Diffusers 默认 latent 回写精度。H1 的实现适配没有引入 Distillation++、改动模型参数或评测公式。

## 冒烟生成、计时与客观评分

使用正式面板之外的一条狗与猫在花园中行走的 prompt，seed=2026101900；两组均为 1280×720、129 帧、24 FPS。成片无非有限值，双解码器帧数、尺寸、FPS 与内容非恒定检查通过。

'''+ '\n'.join(timing)+'''

计时是本次单对冒烟的 GPU 同步去噪与解码实测；加载、共享文本编码、GPU 占用查询、输出校验及编码写盘另存原始记录。它不是正式实验的稳定性能统计，也不等于包含加载的部署延迟。

'''+f'按当前固定面板各 192 条、GPU 6 串行，以这对冒烟耗时线性估算，H2 **仅去噪与解码约 {projected_hours:.1f} 小时（{projected_hours/24:.2f} 天）**，尚未计入模型加载、评测、资源等待和重试。此为单样本工期预算，不是精确承诺；本轮不自行缩减样本、降低分辨率或改用其他 GPU。'+'''

'''+ '\n'.join(scores)+'''

上表为 10 个链路验收分数，无置信区间、无教师资格判定。正式阶段仍按三个官方任务库各自对应的指标评测。动态程度只作护栏，不设容许下降阈值。

Tag2Text/GRiT 均离线加载成功。BERT 路径在本人适配层指向已有本地资源；Tag2Text 的非学习位置缓冲由原生实现生成。GRiT 报出的 106 个缺失键均逐项证明是已加载参数对象的第二个共享别名，没有跳过真正缺失的学习参数。所有生成/评测的网络访问尝试均为 0；评测保留官方预处理、阈值和 16 帧采样，保存了场景描述与对象检测标签。

## 归档与阶段边界

视频已复制并核对 SHA-256：

- `Evaluation_Videos/hya_h1_20261019/T50/H1_smoke.mp4`
- `Evaluation_Videos/hya_h1_20261019/S5/H1_smoke.mp4`

原始运行记录：`runs/hya_h1_20261019/`；本目录 `H1_ACCEPTANCE.json`、`smoke_scores.csv` 保存验收和原始分数。仅同步小型代码、配置、日志摘要与核心结果至本地，视频、latent、模型与环境保留远端。

H1 只在 GPU 6 串行执行，没有尝试其他卡。早期依赖/CPU 检查中的源码文档链接、工具模块导入及 GRiT 共享别名检查问题已解决并保留失败记录；没有用失败样本替代成功结果。下一步须用户授权后才启动 S2-HYA/H2 的 384 视频教师资格实验；只有目标维度教师优势成立且研究者确认，才进入 S3。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps({'status':'H1_passed','result_directory':str(out),'archives':archives,'metric_values':len(rows),'h2_started':False,'h3_started':False}),flush=True)

if __name__=='__main__':
    main()
