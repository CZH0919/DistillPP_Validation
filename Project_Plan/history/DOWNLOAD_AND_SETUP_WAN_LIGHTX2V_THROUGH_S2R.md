# 资源准备纠正与剩余步骤

> 当前状态：用户已完成清理、评测资源下载及环境安装；S1 已完成真实验收。无需再次下载或安装。下文保留为复现/故障恢复说明，最新结果见 `Project_Progress/Stage_Progress/S1_IMPLEMENTATION_ACCEPTANCE.md`。

适用终端：VS Code Remote SSH 的远端 Bash。项目：`/data/chenjiayu/changzihe/DistillPP_Validation`。

**本文件替代此前的下载步骤。生成模型不再下载；只读复用已有教师、学生、T5、VAE 与 tokenizer。不要恢复旧原生教师分片的下载，也不要另下学生 safetensors。**

## 1. 已有资源与证据

| 角色 | 直接读取的已有路径（相对 `/data/chenjiayu/models/`） | 使用方式 |
|---|---|---|
| Teacher | `Wan2.1-T2V-14B-Diffusers/transformer/` | 官方原始教师；加载时按官方规则映射到原生 Wan 参数名，不生成另一份权重文件 |
| Student | `Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt` | 固定 LightX2V 明确支持原版 PT；设置现有目录为 model_path，不设置 dit_original_ckpt / transformer_model_path 覆盖 |
| 公共组件 | 同一 `StepDistill-CfgDistill` 目录下的 T5、VAE、`google/umt5-xxl/` | T/S 使用相同原生组件；直接读取，不复制、不在共享目录新建链接 |

`RESOURCE_MANIFEST.json` v2 固定仓库、revision、24 项文件及哈希。**本轮 24 项完整文件校验已全部通过**，报告为 `Project_Progress/Stage_Progress/resource_check_20261016_091241_1370827.json`；用户无需重复检查。模型组强制 **reuse-only**：即使不加 `--verify-only`，也不能下载、创建、修复或覆盖任何模型文件；缺项或校验不符就停止。

已完成结构核查：12 个教师分片、原生模型和 PT 学生的 1,095 个参数一一对应且形状全部匹配；教师 F32、学生 BF16。与误下载原生教师第一分片的 189 个张量做了共 24,192 bytes/份的抽查，全部一致。抽查不等于全部张量字节等价，meta 核查也不等于真实模型加载。完整文件哈希结果另见 `Project_Progress/Stage_Progress/resource_check_*.json`。

S1 需要实现并验收**加载层**映射，保留原生 Wan 前向、Flow UniPC、负向条件与计时协议；不为解决文件封装问题悄悄换采样实现。现有官方学生 PT 作为本次冻结版本，不声称已证明它与另一仓库 safetensors 的全部张量相同。

## 2. 处理这次误下载

核查确认新目录 `/data/chenjiayu/models/Wan2.1-T2V-14B` 只有六个本轮新增对象：LICENSE、README、config、VAE 软链接、第一完整分片、第二分片的 `.changzihe.part`。合计占用 **12.386 GiB**。VAE 原件没有复制或覆盖；另一个新学生目录尚未创建。

先预览，再执行清理：

```bash
source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate changzihe
cd /data/chenjiayu/changzihe/DistillPP_Validation
python Project_Plan/cleanup_duplicate_download.py
python Project_Plan/cleanup_duplicate_download.py --apply
```

清理脚本仅接受核查记录中的六个对象，并检查真实路径、目录/文件 inode、大小、修改时间、属主、链接目标、下载锁和文件占用。出现额外文件或状态变化会停止。它逐项 unlink，最后只删除空目录；不使用递归删除。VAE 项只解除新软链接，保留已有原件。清理结果保存到阶段记录。Codex 仅运行预览，尚未执行 `--apply`。

## 3. 只下载仍缺的评测资源

本轮在共享模型目录及项目评测目录内做了限深核查，未发现以下八个目标文件；不将此结论外推为已经搜索其他人的工作目录。最大新增下载量 **3,129,674,761 bytes，约 2.915 GiB**。

| 资源 | 用途 | 目录（相对 `resources/evaluation/`） |
|---|---|---|
| DINO ViT-B/16 | subject_consistency | `dino_model/` |
| MUSIQ-SPAQ | imaging_quality | `pyiqa_model/` |
| AMT-S | motion_smoothness | `amt_model/` |
| RAFT-Things | dynamic_degree | `raft_model/models/` |
| OpenAI CLIP ViT-L/14，224px | aesthetic_quality | `clip_model/` |
| LAION V1 linear head | aesthetic_quality | `aesthetic_model/emb_reader/` |
| VBench 打包的 ViCLIP InternVid-10M-FLT | overall_consistency | `ViCLIP/` |
| CLIP BPE 词表 | ViCLIP/CLIP 分词 | `ViCLIP/` |

```bash
python Project_Plan/download_resources.py --group evaluation --plan-only
python Project_Plan/download_resources.py --group evaluation
```

默认 HF-Mirror；固定原始仓库、revision、大小和哈希，镜像只作传输来源。已有目标先校验并跳过，不重复下载；校验失败不覆盖。中断保留 `.changzihe.part`，重跑**评测组**命令可续传。每次网络请求有有限超时；脚本不配置代理、DNS、token 或全局镜像。

若镜像确实报错，可显式换官方端点，仍使用相同清单与断点文件：

```bash
python Project_Plan/download_resources.py --group evaluation --endpoint https://huggingface.co
```

`HASH_START/HASH_PROGRESS` 是本地文件校验，不是下载卡住。成功下载后已做完整哈希，无需立即重复校验。

## 4. 只补评测软件

生成环境 `changzihe` 已安装成功，**不要重跑 generation 安装，也不要为清理重复权重而卸载已安装依赖**。实际核心版本为 Torch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 4.51.3、Diffusers 0.35.1、LightX2V 0.5.0；安装曾把 protobuf 从 5.29.5 调整为 6.33.6，已记录到环境报告，不能称环境完全没有变化。

`changzihe_eval` 尚未创建。固定 PyIQA 0.1.13 要求 Transformers 4.37.2，因此在同一环境根目录建立本人的独立评测环境，避免降级通用生成环境。所需小型源码包已在项目内，无需远端再克隆 GitHub。

先完成八项资源下载，再执行：

```bash
bash Project_Plan/prepare_software.sh evaluation
```

该命令仅维护 `/data/chenjiayu/miniconda3/envs/changzihe_eval`，使用 TUNA 软件源，安装软件包及 CUDA 运行库；**软件包下载量不包含在前述 2.915 GiB 权重/词表预算中**。按需预留约 30 GiB 供评测环境、软件缓存和临时文件；实际以安装报告和实时磁盘为准。不会重建 `changzihe`、修改 base/其他人环境或启动 GPU。CLIP/VBench/DINO 源码固定版本，运行入口后续显式设置对应 PYTHONPATH。

已知 decord 0.6.0 wheel 的内部标签不一致，`pip check` 会报 `decord 0.6.0 is not supported on this platform`。脚本仅允许这一条已记录告警；其他错误返回非零。不能把它写成 pip check 全绿；视频解码与六指标完整运行仍需 S1 验收。

## 5. 交回信息和停止点

保留 `CLEANUP_COMPLETE`、评测资源报告路径、`SOFTWARE_EXIT_CODE` 和软件日志路径。准备成功后继续 S1 接口/加载/计时/六指标验收；本文件没有提供推理、评分或训练启动命令，未建立任何教师优势证据。

如需日后重新完整检查已有模型，可手动执行下式（约读取 90.918 GiB，本地磁盘 I/O，无网络下载）；已有成功报告且文件状态未变时，不必反复重读：

```bash
python Project_Plan/download_resources.py --group models --verify-only
```

小文件源码依据：[LightX2V 原版 PT 加载分支](https://github.com/ModelTC/LightX2V/blob/cc9429192971815afff8368f74bd10adcc6fed48/lightx2v/models/networks/wan/distill_model.py)、[Diffusers v0.35.1 官方 Wan 转换规则](https://github.com/huggingface/diffusers/blob/v0.35.1/scripts/convert_wan_to_diffusers.py)。所有资源精确来源见 v2 清单。
