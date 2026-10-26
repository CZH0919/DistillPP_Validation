# 纠偏实验：进度查看与执行入口

## 已完成：第二/第四步各三个λ

**2026-10-25 09:30:37已完成并复核：288/288视频、480/480分数，全部记录进程已退出。**结果见 `../Result/HYA_H3_STEP24_V1/REVIEW_AND_INTERPRETATION.md`；以下命令可读取完成状态，无需持续刷新或重复启动。

本批于2026-10-25 00:42:45（北京时间）启动后台队列。新增六组各48视频、80分数，合计288视频、480分数；并发上限4，GPU卡号逐次按实际空闲情况选择。首批使用GPU 1–4；该快照不构成固定卡号要求。科学计划已授权执行、并发字段为空，实际运行配置冻结为4。

在VS Code远端Bash中查看状态（解释器使用绝对路径，无需切换当前Conda环境）：

```bash
cd /data/chenjiayu/changzihe/DistillPP_Validation
/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h3_step24_common.py --status
```

每30秒刷新一次：

```bash
watch -n 30 '/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B /data/chenjiayu/changzihe/DistillPP_Validation/tools/hya_h3_step24_common.py --status'
```

`Ctrl+C`只退出查看，不停止后台实验。主要字段：`new_videos`/288、`videos_by_group`（各48）、`new_scores`/480、`active`（任务/GPU/PID/当前阶段）、`errors`。视频全部生成后才开始评分，因此生成阶段评分数为0是正常的。模型首次加载会按既有锁串行进入CPU权重重建，四个worker不必同时开始GPU计算。

查看调度日志和GPU情况：

```bash
tail -n 40 /data/chenjiayu/changzihe/DistillPP_Validation/runs/hya_h3_step24_v1/controller.log
nvidia-smi
```

`running`表示任务运行中；`waiting_for_idle_resources`表示等待空闲卡/内存；`completed`表示生成、评测和合并整理全部完成；`needs_attention`、`controller_failed`或`finalization_failed`需查看本批日志。控制器和worker均使用独立session、`/dev/null`输入与文件日志，关闭VS Code/SSH/Codex不影响后台任务，不需要Codex在线或定时检测。遇占用冲突仅退出自身worker，保留完成结果，冷却并优先选择冲突较少的空闲卡。

本批队列已完成并停止，不要重复运行`--prepare`或`--launch`，也不要修改冻结的本批协议与脚本。

核心算法入口为 `tools/hya_h3_step24_worker.py`，计时与评测口径沿用旧批次；新结果以完整库均值和固定λ的位置对比为主，分层仅保留补充数据。新视频将分别归档至 `Evaluation_Videos/hya_h3_step24_v1/step2_lambda_0p1/0p3/0p5` 和 `step4_lambda_0p1/0p3/0p5` 对应六个目录；合并报告进入 `Result/HYA_H3_STEP24_V1/`。

## 已完成：首步 0.5 + 第三步 0.1/0.3/0.5

**本批于 2026-10-21 13:51:55（北京时间）完成，2026-10-22 已复核。192 条视频、320 个新分数齐全，全部记录进程已退出。**四组各 48 条视频、80 个分数；旧基线和首步两组直接复用。并发上限 4，卡号在每次派发时按实际空闲情况选择；资源不足时降低并发或等待，冲突时只退出自身任务、保留已提交的组并暂避冲突卡。

在 VS Code 的远端 Bash 查看实时状态：

```bash
cd /data/chenjiayu/changzihe/DistillPP_Validation
/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h3_timing_common.py --status
```

本批协议、代码和原始结果保持冻结；不要重复启动或修改冻结配置/脚本。以下后台机制是本批实际运行方式：控制器与 worker 使用独立 session、文件日志及 `/dev/null` 标准输入，关闭 VS Code/SSH/Codex 不影响后台计算；无需 Codex 持续在线。`--launch` 在既有控制器或 worker 仍存活时拒绝重复启动；恢复时保留已提交的视频和评分。

状态字段：`new_videos`/192、`videos_by_group`（每组 48）、`new_scores`/320、`active`（GPU/PID/阶段）、`errors`。评分在视频全部生成后开始；模型初次加载期间部分卡可能暂时利用率为零。`running` 为运行中，`waiting_for_idle_resources` 为等待空闲资源，`completed` 为生成/评分/分析全部完成；`needs_attention`、`controller_failed` 或 `finalization_failed` 需要查看本批日志。

每 30 秒刷新一次（Ctrl+C 只退出查看，不停止实验）：

```bash
watch -n 30 '/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h3_timing_common.py --status'
```

查看后台调度日志：

```bash
tail -n 40 runs/hya_h3_timing_v1/controller.log
```

- 新视频：`Evaluation_Videos/hya_h3_timing_v1/step1_lambda_0p5/`、`step3_lambda_0p1/`、`step3_lambda_0p3/`、`step3_lambda_0p5/`。
- 复核与分析：`Result/HYA_H3_TIMING_V1/REVIEW_AND_INTERPRETATION.md`、`REVIEW.json`；原始 `REPORT.md` 和各 CSV 保留。第三步 λ=0.3 满足本面板双目标均值改善；第三步 λ=0.1 的一条首预测逐位差异及敏感性分析单独披露。
- 核心代码：`tools/hya_h3_timing_worker.py` 的 `generate_one`，复用 `tools/hya_h3_math.py` 的原公式。
- CPU 检查：`CUDA_VISIBLE_DEVICES="" /data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h3_timing_selftest.py`；资源/复用身份只读检查：同解释器执行 `tools/hya_h3_timing_common.py --check`。

## 已完成首步两组：历史入口

**本轮已完成并复核：λ=0.1、λ=0.3 各 48 条视频，合计 96 条纠偏视频、160 个新分数。**所有已记录进程已退出。以下命令可读取完成状态，无需持续刷新。已完成的 λ=0.1 视频复用 1 条，本轮实际新增生成 95 条。仅使用确认空闲的 GPU 5–7，并发上限 3。

在 VS Code 远端 Bash 执行：

```bash
cd /data/chenjiayu/changzihe/DistillPP_Validation
/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h3_common.py --status
```

输出包括总视频数、两组各自数量（各 48）、分数数量（总计 160）、活跃任务的 GPU/PID、当前采样步骤与异常。视频计数包含复用的 1 条；生成未齐时评分为 0 是正常阶段顺序。

如需每 30 秒刷新，仅在自己的终端运行：

```bash
watch -n 30 '/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h3_common.py --status'
```

任务派发和后台错误：

```bash
tail -n 40 runs/hya_h3_flow_v2/controller.log
```

`running` 表示任务活跃；`waiting_for_idle_resources` 表示等待 5–7 中有空闲卡；`needs_attention`、`controller_failed` 或 `finalization_failed` 表示需要检查记录，不要重复启动控制器。`completed` 表示本轮生成、三项评测和报告整理完成。

后台控制器与 worker 通过独立 session 运行，标准输入为 `/dev/null`，日志写本项目；关闭 SSH、VS Code 或 Codex 不会停止计算。没有 Codex 定时巡检。

- 视频：`Evaluation_Videos/hya_h3_flow_v2/lambda_0p1/`、`lambda_0p3/`。
- 结果：`Result/HYA_H3_FLOW_V2/REPORT.md`、`scores.csv`、`paired_scores.csv`、`timing.csv`、`summary.json`；人工整理的复核与分析见 `REVIEW_AND_INTERPRETATION.md`、`REVIEW.json`。
- 计划与配置：`Project_Plan/HYA_H3_DESIGN.md`、`HYA_H3_PROTOCOL.json`。
- 算法：`tools/hya_h3_worker.py` 的 `generate_one`、`tools/hya_h3_math.py`。

两组完成后停止；是否改变介入时机或补充 prompt/指标，等待研究者决定。不要运行旧批次 `hya_h3_20261020` 的队列。
