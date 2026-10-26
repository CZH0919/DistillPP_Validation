# H2 已完成：结果入口与历史进度检查

**2026-10-20 完成复核：64 对/128 视频/224 分数齐全，后台进程已结束。**按研究者均值比较规则，主体一致性和多物体教师取胜；当前结论见 `../Result/HYA_H2_20261019/REVIEW_AND_INTERPRETATION.md`、`QUALIFICATION_DECISION.json`，S3 未启动。

下方为本批次只读查看入口及历史后台运行说明，不是重新启动指令；完成后无需持续刷新或跟随日志。

运行批次：`hya_h2_20261019`。64 条 prompt × 1 seed，T50/S5 合计 128 视频、224 个指标值。

## 查看进度（远端 Bash）

```bash
cd /data/chenjiayu/changzihe/DistillPP_Validation
/data/chenjiayu/miniconda3/envs/changzihe/bin/python -B tools/hya_h2_resume_parallel_20261019.py --status
```

如需每 30 秒刷新：

```bash
watch -n 30 /data/chenjiayu/miniconda3/envs/changzihe/bin/python -B /data/chenjiayu/changzihe/DistillPP_Validation/tools/hya_h2_resume_parallel_20261019.py --status
```

按 `Ctrl+C` 只停止当前显示，不会停止后台任务。本次恢复新增了单对任务，请使用上方更新后的 `--status` 入口；旧的 `hya_h2_status.py` 不识别这些新任务。状态包括单臂视频验收槽位、已经完整配对并归档的视频、分数数量、控制器是否存活，以及每卡任务/采样步数/日志路径。单臂验收槽位可能包含待补齐配对或历史重试，正式完成量以“成对归档”和“配对”计数为准。

## 查看日志和 GPU

```bash
cd /data/chenjiayu/changzihe/DistillPP_Validation
tail -n 60 -f runs/hya_h2_20261019/controller.log
# 单个 worker 的日志路径由上面的 status 命令打印；对该路径执行 tail -n 60 -f 即可。
nvidia-smi
```

控制器日志记录任务派发；详细去噪步骤在各 worker 日志和 progress.json 中。不要仅按 GPU 利用率判断是否完成，应同时看控制器、配对和分数计数。

## 后台方式和资源边界

启动器通过 Python `Popen(start_new_session=True, close_fds=True)` 创建新会话，stdin 接 `/dev/null`，stdout/stderr 写项目日志，控制器忽略 SIGHUP；worker 也独立于 SSH 终端。关闭 VS Code、SSH 窗口或 Codex 不影响正常继续，不依赖 Codex 定时巡检。服务器关机、重启或进程被终止仍会中断；已提交数据、尝试记录和任务身份保留，可经检查后断点接续，不要手动重复派发任务。

原始阶段按四卡、每块两个 prompt 执行；剩余 6 对已拆为单对任务。六卡尝试发生新占用后，按用户最新要求，**新任务仅派到 GPU 3、6**。交接时已经在其他卡正常运行的 worker 保留自然完成，但不再向那些卡补派任务；因此过渡期间活跃卡数可能超过两张。每卡一个 worker，实际加载前再次核验空闲，同对 T50/S5 留在同一物理 GPU。原先三个缺失块在其两个配对均完成后触发对应评测；其他卡可继续生成。资源冲突仅退出自身受影响进程，300 秒后重查；新任务只在 3、6 的实际空闲条件下接续，连续两次冲突仍会暂停相关配对并报告。普通实现/数值失败暂停相关任务，不自动换算法、seed 或指标。

## 输出与停止点

- 持续状态：`runs/hya_h2_20261019/controller.json`。
- 成对提交：`runs/hya_h2_20261019/pairs/`；指标提交：`scores/`。
- 已完成配对视频按组归档：`Evaluation_Videos/hya_h2_20261019/T50/`、`S5/`。
- 完整验收与分析：`Result/HYA_H2_20261019/REPORT.md`、`statistics.json`、`raw_scores.csv`、`paired_scores.csv`、`timing_per_video.csv`。
- `Result/HYA_H2_20261019/COMPLETE.json` 仅在 128 视频/224 分数完整且统计、归档校验通过后生成。
- 控制器 `completed` 表示本轮 H2 全部完成；`needs_attention`、`controller_failed`、`finalization_failed` 需要先检查记录。后台不自动启动 S3，不自动下载或安装资源。

H2 的明确启动授权记录在该批次 `manifest.json` 中。V2 科研协议和 H1 验收契约保持冻结；旧准备阶段的“未授权启动”字段不替代用户本轮明确授权。
当前调度交接记录：`runs/hya_h2_20261019/resume_stable_20261019/`。状态中的运行时长从最初 H2 启动计起，包含暂停与恢复时间；查看命令不受控制器交接影响。
