# DistillPP Validation

Distillation++ 推理侧纠偏迁移研究的核心代码、配置和轻量结果备份。

研究问题：在真正蒸馏得到的双向少步视频模型推理过程中，由原始多步教师进行有限纠偏，改善学生生成质量，同时保留相对完整教师的速度优势。先在目标质量维度建立教师相对学生的实测均值优势，再验证迁移效果。

## 当前实验状态

HunyuanVideo T2V 13B（T50）→ AccVideo（S5）的第 1–4 步 × λ=0.1/0.3/0.5 共 12 个纠偏设置已完成。固定 48 条 prompt、seed=2026101901，结合教师/学生共 14 组、672 条视频对应的 1120 个指标分数。生成设置为 1280×720、129 帧、24 FPS。

主要质量指标是 `subject_consistency` 与 `multiple_objects`，`dynamic_degree` 作为运动护栏。数据以完整任务库均值比较，分层统计仅作补充。结果属于当前固定任务集、单 seed 的初步验证；数值一致性例外及敏感性分析已记录。未启动下一轮 seed 确认实验。

- [完整实验计划](Project_Plan/EXPERIMENT_PLAN.md)
- [迁移公式与设计](Project_Plan/HYA_H3_DESIGN.md)
- [总体进度](Project_Progress/Total_Progress.md)
- [完整 4×3 结果复核与分析](Result/HYA_H3_STEP24_V1/REVIEW_AND_INTERPRETATION.md)
- [原始自动汇总](Result/HYA_H3_STEP24_V1/REPORT.md)

## 目录与核心入口

| 路径 | 内容 |
|---|---|
| `tools/` | 生成、调度、评测、校验、统计和绘图脚本 |
| `Project_Plan/` | 实验协议、prompt、数学设计、依赖与资源清单、固定版本参考源码 |
| `Project_Progress/` | 总体进度、阶段记录与验收结果 |
| `Result/` | 原分、配对统计、真实计时、复核报告及图表 |
| `runs/` | 已同步到本地的轻量运行配置、清单、评分与证据元数据 |

算法入口：

- [`tools/hya_h3_math.py`](tools/hya_h3_math.py)：Flow Matching 端点转换、重加噪与纠偏公式。
- [`tools/hya_h3_step24_worker.py`](tools/hya_h3_step24_worker.py)：第二/第四步单次教师介入与既定评测流程，核心函数为 `generate_one`。
- [`tools/hya_h3_step24_common.py`](tools/hya_h3_step24_common.py)、[`tools/hya_h3_step24_controller.py`](tools/hya_h3_step24_controller.py)：冻结协议核验、资源准入、后台调度与完成结果复用。
- [`tools/hya_h3_step24_finalize.py`](tools/hya_h3_step24_finalize.py)：合并第 1–4 步、三个 λ 与教师/学生结果。
- [`tools/analyze_hya_h3_step24_results.py`](tools/analyze_hya_h3_step24_results.py)：从验收导出的元数据 bundle 独立复算。使用时需提供对应 `--bundle`；该工具不加载模型。

前两批的 `hya_h3_*`、`hya_h3_timing_*` 及 H1/H2 支撑脚本保持原样。历史 Wan2.1→LightX2V 工作与负结果也保留，不能将不同模型或阶段的数据混作同一面板。

## 结果数据与图表

最新完整面板位于 [`Result/HYA_H3_STEP24_V1/`](Result/HYA_H3_STEP24_V1/)：

- `scores.csv`：1120 个指标分数。
- `paired_scores.csv`：纠偏组相对教师/学生的逐样本配对数据。
- `position_contrasts.csv`：固定 λ 的介入位置比较。
- `timing.csv`：672 条视频对应的真实计时记录。
- `correction_magnitudes.csv`：实际纠偏幅度记录。
- `summary.json`、`REVIEW.json`：自动统计与独立复核。
- `COMPLETE.json`：原始自动结果的 SHA-256 清单；后续复核报告和图表为独立补充文件。

图表入口：

- [14 组完整数据表](Result/HYA_H3_STEP24_V1/Figures/HYA_H3_14_GROUPS_SUMMARY.png)
- [介入步骤 × λ 热图](Result/HYA_H3_STEP24_V1/Figures/HYA_H3_STEP_LAMBDA_EFFECTS.png)
- [介入步骤 × λ 变化曲线](Result/HYA_H3_STEP24_V1/Figures/HYA_H3_STEP_LAMBDA_CURVES.png)

已具备 NumPy、Matplotlib、Pillow 和中文字体的 Python 环境可在仓库根目录重绘图片，以下命令不使用 GPU：

```bash
python tools/plot_hya_h3_14_groups.py --project .
python tools/plot_hya_h3_comparison_figures.py --project .
```

中文字体未被自动找到时，通过 `--fontfile` 指定本机字体路径；依赖清单位于 `Project_Plan/`。源码会从 CSV 重新聚合，并与已核验的 JSON 统计比对。

## 备份范围与恢复边界

本仓库对应 Windows 本地目录 `D:\Research\PK_8H200\DistillPP_Validation` 的选定内容。保留核心代码、算法、配置、prompt、依赖/资源清单、报告、图表、CSV/JSON 结果和已有本地运行元数据。少量固定版本源码压缩包与既有核心证据压缩包原样保存，来源、revision 和校验值以相应清单为准。

模型/评测权重、视频、latent、Conda 环境、包缓存及散落运行日志不作为普通 Git 文件备份；`.gitignore` 防止后续误加入这些材料或认证文件。没有从远端补拉整套运行目录或大型资源，仓库不是可直接启动计算的完整服务器镜像。

历史记录中的绝对路径保持原样：业务目录为 `/data/chenjiayu/changzihe/DistillPP_Validation`，共享模型目录为 `/data/chenjiayu/models`。在其他机器恢复时，需要另外准备资源，并核对路径、依赖和实验授权；已完成批次不应重复启动。

`.gitattributes` 禁止 Git 自动转换换行符，避免改变冻结代码、配置及结果的原始字节和 SHA-256。备份不修改原始实验算法、协议或结果。第三方参考源码沿用各自的来源与许可证。
