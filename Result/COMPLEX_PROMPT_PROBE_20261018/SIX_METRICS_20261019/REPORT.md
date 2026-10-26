# CP01 复杂 prompt 单对视频：六指标补充评测

2026-10-19 按用户要求完成。直接评测既有 `CP01_seed2026101801` 的 T50/S4 两条归档视频，没有重新生成或筛选 seed。1 条 prompt、1 个 seed、2 条视频，每条全部六项，共 12 个有效分数。用户“学生指令遵循更好、总体难以取舍”的反馈仅作为非结构化观察背景，不转换为人工评分。

本次是独立补充观察，不属于 S2-R 官方分库面板；用户明确要求这两条视频各算六项，因此未改变 S2-R 的“每库三项”规则，也不合并历史样本。S3 未启动。

## 结果

沿用此前的归一化口径：MUSIQ 原值除以 100；LAION 美学函数返回值已经除以 10，不重复缩放；其余保留函数返回值。表中数值是指标分数，不是准确率，也不用于计算总分。

| 指标 | 教师 T50 | 学生 S4 | T50−S4 |
|---|---:|---:|---:|
| 成像质量 `imaging_quality` | 0.763507 | 0.773655 | -0.010148 |
| 主体一致性 `subject_consistency` | 0.908098 | 0.946145 | -0.038046 |
| 运动平滑度 `motion_smoothness` | 0.988802 | 0.992423 | -0.003621 |
| 动态判定（护栏） `dynamic_degree` | 1.000000 | 0.000000 | +1.000000 |
| 美学质量 `aesthetic_quality` | 0.570567 | 0.565696 | +0.004871 |
| 语义一致性（截断文本） `overall_consistency` | 0.118441 | 0.164691 | -0.046250 |

逐视频原始返回值、原始量纲和归一化值完整保存在 `metrics_per_video.csv`，差值保存在 `paired_differences.csv`。

## 可以与不可以得出的结论

- **实测事实**：学生成像质量、主体一致性、运动平滑度得分更高；教师美学得分略高。语义一致性也是学生更高，但只对应下述截断后的前缀。
- **运动护栏**：教师 `dynamic_degree=1`，学生为 `0`。该实现根据 RAFT 光流的强度与满足条件的帧对数量给出二元结果。0 表示未达到算法的动态判定条件，不表示所有像素完全不动；1 也不能证明动作正确或质量更好。两者均非可直接排序的“运动质量分”。
- **解释边界**：学生较高的主体一致性/平滑度与较低的动态判定应一起呈现。运动差异可能影响稳定性类得分，这是解释假设，本次没有识别这种因果关系。不得把学生的稳定性分数直接等同于复杂动作完成得更好。
- **不能建立总体排序或教师资格**：这里只有一对样本，不做置信区间、显著性检验或“显著胜出”判断；也不能把教师美学的一个正差值当作正式教师优势前提。没有简单 prompt 的同条件对照，仍不能归因于“复杂 prompt”的效果。

## 语义一致性的严重覆盖限制

两模型生成时使用完整同一 prompt，T5 编码 286 tokens，未截断。**生成编码器 T5 与评测编码器 ViCLIP 是不同组件；这里发生的是评测文本截断，不是生成 prompt 被截断。**

同一完整文本在 ViCLIP 下为 **261 tokens（含起止标记）**；固定实现仅接收 32 个位置，即起止标记加前 30 个内容 token。沿用 S2-R 的官方原生 `truncate=True`；没有事后改写文本、挑选短摘要或更换评测权重。实际有效内容为：

> a single continuous five - second , eye - level wide shot inside a sunlit pottery workshop . exactly two adults stand behind an oak table : a man

它覆盖场景与部分人物存在信息，**未覆盖倒茶、停止后移杯、物体颜色及左右关系、猫的遮挡运动等主要指令**。因此，这个分数不能验证或反驳用户对完整指令遵循的观察，不能称为“完整复杂 prompt 语义符合度”。原文、未截断长度、有效 token IDs、有效 token SHA-256 均保存在 `summary.json` 和原始结果中。

## 实现、验收与运行

- 原样调用既有 `evaluate_s2r.load_metric`；六评分函数、权重、预处理和归一化保持不变。VBench 固定版本 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；环境 `changzihe_eval`。配置见 `Project_Plan/COMPLEX_PROMPT_EVALUATION_20261019.json`。
- 8 项评测资源在启动前重新完整哈希校验，评分后大小/mtime 不变；276 个配置/代码/源码文件冻结核验通过。两条视频 SHA-256 与生成验收一致，启动前再次完整解码 81 帧，832×480、16 FPS，评分后哈希复核一致。
- 仅在确认空闲的 GPU 6 上串行评测；六 worker 退出码均为 0，无评分失败、外部占用冲突或网络连接尝试。所有本次登记进程已退出。未创建定时检测、下载、安装或计算容器。
- 一次 CPU 预检曾退出 1：新检查代码把源码归档误按独立 Git checkout 查询，读到上级仓库 HEAD，未启动 GPU。随后只修正新预检逻辑，改为固定归档 SHA-256 与逐文件字节校验，预检退出 0；既有评测源码没有修改。
- 六项任务墙钟合计约 140.1 秒，实际评分合计 18.934 秒；模型加载与各项峰值显存另列 `evaluation_timing.csv`。这些是评分成本，不是视频生成速度。
- 本地 12 个逐视频分数与 6 项差值完成独立表格复算，30 个数值核对最大绝对差为 0。同步仅包含脚本、配置、CSV/JSON/Markdown；视频、权重、latent 和环境保留远端。

## 文件位置

远端结果目录：`/data/chenjiayu/changzihe/DistillPP_Validation/Result/COMPLEX_PROMPT_PROBE_20261018/SIX_METRICS_20261019/`。

远端原视频：`/data/chenjiayu/changzihe/DistillPP_Validation/Evaluation_Videos/complex_prompt_probe_20261018/{T50,S4}/CP01_seed2026101801.mp4`。

数据：`raw_metric_reports.json`、`metrics_per_video.csv`、`paired_differences.csv`、`summary.json`；验收：`integrity.json`、`LOCAL_REVIEW.json`；运行记录：`runs/complex_prompt_eval_20261019/`。完整英文 prompt 保留在原生成配置和逐视频结果中。冻结的生成配置及原验收记录中 `metrics=[]` 是生成当时的事实，保留不改；新增评测以本报告和新配置为准。
