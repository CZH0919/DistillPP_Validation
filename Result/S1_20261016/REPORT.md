# S1 实现与验收报告

结论：**S1 通过；正式 S2 教师—学生资格实验未启动（0/96）。**本批仅使用一个独立非正式 prompt、一个 seed，生成 T50/S4 各一条视频并得到全部 12 项指标记录。结果证明当前实现链路可运行，不证明教师优势、迁移收益或正式速度优势。

## 1. 资源与依赖

- 用户已完成误下载目录的精确清理；原共享模型、VAE/T5 原件保留。
- 24 项生成资源沿用此前完整哈希通过的证据，并逐项确认大小/mtime 未变；八项评测资源本轮重新完整计算 SHA-256，全部匹配。
- 五份固定源码与归档逐文件核对通过。生成环境使用 `changzihe`，评测使用 `changzihe_eval`；无需新增模型、评测权重或依赖安装。
- 两个环境的 `pip check` 均只剩已知 decord 0.6.0 内部 wheel 标签告警，退出码为 1，不能写成全绿。实际两条视频已分别由 decord 与 OpenCV 完整解码，81 帧、832×480、16 FPS 均符合要求。
- 六指标模块导入和真实权重加载/计算全部通过。FA3、FA4、Sage 等可选后端的未安装提示不影响所选 FA2 路径；没有因此安装额外内核。

详细证据：`Project_Progress/Stage_Progress/S1_PREFLIGHT.json`、`S1_CPU_CONTRACT.json`。

## 2. 实际实现与精度

Teacher 直接从已有官方 Diffusers 12 个分片读取，按官方重命名规则在内存映射到原生 WanModel；1,095 个键，missing/unexpected 均为 0，不另存转换权重。Student 通过固定 LightX2V 原版 PT 分支读取已有 `distill_model.pt`，核对 1,095 个 BF16 权重张量及 80 个 FA2 attention 实例。去噪分别使用原生 Wan Flow UniPC 和 LightX2V WanStepDistillScheduler。

| 项目 | Teacher T50 | Student S4 |
|---|---|---|
| 权重/前向 | 原生 FP32 参数存储；BF16 autocast；保留原生 FP32 时间嵌入、归一化/调制与输出头 | 原版 BF16 PT；原生 LightX2V 前向；torch RMSNorm/LayerNorm/modulation，FA2 |
| 持久 latent | FP32 | 从共享 FP32 初始化，原生 step_pre 转 BF16；更新计算 FP32，结果写回 BF16 |
| CFG / 预测数 | CFG=5；50 步，100 次条件分支预测 | 无 CFG；4 步，4 次预测 |
| 时间表 | 50 个原生 timestep；首个 999、最后 92；sigma 首值约 0.99979985，终点 0 | timestep `[1000,937.5,833.333313,625]`；sigma `[1,0.9375,0.833333313,0.625]`，最后一步到 0 |
| 公共组件 | 两组使用完全相同的原生 T5 BF16、VAE FP32、tokenizer 及其文件 | 同左 |

旧计划“学生全程 FP32 状态”的假设已纠正。没有修改原生 scheduler 去迎合错误假设。学生初始原生 BF16 转换的最大绝对误差为 **0.0155568**；后续 S3 必须测量 BF16 回写后的实际纠偏量，不能以 FP32 中非零代替真实注入。若未来改变状态精度，须另立基线与确认口径。

源噪声：独立 CPU FP32 generator、seed `2026101601`、shape `[16,21,60,104]`。两组源噪声 SHA-256 完全相同：`d9865a02b66879866a03eea4daf59f5b90f393ff3b5f3958955b90b991b3ba06`；两组正向 T5 条件为 `[25,4096]`、BF16，张量哈希也相同。这里的一致性指源噪声和文本条件，不表示不同原生精度转换后的状态全程相同。

冒烟 prompt（不属于 16 个正式 prompts，ViCLIP 含特殊 token 共 24，未截断）：

> A blue toy boat drifts across a garden pond, leaving gentle ripples in soft daylight. Static camera.

两组均为 832×480、81 帧、16 FPS；统一 libx264、CRF 18、yuv420p；无提示词增强、量化、近似缓存、插帧或后处理。

## 3. 六指标接口结果

下表仅为单个配对冒烟样本的有效性检查。不给出教师资格、显著性、质量优胜或总体优胜结论；不加入人工质量评分。

| 指标 | T50 | S4 |
|---|---:|---:|
| imaging_quality | 0.667522 | 0.654636 |
| subject_consistency | 0.963911 | 0.965597 |
| motion_smoothness | 0.985560 | 0.982569 |
| dynamic_degree | 0 | 0 |
| aesthetic_quality | 0.628235 | 0.770574 |
| overall_consistency | 0.218174 | 0.201875 |

原始数值和归一化数值见 `metrics.csv`；MUSIQ 原值只除以 100 一次，美学源函数已除以 10，未再次归一化。dynamic_degree=0 是有效的原版 RAFT 动态判定，不是缺失值，也不用于自动通过/拒绝。该项仅监测运动，不按越高越好排名；取舍仍由研究者决定。

评测使用固定 VBench 定义与预处理：MUSIQ 的 longer 预处理，DINO 逐帧一致性，AMT-S 运动平滑性，RAFT 动态判定，OpenAI CLIP+LAION V1 美学，以及 ViCLIP 八段中间帧与完整 prompt。模型处于 eval 模式，所有文件从明确本地路径加载，记录中的网络访问尝试均为 0。

## 4. 计时与显存接口

| 单次冒烟记录 | T50 | S4 |
|---|---:|---:|
| 去噪，秒 | 506.921 | 19.335 |
| VAE，秒 | 11.108 | 11.149 |
| 去噪 + VAE，秒 | 518.029 | 30.484 |
| 峰值已分配显存，GiB | 71.420 | 44.786 |
| 峰值预留显存，GiB | 73.826 | 46.805 |

这是首次完整生成、包含 S1 诊断操作的 CUDA 同步计时；没有正式的充分预热、重复测量、平衡顺序或统计区间，**不能当作正式速度优势估计**。文本编码和模型载入另行记录在 `timing.csv` 与原始 JSON；不使用 50/4 步数比估算速度。

有限共同驻留探测：单张空闲 H200 上同时保留 Teacher、Student、T5、VAE，执行一次教师 CFG 查询（两次分支预测）、一次学生预测及一次 VAE 解码。没有融合教师预测、没有运行纠偏算法、没有导出额外视频。

- 全部模型驻留：90.937 GiB。
- 探测峰值已分配：98.051 GiB；峰值预留：100.086 GiB。
- PyTorch 报告设备可用总量：139.801 GiB；相对峰值预留仍余 39.715 GiB。
- 无 offload、无 OOM；输出 shape 与有限数检查通过。

这证明当前固定形状的有限共同驻留可行，不替代未来完整纠偏实现的资源与计时验收。

## 5. 失败、安全处理与证据保留

1. 学生 attempt01 在首步精度断言处停止：先前错误地要求持久 latent 为 FP32。核实原生 BaseScheduler 的 BF16 转换后，仅修正验收断言和计划描述；保持原生采样行为、模型、prompt、seed、时间表不变。attempt02 完整通过，失败记录保留。
2. dynamic_degree attempt01 结束检查时发现 0 号卡出现另一未归属本任务的 PID。脚本主动退出本人的评测进程，没有终止其他进程。重新确认 1 号卡空闲后，以完全相同输入重跑该项并继续其余两项，全部通过；失败记录保留。
3. CPU 数值检查不能直接导入要求 CUDA 平台的 LightX2V 顶层；采用固定源码中未改动的数值方法隔离检查，GPU 全类运行另行通过。FP32 Euler 形式与原始端点更新的最大差为 `2.3842e-7`，并验证原生 step_pre 的 BF16 转换。
4. 原始教师 JSON 保留启动时计划的旧说明字段；其明确的 `load_acceptance`、已保存执行代码和实际 dtype 记录为准。汇总 `acceptance.json` 采用纠正后的配置文字，保留原始报告及其哈希，不改写原始数值。

所有 GPU 工作串行使用当时空闲卡；7 号卡始终有驻留进程，未使用。每次启动两次检查，运行中检查选卡上的其他进程；项目锁只防止自身重叠，不冒充集群预留。最终快照显示本次使用的 0、1 号卡已释放。没有修改 SSH/DNS/主机密钥、他人目录/环境、共享权重，也没有操作旧服务器。

## 6. 交付与下一阶段

远端项目根：`/data/chenjiayu/changzihe/DistillPP_Validation`。

- 结果：`Result/S1_20261016/acceptance.json`、`metrics.csv`、`timing.csv`、本报告。
- 阶段与总体记录：`Project_Progress/Stage_Progress/S1_IMPLEMENTATION_ACCEPTANCE.md`、`Project_Progress/Total_Progress.md`。
- 执行证据：`runs/s1_20261016/` 的成功/失败 JSON、代码快照；阶段日志在 Project_Progress。
- 分组视频：`Evaluation_Videos/s1_20261016/T50/` 与 `S4/`，均为 `S1_smoke_seed2026101601.mp4`；归档与原始文件 SHA-256 一致。
- 实现：`tools/run_s1_arm.py`、`evaluate_s1.py`、`check_s1_residency.py`、`s1_common.py`、`s1_preflight.py`、`validate_s1_contract.py`、`finalize_s1.py`。

本地仅同步小型代码、配置、日志摘要及核心结果；未同步模型、评测权重、视频、latent、环境或缓存。当前停止于 S1，等待下一阶段指令；S2 的 16 prompts × 3 seeds × 2 组正式视频仍为 0/96，尚未建立本批教师优势证据。
