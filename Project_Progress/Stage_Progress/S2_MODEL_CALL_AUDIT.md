# S2 模型身份、调用与结论边界复核

范围：2026-10-16，回应“是否权重/调用有误，为什么教师未优于学生”。本次仅进行了本地证据与官方资料阅读、远端有界只读核查；未加载模型、启动 GPU 推理、下载、安装或改变 S2 配置/评分/样本。原始 S2 数据与交付清单保持不变。本文件是追加审查意见。

## 当前判断

**没有发现拿错权重、文件校验失败、教师漏载参数或把普通模型误当四步学生的证据。但此前验收证明的是资源身份、实际可运行性和内部重复性，尚未通过独立官方入口证明整个适配调用数值等价。不能将“已通过 S1”解释为已经排除所有实现偏差。**

固定 S2 协议下，五项质量指标没有一项建立教师优势；只有主体一致性支持学生更高，其余四项区间跨零。不能概括为教师全面低于学生，也不能称两者等效。此结论只针对所测 checkpoint、参数、16 条短 prompt 和三个 seed，不能宣称已测得原始 Wan 的最佳能力。S3 未开始。

## 已核对事实

| 对象 | 实际身份与调用 | 核对依据 |
|---|---|---|
| Teacher | `Wan-AI/Wan2.1-T2V-14B-Diffusers`，revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`；复用现有 transformer 的 12 个 FP32 shard，在内存映射到原生 WanModel | 既有 24 项完整文件校验记录；1095 个键完整加载、形状检查；本轮 AST 提取官方 Diffusers v0.35.1 转换表，与计划的有序重命名规则和空 special-remap 逐项相同 |
| Teacher 参数 | 50-step 原生 FlowUniPC，shift=5，CFG=5；条件/负条件各一次，共 100 次 DiT 调用 | 与固定版本 `wan/text2video.py` 的 T2V 默认采样路径静态对照一致；负提示词与 shared_config 相同；FP32 参数、BF16 autocast、FP32 特殊算子与原生实现一致 |
| Student | `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`，revision `b9baaa9a0c29226dea39043db647d1ced950bbea`；原始完整 BF16 `distill_model.pt`，同为 14B | 原生 `WanDistillModel` 明确支持此旧 PT 分支；未设置会覆盖此分支的 checkpoint 参数；加载审计为 1095 键，不是量化模型、LoRA 或教师强行四步 |
| Student 参数 | DMD2 四步，名义步表 `[1000,750,500,250]`、shift=5、CFG 关闭；原生调度器实际时间为 `[1000,937.5,833.3333,625]` | 固定 LightX2V 配置及调度源码；原生 BF16 状态与 FP32 更新后回写得到保留 |
| 公共组件 | 同一份 Wan T5、tokenizer、原生 FP32 VAE | 虽位于学生资源目录，T5 与 VAE 的 SHA-256 分别与原始 Wan 官方仓库元数据相同，不能仅因目录名将其误认成“学生特有 VAE/T5” |
| 输入与成片 | 两组共同 CPU FP32 源噪声和 T5 正向条件，832×480、81 帧、16 FPS，同一编码/评测口径 | S2 逐对哈希、双解码器及 96 条成片身份记录；没有按得分更换样本 |

权重证据入口：`resource_check_20261016_091241_1370827.json`（24 项完整校验成功）、`S1_PREFLIGHT.json`、`Result/S1_20261016/acceptance.json`、`runs/s2_20261016/final_safety_audit.json`。

本次通过正规执行审批以 SSH 只读核查 `chenjiayu@dasys-h200x8`，退出码 0：授权项目真实路径正确；24 项生成资源的大小/mtime 与完整校验时均一致。另读两份小模型配置，14B 架构参数相容。11 个运行源码/配置 SHA 与本地对应文件一致，包括两个自定义运行脚本、键名映射、Wan 模型/T5/VAE/调度代码及 LightX2V 学生加载/调度代码。本次没有重新全读约 97.6 GB 权重计算哈希；“状态一致”与“本轮重新完整哈希”严格区分。

## 尚未闭合的实现与配置问题

1. **独立入口等价验证缺失。** Teacher 使用“官方 Diffusers 权重 → 原生 WanModel”的自定义加载桥接，采样循环由项目包装；Student 直接调用原生 DiT/调度器，同时共用 Wan 的 T5/VAE。S1 与 S2 的预热 latent 一致，只证明同一实现重复运行稳定。应独立验证键映射后的预测、原生完整采样入口与包装循环、公共组件替换的数值差异，不能靠“视频能生成、键数正确”替代。
2. **480P 的 shift 依据存在差异。** 固定版本原生 `generate.py` 对 T2V 默认 shift=5，`WanT2V.generate` 也默认 5；同版本 README 的 Diffusers 示例注明 `5.0 for 720P, 3.0 for 480P`。S2 的 5 符合所采用原生入口默认，因此不能直接宣布参数用错；但也不能称已经验证 480P 最佳教师设置。shift=3 的效果尚未实测。学生的训练固定 shift=5 不应跟随教师任意更改。
3. **这是基础短提示词协议。** Wan 官方建议扩写 prompt，而 S2 按冻结计划关闭扩写，16 条文字主要是短动作描述。480P 本身是官方支持分辨率，不属于模型误用；短提示词/不扩写是实验边界，其对师生相对差异的影响仍未知。若将来增加扩写，需对两组提供同一冻结扩写文本。
4. **同 seed 不等于同噪声。** 官方原生入口通常用 CUDA RNG，S2 为严格配对而用 CPU RNG 产生共同 FP32 噪声；均为高斯噪声方案，但具体 tensor 不同。独立入口复核必须注入同一实际 tensor，不能仅复用 seed 数值后将视频差异误判成调用错误。

## 官方表述与结果解释

LightX2V 的[模型格式说明](https://lightx2v-en.readthedocs.io/en/latest/getting_started/model_structure.html)确实把原始完整 Wan 模型描述为“highest quality”，旁列原始精度、无量化损失、组件完整等特点。但该页未提供当前两个 checkpoint 在我们六指标、16×3 面板上的统计比较；不能由概括性表述推出每个指标上教师必然更高。

所用学生的[官方模型卡](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)明确为基于 Wan2.1-14B 的双向四步与 CFG 蒸馏；[蒸馏说明](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html)将目标描述为减少步骤并保留质量。理论上不存在“蒸馏后任意客观评分只能下降”的一般保证；这不是本次学生得分变化的因果证明。

| 指标 | T50 | S4 | 既定配对统计结论 |
|---|---:|---:|---|
| imaging_quality | 0.700690 | 0.704683 | 99% 区间跨零 |
| subject_consistency | 0.971771 | 0.978664 | T−S = −0.006893；99% CI [−0.012652, −0.001480] |
| motion_smoothness | 0.987733 | 0.987840 | 99% 区间跨零 |
| aesthetic_quality | 0.567752 | 0.581071 | 99% 区间跨零 |
| overall_consistency | 0.206526 | 0.202661 | 教师均值较高，但 99% 区间跨零；此指标是文本视频语义代理 |
| dynamic_degree | 29/48 | 20/48 | 教师动态判定比例高 18.75 个百分点，95% CI [+2.08, +35.42] |

**待验证解释：** 学生动态判定比例较低，而主体一致性评价相邻帧及首帧 DINO 特征相似度，较少运动可能有利于该指标；不能据此断言学生全为静态视频，也未证明运动变化解释了全部一致性差异。[LightX2V 低延迟文档](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_latency.html)也提示步数蒸馏可能损失视频动态。当前结果与这种可能性相容，不构成因果识别。运动护栏仍不按越高越好排序。

仅 16 个独立 prompt 簇的探索性面板对小差异仍有不确定性；增加 seed 不等于增加独立 prompt。分层均值也有不同方向（例如场景类的成像、人物动作类的美学均值偏向教师），这些事后子组描述不改变预设资格结果。

## 上轮建议（未执行，已不作为当前推进顺序）

2026-10-17 已完成基于既有完整校验、原生源码和 96 条实际运行记录的契约复核，未发现已知生成调用错误。当前按核查后修正官方 prompt/指标设置推进，具体见 `Project_Plan/OFFICIAL_QUALIFICATION_V2.md`。下面保留原先未执行的建议作为记录，不自动启动官方入口对照测试。

先在独立于正式面板的固定验收输入上，使用现有资源，对照官方入口和项目入口的同一噪声/条件、单步预测、实际 sigma、最终 latent 与 VAE 输出；教师须覆盖映射桥接，学生须覆盖原生 runner。先定位数值差异来自何处，再定义恰当误差准则，不以“结果看起来差不多”验收。

实现等价问题闭合后，再单独评估教师 shift=3 与 5 等明确官方配置差异；不能以反复调参直至教师胜出为目标。若改变正式教师设置，保留本轮数据，记录选择依据，在独立新样本上确认；不事后改写 S2、删指标或挑有利子组。保持六指标、客观评测和动态护栏的原约定。此前对实现验收的表述需要收窄到已完成的证据范围。

固定官方源码参考：[Wan 原生入口](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py)、[Wan README（提示词扩写及 Diffusers shift 注释）](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/README.md)、[Diffusers 键名转换](https://github.com/huggingface/diffusers/blob/v0.35.1/scripts/convert_wan_to_diffusers.py)、[LightX2V 原 PT 加载](https://github.com/ModelTC/LightX2V/blob/cc9429192971815afff8368f74bd10adcc6fed48/lightx2v/models/networks/wan/distill_model.py)。
