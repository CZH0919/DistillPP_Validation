# Wan2.1 → LightX2V：教师资格验证与 Distillation++ 迁移计划

**2026-10-17 当前主线：寻找能够建立教师优势的模型组合，再验证 Distillation++ 对真正蒸馏的双向少步视频模型的推理纠偏是否有效，并保留相对完整教师的真实速度优势。**推进顺序为权重/生成调用核查、官方依据下的 prompt/指标修正、配对资格验证、研究者确认后的迁移。现有基线契约核查已通过，未发现已知调用错误；不把核查扩展成独立研究任务。

当前生效协议见 [OFFICIAL_QUALIFICATION_V2.md](OFFICIAL_QUALIFICATION_V2.md)，输入与设置见 `QUALIFICATION_V2.json`：官方两类任务库**各 16 条 prompt×3 seeds，每条只评所属库三项**，使用固定公开扩写。资格阶段 T50/S4 共 **192 条视频、576 个指标值**。通过资格并获准迁移后，S3/S4 沿用同一 prompt/seed 配置，不新增或扩大面板。**S2-R 已执行完成，完整性验收与本地复算通过；五质量指标 0/5 建立教师优势，资格未通过，S3 未开始。**结论见 [完成复核与研究解读](../Result/S2R_20261017/REVIEW_AND_INTERPRETATION.md)。下文保留执行前约定；原 S2 及 S2-R 冻结输入、科学代码和原始结果不改写。

历史状态：**旧 S2 已完整执行，五质量指标 0/5 建立教师优势，资格未通过。**96 条视频、576 个分数及分析见 `Result/S2_20261016/REPORT.md`；当时的排程与计时细则保留在 `S2_EXECUTION.json` 和 `S2_EXECUTION_NOTE.md`。它们不作为新批次配置直接执行。S3–S5 保留为条件性方案，不因旧 S2 动态比例更高而替代质量判据。

**资源准备与 S1 已完成：复用现有官方 Diffusers 教师及 LightX2V 原版 PT；24 项生成资源的完整校验与状态复核、八项评测资源的完整校验均通过，误下载目录已清理。RESOURCE_MANIFEST v2、QUALIFICATION_V2 内的 baseline 和下表定义当前身份与精度设置。两组真实加载、完整冒烟生成、六指标及共同驻留探测已通过；这不构成教师优势或迁移有效性证据。详见 `Project_Progress/Stage_Progress/S1_IMPLEMENTATION_ACCEPTANCE.md`。**

## 1. 问题、证据与结论范围

**问题：在真正蒸馏的双向少步视频模型推理过程中，由原始多步教师有限纠偏，提高学生的目标质量，同时保留相对完整教师的实际速度优势。**本批选用 Wan2.1-T2V-14B 与基于它蒸馏的 LightX2V 双向四步完整 BF16 模型，两者均为 14B。学生不是把教师直接缩短为四步，也不是因果/流式生成模型。

先用自身数据建立教师在目标质量维度上的优势；未建立时停在资格验证，不转而论证“更弱教师的局部价值”。只做推理侧干预，不训练、再蒸馏或加入人工视频质量评分。

| 证据类型 | 本批应如何表述 |
|---|---|
| 论文结论 | Distillation++ 在论文所测图像蒸馏模型上有效；没有证明 Wan 视频或 Flow Matching 迁移有效 |
| 固定代码事实 | LightX2V 四步、CFG 蒸馏、双向配置及确定性 Flow 更新可由源码核对；S1 验收和 S2 正式生成/六指标均已完成 |
| 代数推导 | 第 5 节由线性 Flow 路径与 Euler 更新推得；不是质量改善或收敛证明 |
| 研究假设 | S2 未建立预设目标教师优势；一次早期教师纠偏的质量/速度有效性尚未实施和验证 |
| 旧实测 | LTX 2B 0.9.6 首轮纠偏成像差 −0.006150，调整区间为负；次轮教师对 S8/S4 的五质量指标优势 0/10。不能外推所有模型无效。13B→2B 多尺度路线暂停，未执行 |

不将旧 LTX 或旧 S2 的视频/分数并入 V2。V2 文本来自固定 VBench 官方两库，排除旧 S2 已用语义锚点；旧 S2 耗时只用于粗略预算。详细研究背景见本地 `codex_note/RESEARCH_READING_NOTES_20261016.md`。

## 2. 阶段与停止点

| 阶段 | 工作与产物 | 进入下一阶段的条件 |
|---|---|---|
| S0 资源准备 | 用户执行下载/安装指令；核对文件大小、哈希、仓库 commit、环境清单 | 权重身份与路径正确；缺项明确，不能用相似名称替代 |
| S1 实现验收 | 原生 T50/S4 接口、时间表、共同 latent、解码/计时与六指标离线加载；独立非正式 prompt 各 1 条冒烟视频 | 两组流程完整；六项有有效逐视频结果；保存运行配置及实际 attention、显存；不将冒烟样本并入正式数据 |
| S2 旧教师资格 | 已完成：16 prompts × 3 seeds × 2 组 = **96 视频、576 个指标值** | 未建立教师优势；保留原始结果，不进入迁移 |
| S2-R 官方依据资格验证 | 每库 16 条，共 32 prompts × 3 seeds × 2 组 = **192 视频、576 个指标值** | 按第 4 节建立候选教师优势，提交报告，**等待用户确认目标与 S3 授权** |
| S3 条件性迁移开发 | 第 5 节代数/接口验收、λ=0 与 teacher-discard 控制；从同一面板固定取 4 prompts × 2 seeds 作有限参数选择 | 控制通过，有有效注入；每条只评所属三项，完整报告各候选结果；研究者在 S4 前冻结干预配置 |
| S4 同面板迁移效果验证 | 沿用完整同一 32 prompts × 3 seeds，比较 T50/S4/S4_T1：**288 条比较视频、864 个指标值**；符合条件的 192 条基线复用，纠偏组补齐至 96 条 | 按第 6 节报告同面板质量变化、真实速度与限制；由研究者判断取舍与可行性，不称独立确认 |
| S5 整理归档 | 六维结果、配对统计、成本/失败记录、局限；按组复制视频并校验 SHA-256 | `Result/` 报告、`Evaluation_Videos/` 索引、总体与阶段进度一致 |

每个指标实际对应 16 个 prompt 簇、48 个 seed 配对，不是 32 个独立 prompt 簇，也不是 48 个独立 prompts。该规模用于本轮探索性资格筛查，不构成功效保证。区间跨零如实记为“不确定”；不依据结果加 seed、挑换文本、改变 CFG 或扩大面板直到显著。资格和迁移使用同一输入有利于可比性及基线复用，但固定 prompt 本身不保证教师优势；同面板调参后的效果不能写成未见 prompt 的泛化结果。

## 3. 基线冻结方案

| 项目 | T50：原始教师 | S4：蒸馏学生 |
|---|---|---|
| 权重 | 已有 `Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer`，revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`；12 个 F32 分片 | 已有 `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`，revision `b9baaa9a0c29226dea39043db647d1ced950bbea`；`distill_model.pt`，BF16 |
| 实现 | Wan2.1 commit `9737cba9c1c3c4d04b33fcad41c111989865d315` | LightX2V commit `cc9429192971815afff8368f74bd10adcc6fed48` |
| 采样 | 原生 Flow UniPC，50 步，shift=5，CFG=5 | 原生 step-distill/DMD2，4 步，shift=5，`enable_cfg=false` |
| 时间 | 保存实际 scheduler 时间数组，不能由整数步数推测 | 名义 `[1000,750,500,250]`；实际 σ 为 `[1,0.9375,0.833333…,0.625]`，终点 0 |

共同设置：832×480、81 帧、16 FPS、batch=1；主要 DiT 计算采用 BF16，保留各自原生精度路径。**S1 已查明：Teacher 原生实现保留 FP32 参数、通过 BF16 autocast 前向，并在时间嵌入、归一化/调制与输出头显式使用 FP32；Student 原版 PT 参数为 BF16。**两组共用同一份原生 Wan T5/VAE/tokenizer 文件与同一原生组件实现：T5 BF16，VAE 参数和解码 FP32，不使用 LightX2V 的另一套默认 VAE 精度。完整文件身份见 `RESOURCE_MANIFEST.json`。

教师加载时按 Diffusers v0.35.1 官方转换规则逆映射参数名，直接读取已有 FP32 分片，保留原生参数精度，不另存权重副本。原生 Wan 前向和采样协议保持，不能整体将参数转 BF16 后破坏显式 FP32 路径。1,095 个键的映射和真实加载均已核对。学生使用固定 LightX2V 的原版 PT 加载分支与原生 `WanStepDistillScheduler`，`model_path` 为现有学生目录，不设置 `dit_original_ckpt`/`transformer_model_path` 覆盖。不声称现有 PT 与另一仓库 safetensors 已逐张量验证相等。S1 为统一条件/解码，直接调用两框架的原生 DiT 和 scheduler，公共 T5/VAE 走同一原生 Wan 组件；没有改变去噪更新式。

统一使用已通过 S1 验收的 **FlashAttention 2.8.3**：LightX2V 固定示例的两项 attention 配置由 FA3 显式覆盖为 `flash_attn2`，Wan 记录实际采用 FA2。若运行时选到其他后端，停止相关任务核查，不把不同后端的耗时混报。V2 的 prompt enhancement 仅指读取固定公开扩写文本；不运行在线改写器。继续关闭量化、近似缓存、稀疏加速、torch.compile 和插帧。

教师使用固定 Wan 负向 prompt（见 `QUALIFICATION_V2.json` 的 `baseline.negative_prompt`，与旧基线一致）；学生无 CFG，不声称其负向条件实际参与计算。不同原生采样器和部署实现属于这组系统比较的组成部分，不能把所有差异仅归因于蒸馏训练。

`QUALIFICATION_V2.json` 是新资格和迁移的唯一活动输入配置：从官方 72/93 条库中排除旧 S2 锚点，按既定 SHA-256 顺序各取前 16 条、规范化原文去重；不依据分数挑选。只有 `screen_selected=true` 的 32 条参与实验，其余条目只保留来源。生成读取公开 `generation_prompt`，语义评分读取原始 `eval_prompt`，两者及指标映射不得随方法改变。三个 seeds 固定为 `1529987743`、`1993624635`、`1454077509`；`screen.active_panel_sha256` 冻结文本、映射和 seeds，迁移引用同一指纹。`PROMPTS_QUALIFICATION.json` 与 `BASELINE_SETTINGS.json` 保持旧 S1/S2 证据原样；V2 基线设置读取新 JSON 内的 `baseline`。

所有阶段/方法逐对读取**同一份 CPU FP32 源 latent**，使用专用 RNG 并保存/核对张量哈希，不能仅设置相同 seed 后假定不同程序会产生相同噪声。**Teacher 保持 FP32 采样状态；LightX2V 原生 `BaseScheduler.step_pre` 在 `DTYPE=BF16、SENSITIVE_LAYER_DTYPE=None` 下把持久状态转 BF16，`step_post` 以 FP32 计算后写回 BF16。**保留这一原生行为，记录转换前/后的 dtype、哈希和误差；不能声称两组状态全程等精度。学生的 `keep_latents_dtype_in_scheduler` 保持原生 false；RMSNorm、LayerNorm、modulation 选已有 torch 实现，FA2 保持不变，不安装额外量化/稀疏内核。教师纠偏噪声使用独立 RNG 流，同一 prompt/seed 的不同 λ 候选共用同一纠偏噪声；方法名/λ 不改变源噪声。保存 prompt、seed、shape、dtype、checkpoint、配置与视频 SHA。

统一导出 libx264、CRF=18、yuv420p，保持 81 帧和 16 FPS，无补帧/后处理。评测使用这份固定编码视频；编码差异不能作为方法效果。生成文本超过 T5 上限、视频损坏或帧数不符须显式失败，不能私自改写文本或重复帧。ViCLIP 的原生 32-token 限制另按下节显式记录，不声称完整长文本都参与了语义评分。

## 4. 六指标与教师资格判据

使用固定 VBench commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` 的六维定义与明确离线路径；不是完整 VBench 官方榜单。所有分数保留原始值与一次归一化后的值，不能双重归一化。

| 指标 | 固定模型/含义 | 用法 |
|---|---|---|
| imaging_quality | MUSIQ-SPAQ；原值 /100 一次 | 质量，越高越好 |
| subject_consistency | DINO ViT-B/16，原版时序主体特征一致性 | 质量，越高越好；可能受少运动影响 |
| motion_smoothness | AMT-S，原版运动平滑性 | 质量，越高越好；不能代替运动幅度 |
| dynamic_degree | RAFT-Things，逐视频动态标记及组内比例 | **运动变化监测（护栏），不按越高越好排名，不设自动通过/淘汰阈值** |
| aesthetic_quality | OpenAI CLIP ViT-L/14 224px + LAION V1 linear；原值 /10 一次 | 质量，越高越好 |
| overall_consistency | ViCLIP InternVid-10M-FLT，八段中间帧与原始任务文本的相似度；沿用并记录原生 32-token 限制 | 视频文本语义代理，**不是综合质量总分** |

| 活动任务库 | prompt 数 | 每条视频唯一评测集合 |
|---|---:|---|
| 主体与运动库 `motion`，来自官方 subject_consistency 集合 | 16 | subject_consistency、motion_smoothness、dynamic_degree |
| 成像与语义库 `visual_semantic`，来自官方 overall_consistency 集合 | 16 | imaging_quality、aesthetic_quality、overall_consistency |

每条视频只计算并保留所属库三项，项目层面保留六指标。不计算跨库诊断分数，未分配项记为“不适用”，不视为缺失。保留固定预处理、帧选择、模型 `eval()` 与同一解码口径，禁止运行时隐式下载。三项应测指标须全部有效；失败只按原配置重试并留记录，不能换 seed 补样、均值填补或静默删去难样本。

原始任务文本是 ViCLIP 评分锚点，公开扩写只作生成条件。完整 165 条的 T5 生成文本均不超 512 tokens；活动面板的 `VB0724` 原文超出 ViCLIP 32-token 上限，沿用官方原生截断，保存完整原文、未截断长度、标记和有效 token 哈希，不静默改写或删样。预先另报未截断语义子集作为敏感性描述，不替代主统计。详见 V2 专项协议。

令 \(\mathcal P_{g(m)}\) 为指标 m 对应库的 16 条 prompt，\(d_{mis}=q_m(T)_{is}-q_m(S)_{is}\)。先在同 prompt 内平均 3 seeds，再在该库等权平均：

\[
\bar d_{mi}=\frac13\sum_{s=1}^3d_{mis},\qquad
\widehat\Delta_m=\frac1{16}\sum_{i\in\mathcal P_{g(m)}}\bar d_{mi}.
\]

以 **prompt 为重采样簇**，在每个指标对应库内有放回抽样，带上该 prompt 的全部 seeds/方法；10,000 次、NumPy PCG64 seed=20261017、float64、线性分位数。每项有 96 个视频值、48 个 seed 配对、16 个 prompt 簇；报告逐视频值、逐 prompt 差与区间，不把 seeds 或帧当成独立 prompts，也不混合两库均值。bootstrap 是小样本近似推断，不宣称精确覆盖率。

预设五项质量比较族；使用 Bonferroni 的逐项 **99% 双侧区间**（族 α=0.05）。资格要求至少一项的教师−学生下界 >0、全数据完整且不存在评分饱和/解码异常等失真。若多项通过，按预设顺序 `imaging_quality → subject_consistency → overall_consistency → aesthetic_quality → motion_smoothness` 提议目标，用户确认其科学意义；不选点估计最大的维度再假装预设。

`dynamic_degree` 在主体与运动库完整报告逐视频标记、组内动态比例、配对比例差、动态→静态与静态→动态配对数，以及 95% prompt 簇区间；不设置下降幅度的自动拒绝或容许界。若该库质量提升同时伴随运动减少，并列呈现取舍。由于成像与语义库不评该项，不能据另一库的动态结果断言其运动也不受影响。**各指标升降幅度是否可接受、是否具有实际研究价值，由研究者根据完整数据决定。**最低有意义的质量增益无跨指标通用阈值，不捏造；资格是统计候选与用户审查门槛。五项均未通过就停下，报告无优势证据或不确定，不宣称等效或全面排序。

## 5. 通过资格后的迁移候选与数学边界

以下为 **S3 的待验收候选**，不是已实现算法或本轮执行任务。论文的核心链条是“学生干净端点 → 重新加噪 → 原教师去噪端点 → 端点融合 → 交回学生采样器”。论文采用 VP 参数化；Eq. (10) 的噪声引导近似不是所有采样器的精确公式，视频/Flow 有效性须重新验证。

采用噪声 1 → 数据 0 的线性路径，σ>0：

\[
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,\quad v=\epsilon-z_0,
\quad \hat z_0^S=z_\sigma-\sigma v_S,
\quad z_{\rm next}=z_\sigma+(\sigma_{\rm next}-\sigma)v_S.
\]

首个学生步查询一次教师，初始候选 r=σ_next=0.9375；独立 ξ∼N(0,I)：

\[
y_r=(1-r)\hat z_0^S+r\xi,\quad
\hat z_0^T=y_r-rv_T^{\mathrm{CFG}=5}(y_r,r,c),\quad
\hat z_0^c=(1-\lambda)\hat z_0^S+\lambda\hat z_0^T.
\]

在固定当前状态的约束下，把融合端点映回速度并调用同一确定性更新：

\[
v_c=\frac{z_\sigma-\hat z_0^c}{\sigma},\qquad
z_{\rm next}^{c}=z_{\rm next}
+\lambda\frac{\sigma-\sigma_{\rm next}}{\sigma}(\hat z_0^T-\hat z_0^S).
\]

因此首步注入系数为 **0.0625λ**（λ=0.1 时为 0.00625），不能漏掉步长因子。端点恒等式 `ẑ₀ˢ−ẑ₀ᵀ = r[v_T−(ξ−ẑ₀ˢ)]` 仅说明残差关系，不能忽略教师 Jacobian 就宣称精确 SDS 梯度或必然下降。一般时刻“固定当前状态改速度”和“保留学生噪声方向仅替换端点”并不等价；首步 σ=1 下两者在本线性路径中一致，本批不据此外推后续介入。

实际教师输入必须用其原生 timestep 编码/输入缩放表示同一 r，核对 VAE latent 归一化与条件语义，不能直接把整数 750 传入教师并当成 r=0.9375。上述等式是实数/FP32 算术层的候选推导，尚未包含 S1 已发现的学生 BF16 持久状态回写。S3 须记录 FP32 候选增量及 BF16 回写后的实际增量，检查舍入是否抑制或消除纠偏；不能把 FP32 中非零的改变量当成实际注入证据。λ=0 必须恢复冻结的原生学生状态轨迹。若拟把学生状态改为 FP32，必须另立相应基线、明确比较口径并在 S3 前确认，不能将这种数值改变混入教师纠偏收益。

S3 不新增开发 prompt。按已冻结排序，从同一面板各库取前两条：`VB0309`、`VB0273`、`VB0761`、`VB0713`，各用前两个既定 seeds `1529987743`、`1993624635`；共 4 prompts×2 seeds。在此固定子集比较 λ∈{0.02,0.05,0.1}，固定第一步、一次查询与 r，不扫描时机/重加噪强度。每条仍只评所属库三项；目标增益仅在目标指标所属库计算，因此每个目标实际只有 2 个开发 prompt，不作精确性或泛化保证。全部候选的对应指标原值、配对变化及运动库动态变化完整记录，不按动态比例阈值淘汰候选。

按已选目标的平均配对增益给出推荐，相同得分优先较小 λ；研究者综合完整数据决定候选及是否进入 S4，并在完整面板运行前冻结。若候选没有正向平均增益，照实报告，不自行换 prompt、扩 seeds 或扩大网格追分。S3 是同面板子集调参，其选择信息必须进入最终报告；S4 不能因为覆盖更多原面板条目而被称为独立确认。

必做控制：λ=0 与“教师照常查询但丢弃结果”应与基线学生 S4 的状态/输出一致；独立 RNG 保证查询不扰动学生；非零 λ 保存端点差、实际 latent 增量及有限数检查。控制也使用上述既定子集。一次 CFG 教师查询通常包含条件/无条件两次网络前向，必须同时报告查询数、分支前向数和耗时。四步学生没有经过验证的额外第五步，不把任意 S5/S8 当成天然合法的等预算对照。

## 6. 同面板效果验证、速度与资源

S4 沿用资格阶段完整的 32 条 prompt×同三个 seeds，原始评分文本、公开生成扩写、初始噪声和所属指标映射均不变。T50/S4/S4_T1 共 288 条比较视频，每条三项，共 864 个指标值；逐指标报告 T50−S4（资格基线引用）、S4_T1−S4（纠偏收益）及 T50−S4_T1（剩余差距）。不另抽样、不替换困难条目、不为迁移增加独立 prompt。

基线复用必须核对模型身份、生成/评分代码与设置、文本/条件/初始噪声哈希、成片哈希和分数完整性。匹配时复用资格阶段的 192 条 T50/S4 成片与 576 个分数，最终比较只需补齐至 96 条 S4_T1 成片、288 个分数；S3 中选中候选若所有配置/输入/实现相同，也可复用其成片。S3 控制、未选中候选、预热和失败重试单独记成本，不将“补齐至 96 条”误写为迁移阶段总计算量。发现实现或设置实质变化时先分析影响，不能冒充同一基线直接拼接。

质量与速度分别报告点估计和按 prompt 簇的描述性 95% 区间，质量仍限各自 16 条库，重采样带上同一 prompt 的全部 seeds/方法。目标和 λ 已使用本面板选择，常规 bootstrap 没有计入这一选择过程：**结果是同面板上的迁移效果验证，不是独立确认；区间不提供选择后的标称覆盖率保证，也不能证明未见 prompt 的泛化。**复用 T50/S4 不构成第二次独立教师优势证据。即使描述性区间均支持正向改善与速度优势，也须注明这一范围，不能表述为独立确认性的联合显著结论。

完整报告项目六指标在各自库上的逐样本结果、组内汇总、配对变化与区间，以及热态速度比 `median(T50)/median(S4_T1)`、学生额外开销及其 prompt 簇区间。速度同时按两个库分别报告，整体等权面板汇总只能作为总体耗时描述。主体与运动库报告 `S4_T1−S4` 动态比例差和动态转移配对数，**不设置数值容许界，也不据此自动判定成功、失败或淘汰候选**；成像与语义库不额外计算运动指标，不据此声称其运动保持。增益、其他质量退化及实际取舍全部呈现，由研究者决定研究价值与可行性，不用一个胜出维度声称全面胜出。

计时在同型号、同资源数的空闲 H200 上进行；预热也使用活动面板中的固定输入，重复预热不计正式样本。随机/平衡组别顺序，记录同一时间块负载。视频和质量分数可复用，不代表历史耗时一定可直接比较：模型驻留、加载/offload、软硬件或负载条件不匹配时，在同一冻结输入上做匹配计时核查，不新增 prompt。主口径为热态在线“去噪+VAE”，CUDA 同步计时；另报文本编码、模型载入、CPU/GPU 转移、视频编码与冷启动。纠偏教师调用/换入成本计入在线时间；若不能共同常驻，启用真实 offload 后重新冻结一致的资源/计时口径，不能隐藏代价。报告中位数、IQR、逐样本时间、峰值显存、学生额外开销及有效前向数，不能仅按 **50/4** 去噪步数比例推算加速。Teacher 启用 CFG，每步通常需要条件与无条件两次预测；Student 关闭 CFG，单步计算量与教师不同。加速结论以统一计时口径下的**真实 GPU 推理实测耗时**为准；匹配条件不足时只作参考，不宣称已确认速度收益。

用户已确认可直接使用空闲卡；无需每次 Slurm 申请。启动前重新检查 GPU 进程/显存/利用率，明确 `CUDA_VISIBLE_DEVICES`，先单卡单任务；该变量不构成独占预留，不抢占或终止他人任务。S1 已完成有限共同驻留探测，新批次仍按真实负载与激活核算准入。CPU 初始最多 4 workers，保留其他任务 RAM。现有生成及六项评测资源已齐，本次配置调整不新增权重或安装；视频/latent/归档另按阶段预算。每阶段前检查实际剩余空间，不以旧重复下载预算代替增量核算；不得清理他人文件。

## 7. 目录、可复现产物与本轮边界

远端根：`/data/chenjiayu/changzihe/DistillPP_Validation`。本地镜像：`D:\Research\PK_8H200\DistillPP_Validation`。

| 位置 | 内容 |
|---|---|
| `Project_Plan/` | 本计划、冻结设置/prompt/资源清单、下载与软件准备指令 |
| `Project_Progress/Total_Progress.md`、`Stage_Progress/` | 全局状态、各阶段配置/失败/验收/用户决定 |
| `src/`、`configs/`、`runs/<batch>/` | 执行代码、配置、原始输出/日志；S1 已验收，S2 已实现 |
| `resources/evaluation/` | **本批评测权重/词表**，按用户最新指定放在实验目录 |
| `/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers/`、`Wan2.1-T2V-14B-StepDistill-CfgDistill/` | 直接只读复用已有教师、学生及共同组件；不新增另一种封装目录 |
| `Evaluation_Videos/<batch>/<arm>/<prompt>_seed<seed>.mp4` | 阶段结束按组复制一份完整视频，核对哈希并保存索引；不覆盖旧批次 |
| `Result/<stage>/` | 逐视频所属库三项指标 CSV（项目共六项）、配对统计 JSON/CSV、速度与护栏、小图及分析 Markdown |

共享资源复用、原版 PT 加载和六指标已经完成 S1 真实验收；误下载新目录已由用户按精确清单清理，VAE 原件保留。本批没有新增生成权重、转换权重副本、依赖安装或更改他人环境。原始失败记录和重试均保留在 `runs/s1_20261016/`，两条有效视频已按 T50/S4 组归档。

本地同步只含代码/脚本、配置/文档、核心统计和必要小图/摘要；不含模型、评测权重、大量视频/latent、Conda 环境或缓存。每组结果保留全部配对条件与失败记录，不能只挑好视频归档。

资源下载和软件安装由用户手动完成。Codex 已完成 S1、旧 S2，以及 S2-R 的生成、对应指标评测、配对统计、真实耗时、视频归档和完成复核。**旧 S2 与本轮 S2-R 均为五质量维度 0/5 建立教师优势；S2-R 执行验收成功，资格未通过。**本轮使用独立适配入口，保留活动 32 条、每库三指标、分离生成/评分文本及面板指纹核验；冻结科学输入和原始结果不变。不制作 PDF 或图像，核心数据与分析以 CSV、JSON、Markdown 交付。准确进度见 `../Project_Progress/Total_Progress.md`，当前分析见 `../Result/S2R_20261017/REVIEW_AND_INTERPRETATION.md`；S3 未开始，未实施教师纠偏、训练或迁移实验。

## 8. 主要原始来源

- [Distillation++ v1 论文](https://arxiv.org/abs/2412.08871v1)：本地 PDF 与公式阅读记录已核对。
- [原始教师的官方 Diffusers 文件](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/38ec498cb3208fb688890f8cc7e94ede2cbd7f68)、[固定原生教师实现](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py)、[官方参数名转换](https://github.com/huggingface/diffusers/blob/v0.35.1/scripts/convert_wan_to_diffusers.py)。
- [LightX2V 四步配置](https://github.com/ModelTC/LightX2V/blob/cc9429192971815afff8368f74bd10adcc6fed48/configs/distill/wan21/wan_t2v_distill_model_4step_cfg.json)、[Flow scheduler](https://github.com/ModelTC/LightX2V/blob/cc9429192971815afff8368f74bd10adcc6fed48/lightx2v/models/schedulers/wan/step_distill/scheduler.py)、[本批原版学生文件](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/tree/b9baaa9a0c29226dea39043db647d1ced950bbea)、[PT 加载入口](https://github.com/ModelTC/LightX2V/blob/cc9429192971815afff8368f74bd10adcc6fed48/lightx2v/models/networks/wan/distill_model.py)。
- [模型双向/蒸馏说明](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v)、[所引用双向训练配置](https://github.com/GoatWu/Self-Forcing-Plus/blob/67b8016498b5fc3b7ce753df7c8c97820acd6baa/configs/self_forcing_14b_dmd.yaml)。
- [固定 VBench 实现](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490)。资源下载镜像只改变传输端点，不改变固定版本与校验值。
