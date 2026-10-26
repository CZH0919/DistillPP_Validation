# H2 scene 专项核查

**结论：未发现本轮 scene 的 prompt 选用、官方扩写与标签对应、抽帧、逐视频计分或汇总错误。32 条视频的 512 个已保存抽样帧描述，经官方函数和独立逐词逻辑复算，全部与原分数一致。教师 41.40625、学生 53.515625（原分×100）无需更正。**

核查日期：2026-10-20。范围为远端只读核对、官方固定版本源码对照及本地 CPU 复算；未重新生成视频、运行 Tag2Text GPU 推理、安装依赖或进入 S3。

## 1. 论文数字没有抄错，但比较范围不同

| 比较 | 教师 HunyuanVideo 720p | AccVideo 720p | 教师−学生 |
|---|---:|---:|---:|
| AccVideo 论文表 5 | 53.88 | 50.75 | +3.13 |
| 本轮 scene 16×1 | 41.40625 | 53.515625 | −12.109375 |
| 本轮减论文 | −12.47375 | +2.765625 | — |

论文在评测设置中说明使用 946 条提示词、每条采样 5 次；表 2 说明 HunyuanVideo-720P 的数值取自官方 VBench 榜单，AccVideo-720P 由作者测试。因此，这两项参考分数不是作者在我们这 16 条 prompt、固定 seed 下的一次配对实验结果。[AccVideo 论文 §5、表 2/5](https://arxiv.org/html/2503.19462v1)

VBench 官方规范的 scene 库有 86 条、每条采样 5 个视频，即完整标准面板为 430 条/模型；本轮是预先从该库选出的 **16 条×1 seed，16 条/模型**。整个项目的 64×1 并不意味着 scene 有 64 个样本。[VBench prompt 与采样规范](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/prompts/README.md)

这些事实说明两个实验的均值不应预期逐值一致；它们本身尚不能证明与论文的全部差异究竟有多少来自 prompt 子集、seed、环境或具体生成结果。核查所用公开材料未提供与本轮对应的论文逐 prompt/seed 分数，故不臆测具体占比。

## 2. prompt 和标签核查

1. 从官方固定 commit 重新取得 `all_dimension.txt`、`hunyuan_all_dimension.txt` 和 `VBench_full_info.json`，字节哈希与本轮使用文件一致。946 行原始 prompt、扩写行与元数据均对齐。
2. scene 的 86 条原始文本与官方 `prompts_per_dimension/scene.txt` 完全一致。按冻结的 SHA-256 排序规则重新选择，得到相同的 16 条，未发现越界选库、错行或按结果换 prompt。
3. 每条实际生成记录中的文本与对应官方 Hunyuan 扩写逐字一致；T50/S5 使用同文、同 seed、同输入。LLM 含模板最长 123 tokens（上限 351），CLIP 最长 28（上限 77），不存在这批 scene 文本被截断的问题。
4. 评测使用原始任务的官方场景标签；对照 `load_dimension_info` 得到的 `auxiliary_info`，与 H2 传入字典一致。没有把扩写全文当作标签，也没有把前期 garden 冒烟标签误用于正式样本。

AccVideo 官方 README 明确说明其 Hunyuan 版本使用 VBench 提供的 Hunyuan 扩写 prompt；我们的文本来源符合这一说明。[AccVideo 固定版本 README](https://github.com/aejion/AccVideo/blob/a2752d2f6990a1363713f1d57907bd28e3ad365c/README.md#-vbench-results)

官方标签和扩写中的原有细节也被保留。例如原始标签为 `carrousel`，扩写使用另一种拼写 `carousel`；`restaurant` 的官方扩写描述备餐和服务场所，没有直接包含这个场景词。它们可能使生成或关键词匹配更敏感，但不是我们错配或自行改写文本造成的。

## 3. scene 实际如何计分

当前固定的官方实现：

1. 从视频均匀分段取 16 个中间帧；129 帧视频索引为 `3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,124`。
2. 使用官方 384×384 解码/变换、ImageNet 归一化及 Tag2Text Swin-B 生成每帧的短文本描述。
3. 将场景标签按空格拆词，要求**每个词均以字面子串形式出现在该帧描述中**，该帧才计 1；没有同义词扩展或语义相似度判断。
4. 每条视频分数为命中帧数/16；本轮每条抽帧数相同，所以官方帧汇总与等权视频均值一致。

这意味着 scene 更接近“自动描述中的场景关键词命中率”，不能直接等同于完整场景理解或人工整体观感。[官方 scene.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/scene.py)

已核对运行中的 `scene.py`、`utils.py` 与官方固定版本字节相同；使用官方 Tag2Text 权重和构造设置，加载记录没有缺失的学习参数或意外键。离线适配只处理资源路径和兼容加载，没有修改 scene 匹配规则。资源大小/mtime 与启动时完整哈希凭据一致。本轮没有重新运行 caption 模型，也不将上述核查表述为与论文运行环境的逐位等价证明。

## 4. 原始逐帧记录解释了哪些低分

以下为保留的自动 caption 及按官方规则计算的结果，用于解释计分，不是新增人工视频质量评分。

| 场景 | 教师命中 | 学生命中 | 从保存的 caption 可以确认的计分原因 |
|---|---:|---:|---|
| classroom | 0/16 | 16/16 | 教师描述为老师给学生上课，未出现 `classroom`；学生描述中含该词 |
| bathroom | 1/16 | 16/16 | 教师多数帧描述了 toilet、sink、shower，却未出现 `bathroom` |
| forest road | 16/16 | 0/16 | 学生描述用了 `woods`，不含 `forest`，即使含 road 也不能命中 |
| indoor library | 0/16 | 0/16 | 学生 caption 含 library/bookshelves，但缺少官方标签要求的 `indoor` |
| carrousel | 0/16 | 0/16 | 双方 caption 使用 merry go round，没有出现官方标签字串 |

这些例子显示关键词漏计对双方都可能发生。它们能解释**为什么既有 caption 得到这些分数**，并不能仅凭 caption 判定相应视频在视觉上一定正确或错误。

本轮差值的较大贡献项是 classroom、bathroom、windmill：分别使总体教师−学生均值变化 −6.25、−5.859375、−4.6875 分；这是分解现有 16 条均值，不是删除这些样本后的新评测。windmill 的教师 caption 多为 water tower / observation tower / smoke stack，可能涉及生成内容或识别问题；本次不据此单独认定是哪一环节。

## 5. 全部 scene 样本与复算

下表均为每条视频 16 帧中的命中数；完整生成文本、官方行号及对应关系见 `PROMPT_AND_SCORE_AUDIT.json`，全部 512 条帧描述及缺失关键词见 `FRAME_CAPTION_AUDIT.json`。

| ID 后缀 | 官方场景标签 | 教师命中/16 | 学生命中/16 |
|---|---|---:|---:|
| 01 | lighthouse | 16 | 16 |
| 02 | industrial area | 0 | 0 |
| 03 | mansion | 16 | 16 |
| 04 | carrousel | 0 | 0 |
| 05 | office | 10 | 16 |
| 06 | basement | 3 | 0 |
| 07 | classroom | 0 | 16 |
| 08 | glacier | 16 | 13 |
| 09 | iceberg | 16 | 16 |
| 10 | indoor library | 0 | 0 |
| 11 | restaurant | 0 | 0 |
| 12 | forest road | 16 | 0 |
| 13 | windmill | 0 | 12 |
| 14 | bathroom | 1 | 16 |
| 15 | cafeteria | 12 | 10 |
| 16 | supermarket | 0 | 6 |
| 合计 | 16 条视频/模型 | **106/256** | **137/256** |

官方 `check_generate` 及独立逐词判定对 512 个帧描述的复算一致，32 个视频分数和组均值最大差为 **0**。零分均可追溯到实际描述和匹配判定，不是失败补零、漏评分或组别错置。

**处理结果：保留 scene 原分数，当前两项教师均值胜出的资格结论也不变。**本次差异没有显示出需要修正 prompt 接线或计分代码的证据；后续使用 scene 时，按其场景关键词命中定义解释，并与整体场景语义表现作区分。

核查文件：`VALIDATION.json`、`PROMPT_AND_SCORE_AUDIT.json`、`FRAME_CAPTION_AUDIT.json`；官方源文件及下载/远端核对凭据在 `reference/`。复算脚本为 `../../../tools/hya_h2_scene_audit_20261020.py`。本次新增资料由 `DELIVERY_SHA256.json` 记录，原始 H2 数据与完成标记保持不变。
