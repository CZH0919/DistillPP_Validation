# HYA 720p：资源核查与计划冻结

日期：2026-10-19。范围：用户已选定 HunyuanVideo → AccVideo 720p；本轮只核查资源、更新计划、准备手动下载指令。**H1 实现验收、H2 正式资格与 H3 迁移均未启动。**

已完成：

- 有界只读核查 h200 的授权根目录、共享模型命名和本项目评测文件；未读取他人工作文件或任务日志。共享盘检查时剩余约 892.33 GiB；这是快照，下载/计算前必须重查。
- 现有 HunyuanVideo 27 个运行文件完整哈希与固定社区 Diffusers 转换仓库匹配；DINO、RAFT 两项与冻结资源身份匹配，共 29 项、约 39.365 GiB 可以复用。没有下载原生教师的第二份 PT，没有生成转换副本。
- 已装 Diffusers 0.35.1 源码包含 Hunyuan 单文件映射入口。这是静态兼容依据；尚未执行 AccVideo 实际权重解包、加载或与原生实现的数值对照。
- 共享模型根与项目资源/软件缓存的有界文件名盘点未发现所需 AccVideo、CLIP ViT-B/32、Tag2Text、GRiT、BERT。固定缺失清单为 11 个文件（含许可/小配置），31,333,237,562 字节，约 29.181 GiB；模型只新增公共 `models/AccVideo/`，评测放项目 `resources/evaluation/`。
- 原计划已切换到 HYA；旧 Wan 计划保存为有明确历史用途的一份文档，旧基线 JSON、原始实验数据、源码和结果不改写。通用 CODEX_HANDOFF 不因项目切换增加流水记录或备份。
- 冻结 32 条主体/运动、16 条场景、16 条多物体官方 prompt，各 3 seeds。分别评 2/2/1 项；T50/S5 共 384 视频、672 个逐视频指标值。全部原文、Hunyuan 扩写、辅助标签、源行号、哈希选择排序均保存。
- CPU tokenizer 检查最长 LLM 原始文本 27 tokens、CLIP 含特殊符号 28 tokens，未达到上限。此项没有初始化 CUDA；完整生成模板仍待 H1。
- 下载脚本本地 6 项安全检查通过：正确文件不重下、错误文件不覆盖、复用文件缺失不补下、路径边界、重定向防护、已有目标不被发布覆盖。manifest、CSV/JSON、官方维度归属和源文件哈希检查通过。

仍待完成：用户手动下载；detectron2/GRiT 与 Tag2Text 新增软件依赖的兼容性准备；教师转换版/学生 PT 实际加载与原生采样语义核对；GPU 6 单卡 720p/129 帧的显存、运行和五指标离线验收。旧 VBench 安装说明涉及较旧 CUDA，不能声称本机 CUDA 12.8 的 detectron2 构建已经通过。

资格规则：四个预声明质量维度的 prompt 簇配对统计，报告名义 Bonferroni 校正 98.75% bootstrap 区间；动态单独作护栏并完整报告，不设自动容许阈值。论文数据支持候选模型，尚无本地 HYA 教师优势证据。后续是否推进由研究者确认；通过后仍使用同一固定 prompt/seed 面板。

入口：`Project_Plan/EXPERIMENT_PLAN.md`、`HUNYUAN_ACCVIDEO_PROTOCOL.json`、`HUNYUAN_ACCVIDEO_PROMPTS.md`、`HUNYUAN_ACCVIDEO_RESOURCES.json`、`DOWNLOAD_AND_SETUP.md`。审计数据为本目录 `HYA_RESOURCE_INVENTORY_20261019.json`、`HYA_PROMPT_TOKEN_CHECK_20261019.json`、`HYA_PLAN_STATIC_CHECK_20261019.json`。同步校验清单单独记载，不将“已同步”替代运行验收。
