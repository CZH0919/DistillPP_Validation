# HunyuanVideo → AccVideo 720p：资源与环境准备记录

本页只用于新 HYA 计划。旧 Wan/LightX2V 资源保持原样，旧说明保存在 `history/DOWNLOAD_AND_SETUP_WAN_LIGHTX2V_THROUGH_S2R.md`。**H0 资源下载由用户完成，40 项文件已完整校验；H1 必要依赖安装、两条冒烟及五指标验收均已完成。当前无需再下载或安装资源，正式 H2 尚未启动。**下述历史下载命令仅保留作缺项维护参考，适用于 VS Code Remote SSH 的远端 Bash。当前验收结论见 `../Result/HYA_H1_20261019/REPORT.md`。

## 1. H0 盘点依据与已完成的新增资源

2026-10-19 有界核查共享模型根目录和本人项目资源：27 个现有 HunyuanVideo 运行文件及 DINO、RAFT 的完整哈希通过。搜索现有模型/评测/本项目缓存文件名未发现 AccVideo、CLIP ViT-B/32、Tag2Text、GRiT 或 bert-base-uncased；该结论针对核查的已授权资源位置，不表示读取了全服务器所有用户目录。

| 资源 | 处理 | 路径/增量 |
|---|---|---|
| HunyuanVideo 教师、LLM、CLIP 文本编码器、tokenizer、VAE | 只读复用；不下载原生 PT 第二份 | `/data/chenjiayu/models/HunyuanVideo/` |
| DINO、RAFT | 只读复用，完整哈希通过 | 项目 `resources/evaluation/dino_model/`、`raft_model/` |
| AccVideo 五步 DiT | 仅下载该 PT 和三份许可/说明文件 | `/data/chenjiayu/models/AccVideo/`，约 23.881 GiB |
| CLIP ViT-B/32 | 下载；已有 ViT-L/14 不能代替 | `resources/evaluation/clip_model/ViT-B-32.pt`，约 0.330 GiB |
| Tag2Text | 下载 | `resources/evaluation/caption_model/tag2text_swin_14m.pth`，约 4.171 GiB |
| GRiT | 下载；已有 GroundingDINO/SAM 不能代替 | `resources/evaluation/grit_model/grit_b_densecap_objectdet.pth`，约 0.389 GiB |
| BERT 辅助模型与 tokenizer | 下载单份 safetensors 及 3 个小配置；不用同时下载 bin | `resources/evaluation/bert-base-uncased/`，约 0.410 GiB |

上述资源现已全部到位并校验通过。模型/评测权重及其小文件实际新增合计 **31,333,237,562 字节，约 29.181 GiB**；不含软件依赖/输出。盘点时空闲约 892.33 GiB，下载脚本会重新计算缺失量并检查额外 50 GiB 余量。只使用一份可续传临时文件，校验后原子发布，不为安装另复制一套模型缓存。

固定清单为 `HUNYUAN_ACCVIDEO_RESOURCES.json`：每项含仓库、不可变 commit、文件名、长度和 SHA-256/Git blob SHA-1。CLIP 与 OpenAI 原始 URL 哈希一致；Tag2Text 与作者原始 Space 的 LFS 哈希一致；GRiT 来自 OpenGVLab；镜像仅用于传输，不能改变内容身份。

## 2. 远端 Bash 命令

```bash
conda activate changzihe
cd /data/chenjiayu/changzihe/DistillPP_Validation

# 先显示现在还缺什么；只读，不开始下载。
python -B Project_Plan/download_hunyuan_accvideo_resources.py --inventory-only

# 仅在重新盘点确认缺项、且获准补齐时使用；本轮已下载完成。
python -B Project_Plan/download_hunyuan_accvideo_resources.py \
  --download --endpoint https://hf-mirror.com

# 完成后独立复核。成功末尾应输出 ALL_SELECTED_ASSETS_VERIFIED。
python -B Project_Plan/download_hunyuan_accvideo_resources.py --verify-only
```

下载可以分开操作，加 `--group student` 或 `--group evaluation`；student 会先检查已有 Hunyuan 复用文件，不下载这些文件。默认 `all` 即全部。本脚本只导入 Python 标准库、调用系统 curl，不初始化 CUDA、不安装包。

下载前先检查所有已有目标的哈希；不符时立即停，不覆盖、不自动删文件或重下。复用文件缺失时也停止，不把它当作下载第二份的理由。新增共享写入只允许 `AccVideo/accvideo-t2v-5-steps/diffusion_pytorch_model.pt`、`LICENSE`、`Notice`、`README.md`。如新增了另一 AccVideo 目录/权重或出现路径符号链接，脚本先停，重新核查复用关系。完整下载后保留发布文件，不留下第二份 `.part`。

网络中断保留 `.changzihe_hya.part`，重新运行原命令继续；每个网络尝试有连接/总时限和有限重试。不要删除整个共享模型目录。若镜像持续失败，保留实际 curl 退出码；可以**明确手动改用同一仓库/commit 的官方域名**：

```bash
python -B Project_Plan/download_hunyuan_accvideo_resources.py \
  --download --endpoint https://huggingface.co
```

不关闭 TLS 验证、不更改 DNS/SSH 配置、不改 IP 绕过。哈希失败、无续传支持、403 或磁盘不足时暂停相关下载并返回错误，不能将 HTML 错误页/截断文件当作模型。许可和使用条款随 AccVideo 同步到模型目录；HF README 旧示例仍可能写 50 步，本轮学生必须按固定协议明确 5 步。

## 3. H1 已完成的软件依赖与实现验收

生成环境 `changzihe` 保持原有 torch 2.8.0+cu128、Transformers 4.51.3、Diffusers 0.35.1、FlashAttention 2.8.3，没有安装或降级包。两组使用固定原生 Hunyuan DiT/Flow 调度器；教师现有 Diffusers 分片仅在内存中无损重组，学生原生 PT 严格加载，不保存另一套大权重。实际精度、时间数组、公共组件及编码配置见 `HYA_RUNTIME_CONTRACT.json`。

评测环境 `changzihe_eval` 已增加 21 个锁定依赖和 Detectron2（合计 22 个包），原有 110 个包版本均未改变。Detectron2 来自 commit `1e3e13bbf607b54f62205c4c33922521822fb298`，本机以 GCC 13.3、CUDA 12.8、SM90 编译并通过 GRiT 实际评分。新增包入口为 `requirements_hya_evaluation.txt`，精确下载 URL/哈希、解析结果、安装日志和编译 wheel 身份保存在 `runs/hya_h1_20261019/dependencies/`，不要照搬全量旧 requirements 或重复运行旧 Wan 安装脚本。

Tag2Text/GRiT 的 BERT 调用已在本人适配层指向本地辅助资源；它们和其余三指标均完成离线评分，网络访问尝试为 0。Tag2Text 的非学习位置缓冲由原生实现生成；GRiT 缺失的 106 个别名键逐一证明与已加载参数共享同一对象，没有跳过真正缺失的学习参数。

两个环境的 `pip check` 仍返回已有 Decord 标签告警，退出码 1。实际 wheel 元数据声明 `cp36-cp36m-manylinux2010_x86_64`，与当前 Python 3.10 标签不符；没有改写标签掩盖告警。两条 129 帧视频已通过 Decord/OpenCV 双解码及五指标实际运行，当前使用路径验收通过；不能声称 pip check 已退出 0。

H1 已完成并按组归档视频、同步结果。下一步仅在用户授权后开始 S2-HYA/H2 正式教师资格实验；H1 冒烟不构成教师优势证据，也不授权进入 S3。
