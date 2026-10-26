# S0：已有模型复用核查

> 本页保留初查事实；后续完整核验已完成：24 项生成文件哈希通过，1,095 个键/全部形状映射通过，清单改为复用专用 v2。当前结论及补救指令见 `S00_DOWNLOAD_REMEDIATION.md`。本页“尚未校验”“下载暂缓”等措辞仅描述初查时点。

用户要求避免重复下载。此前仅因封装不同而将另一套教师/学生文件列为新下载，必要性核查不充分；现更正为先核验和复用已有资源。

## 已核实的事实

| 资源 | 服务器已有路径 | 核查结论 |
|---|---|---|
| 教师 | `/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers` | 官方原始 Wan2.1-T2V-14B 的 Diffusers 封装；transformer 有 12 个分片，抽查文件头为 F32，配置为 40 层、40 heads、每 head 128 维、16 latent channels |
| 学生 | `/data/chenjiayu/models/Wan2.1-T2V-14B-StepDistill-CfgDistill/distill_model.pt` | 28,577,333,801 bytes；模型说明明确四步、无推理 CFG、双向蒸馏；只读 ZIP/pickle 元数据得到 BF16 storage 和原生 Wan 参数名，没有反序列化或加载张量 |
| VAE | 同一学生目录的 `Wan2.1_VAE.pth` | 新教师目录已有指向此文件的软链接，不是复制一份权重；下载脚本设计为校验成功后才创建链接，本轮未重新读取全部字节 |
| T5 | 同一学生目录的 `models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 bytes；下载清单已有先校验后复用规则，不应重复拉取 |

共享目录缓存 metadata 与官方 HF 固定 revision 的元数据相符：

- 教师仓库 `Wan-AI/Wan2.1-T2V-14B-Diffusers`，revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`。抽查第一分片 4,863,171,808 bytes；官方 SHA-256 `63eb07f03808d8ec9b6bfc1b4f4968d7d1253f2569fec18fae114b4530554181`，与缓存记录一致。
- 学生仓库 `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`，revision `b9baaa9a0c29226dea39043db647d1ced950bbea`。官方 PT SHA-256 `13fec11400e094785dd9dae25a928a3acfb3cfc61168ab1f388344f06bef4db7`，与缓存记录一致。

以上是文件存在、格式与来源元数据证据，**不是本轮完整文件哈希通过或运行验收通过**，也没有证明 PT 与新 safetensors 逐张量相等。

## 源码事实与处置

固定 LightX2V commit `cc9429192971815afff8368f74bd10adcc6fed48` 的 `lightx2v/models/networks/wan/distill_model.py` 中，`WanDistillModel._load_ckpt` 明确引用上述原版学生仓库，并在 `model_path/distill_model.pt` 存在且没有 `dit_original_ckpt` / `transformer_model_path` 覆盖时，使用 `torch.load(..., map_location="cpu", weights_only=True)` 加载。

因此学生不需要仅为换成 safetensors 再下载；后续使用已有目录与该代码分支，须先完整哈希核验并在 S1 做实际加载验收。此处只读源码，未运行这段加载代码。

教师已有官方同模型的 Diffusers 文件，但原生 Wan loader 的参数名与分片组织不同。优先在加载层核验映射或兼容读取；不能直接改后缀、软链接成原生分片后宣称兼容，也不能仅因格式差异就默认必须再次下载约 57 GB。具体方式及最终资源清单在 S1 前明确，不偷偷改变研究模型或采样协议。

只读快照时，新原生教师的第一分片已完成，第二分片为 `.part`，新学生 safetensors 目录尚不存在。已建议用户按 Ctrl+C 暂停模型下载；没有停止进程、删除部分下载或修改共享权重。保留已完成和部分文件，待复用决策后再由用户决定是否清理。

资源准备说明与计划已标注暂缓原模型下载清单。没有修改正在运行的下载脚本或资源清单，也未将“建议暂停”写成进程已经停止。

来源：[固定 LightX2V PT 加载代码](https://github.com/ModelTC/LightX2V/blob/cc9429192971815afff8368f74bd10adcc6fed48/lightx2v/models/networks/wan/distill_model.py)、[原版学生](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill/blob/b9baaa9a0c29226dea39043db647d1ced950bbea/distill_model.pt)、[官方 Diffusers 教师](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/38ec498cb3208fb688890f8cc7e94ede2cbd7f68)。
