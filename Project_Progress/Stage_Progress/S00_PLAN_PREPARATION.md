# S00：计划与资源指令

> 历史准备记录。第 6 项旧清单没有充分核查已有教师/学生的加载兼容性，已撤销。当前为清单 v2：24 项生成文件全部只读复用、完整哈希通过，新增生成模型下载为零；仅余八项评测资源和评测软件。执行以 `S00_DOWNLOAD_REMEDIATION.md` 和新版 `DOWNLOAD_AND_SETUP.md` 为准。

日期：2026-10-16。范围：计划、文档、小型源码/脚本与有界只读核查；没有模型推理/评测/训练。

1. 主环境已核对：`/data/chenjiayu/miniconda3/envs/changzihe`，Python 3.10.22，Torch 2.8.0+cu128，FA2 2.8.3，Transformers 4.51.3。六指标所选 PyIQA 0.1.13 依赖 Transformers 4.37.2，故提供统一 Conda 父目录内的独立评测环境指令，尚未创建。
2. 项目根、Project_Plan、Project_Progress 的 realpath 与授权路径一致，chenjiayu:dasys-h200、755；共享 models 根相同。新教师/学生目标仓库名目录、changzihe_eval 当次核查不存在。
3. 现有共享 T5/VAE 缓存 metadata 的校验值匹配目标：T5 `7cace0da…`、VAE `38071ab5…`。这不是实际文件校验；手动下载脚本先校验实际字节后才能复用。没有改动旧共享资源。
4. h200 HEAD-only：教师 config HTTP 200/1.280s，学生 BF16 HTTP 200/4.628s，MUSIQ HTTP 200/1.702s；三个 curl 退出码 0，SSH 退出码 0。没有下载响应正文或测试吞吐，不能推广成全站/完整传输保证。
5. 固定官方 Wan/LightX2V 源码在本地导出；VBench/DINO/CLIP 复用本地已固定的小型源码归档，重新核验包 SHA。所有交付包均不含生成/评测权重或视频，服务器不必拉取 GitHub。
6. 已提供 25 文件资源清单、16-prompt 面板、固定基线参数、离线源码及软件/下载脚本。生成 90.918 GiB，评测 2.915 GiB，可复用组件 11.054 GiB；软件/缓存另外预算。
7. 本地检查包括脚本解析、JSON/哈希/源码包成员/路径核对、计划下载清单干运行；远端只做 Bash 语法及同步文件 SHA 核对。未把准备脚本的“语法通过”报告为安装或模型验收通过。

下一步：用户按 `Project_Plan/DOWNLOAD_AND_SETUP.md` 手动准备资源，返回准备日志/资源报告；再完成 S1。S2 通过且用户确认后才能实施迁移。
