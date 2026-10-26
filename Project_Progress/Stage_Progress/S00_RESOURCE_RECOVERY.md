# S0 资源准备中断与恢复

> 历史记录：下述继续模型下载的建议已撤销。当前处置以 `S00_DOWNLOAD_REMEDIATION.md` 和 `Project_Plan/DOWNLOAD_AND_SETUP.md` 为准；生成资源只读复用，不能恢复旧清单下载。

用户提供的终端日志显示：

- LightX2V 0.5.0 与生成侧声明依赖安装成功，`SOFTWARE_EXIT_CODE=0`；Torch 保持 2.8.0+cu128，CUDA 未初始化。decord 的已知 wheel-tag 告警被记录，未将 pip check 描述为无告警通过。
- pip 解析将 protobuf 从 5.29.5 调整为 6.33.6；完整软件状态以本次生成阶段的 pip report/freeze 为准，不仍引用安装前的完整环境快照。
- 教师 LICENSE/README 下载完成，随后在共享 VAE 文件的 SHA-256 读取循环中出现 `^C` / `KeyboardInterrupt`；日志中该步骤没有 HF HTTP、TLS 或网络超时报错。
- 后续直接运行 `condabin/conda activate` 触发 shell 初始化提示。恢复命令通过 `source .../etc/profile.d/conda.sh` 使用 shell 函数，不需要执行 `conda init`。

有界远端核对确认：目标目录仅有上述两个普通文件；共享 VAE/T5 的大小与清单一致，但本轮未读取其全部内容验证哈希；资源下载锁空闲；原脚本因未捕获 KeyboardInterrupt，没有生成本次 resource_check 报告。不能将“尺寸匹配”写成“哈希通过”。

修复下载工具：大文件哈希开始时显示路径/大小，每 5 秒提供独立进度提示；即使读取等待，也可看到无进展时长。中断时保存报告并返回 130，保留已完成文件和下载临时文件。原模型版本、哈希、目标路径、校验要求和网络端点不变。

后续由用户执行：模型下载 → 评测资源下载 → 评测软件准备。无需重装生成环境。本次只修改下载工具与项目说明，不代替用户下载权重，不安装软件或运行 GPU 工作。
