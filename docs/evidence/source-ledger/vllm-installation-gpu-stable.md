# vLLM GPU Installation 与 Docker（官方文档）

Source: vLLM official documentation, “GPU Installation” and “Using Docker.”  
URL: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/  
URL: https://docs.vllm.ai/en/stable/deployment/docker/

Supports:

- vLLM 的 Python、GPU 后端和 wheel 兼容条件需要按目标版本核对。
- 官方文档提供隔离环境、预编译 wheel、源码安装和 `vllm/vllm-openai` 镜像路径。
- Docker 运行需要 GPU 设备、共享内存以及模型/cache 挂载等运行条件。

Does not support:

- 把书中某组 Python、PyTorch、CUDA 与 vLLM 版本永久当作兼容矩阵。
- 使用 `latest`、nightly 或未锁定源码分支作为可复现生产制品。
- 宣称单一安装方式适用于 CUDA、ROCm、XPU、Metal 和 Windows。

Owner: 第4章作者  
Purpose: 支撑可复现的 vLLM 安装与容器边界  
Evidence grade: A  
Assumptions: 出版时需记录核对日期；生产制品使用 tag 与 digest 双重锁定  
Open questions: 是否在仓库中增加机器可读的已验证版本矩阵  
Handoff: 环境与发布负责人
