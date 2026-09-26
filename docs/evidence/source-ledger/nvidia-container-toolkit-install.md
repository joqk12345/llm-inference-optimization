# NVIDIA Container Toolkit 安装（官方文档）

Source: NVIDIA documentation, “Installing the NVIDIA Container Toolkit.”  
URL: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Supports:

- 宿主机需要受支持的 NVIDIA 驱动、容器引擎和 NVIDIA Container Toolkit。
- Docker 运行时可用 `nvidia-ctk runtime configure --runtime=docker` 配置，并重启 Docker。
- rootless Docker、containerd、CRI-O 和 Podman 的配置路径不同。

Does not support:

- 为所有发行版手写同一份 `/etc/docker/daemon.json`。
- 仅凭容器内显示的 CUDA toolkit 版本判断宿主驱动兼容性。
- 不核对平台支持与版本就复制固定 CUDA 基础镜像。

Owner: 第4章作者  
Purpose: 支撑 GPU 容器运行时配置与排障边界  
Evidence grade: A  
Assumptions: 使用 latest 文档入口，发布前记录实际 Toolkit 版本  
Open questions: 是否补充 Kubernetes/containerd 的独立验证矩阵  
Handoff: 环境与发布负责人
