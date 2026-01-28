# 参考资料汇总

> 本文档记录《LLM推理优化实战》一书编写过程中引用的所有参考资料。

---

## Reference materials (books, papers, reports)

### 学术论文与报告

#### AI Agent系统
- **Manus Blog** - Context Engineering for AI Agents: Lessons from Building Manus
  - 作者：Yichao 'Peak' Ji
  - 发布日期：2025年7月18日
  - 核心内容：
    - KV-Cache优化在Agent系统中的应用
    - Context Engineering六大核心原则
    - Agent系统的成本优化策略
  - 引用章节：6.7.8, 10.6.5, 11.1.7
  - URL: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus

#### MoE架构与推理
- **Large-scale Expert Parallelism (大EP)**
  - 来源：vLLM Blog
  - 发布日期：2025年12月17日
  - 核心内容：大规模MoE服务架构
  - 引用章节：11.8.1
  - URL: https://blog.vllm.ai/2025/12/17/large-scale-serving.html

- **EPD (Expert-Parallel Data Parallelism)**
  - 核心内容：专家并行与数据并行的结合
  - 引用章节：11.8.2

- **Elastic Expert Parallelism**
  - 核心内容：弹性专家并行
  - 引用章节：11.8.3

- **MoonCake**
  - 核心内容： disaggregated架构
  - 引用章节：11.8.4

#### 性能分析
- **vLLM Profiling Documentation**
  - 维护者：vLLM Community
  - 核心内容：
    - PyTorch Profiler集成
    - NVIDIA Nsight Systems profiling
    - Offline inference和Server mode profiling
  - 引用章节：10.5.5
  - URL: https://docs.vllm.ai/en/stable/contributing/profiling/

#### 量化技术
- **Quantization-Aware Training (QAT)**
  - 引用章节：8.2.2
  - 核心内容：训练时量化，SGLang团队验证

- **INT4 Quantization (W4A16)**
  - 引用章节：8.3.2
  - 核心内容：4-bit权重，16-bit激活

#### 投机采样
- **Eagle Series**
  - 引用章节：9.3.4
  - 核心内容：Eagle, Eagle 2, Eagle 3投机采样

- **vLLM Speculators v0.3.0**
  - 引用章节：9.7.7
  - 核心内容：端到端Eagle 3训练，NVIDIA官方支持

#### 行业实践（2025"青稞"AI嘉年华）
- **朱立耕@NVIDIA**
  - 异构硬件部署（11.2.2）
  - 容灾和混部（11.2.4）
  - 精度对齐实践（11.8.4）

- **张明星@清华**
  - Agent环境复杂性（11.1.3）
  - SPMD到Event Driven（11.9.2）

- **张博涵@浙大**
  - Checkpoint管理（11.3.4）
  - Video Generation挑战（11.4.4）
  - Sparse Attention vs Linear Attention（11.6.3）
  - 算法和系统的co-design（11.9.3）

- **朱子林@质朴**
  - RL系统部署挑战（10.10.2）
  - 容灾和混部（11.2.4）

- **刘海超@vLLM**
  - 技术栈深度（11.9.1）
  - 前端性能优化（11.7.5）

---

## Web resources

### 官方文档

#### vLLM
- **vLLM Official Documentation**: https://docs.vllm.ai/
  - Profiling Guide: https://docs.vllm.ai/en/stable/contributing/profiling/
  - Prefix Caching: https://docs.vllm.ai/en/stable/serving/prefix_caching.html

#### SGLang
- **SGLang GitHub**: https://github.com/sgl-project/sglang
  - INT4推理支持

#### NVIDIA工具
- **NVIDIA Nsight Systems**: https://developer.nvidia.com/nsight-systems
  - GPU系统级性能分析工具

- **NVIDIA Nsight Compute**: https://developer.nvidia.com/nsight-compute
  - GPU kernel级深度分析工具

### 云厂商资源

#### 阿里云
- **使用Nsight Systems进行性能分析**: https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/using-nsight-system-to-realize-performance-analysis
  - 训练优化案例：542 → 3173 samples/s（5.85x提升）
  - 7项关键优化技术

### 开源项目

#### vLLM插件系统
- **vLLM Plugin System Blog**: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
  - 引用章节：10.11
  - 核心内容：无需fork即可定制vLLM

### 技术博客

#### vLLM Official Blog
- **Large-scale Serving (MoE)**: https://blog.vllm.ai/2025/12/17/large-scale-serving.html

#### Manus
- **Context Engineering for AI Agents**: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
  - Agent系统最佳实践
  - KV-Cache优化策略
  - 成本优化实战案例

---

## 按章节索引

### 第6章 - KV Cache优化
- **6.7.8** Agent系统的KV Cache优化实战
  - Manus Blog - Context Engineering

### 第7章 - 请求调度策略
- **7.7** Prefill-Decode分离（PD分离）
  - vLLM、SGLang社区合作

### 第8章 - 量化技术
- **8.2.2** QAT (SGLang团队验证)
- **8.3.2** INT4 (W4A16)
- **8.4.2** SGLang INT4推理
- **8.4.3** NVIDIA Model Optimizer
- **8.8** 精度对齐：Train vs Inference (朱立耕@NVIDIA)

### 第9章 - 投机采样
- **9.3.4** Eagle系列
- **9.7** Eagle 3 with SGLang (NVIDIA官方支持)
- **9.7.7** vLLM Speculators v0.3.0

### 第10章 - 生产环境部署
- **10.5.5** 性能分析工具与实战
  - vLLM Profiling Documentation
  - NVIDIA Nsight Systems
  - NVIDIA Nsight Compute
  - 阿里云性能分析案例

- **10.6.5** Agent系统的成本优化策略
  - Manus Blog - Claude Sonnet定价

- **10.10** RL系统部署
  - 朱子林@质朴、朱立耕@NVIDIA

- **10.11** vLLM插件系统
  - vLLM Plugin System Blog

### 第11章 - 高级话题
- **11.1** Agent基础设施
  - 张明星@清华、朱立耕@NVIDIA

- **11.1.7** Context Engineering最佳实践
  - Manus Blog (完整六大原则)

- **11.2** 异构硬件部署
  - 朱立耕@NVIDIA、朱子林@质朴

- **11.3** MoE模型推理优化
  - 张博涵@浙大
  - Large EP、EPD、Elastic Expert Parallelism、MoonCake

- **11.4** 多模态模型推理
  - 张博涵@浙大 (Video Generation)

- **11.6** Flash Attention
  - 张博涵@浙大 (Sparse Attention)

- **11.7** 自定义算子开发
  - 刘海超@vLLM

- **11.8** 技术发展与展望
  - 张明星@清华 (SPMD → Event Driven)
  - 张博涵@浙大 (算法和系统的co-design)
  - 刘海超@vLLM (技术栈深度)

---

## 添加日期记录

- **2025-01-29**: 添加Manus Blog资源（6.7.8, 10.6.5, 11.1.7）
- **2025-01-29**: 添加性能分析资源（10.5.5）
- **2025-01-29**: 添加2025"青稞"AI嘉年华专家观点

---

## 说明

### 文档用途
- 追踪所有引用的资料来源
- 便于后续更新和验证
- 提供读者延伸阅读参考

### 更新原则
- 每次添加新内容时同步更新此文件
- 记录添加日期，便于追溯
- 按章节索引，方便查找

### 未分类资源
如果遇到难以分类的资源，可以临时添加到对应章节的"其他"类别中，并在review时整理。
