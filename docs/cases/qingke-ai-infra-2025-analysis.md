# 2025"青稞"AI嘉年华 - Infra专题研讨会核心内容分析

**来源**: 2025年"青稞"AI嘉年华基础设施专题研讨会嘉宾发言逐字稿
**日期**: 2026年1月27日整理
**参会嘉宾**:
- 张明星（清华大学副教授，MoonCake、Kitchen开源项目发起人）
- 张博涵（浙江大学研究员，高效算法和infra co-design）
- 刘海超（清华大学博士，vLLM项目维护）
- 朱子林（质朴，RL infra工程师，slime框架）
- 朱立耕（NVIDIA research，多模态reasoning）
- 张浩（清华大学毕业，alite开源框架）

---

## 一、2025年最重要的技术进展

### 1.1 DeepSeek V3的震撼（张明星、张博涵、朱立耕）

**核心意义**:
- 第一次有了能够媲美ChatGPT的开源模型
- 展示了大规模MoE的训练范式
- 揭示了大规模MoE推理的tricks
- 颠覆了行业格局

**影响**:
- 之前的dance模型范式快速切换到MoE模型
- 各大厂对infra的重要性形成共识
- 算力、infra侧、算法和数据的三方面co-design才是王道

### 1.2 PD分离（Prefill-Decode分离）的普及（张明星）

**时间线**:
- 2025年初：概念提出
- 2025年中：社区合作（vLLM、SGLang等）
- 2025年底：几乎所有厂商都在用

**意义**:
- 从概念到生产落地只用了一年
- 成为推理优化的标准架构
- 开源社区的协作推动技术快速成熟

### 1.3 RL（Reinforcement Learning）Info的兴起（张明星、朱子林、张浩）

**时间线**:
- 上半年（1-6月）：推理集群化
- 下半年（7-12月）：RL info如何scaling up/scaling out

**关键框架**:
- Open RHF（早期）
- verl
- slime（质朴）
- arewe
- veRL

**核心挑战**:
- 训练和推理的深度结合
- 需要training team和inference team更紧密地协作
- 不同于传统RL的主线任务，RL更强调flexibility和场景多样性

### 1.4 多模态的爆发（张博涵、朱立耕）

**代表模型**:
- Google: Gemini 2.0、Gemini Nano、Gemini Flash系列
- OpenAI: GPT-4V、Sora
- 国内: AutoJam、多宝书记

**核心洞察**:
- 原生多模能力是模型能力的撬动杠杆
- 展示了agent的实际价值
- 训练成本指数级下降，垂直领域落地越来越近

**技术挑战**:
- Diffusion模型的训练系统缺失
- Video generation的I/O bottleneck严重
- Diffusion的训练推理分离是否成立（张博涵提出疑问）

### 1.5 Agent的兴起（朱立耕、刘海超）

**代表案例**:
- Google: NotebookLM、Gemini Flash、Gemini Nano
- 国内: AutoJam、多宝书记

**核心价值**:
- AI可以作为科研助手（张博涵: Gemini完全可做他的RA）
- 展示了整体system的重要性
- 在复杂场景下完成的任务能力

**技术挑战**:
- 开源agent system还是负数状态（朱立耕）
- 需要scalable and easy to use的sandbox system
- 需要处理文件系统、网络、虚拟机等复杂环境

---

## 二、RL系统的关键问题

### 2.1 当前RL系统的核心挑战（张明星、朱子林、刘海超）

#### 2.1.1 缺少像pretrain那样的"主线"（张明星）

**对比**:
- Pretrain: 主线明确，大家卷MFU（Model FLOPS Utilization）
- RL: 多种不同场景，需要flexibility、convergence、efficiency的trade-off

**挑战**:
- 不同RL实验需要不同的workload和workflow
- Agent environment需要根据场景不断定义
- 端到端迭代速度慢（从idea到投产）

**需要的支持**:
- 灵活组合的框架
- 中间脚手架工具
- 更基础的infra支持

#### 2.1.2 CPU的关注度不够（张明星、刘海超）

**问题**:
- 未来agent的environment会更复杂（docker、VM、网络、workflow）
- 开源生态agent system基本是负分
- CPU workload会变多，需要降低CPU开销

**案例**:
- Agent需要复杂的文件系统操作
- 可能需要嵌套VM
- 网络调用、workflow编排

#### 2.1.3 Diffusion RL的训练系统缺失（张博涵）

**尴尬现状**:
- 做算法的人: inference太慢，训练时间太长，等不起
- 做系统的人: Diffusion算法还没成熟，等算法成熟再说

**技术疑问**:
- Diffusion的训练推理分离是否符合第一性原理？
  - 训练: computation bound
  - 推理: I/O bound
- 训练和推理都打得很满，分离的动机是什么？

**市场空白**:
- Video generation没有好的开源训练框架
- 市面上没有很好的Diffusion RL系统

### 2.2 RL系统的泛化问题（张浩）

#### 2.2.1 核心痛点：跨环境不泛化

**案例**:
- 训练任务: 数一个word里有多少个T或R
- 训练效果: 7B模型能将512长度字符串数得非常准确
- 泛化测试: 将stream里的A换成B
- **结果**: 完全无法泛化，千位上还是原来的数

**影响**:
- 无法搭建普适的环境适应所有下游任务
- 只能针对关键任务（如机器人）搭建专门环境
- 限制了RL的通用性

#### 2.2.2 解决方向

**需要**:
- 能够显式泛化的算法出现
- 显式训练A，在B上有提升
- 从环境搭建到算法改进都需要努力

**机会**:
- 一些关键任务可能体现出环境搭建的差异
- 不期待大公司做出普适环境吃掉所有任务

### 2.3 RL系统的资源分配问题（朱立耕）

#### 2.3.1 训练和Rollout的资源不匹配

**现状**:
- 传统做法: 训练和Rollout用同样的卡数（如128卡）
- 问题: GPU空置，利用率非常低

**挑战**:
- 训练阶段: 可能只需要64卡
- Rollout阶段: 可能需要256或512卡
- 需要动态调整资源分配

**需求**:
- 给一组pod（如1024张卡）
- 动态调整train和rollout的卡数
- elastic dynamic resource allocation

#### 2.3.2 开源Sandbox System缺失（朱立耕）

**问题**:
- 在公司内部搭建Jupyter agent就很难
- 需要manage K8S、自动起virtual environment
- 学术界几乎没有使用经验

**需求**:
- Scalable and easy to use的sandbox system
- 像inference engine一样给个URL，发HTTP request
- 能够处理CPU资源和调度

---

## 三、系统架构的演进

### 3.1 从SPMD到MPMD（刘海超、朱子林、张明星）

#### 3.1.1 变化的本质

**之前（Pretrain时代）**:
- SPMD（Single Program Multiple Data）
- 1万张卡跑一样的东西
- 主要是GPU workload

**现在（RL时代）**:
- MPMD（Multiple Programs Multiple Data）
- 训练和推理是不同的workload
- CPU和GPU混合

**代表**:
- DeepSeek V3: PP + EP的变形
- MoE: 不是SPMD范式
- RL: 训练和推理分离

#### 3.1.2 从Workflow到Event Driven（张明星）

**Workflow模式**:
- 事先program好所有的策略
- 一起执行

**Event Driven模式**:
- 底层是大的data flow
- 不同event互相driven
- 异步感知环境状态并调整

**优势**:
- 更适合batch size达不到的场景
- 可以提升GPU和CPU的混合利用率
- 可以处理agent的密度和切换

**代价**:
- 编程复杂性高很多
- 需要非常好的infra支持

#### 3.1.3 容灾和混部的机会（朱子林）

**之前的问题**:
- NCCL/MPI通信不太能容灾
- 一个节点挂了就整体夯死
- 大家全杀掉重启

**现在的机会**:
- 推理engine可以独立操作
- 推理之间没有异构通信
- 可以做容灾、混部、扩缩容

**应用场景**:
- 潮汐队列: 白天推理，夜间RL
- SMP和RL的大集群可以混用
- 提升整体硬件利用率

### 3.2 异构化的机遇和挑战

#### 3.2.1 不同硬件用于不同workload（朱立耕）

**训练vs推理的算力差异**:
- 训练: flops per byte ≈ 10^5
- 推理: flops per byte ≈ 80
- **差距: 2-3个数量级**

**应用**:
- H100训练 + H200推理
- 国产卡推理 + NV训练
- 不同硬件可以更好利用

**之前的问题**:
- 大家都在SPMD时不会考虑
- 物理上在同一集群但权限不同
- 现在可以把这些卡更好利用起来

#### 3.2.2 跨集群训练的挑战（刘海超）

**问题**:
- 模型特别大后checkpoint存储都是问题
- 多集群训练时很多假设不成立
- 需要考虑storage、网络、编解码等

**复杂度**:
- Video的encode/decode
- 多模态需要不同的节点类型
- 问题本身内置复杂度

#### 3.2.3 MoE的checkpoint挑战（张博涵）

**问题**:
- T级别模型的checkpoint巨大
- 加载和保存非常缓慢
- 故障恢复困难

**可能的解决方案**:
- Partial checkpoint保存和加载
- 屏蔽掉挂掉的部分专家快速恢复
- 利用CPU和GPU的不同特性加速

---

## 四、精度问题：Info vs Algorithm

### 4.1 低精度的现状（刘海超、张明星、朱立耕）

#### 4.1.1 推理vs训练的容忍度

**推理（刘海超）**:
- 低精度看不出特别大的掉点
- Beast buy上测试performance
- 不是核心问题

**训练（张明星）**:
- 低精度训练容易不稳定
- 但不完全是精度问题
- 算法本身也有影响

**朱立耕（NVIDIA）**:
- 从FP32 → FP16 → BF16 → FP8都见过类似问题
- 随着算法stabilize和config摸清楚，问题可以解决
- 低精度收益还是很大的

#### 4.1.2 精度对算法的影响（朱立耕）

**核心问题**:
- 训练和推理的算子精度不对齐
- 训练用自定义kernel，推理用Flash Attention
- 有numerical gap

**大团队的做法**:
- Train和inference的算子在同一个大的wrapper里维护
- 精度问题就不是问题

**开源社区的问题**:
- Train和inference是两帮人做
- 算子没对齐导致accuracy不稳定

#### 4.1.3 不同任务对精度的要求（张博涵）

**LLM**:
- 离散采样
- 对低精度容忍度更高

**Diffusion**:
- 连续空间采样
- 误差累积严重
- FP4可能掉10-20个点，效果没法看

**FP8**:
- 相对实用
- 但相对BF16/FP16提速有限（1.5倍）
- 达不到理论上的optimal

### 4.2 低精度的挑战（刘海超、朱子林）

#### 4.2.1 软件抽象复杂度（刘海超）

**之前**:
- BF16/FP16: 一个tensor就是一个数据

**FP8**:
- 一个weight变成3个tensor
- 需要metadata描述layout

**FP4**:
- 需要3个tensor表示
- 需要padding、pack等操作
- PyTorch最少1 byte，需要pack 2个FP4

**问题**:
- 整个软件生态需要演进
- 用户心智负担大
- 如何在收益和复杂度间平衡

#### 4.2.2 训练范式需要重新调（朱子林）

**问题**:
- 每多一个新精度，整个训练范式要从头调一遍
- 大家先怪精度，然后一点点找细节问题

**关键问题**:
- Scaling到底有多重要？
- 1000张卡跑INT4的400B模型
  vs
- 200张卡跑BF16的200B模型
  哪个效果更好？

**结论**:
- 这是和算法联合的问题
- 不是单纯硬件/系统的问题

---

## 五、算法和系统的未来发展

### 5.1 算法侧需要关注的（张博涵、朱立耕）

#### 5.1.1 推理优化（张博涵）

**核心问题**:
- Reasoning length越来越长
- License非常高

**方向**:
- 大小模型协同
- 降低reasoning length

#### 5.1.2 Speculative Decoding的范式问题（张博涵）

**当前范式（MTP: Multi-Token Prediction）**:
- Diffusion M、Medusa等通过多头并行采样
- 采样空间应该是V×V（联合分布）
- 但实际是2×V（独立采样）

**问题**:
- Scaling up有很大问题
- 不是从联合分布采样

**未来方向**:
- 新的MTP并行采样范式
- 相对AR接近无损
- 配合speculative decoding

#### 5.1.3 Long Context的Attention（张博涵、刘海超）

**趋势**:
- 大厂逐渐放弃linear attention
- 收敛到sparse attention

**原因**:
- Agent场景是multi-turn的long context
- 理想情况：全存，sparse retrieval
- Make sense

**挑战**:
- 在long context reasoning场景下
- 怎么把sparse attention做不掉点？
- 例如：Needle In A Haystack（大海捞多针）
  - Claude 3精度只有20-30%

#### 5.1.4 视觉模型的新范式（刘海超）

**范式**:
- Block Fusion的新范式
- 目标: 实时可交互、符合物理规律的长视频生成

**应用**:
- 巨量（Jiyang）、支架等提供数据引擎

### 5.2 系统侧需要关注的（刘海超、朱子林、张明星）

#### 5.2.1 新workload支持（刘海超）

**Reasoning**:
- 输出更长，怎么做？

**新硬件**:
- GB300等新TPU/GPU
- 如何支持？

**新算法**:
- Linear attention
- Hybrid models

**vLLM的工作**:
- 如何在vLLM里高效支持这些算法

#### 5.2.2 前端性能优化（刘海超）

**问题**:
- Python写web service性能差
- 需要加rest

**Inference的CPU优化**:
- 以前只关注GPU
- 现在CPU workload变多
- 需要做很多compaction

**是否用C++**:
- PyTorch也在考虑
- 需要联合优化

#### 5.2.3 框架设计的灵活性（朱子林）

**RL框架的设计原则**:
- 给算法老师足够大的设计空间
- 把infra问题模块化抽离
- 同时有训练框架和推理框架

**组成**:
- Rollout和驱动的联合
- 参数更新
- 推理生成数据传回

**未来接入**:
- Diffusion的训推结合
- 多模态场景

#### 5.2.4 训练框架的稳定性和性能（朱子林）

**永恒的追求**:
- 训练框架的稳定性
- 训练框架的性能
- 最大程度压榨硬件性能

### 5.3 硬件侧需要关注的（朱立耕、刘海超）

#### 5.3.1 NVLink和大型机（刘海超）

**GB200 NVL72**:
- NVLink网络
- 大型机的优势

**挑战**:
- 如何支持？
- 前端怎么做？
- 如何压榨性能？

#### 5.3.2 不同精度的硬件（朱立耕）

**H100推理**:
- Flops per byte ratio不匹配
- 计算浪费

**NV专用推理硬件**:
- Blackwell
- L40s、L40s系列
- 根据workload设计

#### 5.3.3 国产卡的机会（朱立耕）

**当前格局**:
- Training: NV的privilege
- Inference: 可选硬件很多
- 国产卡、全连MV7等

**RL的天然优势**:
- 两种workload的混合
- 天生适合用不同硬件

---

## 六、技术趋势总结

### 6.1 上半年vs下半年（张明星）

**上半年（1-6月）**:
- 推理集群化
- PD分离普及
- 大规模EP

**下半年（7-12月）**:
- RL info如何scaling up/scaling out
- Agent的兴起
- 多模态的爆发

### 6.2 从Training到Inference再到System（张浩）

**三个阶段**:
1. Training team: 专注mega双上的修改
2. Inference team: 对接算子优化
3. System team: 两边team更close

**RL带来的变化**:
- 要求两边team更紧密协作
- 共同push一些事情
- 从使用者变成参与者

### 6.3 Infra的重要性共识（张博涵）

**之前**:
- 大厂领导可能不鸟infra

**现在**:
- 不用说了，已经形成共识
- 算力、infra侧、算法和数据的三方面co-design才是王道

### 6.4 算法和系统的gap（张博涵）

**Diffusion RL的尴尬**:
- 做算法的: infra太慢
- 做系统的: 算法还没成熟
- 两边大眼瞪小眼

**需要**:
- 算法和系统更紧密的合作
- 共同推动领域发展

### 6.5 从User到Contributor（朱立耕、张浩）

**VLM和SGLang时代**:
- 更像是使用者

**现在**:
- 需要关注功能开发和维护
- 从使用者变成参与者
- 贡献到开源社区

### 6.6 技术栈变深（刘海超）

**2024年**:
- 从框架层面优化
- 底层工具提供好

**2025年**:
- 技术栈变得非常深
- 需要到RDMA、networking层面
- 需要懂算法、硬件、系统
- 需要联合优化

**变化**:
- 以前: vLLM是一个python package
- 现在: 需要配置interconnect、driver、OS
- 从简单应用到class级别的工作

---

## 七、对书籍的启示

### 7.1 可以融入的内容

#### 第1章（商业价值/技术全景）
- **补充**: 2025年技术趋势
  - PD分离的普及
  - RL info的兴起
  - Agent和多模态的爆发
  - 从SPMD到MPMD的演进

#### 第3章（LLM推理原理）或新增章节
- **新增**: RL推理原理
  - RL vs 传统推理
  - Training和Rollout的分离
  - Rollout的特殊挑战（batch size、length）

#### 第4章（GPU基础）
- **扩充**: 异构硬件
  - 训练vs推理的算力差异
  - H100 vs H200的应用
  - 国产卡的机会

#### 第5章（KV Cache）
- **补充**: Long Context的挑战
  - Sparse attention
  - Linear attention的局限
  - Agent场景的multi-turn long context

#### 第6章（Continuous Batching）
- **扩充**: PD分离（Prefill-Decode分离）
  - 为什么PD分离成为标准
  - vLLM和SGLang的实现
  - 生产环境的实践经验

#### 第7章（量化）
- **补充**: 精度对算法的影响
  - LLM vs Diffusion的精度差异
  - Train和Inference算子对齐
  - 低精度的软件抽象复杂度

#### 第8章（投机采样）
- **扩充**: Speculative Decoding的新范式
  - MTP（Multi-Token Prediction）的问题
  - 联合分布采样的必要性
  - 大小模型协同

#### 第9章（生产部署）
- **新增**: RL系统部署
  - Scalable sandbox system
  - Elastic resource allocation
  - Train和Rollout的资源动态分配
  - 容灾和混部

#### 第10章（高级话题）
- **新增**: Agent Infra
  - Agent system的缺失
  - 文件系统、网络、VM的管理
  - CPU的重要性
- **新增**: Diffusion推理
  - Diffusion RL的挑战
  - Video generation的I/O bottleneck
- **新增**: 多模态推理
  - 原生多模模型
  - Video encode/decode
  - 异构节点类型

### 7.2 新增的实战案例

#### 案例1: PD分离的实践
- vLLM的PD分离实现
- SGLang的PD分离优化
- 生产环境的性能提升

#### 案例2: RL系统搭建
- Slime框架的使用
- Train和Rollout的分离
- 资源动态分配

#### 案例3: Agent环境搭建
- Docker环境管理
- 文件系统操作
- 网络调用和workflow

#### 案例4: 异构硬件部署
- H100训练 + H200推理
- 国产卡推理的实践
- 跨集群的挑战

### 7.3 需要强调的观点

#### 1. 技术栈越来越深（刘海超）
- 不是简单的python package
- 需要深入到RDMA、networking
- 需要懂算法、硬件、系统

#### 2. 从SPMD到Event Driven（张明星）
- 事先编排到动态调度
- 更适合RL和Agent场景
- 编程复杂性高但更灵活

#### 3. 算法和系统要co-design（张博涵、朱立耕）
- 不是系统等算法成熟
- 也不是算法等系统优化
- 需要同步螺旋式上升

#### 4. 精度问题不是纯技术问题（朱子林）
- 需要算法和系统共同解决
- 算子对齐比低精度本身更重要
- 大团队的优势：统一维护算子

#### 5. CPU的重要性被低估（张明星）
- Agent环境需要大量CPU
- 开源生态CPU支持是负分
- 需要更多关注CPU优化

#### 6. 泛化是RL的核心挑战（张浩）
- 跨环境泛化失败
- 需要算法突破
- 影响RL的通用性

#### 7. Infra的重要性已成共识（张博涵）
- 算力、infra、算法、数据缺一不可
- co-design才是王道
- DeepSeek V3证明了这一点

---

## 八、金句摘录

### 张明星（清华大学）
1. "2025年发生的这些事情，其实大家回看会发现，PD分离、大EP这些概念现在总感觉已经是well known，但其实真的就是2025年发生的。"
2. "上半年大家推理的集群化，下半年的主题基本上都是RL info怎么去scaling up scaling out RL。"
3. "RL和pretrain最大的区别在于RL没有像pretrain那样的主线。"
4. "大家对CPU的关注目前还是不够的。"
5. "未来可能是event driven的这样的一个方法，但代价是编程复杂性会高非常多。"

### 张博涵（浙江大学）
1. "DeepSeek V3开年王炸，让所有的大厂对infra的重要性形成了共识。"
2. "算力、infra侧、算法和数据，这四方面co-design才是王道。"
3. "Gemini完全可做我的科研助手了，可以少雇一些inference了。"
4. "做算法和系统这两拨人有点大眼瞪小眼，现在有点尬住了。"
5. "Diffusion的训练推理分离，第一性原理上成不成立？"

### 刘海超（vLLM）
1. "2024年的时候我们做推理优化，更多的还是从框架层面。今年其实整个优化的技术栈非常的深。"
2. "我们需要到RDMA、networking的层面去做customized的优化。"
3. "随着workload变得越来越重要之后，各方面大家都在尝试进行优化。"
4. "现在的创新感觉越来越是纠缠在一起的，你需要懂很多。"

### 朱子林（质朴）
1. "从去年年底OpenAI发布到后面DeepSeek R1的论文，让大家发现了一种新的训练范式。"
2. "RL框架今年对我来说最大的变化。"
3. "我们现在还是更多的是用一些固有的底层工具，今年大家都会去往更底层更第一性原理去研究。"
4. "训练框架的稳定性、性能永远是一个紧迫的问题。"
5. "开源agent system可能还是一个负数的状态。"

### 朱立耕（NVIDIA）
1. "DeepSeek V3告诉我们两件很重要的事情：大规模MoE的训练怎么去做，大规模MoE的推理有哪些tricks。"
2. "上半年大家从dance模型快速switch到MoE模型。"
3. "训练和推理两个team需要做更加close下去。"
4. "从使用者逐渐变成一个参与者。"
5. "从FP32到FP16，到BF16，再到FP8，每次过渡都见过类似问题，但都可以逐渐解决。"

### 张浩（alite）
1. "DeepSeek R1的出现说明预训练和后训练都还有非常大的空间。"
2. "我们不光是在训练一个模型，是在训练一个系统。"
3. "多模态是撬动模型能力的非常大的杠杆。"
4. "26年我们会在agent方向看到百花齐放。"
5. "RL跨环境以后不泛化，这是最致命的。"

---

## 九、后续行动

### 可以立即添加的内容

1. **第2章（技术全景）**:
   - 补充2025年技术趋势
   - PD分离、RL info、Agent、多模态

2. **第6章（Continuous Batching）**:
   - 新增PD分离小节
   - vLLM和SGLang的实现

3. **第9章（生产部署）**:
   - 新增RL系统部署
   - Sandbox system
   - Elastic resource allocation

### 需要进一步研究的内容

1. **RL推理原理**（可能需要新章节）
2. **Agent Infra**（高级话题）
3. **Diffusion推理**（高级话题）
4. **异构硬件部署**（生产部署）

### 建议的案例研究

1. **vLLM的PD分离实践**
2. **Slime框架的RL系统**
3. **SGLang的异构硬件支持**
4. **Gemini的Agent实现分析**

---

**文档状态**: 待审核
**优先级**: 高
**建议**: 这些内容代表了2025年最新的工业界实践，应该融入到相关章节中
