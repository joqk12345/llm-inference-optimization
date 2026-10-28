# FlashAttention (2022)

Source: https://arxiv.org/abs/2205.14135

Algorithm: https://arxiv.org/html/2205.14135v2

Evidence grade: A（原始论文）

Verified on: 2026-09-26（核对 Algorithm 1；非 GPU 实测）

Supported claims: 分块与在线 softmax 避免物化完整 attention 矩阵；递推必须重缩放历史指数和与输出累积；数学定义保持 dense Attention，算术复杂度仍为 O(N²d)。

Assumptions: 标准 softmax Attention；mask、dropout 与精度需按实现分别处理。

Limits: 本仓库的纯 Python 实现仅验证递推，不验证 GPU IO、吞吐或特定后端。

Open questions: 目标 GPU 的 kernel 数值与性能回归仍需独立执行。
