# CPU 回归检查

在仓库根目录执行：

```bash
python3 -m unittest discover -s tests -v
```

仅需 Python 3.10+ 标准库。覆盖 Attention 数值递推、投机采样接受/修正概率与前缀推进、诊断采样生命周期，以及实验客户端的流式协议、失败计数、质量门槛、环境可比性和原始日志。

HTTP 测试使用内存模拟响应，不需要访问模型服务。通过这些测试不代表 GPU 后端、模型质量或性能已经验证；真实实验见 [第10章配套说明](../code/chapter10/README.md)。
