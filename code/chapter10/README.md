# 贯穿实验：前缀缓存的基线、优化与回归

执行状态：CPU 参考算法和 HTTP 协议测试可在本地验证；尚无真实 GPU 结果。这里提供可执行实验工具与固定工作负载，不提供虚构跑分。

## 问题与控制变量

固定模型、不可变模型/tokenizer revision、GPU、运行时、精度及请求配置，只切换前缀缓存。默认数据为八条共享前缀的查表任务，答案 exact match 是质量门槛；这只能检验本任务，不能替代业务质量评测。

客户端采用闭环并发：完成一个请求才补入下一个，请求计时从 worker 发出请求开始。它不模拟固定到达率，不用于证明线上饱和容量。固定顺序、seed、温度和 workload hash 都写入日志。每次启动服务后先完整预热数据一轮，因此比较的是热缓存场景；冷启动、低复用和缓存驱逐需单独设计工作负载。

## 环境与启动

客户端仅需 Python 3.10+ 标准库。服务端在支持 CUDA 的主机使用 `vllm==0.30.0`，见 [版本适配卡](../../docs/cards/vllm-0.30.0.md)。先完成环境 smoke test，留存依赖锁文件或镜像 digest。小模型用于控制实验成本，不能保证在此任务上通过质量门槛。

将 `environment.example.json` 复制为本地 `baseline.environment.json`，填写真实 GPU、驱动、CUDA、不可变 revision、环境摘要和整个服务实例的每小时成本。`server_args` 记录实际启动参数向量；客户端只能从 `/version` 和 `/v1/models` 核对版本与模型，其他字段需用启动日志和环境记录佐证。

按 manifest 的 `server_args` 启动基线服务，明确使用 `--no-enable-prefix-caching`。保存启动日志，确认配置生效。以下命令在仓库根目录执行，SLO 数字仅是教学实验门槛，应在两组测试前按目标业务一起确定：

```bash
python3 code/chapter10/experiment.py run \
  --manifest baseline.environment.json \
  --output code/chapter10/results/baseline.jsonl \
  --ttft-slo 2 --e2e-slo 5
```

停止基线服务，在同一环境重启候选服务。候选 manifest 只把 `prefix_caching` 改为 `true`，把启动参数替换为 `--enable-prefix-caching`；其他字段和启动参数保持一致：

```bash
python3 code/chapter10/experiment.py run \
  --manifest candidate.environment.json \
  --output code/chapter10/results/candidate.jsonl \
  --ttft-slo 2 --e2e-slo 5

python3 code/chapter10/experiment.py compare \
  --baseline code/chapter10/results/baseline.jsonl \
  --candidate code/chapter10/results/candidate.jsonl
```

重跑时使用新的结果文件名；脚本拒绝覆盖原始证据。至少交替运行三组 A/B，检查波动和重复性，并补低前缀复用负载。不能把本例的一次通过当作上线批准。

## 指标与验收

- 每个请求保留输出、标准答案、usage token 数、完成原因、错误以及客户端端到端耗时；失败不会从质量分母剔除。
- TTFT 为发请求到收到首个非空文本 chunk 的时间。TPOT 为首末非空 chunk 时间差除以 `output_tokens - 1`；这是 chunk 级近似，不是逐 token ITL，单 token 输出记为 null。
- 分位数采用 nearest rank，仅对成功请求计算，并报告样本数；错误率始终单列。默认 80 个测量请求不足以稳健估计 P99，正式实验应扩大样本与重复次数。
- 合格请求同时满足答案正确、TTFT SLO 和 E2E SLO。goodput 的分母包括整段测量墙钟时间。成本按总实例小时费率折算，不包含预热、部署、闲置或工程投入，不能称为完整 ROI。
- 默认门槛为候选无请求错误、质量达到 100% 且不低于基线、goodput 增加、每个合格请求成本下降。门槛失败退出码为 2，保留基线；日志或环境不可比退出码为 1。

HTTP 连接/读超时用于避免无响应连接；它不是服务端取消保证。发生超时后，等服务端在途请求清空或重启后再运行下一组，避免残留工作影响结果。

本脚本不采集 GPU 显存或 profiler trace。需要分析原因时，另存服务器 `/metrics`、GPU trace 和显存记录，与 JSONL 时间戳及 manifest 对齐。不要从客户端延迟推断缓存命中率或 HBM 带宽。

## 迁移到其他优化

第6章可直接复用本实验观察前缀缓存。第7章调度、第8章量化、第9章投机需要各自的控制变量与任务质量门槛；当前 `compare` 刻意只接受前缀缓存差异，不能把不相关的实验混合比较。量化应补校准记录及代表性长上下文评测，投机应补接受长度与验证耗时。
