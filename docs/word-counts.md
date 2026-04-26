# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 15760 | 337 | 7004 | 6108 | 3802 | 757 |
| `chapters/chapter02-technology-landscape.md` | 40745 | 762 | 19167 | 16749 | 9141 | 2059 |
| `chapters/chapter03-gpu-basics.md` | 25749 | 639 | 13342 | 11027 | 5288 | 1460 |
| `chapters/chapter04-environment-setup.md` | 35353 | 1441 | 26978 | 21203 | 2943 | 3030 |
| `chapters/chapter05-llm-inference-basics.md` | 41344 | 1490 | 25701 | 20402 | 7253 | 4089 |
| `chapters/chapter06-kv-cache-optimization.md` | 50057 | 1738 | 34114 | 26893 | 7225 | 5185 |
| `chapters/chapter07-request-scheduling.md` | 51991 | 1675 | 33487 | 24321 | 7424 | 3805 |
| `chapters/chapter08-quantization.md` | 61298 | 2362 | 41277 | 32105 | 9059 | 5626 |
| `chapters/chapter09-speculative-sampling.md` | 37185 | 1098 | 19610 | 15708 | 7289 | 2283 |
| `chapters/chapter10-production-deployment.md` | 79063 | 2938 | 58077 | 43759 | 8132 | 5971 |
| `chapters/chapter11-advanced-topics.md` | 63724 | 2106 | 41651 | 34082 | 10082 | 4286 |
| **TOTAL** | 502269 | 16586 | 320408 | 252357 | 77638 | 38551 |
