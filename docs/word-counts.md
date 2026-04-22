# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 23969 | 449 | 9967 | 9019 | 6035 | 754 |
| `chapters/chapter02-technology-landscape.md` | 28019 | 586 | 12550 | 11081 | 6440 | 1163 |
| `chapters/chapter03-gpu-basics.md` | 22501 | 556 | 10886 | 9466 | 4944 | 1132 |
| `chapters/chapter04-environment-setup.md` | 35353 | 1441 | 26978 | 21203 | 2943 | 3030 |
| `chapters/chapter05-llm-inference-basics.md` | 36424 | 1391 | 22817 | 17981 | 6333 | 3655 |
| `chapters/chapter06-kv-cache-optimization.md` | 36954 | 1418 | 26355 | 20476 | 4887 | 4018 |
| `chapters/chapter07-request-scheduling.md` | 45309 | 1560 | 29370 | 22171 | 7135 | 3360 |
| `chapters/chapter08-quantization.md` | 56590 | 2275 | 38064 | 29514 | 8405 | 5023 |
| `chapters/chapter09-speculative-sampling.md` | 36800 | 1096 | 19445 | 15556 | 7188 | 2271 |
| `chapters/chapter10-production-deployment.md` | 73784 | 2779 | 53892 | 40579 | 7626 | 5540 |
| `chapters/chapter11-advanced-topics.md` | 52475 | 1969 | 36330 | 29187 | 7457 | 3914 |
| **TOTAL** | 448178 | 15520 | 286654 | 226233 | 69393 | 33860 |
