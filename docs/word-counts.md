# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 15711 | 338 | 7016 | 6116 | 3761 | 762 |
| `chapters/chapter02-technology-landscape.md` | 32565 | 676 | 15350 | 13313 | 7187 | 1710 |
| `chapters/chapter03-gpu-basics.md` | 25749 | 639 | 13342 | 11027 | 5288 | 1460 |
| `chapters/chapter04-environment-setup.md` | 35353 | 1441 | 26978 | 21203 | 2943 | 3030 |
| `chapters/chapter05-llm-inference-basics.md` | 38969 | 1454 | 24522 | 19358 | 6714 | 3968 |
| `chapters/chapter06-kv-cache-optimization.md` | 40846 | 1501 | 28942 | 22499 | 5426 | 4483 |
| `chapters/chapter07-request-scheduling.md` | 51638 | 1673 | 33326 | 24172 | 7338 | 3794 |
| `chapters/chapter08-quantization.md` | 60908 | 2360 | 41115 | 31952 | 8957 | 5618 |
| `chapters/chapter09-speculative-sampling.md` | 36800 | 1096 | 19445 | 15556 | 7188 | 2271 |
| `chapters/chapter10-production-deployment.md` | 78224 | 2934 | 57724 | 43424 | 7920 | 5955 |
| `chapters/chapter11-advanced-topics.md` | 52475 | 1969 | 36330 | 29187 | 7457 | 3914 |
| **TOTAL** | 469238 | 16081 | 304090 | 237807 | 70179 | 36965 |
