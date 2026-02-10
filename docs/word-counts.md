# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 30988 | 546 | 12288 | 11154 | 8044 | 920 |
| `chapters/chapter02-technology-landscape.md` | 27294 | 558 | 11929 | 10528 | 6391 | 1114 |
| `chapters/chapter03-gpu-basics.md` | 21767 | 528 | 10290 | 8936 | 4881 | 1085 |
| `chapters/chapter04-environment-setup.md` | 34626 | 1416 | 26519 | 20743 | 2875 | 2996 |
| `chapters/chapter05-llm-inference-basics.md` | 48319 | 2012 | 33098 | 25232 | 6198 | 5478 |
| `chapters/chapter06-kv-cache-optimization.md` | 34375 | 1367 | 25370 | 19521 | 4213 | 4026 |
| `chapters/chapter07-request-scheduling.md` | 44777 | 1631 | 30615 | 22717 | 6351 | 3864 |
| `chapters/chapter08-quantization.md` | 55177 | 2251 | 37821 | 29209 | 7957 | 5110 |
| `chapters/chapter09-speculative-sampling.md` | 36267 | 1075 | 19075 | 15176 | 7157 | 2295 |
| `chapters/chapter10-production-deployment.md` | 74240 | 2861 | 55939 | 41512 | 6588 | 5800 |
| `chapters/chapter11-advanced-topics.md` | 50117 | 1930 | 35397 | 28341 | 6866 | 3890 |
| **TOTAL** | 457947 | 16175 | 298341 | 233069 | 67521 | 36578 |
