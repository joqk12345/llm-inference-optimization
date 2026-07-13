# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 15893 | 337 | 7040 | 6150 | 3883 | 751 |
| `chapters/chapter02-technology-landscape.md` | 41174 | 759 | 19172 | 16800 | 9410 | 2012 |
| `chapters/chapter03-gpu-basics.md` | 25694 | 639 | 13316 | 11014 | 5290 | 1447 |
| `chapters/chapter04-environment-setup.md` | 35163 | 1441 | 26901 | 21149 | 2974 | 3006 |
| `chapters/chapter05-llm-inference-basics.md` | 41331 | 1490 | 25695 | 20400 | 7255 | 4085 |
| `chapters/chapter06-kv-cache-optimization.md` | 53853 | 1783 | 36266 | 28816 | 8014 | 5392 |
| `chapters/chapter07-request-scheduling.md` | 52904 | 1689 | 34032 | 24832 | 7634 | 3831 |
| `chapters/chapter08-quantization.md` | 62969 | 2395 | 42152 | 32945 | 9594 | 5639 |
| `chapters/chapter09-speculative-sampling.md` | 36924 | 1098 | 19484 | 15642 | 7304 | 2219 |
| `chapters/chapter10-production-deployment.md` | 79628 | 2852 | 57314 | 43699 | 8790 | 5947 |
| `chapters/chapter11-advanced-topics.md` | 67725 | 2120 | 43521 | 35941 | 11099 | 4504 |
| **TOTAL** | 513258 | 16603 | 324893 | 257388 | 81247 | 38833 |
