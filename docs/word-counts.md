# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 15994 | 335 | 7055 | 6176 | 3936 | 741 |
| `chapters/chapter02-technology-landscape.md` | 42168 | 759 | 19452 | 17090 | 9761 | 2002 |
| `chapters/chapter03-gpu-basics.md` | 26001 | 635 | 13341 | 11105 | 5418 | 1376 |
| `chapters/chapter04-environment-setup.md` | 33681 | 1231 | 24181 | 19493 | 3583 | 2514 |
| `chapters/chapter05-llm-inference-basics.md` | 43515 | 1469 | 26362 | 21129 | 7934 | 4013 |
| `chapters/chapter06-kv-cache-optimization.md` | 55723 | 1733 | 36167 | 28998 | 8897 | 5156 |
| `chapters/chapter07-request-scheduling.md` | 53392 | 1541 | 33144 | 24542 | 8204 | 3568 |
| `chapters/chapter08-quantization.md` | 62188 | 2355 | 40940 | 32122 | 9786 | 5253 |
| `chapters/chapter09-speculative-sampling.md` | 36996 | 1074 | 19540 | 15770 | 7309 | 2178 |
| `chapters/chapter10-production-deployment.md` | 79872 | 2729 | 56316 | 43326 | 9331 | 5796 |
| `chapters/chapter11-advanced-topics.md` | 67905 | 2095 | 43231 | 35740 | 11303 | 4476 |
| **TOTAL** | 517435 | 15956 | 319729 | 255491 | 85462 | 37073 |
