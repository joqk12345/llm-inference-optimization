# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 15994 | 335 | 7055 | 6176 | 3936 | 741 |
| `chapters/chapter02-technology-landscape.md` | 42273 | 750 | 19327 | 17068 | 9859 | 1898 |
| `chapters/chapter03-gpu-basics.md` | 25999 | 635 | 13341 | 11104 | 5418 | 1377 |
| `chapters/chapter04-environment-setup.md` | 33679 | 1231 | 24181 | 19492 | 3583 | 2515 |
| `chapters/chapter05-llm-inference-basics.md` | 33283 | 1040 | 20803 | 16668 | 5712 | 3170 |
| `chapters/chapter06-kv-cache-optimization.md` | 56983 | 1749 | 36797 | 29566 | 9177 | 5211 |
| `chapters/chapter07-request-scheduling.md` | 53390 | 1541 | 33144 | 24541 | 8204 | 3569 |
| `chapters/chapter08-quantization.md` | 63944 | 2309 | 41434 | 32651 | 10351 | 5253 |
| `chapters/chapter09-speculative-sampling.md` | 34305 | 970 | 17662 | 14964 | 7514 | 1927 |
| `chapters/chapter10-production-deployment.md` | 80049 | 2731 | 56459 | 43436 | 9345 | 5828 |
| `chapters/chapter11-advanced-topics.md` | 59179 | 1668 | 35514 | 29861 | 10863 | 3500 |
| **TOTAL** | 499078 | 14959 | 305717 | 245527 | 83962 | 34989 |
