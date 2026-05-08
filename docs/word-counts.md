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
| `chapters/chapter06-kv-cache-optimization.md` | 50816 | 1742 | 34589 | 27310 | 7356 | 5241 |
| `chapters/chapter07-request-scheduling.md` | 52913 | 1687 | 34049 | 24812 | 7592 | 3869 |
| `chapters/chapter08-quantization.md` | 63369 | 2395 | 42445 | 33117 | 9482 | 5757 |
| `chapters/chapter09-speculative-sampling.md` | 37185 | 1098 | 19610 | 15708 | 7289 | 2283 |
| `chapters/chapter10-production-deployment.md` | 80155 | 2952 | 58711 | 44314 | 8345 | 6043 |
| `chapters/chapter11-advanced-topics.md` | 66313 | 2136 | 43140 | 35378 | 10595 | 4464 |
| **TOTAL** | 509702 | 16679 | 324736 | 256128 | 79086 | 39052 |
