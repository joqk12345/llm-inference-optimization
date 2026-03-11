# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 23975 | 449 | 9969 | 9021 | 6037 | 754 |
| `chapters/chapter02-technology-landscape.md` | 27884 | 584 | 12499 | 11035 | 6400 | 1159 |
| `chapters/chapter03-gpu-basics.md` | 22306 | 554 | 10819 | 9402 | 4885 | 1130 |
| `chapters/chapter04-environment-setup.md` | 35110 | 1439 | 26955 | 21170 | 2898 | 3041 |
| `chapters/chapter05-llm-inference-basics.md` | 49029 | 2039 | 33677 | 25754 | 6260 | 5513 |
| `chapters/chapter06-kv-cache-optimization.md` | 35423 | 1394 | 25817 | 19972 | 4512 | 3991 |
| `chapters/chapter07-request-scheduling.md` | 45293 | 1643 | 30798 | 22962 | 6520 | 3778 |
| `chapters/chapter08-quantization.md` | 56118 | 2273 | 38012 | 29440 | 8332 | 5046 |
| `chapters/chapter09-speculative-sampling.md` | 36510 | 1094 | 19383 | 15491 | 7127 | 2275 |
| `chapters/chapter10-production-deployment.md` | 75095 | 2883 | 56151 | 41751 | 6903 | 5756 |
| `chapters/chapter11-advanced-topics.md` | 51207 | 1956 | 35940 | 28803 | 7139 | 3912 |
| **TOTAL** | 457950 | 16308 | 300020 | 234801 | 67013 | 36355 |
