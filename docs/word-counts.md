# Chapter 1-11 字数统计

说明：
- `non_ws` = 去除空白后的字符数（更接近“字数”口径）
- `cjk` = 汉字数量（U+4E00..U+9FFF）
- `words` = 按空白分词的词数（对中文不敏感，仅作参考）

| file | bytes | lines | chars | non_ws | cjk | words |
| --- | --- | --- | --- | --- | --- | --- |
| `chapters/chapter01-introduction.md` | 23417 | 422 | 9429 | 8549 | 6028 | 706 |
| `chapters/chapter02-technology-landscape.md` | 27294 | 558 | 11929 | 10528 | 6391 | 1114 |
| `chapters/chapter03-gpu-basics.md` | 21767 | 528 | 10290 | 8936 | 4881 | 1085 |
| `chapters/chapter04-environment-setup.md` | 34556 | 1412 | 26415 | 20697 | 2892 | 2994 |
| `chapters/chapter05-llm-inference-basics.md` | 48441 | 2011 | 33103 | 25250 | 6254 | 5465 |
| `chapters/chapter06-kv-cache-optimization.md` | 34793 | 1364 | 25197 | 19432 | 4508 | 3937 |
| `chapters/chapter07-request-scheduling.md` | 44604 | 1613 | 30127 | 22369 | 6512 | 3726 |
| `chapters/chapter08-quantization.md` | 55472 | 2241 | 37380 | 28894 | 8326 | 4990 |
| `chapters/chapter09-speculative-sampling.md` | 35835 | 1064 | 18722 | 14908 | 7121 | 2223 |
| `chapters/chapter10-production-deployment.md` | 74386 | 2850 | 55460 | 41150 | 6895 | 5698 |
| `chapters/chapter11-advanced-topics.md` | 50499 | 1923 | 35246 | 28199 | 7133 | 3854 |
| **TOTAL** | 451064 | 15986 | 293298 | 228912 | 66941 | 35792 |
