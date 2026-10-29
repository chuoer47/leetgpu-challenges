# 我的解法

按 `solutions/<难度>/<题号_名称>/` 保存解法，与上游题目分开放置。
用 `python practice/run.py <题号> --init` 复制 CUDA 模板；模板尚未实现，测试失败是正常的。
PyTorch / Triton 使用 `--language pytorch` / `--language triton`。

建议每题附上 `NOTES.md`，记录：

- 算法思路、访存布局、线程划分。
- 每次优化改动、假设和实测结果（GPU、输入大小、运行次数）。
- 正确性、数值精度、边界条件和后续尝试。

历史版本可以存为 `v1.cu`、`v2.cu`，用 `--solution` 指定文件比较。
