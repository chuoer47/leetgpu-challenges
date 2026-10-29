# 在自己的 4090 上刷 LeetGPU

GitHub fork：https://github.com/chuoer47/leetgpu-challenges

4090 目录：`/home/amax01/lingchen/leetgpu-challenges`

Windows 副本：`F:\code_lib\leetgpu-challenges`

## 进入环境

```bash
cd /home/amax01/lingchen/leetgpu-challenges
source practice/env.sh
python --version
nvcc --version
nvidia-smi
```

复用现有 `/home/amax01/miniconda3/envs/leetcuda`，无需重新安装 CUDA。
`env.sh` 同时选择环境内的 host compiler，避免系统 GCC 14 与 CUDA 12.4 不兼容。
本地 CUDA runner 使用动态 cudart，适配现有环境的运行库；暂不支持需要 cudadevrt 的动态并行。
验证环境为 PyTorch 2.5.1+cu124、CUDA Toolkit 12.4、Triton 3.1.0。
显式 `--gpu 3` 优先选择物理 GPU 3，即使已经设置了 `CUDA_VISIBLE_DEVICES`。
未传 `--gpu` 时沿用 `CUDA_VISIBLE_DEVICES`；两者都没有时默认物理 GPU 1。
也可以用 `export CUDA_VISIBLE_DEVICES=2` 设置当前终端的默认卡。
选中的第一张可见卡在程序里叫 `cuda:0`；启动时会打印可见卡设置和 GPU UUID。

## 开始一道题

```bash
# 第一题模板已创建，可直接编辑；换新题时用 --init（不会覆盖已有文件）
# 阅读 challenges/easy/1_vector_add/challenge.html
# 编辑 solutions/easy/1_vector_add/solution.cu，实现 solve
python practice/run.py 1
```

CUDA 的 `solve` 必须以 `extern "C"` 导出，参数顺序和类型以该题
`get_solve_signature()` 为准。Python 解法按相同顺序接受 GPU tensor / 标量，原地写输出。
题目 starter 的类型偶尔与 signature 不完全一致（例如第 1 题的 N）；以 signature 为准。

```bash
python practice/run.py 2 --init
python practice/run.py 1 --language triton --init
python practice/run.py 1 --language triton
python practice/run.py 1 --solution solutions/easy/1_vector_add/v2.cu --runs 30
python practice/run.py 1 --check-only
```

首次使用可运行独立示例验证环境，不会覆盖自己的解法：

```bash
python practice/run.py 1 --solution practice/examples/vector_add.cu
```

本地工具支持单 GPU 的 CUDA、PyTorch、Triton，以及现有 tensor 和随机输入描述对象。
暂不支持 JAX、Mojo、多 GPU 和特殊自定义 checker；个别题目的特殊判题规则需要额外实现。
会检查示例、全部功能测试和大规模性能用例，按题目的 atol/rtol 比较 out/inout，
也检查只读输入是否被修改。编译、生成输入、L2 刷新均在计时外；默认 2 次预热、15 次测速。
每次测速重新生成输入，避免 inout 重复修改改变工作负载。耗时报告含 solve + 同步的墙钟时间
和 CUDA event 时间；墙钟时间含事件记录开销，均不是网站官方成绩。
本工具执行可信的个人代码，不提供沙箱，也不保证捕获所有越界访问。

## 保存成果

```bash
git add solutions practice LOCAL_WORKFLOW.md benchmarks
git commit -m "Add my solution and optimization notes"
git push origin main
```

`origin` 指向你的 fork；`upstream` 指向 AlphaGPU 原仓库。
Windows 和服务器副本通过 GitHub 同步：一端 push 后另一端 `git pull --ff-only origin main`。
服务器已有 GitHub CLI 登录；如 HTTPS 推送需要认证，可运行 `gh auth setup-git`。

## 同步上游新题

先提交自己的改动，保持工作区干净，再运行：

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

保留已有提交；有冲突时解决冲突后再提交。个人解法放在 solutions/，可减少与上游题目冲突。

仓库原有 `scripts/run_challenge.py` 使用网站 API。自己机子测速请使用 `practice/run.py`。

## 已导入的网站成功提交

2026-10-28 从本人账号导入 42 道已完成题目的 65 份成功提交。
入口：`site-archive/2026-10-28/README.md`；本地验证报告：同目录 `REPORT.md`。
默认 `solution` 为每种语言最近一次网站成功提交；各题 `submissions/` 保存原始版本，
`SITE_IMPORT.md` 列出提交时间、语言、GPU 和提交 ID。原始代码不自动修改。

```bash
python practice/run.py 2
python practice/run.py 41 --language pytorch
python practice/test_imported.py --manifest site-archive/2026-10-28/manifest.json --resume
python practice/report_imported.py --manifest site-archive/2026-10-28/manifest.json
```

严格检查会将 out 缓冲区初始化为非零/NaN，检查解法是否完整覆盖输出。
历史版本中依赖输出预先清零的实现，另有零初始化诊断记录；网站曾通过与本地严格通过分别记录。
归档不含 Authorization、登录令牌或账号会话数据。
