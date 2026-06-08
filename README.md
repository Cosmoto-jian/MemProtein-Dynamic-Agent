# MemProtein-Dynamic-Agent

膜蛋白受力形变模拟 + 残基协同运动分析。

输入一个 [OPM 数据库](https://opm.phar.umich.edu) 的膜蛋白结构,模拟它在膜张力(径向拉伸)下的动态形变,再分析各残基的协同运动模式(瞬时相关分析)。Piezo1(6b3r)、视紫红质(1u19)、6w7b 等都已验证可跑。

> 英文文档见 [README.en.md](README.en.md)。

## 目录结构

```
memprotein/          核心代码包(可被程序 / agent 直接 import)
  io.py              读写输入文件(MODEL/targetNode/mass/evector)
  preprocess.py      OPM PDB + 跨膜段文本 → 模型输入
  simulate.py        向量式有限元(VFIFE)动态仿真
  analysis.py        瞬时相关分析 + 四种图
  pipeline.py        端到端编排
cli.py               命令行总入口(run / preprocess / simulate / analyze)
viz/animate.py       PyVista 三维形变动画(需图形界面)
data/
  raw/               原始输入(你下载/准备的 .pdb 和 _tm.txt)
  inputs/            生成的模型文件(自动生成,不进 git)
  results/           仿真结果 .h5 + 分析图(自动生成,不进 git)
clean.sh             一键清除生成产物
```

## 安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
（Python 3.10+,在 3.14 上测试通过。`pyvista`/`vtk` 仅可视化用。)

## 准备输入(每个蛋白两个文件)

1. **下载 OPM PDB** 到 `data/raw/`(把 `6b3r` 换成你的 4 位 PDB ID):
   ```bash
   curl -L -o data/raw/6b3r.pdb "https://storage.googleapis.com/opm-assets/pdb/6b3r.pdb"
   ```
2. **复制跨膜段文本**:在该蛋白的 OPM 页面找 "Transmembrane Secondary structure segments",把每条链那几行(形如 `A - TM segments: 1(33-64), 2(...)`)复制,存成 `data/raw/6b3r_tm.txt`。

## 运行

**一键端到端**(预处理 → 仿真 → 4 张分析图):
```bash
.venv/bin/python cli.py run --pdb data/raw/6b3r.pdb --tm data/raw/6b3r_tm.txt
```

**分步运行**:
```bash
.venv/bin/python cli.py preprocess --pdb data/raw/6b3r.pdb --tm data/raw/6b3r_tm.txt
.venv/bin/python cli.py simulate --ET 100 --Fmax 0.1
.venv/bin/python cli.py analyze --kind all
```

**作为库调用**(给 agent / 脚本用):
```python
from memprotein.pipeline import run
run("data/raw/6b3r.pdb", "data/raw/6b3r_tm.txt")
```

结果在 `data/results/`:`simulation_data.h5` + `instant_corr.png` / `distance_corr.png` / `anchor_corr.png` / `anchor_stack.png`。

**看三维动画**(需图形界面):
```bash
.venv/bin/python viz/animate.py
```

**清除生成产物**:
```bash
bash clean.sh
```

## 分析

相关性分两种:**C^Z**(垂直膜,升降协同,取符号 ±1)和 **C^XY**(膜平面内,夹角余弦 −1~+1)。算位移前会先对每帧做 **Kabsch 刚体对齐**,去掉整体平移/转动造成的假协同。

`cli.py analyze` 出四张基础图:

| 图 | 内容 |
|---|---|
| `instant_corr` | 抽样残基对:相关性 vs 残基间距离(多时刻叠加) |
| `distance_corr` | 距离分箱:平均相关性的 距离 × 时间 热图 |
| `anchor_corr` | 单个锚点 vs 其他全部残基,某时刻的散点 |
| `anchor_stack` | 多个锚点各一行,竖排对照 |

`memprotein.analysis` 里还有更多可编程调用的函数(都支持 `node_subset` 单链限定、`align` 刚体对齐开关):

| 函数 | 内容 |
|---|---|
| `correlation_vs_distance` | 散点:相关性 vs 距离(可传 `pairs` 画全部对) |
| `correlation_hexbin` | 密度图(点多时不糊) |
| `correlation_binned` | 单时刻分箱平均线(看趋势) |
| `binned_multitime` | 多时刻分箱均线叠加;`realtime=` 切换 x 轴用 t=0 距离 / 各时刻实时距离 |
| `chain_nodes(h5, "A")` | 取某条链的节点(从结果文件直接读链信息) |

**单链分析脚本** `instant_one_chain.py`:只分析一条链(适合对称多聚体),顶部 CONFIG 改 `CHAIN` / `TIMES` / `N_SAMPLE`(`0`=全部对)即可,跑 `.venv/bin/python instant_one_chain.py`。

**节点身份**:结果文件 `simulation_data.h5` 里存了每个节点的链(`node_chains`)和残基号(`node_resids`),分析按链/残基选取时直接读它,不会和粗粒化序号错位。

## 关键参数

仿真(`cli.py run/simulate`):`--ET` 总时长(ps)、`--h` 时间步、`--Fmax` 峰值力(pN)、`--E` 杨氏模量、`--A` 截面积、`--zeta` 阻尼、`--ramp-t1`/`--unload-t1` 三角波加载时刻。
预处理:`--cutoff` 弹性网络连接半径(默认 10 Å)、`--include-hetatm`(默认排除脂/配体/UNK 等非蛋白原子)。

## 方法学要点

- **粗粒化**:每个氨基酸残基用其 Cα 当一个珠子(节点)。
- **弹性网络**:距离 <10 Å 的残基对用弹簧连接(单元)。
- **加力**:膜内残基(由 OPM 跨膜段确定)沿膜平面径向往外拉,模拟膜张力。
- **积分**:中心差分显式时间积分(向量化,比逐元素循环快约 26 倍)。
- **分析**:对每帧、每对残基算瞬时运动方向相关性,看协同如何随距离/时间演化。
