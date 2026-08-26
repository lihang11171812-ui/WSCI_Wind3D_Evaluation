# WSCI Wind Evaluation

用于规则网格风场的两套诊断工具：

1. **WSCI**：垂向廓线合理性、空间相关尺度和局部扰动结构的组合评分；
2. **时空—地形相干性**：空间连续性、时间连续性和近地地形边界一致性的组合评分。

> **重要声明**：WSCI、相干性组合方式及其数值权重是本项目定义的工程诊断指标，尚不是 WMO、IEC、CFD 或激光测风雷达行业标准。半变异函数、结构函数、风速方差、幂律廓线及地形无穿透条件具有已有物理/统计基础，但本仓库中的归一化函数和权重需要针对具体数据集进行标定。

## 1. 安装

```bash
git clone https://github.com/<YOUR_ACCOUNT>/wsci-wind-evaluation.git
cd wsci-wind-evaluation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

也可只安装依赖并从源码运行：

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m wind_eval.cli --help
```

## 2. 快速开始

生成示例数据：

```bash
python examples/generate_example.py
```

计算 WSCI：

```bash
wsci-eval examples/example_wind.npz \
  -o results/wsci.json \
  --csv results/wsci_summary.csv
```

等价的统一命令：

```bash
wind-eval wsci examples/example_wind.npz -o results/wsci.json
```

计算时空—地形相干性（假设相邻帧间隔 10 s）：

```bash
coherence-eval examples/example_wind.npz \
  --dt 10 \
  --max-space-lag 8 \
  --max-time-lag 5 \
  -o results/coherence.json \
  --csv results/coherence_summary.csv
```

等价命令：

```bash
wind-eval coherence examples/example_wind.npz --dt 10 -o results/coherence.json
```

## 3. 输入数据格式

输入必须是 NumPy `.npz` 文件。风场数组统一采用以下维度：

```text
(y, x)             单个二维风场
(z, y, x)          单个三维风场
(time, z, y, x)    三维时间序列
```

### 3.1 推荐字段

| 键名 | 维度 | 单位 | 必需性 | 说明 |
|---|---|---:|---|---|
| `u` | 上述任一形状 | m/s | 必需* | 东西向风速；没有 `v` 时也可表示标量风速 |
| `v` | 与 `u` 相同 | m/s | 推荐 | 南北向风速 |
| `w` | 与 `u` 相同 | m/s | 地形项必需 | 垂直风速 |
| `true_speed` | 与 `u` 相同 | m/s | 可选 | 独立参考风速，用于 (P_{var}) |
| `air_mask` | `(z,y,x)` 或 `(t,z,y,x)` | bool | 可选 | `True` 表示有效空气网格 |
| `z_abs_m` | `(z,)` | m | WSCI垂向项推荐 | 每层绝对高度；缺失时使用层索引 |
| `dem_m` | `(y,x)` | m | 地形项必需 | 数字高程模型 |
| `dx_m` | 标量 | m | 推荐 | x方向网格间距，默认1 |
| `dy_m` | 标量 | m | 推荐 | y方向网格间距，默认等于 `dx_m` |

\* 如果没有 `u`，也可使用 `pred` 或 `speed`。当 `u`、`v` 都存在时，评分变量为

\[
U_h=\sqrt{u^2+v^2}.
\]

`true` 或 `reference` 可作为 `true_speed` 的别名。参考场必须与评价场形状完全一致。

示例：

```python
import numpy as np

np.savez_compressed(
    "wind.npz",
    u=u, v=v, w=w,                    # (time,z,y,x)
    true_speed=true_speed,             # 可选
    air_mask=air_mask,                 # (z,y,x) 或 (time,z,y,x)
    z_abs_m=z_abs_m,                   # (z,)
    dem_m=dem_m,                       # (y,x)
    dx_m=30.0, dy_m=30.0,
)
```

### 3.2 时间间隔

时间间隔不从文件自动推断，必须通过命令行传入：

```bash
coherence-eval wind.npz --dt 10
```

表示相邻时间帧间隔为10秒。时间评分至少需要3帧，建议提供10帧以上。

## 4. 输出格式

主输出为 UTF-8 JSON，保存：

- 输入文件和标准状态声明；
- 原始权重及因缺失分项形成的有效权重；
- 各一级、二级指标；
- 半变异函数/时间结构函数的距离、数值和点对数；
- 最终 `0–1` 与 `0–100` 分数。

`--csv` 额外输出便于表格软件读取的分项摘要。命令运行时也会把完整 JSON 打印到标准输出，因此可以重定向：

```bash
wsci-eval wind.npz > wsci_stdout.json
```

## 5. WSCI 定义

### 5.1 总分

\[
S=0.35I_v+0.35I_s+0.30I_d,\qquad WSCI=100S.
\]

### 5.2 垂向廓线合理性

\[
I_v=0.60R^2_{profile}+0.25P_\alpha+0.15P_{smooth}.
\]

对水平平均风速廓线拟合幂律：

\[
\bar U(z)=az^\alpha.
\]

默认合理区间为 (0.05\le\alpha\le0.50)。区间只是一项可调工程先验，不应理解为所有稳定度和下垫面条件下的普适范围。

### 5.3 空间相关尺度

\[
\gamma(h)=\frac{1}{2N(h)}\sum_{\|r_i-r_j\|\approx h}
[U(r_i)-U(r_j)]^2.
\]

\[
I_s=0.55P_{mono}+0.35P_{nugget}+0.10P_{var}.
\]

\[
P_{mono}=\frac{\operatorname{corr}[h,\gamma(h)]+1}{2},\qquad
P_{nugget}=1-\frac{\gamma(h_1)}{\max_h\gamma(h)+\varepsilon}.
\]

有独立参考场时：

\[
P_{var}=\min\left(
\frac{\operatorname{Var}(U)}{\operatorname{Var}(U_{true})+\varepsilon},
\frac{\operatorname{Var}(U_{true})}{\operatorname{Var}(U)+\varepsilon}
\right).
\]

默认 `--missing-policy reweight`：没有真值时剔除 (P_{var})，将0.55和0.35重新归一化。兼容旧计算可使用：

```bash
wsci-eval wind.npz --missing-policy legacy-perfect
```

此时强制 (P_{var}=1)，但这表示“未惩罚”，并不表示方差已被验证为满分。

### 5.4 局部扰动结构

设 (U_{bg}=G_\sigma*U)，(U'=U-U_{bg})：

\[
P_{Rd}=\frac{\sigma(U')}{|\bar U|+\varepsilon},\qquad
C_d=\frac{\operatorname{Var}(G_\sigma*U')}{\operatorname{Var}(U')+\varepsilon},
\qquad I_d=P_{Rd}C_d.
\]

## 6. 时空—地形相干性

\[
C=0.45C_s+0.35C_t+0.20C_g.
\]

### 空间

\[
C_s=0.60P_{mono}+0.40P_{small}.
\]

### 时间

\[
\gamma_t(\tau)=\frac{1}{2N(\tau)}\sum_t[U(t+\tau)-U(t)]^2,
\]

\[
C_t=0.40P_{persist}+0.35P_{mono}+0.25P_{small}.
\]

### 地形

对地形表面 (z=H(x,y))，近地无穿透条件近似为：

\[
w\approx u\frac{\partial H}{\partial x}+v\frac{\partial H}{\partial y}.
\]

评分为：

\[
C_g=\exp\left[-\frac{\operatorname{RMSE}
(w-uH_x-vH_y)}{\operatorname{RMS}(|\mathbf V|)+\varepsilon}\right].
\]

时间或地形项不可用时，程序返回 `null` 并对其余一级权重重归一化，而不是赋予满分。

## 7. 权重为什么这样设置

| 层级 | 权重 | 项目设计理由 |
|---|---:|---|
| WSCI：(I_v/I_s/I_d) | 0.35/0.35/0.30 | 垂向和空间结构为主体，局部扰动略低以减少滤波尺度敏感性 |
| (I_v) | 0.60/0.25/0.15 | 廓线拟合使用全部高度点；指数范围和二阶平滑为辅助先验 |
| (I_s) | 0.55/0.35/0.10 | 多距离箱整体趋势优先；最短距离较易受噪声影响；方差仅作幅度校验 |
| 相干性：空间/时间/地形 | 0.45/0.35/0.20 | 空间覆盖全场；时间受采样间隔影响；地形项主要约束近地层 |

这些权重不是从引用论文直接得到的。建议在正式研究中通过专家标注、扰动试验、全局灵敏度分析或独立验证集重新标定，并报告权重变化对结论的影响。

## 8. 已知限制

- 各向同性空间半变异函数合并x/y方向，不能识别方向性相关尺度；
- 当前 (P_{mono}) 奖励规则增长，也可能给过度平滑场较高分；
- 地形项只检查近地无穿透一致性，不验证山脊加速、分离和尾流位置；
- 雷达径向速度、CFD水平风速和模型目标变量必须先统一物理量与网格；
- WSCI不替代MAE、RMSE、偏差、谱分析、守恒残差和不确定性分析。

## 9. 论文与方法依据

1. Frehlich, R., & Cornman, L. (2002). *Estimating Spatial Velocity Statistics with Coherent Doppler Lidar*. Journal of Atmospheric and Oceanic Technology, 19, 355–366. https://doi.org/10.1175/1520-0426-19.3.355
2. Wang, H., Barthelmie, R. J., Doubrawa, P., & Pryor, S. C. (2016). *Errors in radial velocity variance from Doppler wind lidar*. Atmospheric Measurement Techniques, 9, 4123–4139. https://doi.org/10.5194/amt-9-4123-2016
3. Vogelzang, J., King, G. P., & Stoffelen, A. (2015). *Spatial variances of wind fields and their relation to second-order structure functions and spectra*. Journal of Geophysical Research: Oceans, 120, 1048–1064. https://doi.org/10.1002/2014JC010239
4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

引用1–3支持使用结构函数、空间统计和速度方差诊断风场；引用4支持将物理约束用于数据驱动场重建。它们不规定本项目的WSCI公式或权重。

## 10. 上传 GitHub

```bash
git init
git add .
git commit -m "Initial release of WSCI wind evaluation"
git branch -M main
git remote add origin https://github.com/<YOUR_ACCOUNT>/wsci-wind-evaluation.git
git push -u origin main
```

发布前请在 `LICENSE`、`CITATION.cff` 和 `pyproject.toml` 中补充真实作者、单位和仓库地址。
