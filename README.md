# ICP-PP：CAD 与扫描点云配准、偏差分析

使用 **KISS-Matcher 粗配准 + 鲁棒 Point-to-Plane ICP**，将扫描点云对齐到 STEP CAD，计算点到三角面的距离、局部法向偏差和扫描覆盖，并输出可离线查看的三维图。所有长度单位为 **mm**。

这是研究和工程初步分析代码。拟合残差和内部稳定性不能证明绝对测量精度；项目尚未完成 ≤0.2 mm 检测或机器人打磨的独立物理验证，不生成可直接执行的机器人轨迹。

## 安装与运行

已在 Linux、Python 3.10 环境验证。依赖版本记录在 `requirements.lock.txt`。STEP 和实际扫描数据不随代码发布，需要自备有权使用的数据。

```bash
git clone https://github.com/18258629220/ICP-PP.git
cd ICP-PP
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
cp config.example.json config.json
# 将模型和扫描放到 data/，编辑 config.json 的 step_path 和 scan_path
./run.sh
```

已有 `config.json` 时请直接编辑，不要覆盖本机配置。Debian/Ubuntu 的系统 Python 需要对应的 venv 支持；也可用自带 ensurepip 的 Python 3.10 创建环境。OCCT 网格和完整分辨率分析可能占用较多内存与磁盘，资源需求随输入规模变化。

当前完整管线要求：STEP 显式声明毫米；PLY 为仅含顶点的二进制文件，XYZ 为浮点毫米值，包含 `intensity`、`event_id`、`camera_column` 标量字段。现有报告验证要求至少 10,000 点；完整分辨率路径假设 XYZ 均有限。通用 XYZ-only、ASCII PLY、非有限值和较小点云需要先适配验证逻辑，详见 [数据说明](data/README.md)。

`config.example.json` 提供示例参数，`config.json` 为忽略的本机配置；运行时保存副本到输出目录。门槛用于配准筛查，`example_tolerance_mm = 1.0` 是演示值，不代表正式公差或测量能力。改变输入或参数后需重跑有关阶段，避免混用旧产物。

## 脚本与工作流

| 脚本 | 用途 |
|---|---|
| `01_convert_step.py` | 检查 STEP 单位，按绝对 OCCT 离散参数生成网格 |
| `02_register_and_analyze.py` | 预处理、多个 KISS 候选、两级 ICP、偏差和可视化 |
| `03_validate_and_report.py` | 检查输出一致性，生成中文报告 |
| `04_registration_sensitivity.py` | 初始姿态扰动与交替半数点拟合检查 |
| `05_assess_registration.py` | 分层残差、离群簇、局部几何敏感性、空间分区留出检查 |
| `scripts/check_spatial_holdout_control.py` | 全点重新拟合的对照检查 |
| `scripts/plot_assessment.py` | 绘制残差分布与空间留出对比图 |
| `investigate_registration.py` | 可选 CAD 可见性排查；ROI 阈值为特定几何的实验假设，使用前需调整 |

`run.sh` 顺序执行网格转换、完整配准分析、内部稳定性检查、合成测试与报告，遇错停止。额外分析在主流程完成后手动执行：

```bash
.venv/bin/python 05_assess_registration.py
.venv/bin/python scripts/check_spatial_holdout_control.py
.venv/bin/python scripts/plot_assessment.py
```

也可以按阶段运行：

```bash
.venv/bin/python 01_convert_step.py
.venv/bin/python 02_register_and_analyze.py --stage prepare
.venv/bin/python 02_register_and_analyze.py --stage register
.venv/bin/python 02_register_and_analyze.py --stage analyze
.venv/bin/python 02_register_and_analyze.py --stage plots
.venv/bin/python 04_registration_sensitivity.py
.venv/bin/python -m pytest -q tests
.venv/bin/python 03_validate_and_report.py
```

主流程读取配置中的输出目录。`run.sh` 的日志、绘图缓存和探索性可见性脚本目前使用默认 `output/`，修改输出目录时也需调整这些入口。额外诊断中的 5/2 mm ICP 阈值、1 mm Huber 和若干分层阈值沿用初始实验设置，变更主配准参数后应同步检查。CAD 抽样和显示抽样固定随机种子；KISS 的并行过程仍可能产生小幅候选差异。

## 查看结果

默认输出位于 `output/`，包括：

- `REPORT_zh.md`：自动生成的结果、验证记录和限制。
- `registration_3d.html`：对齐后的 CAD 与扫描。
- `deviation_heatmap_3d.html`、`deviation_heatmap_all_points_3d.html`：筛选后与全体点偏差视图。
- `cad_coverage_3d.html`：扫描覆盖估计；未覆盖区域不能解释为零偏差。
- `T_scan_to_CAD.npy/.txt`：原始输入坐标到 CAD 的刚体变换。
- `scan_aligned.ply`：保持原始点顺序和标量属性，仅变换 XYZ。
- `deviation_points.csv/.npy`：体素质心的诊断字段与质量掩码。
- `deviation_full_resolution.npy`：与原始 PLY 逐行对应的完整分辨率偏差。
- `assessment/`：额外分层统计、空间留出结果与诊断图。

HTML 内嵌 Plotly，可离线查看。真实输入、输出、报告、原始项目方案和本机配置均通过 `.gitignore` 排除，仅代码、示例配置、测试与公开说明进入 Git。

## 几何约定与解释

变换方向为 `p_C = R @ p_S + t`；NumPy 行向量形式为 `p_C = p_S @ R.T + t`。此矩阵只把当前输入扫描映射到 CAD，不等同于机器人手眼标定。

`euclidean_distance_mm` 是点到最近三角面的欧氏距离；`signed_deviation_mm` 是局部三角面法向投影，`absolute_deviation_mm` 是该投影的绝对值，不能与欧氏距离混用。开放曲面或未验证外法向的网格，其正负值不能直接解释为多料或缺料。

三角形编号索引 `cad_mesh_clean.ply`，不直接对应 STEP 面编号。质量掩码结合距离、切向残差与法向一致性，只是分析标记；全体统计同时保留。离群点过滤用于配准，不删除原始测量记录。

```python
import numpy as np
full = np.load('output/deviation_full_resolution.npy', mmap_mode='r')
print(full.dtype, full.shape)
print(full['euclidean_distance_mm'][:10])
T = np.load('output/T_scan_to_CAD.npy')
```

STEP 离散参数不是经认证的全局网格误差上界。最佳拟合会受覆盖、区域权重及真实形变影响；扰动或半数点重复拟合不是独立重复测量。面向 ≤0.2 mm 应用，还需稳定基准、独立参考测量、重复扫描、标定与完整误差预算。

## 验证

无需真实工件数据即可运行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests
.venv/bin/python -m pip check
```

现有 5 项合成测试覆盖大小端交错 PLY、属性与点顺序保持、截断文件拒绝、已知几何的法向符号与欧氏距离、已知刚体变换恢复。它们验证软件数值行为，不证明真实设备精度。Git 历史从当前开源基线开始，先前未提交的开发过程没有虚构历史。

## M7–M8：离线去毛刺路径

新增符号/量值与原始 STEP 曲面抽查、疑似凸起分区、OpenCAMLib 球形工具
接触路径和 CAD 坐标系位姿导出。安装与运行见 [M7–M8 说明](docs/M7_M8.md)。
实际配准精度、材料外侧和工具仍需独立验证；输出是带明确假设的几何预览。

## 开发与版本管理

修改和执行项目请遵循独立的 [代码修改与项目执行守则](PROJECT_RULES.md)。
根目录 [AGENTS.md](AGENTS.md) 为后续编程助手提供持续入口。
每个有意义的修改版本都需要验证、提交并同步；版本摘要见 [CHANGELOG.md](CHANGELOG.md)。

## 方法来源与许可证

- [KISS-Matcher 官方仓库](https://github.com/MIT-SPARK/KISS-Matcher)
- [Open3D RaycastingScene](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html)
- [CadQuery 导入导出文档](https://cadquery.readthedocs.io/en/latest/importexport.html)

本仓库原创代码与文档使用 [MIT License](LICENSE)。依赖包遵循各自许可证；私人 CAD、扫描与报告不包含在代码授权范围内。
