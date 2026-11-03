# ICP-PP：CAD 引导的真实工件分割与基坐标随形规划

原始扫描点云已由用户确认为机械臂基坐标系 `ROBOT_BASE`，长度单位为 mm。
程序将 CAD 定位到该基坐标系，生成真实工件型面标签、随形接触点和刀轴位姿。
原始扫描 XYZ、点序与标量属性保持原样；主要结果统一标为 `ROBOT_BASE`。

当前输出是表面接触框架，不依赖磨头型号，也不是已标定 TCP 或机器人执行程序。
所有位姿保留 `execution_allowed=False`；基坐标来源确认不等于独立的实物精度认证。

## 安装与输入

已验证 Linux/Python 3.10，依赖版本见 `requirements.lock.txt`。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

首次使用从 `config.example.json` 与 `surface_workflow.example.json` 复制本机配置，
填写 STEP/PLY 路径，并确认 `scan_frame="ROBOT_BASE"`。已有本机配置勿覆盖。
局部扫描拟合参数统一放在型面配置的 `mapping` 中；覆盖间距在 `coverage` 中设置。
输入可在仓库外；原始几何、本机配置和输出不随代码公开。
当前扫描支持仅含顶点的二进制 PLY；格式见 [data/README.md](data/README.md)。

## 主流程

1. `01_convert_step.py`：检查毫米单位并生成 CAD 几何。
2. `02_register_and_analyze.py`：配准和偏差分析，并保存 CAD 到基坐标的定位变换。
3. `09_plan_surface_workflow.py`：复用同批配准，完成全点分割、主要型面覆盖与基坐标刀轴输出。

已有配准只在 STEP/扫描哈希与保存矩阵一致时复用。确认坐标含义后，无需为了改输出坐标
重新拟合一次 ICP；使用原矩阵的逆变换即可将 CAD 与路径定位到真实扫描基坐标。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  09_plan_surface_workflow.py --config config.json \
  --surface-config surface_workflow.local.json \
  --run-dir output/real_base_run
```

运行目录必须为空，保留每轮记录。说明见 [型面工作流](docs/SURFACE_WORKFLOW.md)。

## 主要结果

| 文件 | 含义 |
|---|---|
| `path_details/PATH_DETAILS.html` | 三条代表曲线路径的 CAD／实测曲面对照，放大完整 XYZ 刀轴、读取接触位姿 |
| `path_details/cad_scan_surface_comparison.png` / `P*_contact_axes_detail.png` | CAD／实测曲面对照图与连续局部刀轴放大图 |
| `REAL_WORKPIECE.html` / `real_overview.png` | 在机械臂基坐标中查看真实扫描、定位后的 CAD、分割与接触路径 |
| `contact_poses_BASE.csv` | 基坐标接触点 XYZ、刀轴三轴和 XYZW 四元数 |
| `contact_transforms_BASE.npz` | 基坐标下的接触框架齐次矩阵，列向量约定 |
| `surface_field_BASE.csv` | 全部参考采样的基坐标位置、法向和拟合/拒绝诊断 |
| `cad_reference_BASE.csv` | 定位到基坐标的 CAD 参考点，保留面号和 UV |
| `scan_segmented_BASE.ply` | 原始扫描 XYZ 和属性保持不变，附型面标签和显示颜色 |
| `CAD_in_BASE.ply` | 已定位到基坐标的 CAD 网格 |
| `T_CAD_to_BASE.npy/.txt` / `T_BASE_to_CAD.npy/.txt` | CAD 与基坐标的双向刚体变换 |
| `registration_BASE.json` | 坐标来源、变换定义及数值检查 |
| `FINAL_RESULT_zh.md` / `summary.json` | 本次处理数量、有效路径及限制 |

内部保留 CAD 曲面参数域用于面号、UV 和精确裁剪；主输出按基坐标表达。
仅在 `export_cad_audit=true` 时另导出 `contact_poses_CAD_audit.csv`，用于核对内部 CAD 工作区。
`T_BASE_to_CAD` 与兼容文件 `T_scan_to_CAD` 数值相同，`T_CAD_to_BASE` 为其逆矩阵。
位置需要旋转和平移，方向只旋转，姿态左乘 CAD 到基坐标的旋转；原始扫描不能重复变换。

全点分割用保留 CAD 面号的网格加速，精确 STEP 抽查单独记录。
未扫描、边界歧义与支持不足处保留未知，路径不跨缺口连接。
ICP 残差、拟合残差和数值往返误差不等于独立测量精度，不能据此宣称 ≤0.2 mm 达标。

已有运行结果可单独重绘，无需重新配准或规划：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  10_visualize_surface_paths.py --run-dir output/real_base_run \
  --output-dir output/path_visuals_v2
```

打开新目录的 `PATH_DETAILS.html`，选择单条路径即可查看曲面和稀疏三轴。
可重复传入 `--path-id` 指定一至三条路径，`--axis-length-mm` 只改变箭头显示长度。
实测曲面由同型面扫描支持点局部三角化，仅作显示；原始 CSV/NPZ 和路径偏差均不改变。

## 精简与检查

已移除重复的 M7/M8 球头去毛刺、代表性 CAD 曲线入口及其专属 OpenCAMLib/Shapely 依赖。
局部拟合与基坐标导出共用全型面流程，配置不再依赖单独的旧路径示例文件。
`03`–`05` 保留配准报告和必要诊断。关键检查覆盖点序/属性、孔洞/薄壁、面号来源和坐标方向。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests
.venv/bin/python -m pip check
```

修改、验证、版本和同步遵循 [PROJECT_RULES.md](PROJECT_RULES.md)。
原创代码与通用说明采用 [MIT License](LICENSE)，依赖遵循各自许可证。
方法来源：[KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher)、
[Open3D](https://github.com/isl-org/Open3D)、[CadQuery/OCCT](https://github.com/CadQuery/cadquery)。
