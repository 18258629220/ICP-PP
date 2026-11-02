# ICP-PP：CAD 引导的真实工件点云分割与磨抛规划

本项目围绕五个步骤处理真实工件：CAD—扫描配准、CAD 型面拓扑、点云型面分割、
主要型面整体随形磨抛覆盖路径，以及接触点与刀轴映射。输入为毫米 STEP 与扫描 PLY，
输出保留原始点对应关系和被拒绝区域，并提供可离线查看的真实工件三维结果。

当前交付是 CAD/输入扫描坐标系下的接触点与刀轴位姿，不依赖磨头型号。
所有输出保留 `execution_allowed=False`；实际设备绝对精度和机器人执行另行验证。

## 安装

已验证 Linux/Python 3.10。依赖版本见 `requirements.lock.txt`。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

`config.json` 和 `surface_workflow.local.json` 是本机配置，均不随代码公开。
已有配置请直接编辑，勿用示例覆盖；首次使用时复制 `config.example.json` 与
`surface_workflow.example.json`，将本机原始文件路径填入配置。

原始 STEP、扫描和运行输出保留在本地，源文件可以位于仓库外。当前扫描入口支持
仅含顶点的二进制 PLY，XYZ 单位为 mm；配准分析管线还读取 `intensity`、
`event_id`、`camera_column` 字段。格式说明见 [data/README.md](data/README.md)。

## 真实数据主流程

1. `01_convert_step.py`：检查毫米单位，生成 CAD 网格。
2. `02_register_and_analyze.py`：KISS-Matcher 粗配准、鲁棒点到面 ICP、偏差与覆盖分析。
3. `09_plan_surface_workflow.py`：复用同批输入的配准，生成面拓扑、全点标签、覆盖路径与刀轴。

已有配准时先核对 STEP/扫描 SHA-256；只有输入一致才复用已有矩阵。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  09_plan_surface_workflow.py --config config.json \
  --surface-config surface_workflow.local.json \
  --run-dir output/real_surface_run
```

目录必须为空；每轮保留独立目录，不能覆盖旧测量记录。
完整步骤、输出字段和参数见 [型面工作流说明](docs/SURFACE_WORKFLOW.md)。

## 先看哪些结果

| 结果 | 用途 |
|---|---|
| `REAL_WORKPIECE.html` | 首选结果：真实工件分割、主要型面、随形覆盖路径与刀轴的交互三维图 |
| `real_overview.png` | 真实工件分割与规划总览图 |
| `contact_poses_CAD.csv` / `contact_poses_SCAN.csv` | 接触点 XYZ、刀轴三轴及四元数；两种坐标系表达同一批位姿 |
| `point_diagnostics.npy` / `face_labels.npy` | 与原始 PLY 逐行对应的诊断与面标签，可按需读取，不生成巨大 CSV |
| `exact_face_audit.csv` / `.json` | 抽样点与精确 STEP 曲面的对照检查 |
| `face_topology.json` / `cad_face_mesh.npz` | CAD 面、孔洞、邻接及网格面号来源 |
| `surface_field.csv` | 路径附近实际表面拟合、法向、支持质量及拒绝原因 |
| `REPORT_zh.md` / `summary.json` | 真实输入的处理数量、主要型面、有效路径与数值检查 |
| `manifest.json` / `run_state.json` | 输入和源码版本、运行状态和是否完整完成 |

全分辨率分割使用保留 CAD 面号的三角网格加速；精确 STEP 对照抽查单独记录，
不把网格面标签说成每个点都经过精确曲面求解。逐点 CSV 默认关闭，完整 NPY 仍保留全部原始行。

未扫描区域保留为未知；不能理解为零偏差或无需加工。拟合残差、ICP 收敛和
数值往返误差不能代替独立的实物精度验证，不能据此宣称达到 ≤0.2 mm。

## 其他入口与轻量检查

`08_generate_cad_guided.py` 保留指定面和控制点曲线接口；
`03`–`05` 提供报告与配准诊断；`06`–`07` 为原有局部去毛刺几何接口。
相应说明见 [CAD 指定路径](docs/CAD_GUIDED_PATHS.md) 与 [M7–M8](docs/M7_M8.md)。

仓库已移除人工平板/圆柱工件生成器及其端到端演示，测试仅保留必要保障：
点序和属性、孔洞/薄壁、面号来源、坐标与法向，以及分块输出一致性。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests
.venv/bin/python -m pip check
```

## 几何和版本约定

变换固定为 `p_C = R @ p_S + t`，行向量形式为 `p_C = p_S @ R.T + t`。
欧氏距离、法向偏差与接触点坐标分别记录；未经确认的法向正负不能直接解释为余量。
每个实际运行记录输入哈希和代码版本，原始数据不因版本追踪而公开。
修改、验证、提交和同步遵循 [PROJECT_RULES.md](PROJECT_RULES.md)。

原创代码与通用文档采用 [MIT License](LICENSE)，依赖遵循各自许可证。
方法来源：[KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher)、
[Open3D](https://github.com/isl-org/Open3D)、[CadQuery/OCCT](https://github.com/CadQuery/cadquery)。
