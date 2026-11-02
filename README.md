# ICP-PP：CAD 引导的真实工件点云分割与磨抛规划

本项目围绕五个步骤处理真实工件：CAD—扫描配准、CAD 型面拓扑、点云型面分割、
表面磨抛覆盖路径，以及接触点与刀轴映射。输入为毫米 STEP 与扫描 PLY，
输出保留原始点对应关系和被拒绝区域，并提供可离线查看的真实工件三维结果。

当前范围到 CAD/输入扫描坐标系下的几何规划；输出 `execution_allowed=False`。
实际设备绝对精度、磨头接触工艺及机器人执行需另行验证。

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
| `face_topology.json` | CAD 面、孔洞与共享边目录；STEP 面号不同于三角形编号 |
| `scan_face_labels.csv` / `face_labels.npy` | 原始扫描逐点面标签、对应关系与不确定区域 |
| `surface_field.csv` | 实际表面拟合、法向偏差、支持质量及工艺区域 |
| `layer_*/CAD_scan_paths.html` | 可旋转的工件、参考路径、实际表面路径与被拒绝位置 |
| `layer_*/tool_axis_poses_CAD.csv` | 接触点 XYZ、刀轴方向、四元数及示例球心 |
| `REPORT_zh.md` / `summary.json` | 本次真实输入的处理数量、拒绝原因和数值检查 |
| `manifest.json` / `run_state.json` | 输入与源码版本、运行状态和是否完整完成 |

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
欧氏距离、法向偏差和工艺去除深度分别记录；未经确认的法向正负不能直接解释为余量。
每个实际运行记录输入哈希和代码版本，原始数据不因版本追踪而公开。
修改、验证、提交和同步遵循 [PROJECT_RULES.md](PROJECT_RULES.md)。

原创代码与通用文档采用 [MIT License](LICENSE)，依赖遵循各自许可证。
方法来源：[KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher)、
[Open3D](https://github.com/isl-org/Open3D)、[CadQuery/OCCT](https://github.com/CadQuery/cadquery)。
