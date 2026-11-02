# 真实工件基坐标分割与随形接触规划

原始扫描文件的 XYZ 已由用户确认为机械臂基坐标 `ROBOT_BASE`，单位 mm。
本流程以该坐标系交付 CAD 定位结果、分割点云、接触路径和刀轴。
确认坐标来源本身不会提高已有配准精度；不加入额外手眼变换或设备执行指令。

## 坐标与变换

记 `C` 为 CAD 设计坐标，`B` 为原始扫描所在的机械臂基坐标。
原有配准保存 `p_C = R_CB @ p_B + t_CB`；兼容文件 `T_scan_to_CAD.npy`
仍保留，明确等于 `T_BASE_to_CAD.npy`。定位到现场使用其逆矩阵：

```text
T_CAD_to_BASE = inverse(T_BASE_to_CAD)
p_B = R_BC @ p_C + t_BC
n_B = R_BC @ n_C
R_B_contact = R_BC @ R_C_contact
```

位置旋转后加平移；法向、刀轴只旋转；四元数重新由基坐标姿态矩阵计算，顺序 XYZW。
`T_B_contact` 的前三列是接触框架的 X/Y/Z，末列为接触点 XYZ。
工具 +Z 指向表面，+X 沿路径切向，+Y 构成右手系。
接触原点不是已标定 TCP、法兰位置或磨头球心。

原始点云已经在 B 系，输出分割 PLY 时直接保留其 XYZ；不能再对这些点应用一次
CAD 到 B 的变换。主 CSV、图、点云和 CAD 必须使用相同的基坐标约定。
CAD 面号和 UV 仍描述设计曲面，不会因基坐标输出而重新编号或改成三角形编号。

## 处理和运行

1. 核对 STEP/扫描哈希、毫米单位、`scan_frame="ROBOT_BASE"` 和既有配准。
2. 保存双向变换，将 CAD 定位到原始扫描基坐标，并核对数值往返。
3. 内部使用 CAD 面/UV 进行全点分割、精确裁剪与局部扫描拟合。
4. 在有观测支持的主要型面上生成覆盖路径，保留孔洞和缺测处的断开。
5. 将接触点、参考点、法向、完整姿态一致转换为 B 系，生成主输出与基坐标三维图。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  09_plan_surface_workflow.py --config config.json \
  --surface-config surface_workflow.local.json \
  --run-dir output/real_base_run
```

目录必须为空。真实输入、本机配置、图像与结果只保存在本地。
已有矩阵经核对后直接复用逆变换，不为了改变表达坐标重新 ICP，也不缩放点云。
局部拟合默认参数统一保存在 `surface_workflow.example.json.mapping`，由本机型面配置的
`mapping` 覆盖；原有本机配置可保留其拟合参数。`coverage.path_sampling_mm` 统一控制路径采样。
`coverage.face_ids` 可指定型面；自动选择记录面积、支持数量和排除原因，不代表生产区域批准。
行距和采样间距是几何规划参数，当前不依赖磨头型号，也不声明实际接触面积全覆盖。

## 先看这些文件

| 文件 | 作用 |
|---|---|
| `REAL_WORKPIECE.html` | 基坐标真实扫描、CAD、型面颜色、接触路径和刀轴，可切换图层 |
| `real_overview.png` / `contact_axes_detail.png` | 基坐标静态总览和局部姿态 |
| `contact_poses_BASE.csv` | 接触点 XYZ、三轴、XYZW 四元数及 `frame=ROBOT_BASE` |
| `contact_transforms_BASE.npz` | 与 CSV 一一对应的基坐标接触框架矩阵 |
| `surface_field_BASE.csv` | 全部参考采样的映射结果、参考点、法向与拒绝原因；几何字段为基坐标 |
| `cad_reference_BASE.csv` | 基坐标 CAD 参考点以及不变的 STEP 面号、UV |
| `scan_segmented_BASE.ply` | 全分辨率分割标签；原始 XYZ、点序和标量属性保留 |
| `CAD_in_BASE.ply` | 定位到机器人现场坐标的 CAD |
| `T_CAD_to_BASE.*` / `T_BASE_to_CAD.*` | 双向定位矩阵 |
| `registration_BASE.json` | 坐标来源、矩阵含义、软件数值检查 |
| `FINAL_RESULT_zh.md` / `summary.json` | 处理数量、独立路径段与限制 |

`segment_id` 改变即表示路径断开，不能跨孔洞、缺测或姿态跳变直接连接。
所有位姿 `execution_allowed=False`；尚未包含机器人逆解、碰撞、进退刀和速度规划。
内部 CAD 位姿仅在 `export_cad_audit=true` 时保存为 `contact_poses_CAD_audit.csv`，
该审核文件不能与主 CSV 中的基坐标位置或方向混合使用。

## 支撑数据和质量检查

全分辨率分割采用带 CAD 面号来源的网格加速查询，距离、竞争、边界或未离散面附近
会拒绝自动归属。`face_labels.npy` 和 `point_diagnostics.npy` 与原始行号一一对应；
距离和 UV 等标量不会因坐标旋转改变。拒绝标签 -1，未处理标签 -2。
默认不导出数 GB 的逐点 CSV。

`exact_face_audit.csv/json` 将接受和拒绝样本与精确 STEP 投影对照；抽查一致性不等于
全体标签的真实正确率。`face_topology.json` 和 `cad_face_mesh.npz` 保存面、共享边、
孔洞、顶点 UV 和网格来源；`mesh_status=partial` 时未离散面仍有明确记录。

实际接触点和刀轴法向来自同一局部扫描曲面，保留设计与扫描法向诊断。
邻域不跨已归属型面，支持不足、多层歧义、外推和姿态不连续会拒绝或拆段。
CAD 工作区中的审核字段必须明确命名，不能与 `ROBOT_BASE` 主输出混用。

缓存按 STEP/扫描哈希、网格参数与支持采样参数核对；坐标来源确认不会改变这些输入。
运行验收检查 `run_state.json` 的 `status="completed"`、原始点属性/点序、双向矩阵、CSV/矩阵姿态一致性、
CAD 与基坐标往返误差，以及路径没有跨缺口连接。

外部扫描误差、配准误差与模型偏差没有独立测量时保持未知。
当前交付只证明软件几何处理和坐标表达的一致性，不认证 ≤0.2 mm 实物绝对精度。

当前仅保留全型面接触规划入口；旧球头去毛刺、代表性曲线入口及其专属依赖已移除。
精确 STEP 面/UV 几何、全点标签和必要诊断继续保留，检查集中在坐标、属性和有效路径上。
