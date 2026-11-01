# CAD 路径、扫描局部映射与刀轴位姿

本流程先在 STEP 精确曲面上生成参考曲线，沿 CAD 法向在已配准扫描中拟合
局部表面，再导出接触点坐标与完整刀轴姿态。只处理几何；不包含机器人基座
坐标转换、IK、段间连接、进退刀、去除深度或设备执行。

## 运行

先完成主配准管线，保留同一批输入的 `T_scan_to_CAD.npy`、
`registration_quality.json`、`cad_mesh_clean.ply` 与输入元数据。依赖使用现有
`requirements.lock.txt`，本流程不要求 PyVista、ROS 或 OpenCAMLib。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  08_generate_cad_guided.py \
  --mapping-config cad_mapping.example.json \
  --run-dir output/cad_guided_example
```

运行目录必须为空，避免覆盖旧结果。原始 CAD、扫描和输出均保持在忽略目录，
不随源码公开。输入 STEP/扫描哈希必须与上游元数据一致。

## 选择 CAD 路径

默认从面积较大的有限候选面中，通过扫描支持对候选曲线排序，优先保留一条
法向变化明显的曲线，其余优先选择不同面。这里的路线用于几何流程验证，
尚未由用户确认是正式生产去毛刺区域。`max_reference_paths` 控制条数，
`path_sampling_mm` 控制沿线间隔。曲线在曲面的 UV 域中定义，再转换成三维，
因此圆柱等曲面上的曲线可以是空间圆弧。

可以在本机复制参数 JSON，并填写 `selections`，指定零起始 STEP 面编号与
UV 或 XYZ 控制点，两者只能提供一种：

```json
{
  "selections": [
    {"face_id": 0, "uv_controls": [[0.2, 0.3], [0.8, 0.3]]}
  ]
}
```

上述数值仅演示格式，必须使用自己模型的 UV 范围；完整运行仍需示例配置的
其他字段。XYZ 控制点使用 `xyz_controls`，会先投影到指定曲面的 UV 域。
多个控制点按 UV 折线连接，并非三维空间的无约束样条。程序精确裁剪曲面孔洞
和边界，分成独立路径。面编号由当前 STEP 导入遍历确定，与网格三角形编号
不同；模型修改后需重新确认。`face_catalog.json` 和 `selected_paths.json`
记录本次选择；自动选路还可能按扫描支持进一步截短。

## 扫描局部映射

1. 原始点云在输入坐标中体素化，使用已有刚性变换对齐到 CAD。
2. 在 CAD 路径点附近搜索有界邻域，限制切向半径和法向偏移。
3. 检测沿法向分离的多层表面，避免在薄壁两面之间任意选点。
4. 通过鲁棒加权平面/二次曲面拟合局部基体，抑制稀疏凸起和离群值。
5. 检查邻域是否从两维约束接触点、接触点是否处于观测凸包内，以及接触点
   附近是否有足够近的实际观测。凸包检查本身不能发现中央缺测孔，因此同时
   使用 `max_support_gap_mm` 检查，任何小于该阈值的缺失仍受采样分辨率限制。
6. 拒绝支持不足、噪声过大、法向分歧、偏移过大或歧义位置；位移或完整姿态
   不连续处拆段，孤立点不作为有效轨迹。

默认参数只是当前几何示例，不能视为经过设备验证的公差。主刀轴使用 CAD
解析法向，扫描拟合法向及差角只作诊断。该方法仅做法向修正，不能修正切向
边缘错位，也不对点云做非刚性变形。拟合残差、保留点比例不代表绝对定位
精度或面积覆盖率。CAD 路径保留为设计目标；映射点描述当前局部基体，二者
之间差值未被直接解释为去除深度。

## 位姿约定

全部长度为 mm。`T_scan_to_CAD` 满足 `p_C = R_CS @ p_S + t_CS`。
反变换的位置为 `R_CS.T @ (p_C-t_CS)`，法向为 `R_CS.T @ n_C`。
`SCAN_INPUT` 表示原始扫描文件保存的坐标，不默认是相机坐标或机器人基座坐标。

- 位姿原点：映射后的表面接触点；不是已标定 TCP。
- 工具 +Z：指向工件，等于朝工具侧法向的负方向。
- 工具 +X：路径前进方向在切平面的投影；+Y 按右手系构成。
- 四元数：`qx,qy,qz,qw`，主动旋转，将工具局部坐标映射到当前文件坐标。
- 齐次矩阵：列向量约定，前三列为工具 X/Y/Z，末列为接触点。
- `T_C_contact = T_scan_to_CAD @ T_S_contact`。
- 示例球心：`contact + radius * normal`。示例参数不是实际磨头或碰撞验证。

`approach_outward_C` 仅选择每条曲线上法向的统一符号；开放 CAD 的材料外侧
没有独立确认。`segment_id` 不同必须断开，各段间没有规划连接。所有导出行
均 `execution_allowed=False`，供几何研究与后续接口开发使用。

## 输出与验证

| 文件 | 用途 |
|---|---|
| `cad_reference_path.csv` | 原始 CAD 路径及解析法向，保留所有选定样本 |
| `path_mapping_diagnostics.csv` | 全量样本有效性、拒绝原因、拟合残差、修正量 |
| `tool_axis_poses_CAD.csv` | 有效接触点、刀轴三轴、XYZW 四元数，CAD 坐标 |
| `tool_axis_poses_CAD_compact.csv` | 接触点、刀轴方向及四元数精简表 |
| `tool_axis_poses_SCAN.csv` | 同一组位姿转回输入扫描坐标 |
| `tool_axis_transforms.npz` | 接触点与示例球心在两坐标中的齐次矩阵 |
| `CAD_scan_paths.html` | 内嵌 Plotly 的离线交互预览 |
| `CAD_path_overview.png` | CAD 上的路径与被拒绝位置 |
| `path_mapping_detail.png` | 局部路径修正及拟合诊断 |
| `pose_axes_detail.png` | 接触点、法向和工具轴方向 |
| `manifest.json` / `summary.json` | 输入/源码哈希、版本、统计和数值检查 |
| `path_summary.csv` | 每条路径的曲面类型、长度、样本数及法向变化 |

位姿检查包括旋转正交性、右手系、四元数与矩阵一致性、CAD/扫描往返、CSV
位置/方向/球心与矩阵一致性，以及段内距离和姿态步长。合成测试覆盖已知
曲面、刚体变换、缺测孔洞、裁剪孔洞和多层歧义。这些检查验证软件数值行为，
不替代实际设备定位验证。

参考接口：[CadQuery](https://github.com/CadQuery/cadquery)、
[SciPy Rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)、
[Plotly](https://github.com/plotly/plotly.py)。PyVista 交互选点界面暂未集成，
后续可在现有面/控制点接口上添加；自动选路可直接运行。
