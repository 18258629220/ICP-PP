# CAD 型面分割、覆盖路径与刀轴映射

本版本接续用户已有的 08 精确曲线映射，补齐前述方案第 1–5 步的几何工作流。
第 6 步机器人基座/逆解/运动规划、第 7 步加工与复测闭环不在本次范围。
输出始终 `execution_allowed=False`。长度均为 mm，变换仍为 `p_C=R@p_S+t`。

## 五步处理链路

1. 保留 CAD 面、共享边、孔洞和三角网格面号来源。
2. 用配准后的真实扫描建立点到 CAD 型面的对应，保留歧义与未覆盖区域。
3. 在同型面内拟合实际表面位置、法向和曲率。
4. 在有扫描支持的型面上生成覆盖路径，处理孔洞、边界和保护区。
5. 将参考接触点映射至实际表面，输出接触点、刀轴及完整姿态。

## 运行真实数据

先完成现有配准；本流程不重新拟合、不改变 `T_scan_to_CAD.npy`。
校验原始 STEP/扫描哈希及保存的变换，输入改变时拒绝复用旧配准。

```bash
cp surface_workflow.example.json surface_workflow.local.json
# 根据实际几何与工艺编辑本机文件，优先指定 coverage.face_ids 控制覆盖范围。
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  09_plan_surface_workflow.py --config config.json \
  --surface-config surface_workflow.local.json \
  --run-dir output/surface_workflow/my_new_run
```

运行目录必须为空。真实输入、配置、原始点标签、图像与路径结果均留在本地忽略目录。
不要把真实面号/工艺方案写进公开示例配置。

`coverage.face_ids=null` 对每个扫描支持足够的面建立覆盖线；不是确认生产加工区。
复杂模型可能超过路径样本上限，此时明确报错，保留已经写出的分割结果。
应指定型面范围或调整采样，而不是静默截断。分割依然处理全部原始点；
`segmentation_chunk_points` 限制每块诊断表规模，不能改变点序或跳过点。

## 输出关联

| 文件 | 含义 |
|---|---|
| `face_topology.json` | 面/共享边/孔洞/邻接与导入遍历编号 |
| `cad_face_mesh.npz` | 独立按面离散的顶点/三角形、triangle_face_ids、vertex_uv、vertex_face_ids |
| `scan_face_labels.csv` | 与输入 PLY 逐行对应的 source_index、面标签、UV、投影、偏差、边界距离及拒绝原因 |
| `face_labels.npy` | 全点 int32 面标签，拒绝点 -1，分块写入 |
| `support_provenance.csv` | 拟合支持点到原始行号的来源；每面/体素保留真实点，不跨薄壁平均 |
| `cad_coverage_reference.csv` | 所有参考路径样本及精确 CAD UV |
| `coverage_sampling.json` | 每面线数、采样度量行距及非认证标记 |
| `surface_field.csv` | 沿覆盖线采样的实际局部型面、法向、误差、保护/余量/未知及 region_id |
| `layer_*/path_mapping_diagnostics.csv` | 每层全部样本与最终有效性；断点不会被连回 |
| `layer_*/tool_axis_poses_CAD.csv` / `SCAN.csv` | 接触点、工具三轴、XYZW 四元数、示例球心 |
| `layer_*/tool_axis_transforms.npz` | CAD/输入扫描坐标接触框架矩阵 |
| `summary.json` / `manifest.json` | 统计、数值检查、输入与代码哈希、commit/dirty 状态及耗时 |

`surface_field.csv` 是覆盖线上的采样场，不是每个 UV 位置都有可信数据的连续全域曲面。
`face_id=-1` 不等于没有缺陷；置信度只是几何诊断分数，未经概率标定。
STEP 面号不能用作原有 clean mesh 的三角形索引；新 NPZ 自己保存对应关系。
CAD UV 不等于 Open3D 的三角形重心坐标。
个别有效精确面可能没有 OCCT 三角网格；目录会记录 `mesh_status=partial`、
`mesh_warnings` 与未离散面号，精确面仍参与分割，后续面号不会重新编号。
此时新离散网格不是完整几何包络，不能用作完整碰撞或覆盖证明。
`run_state.json` 记录运行中、失败或完成；中断时 NPY 未处理行保持 -2，拒绝行是 -1。

## 参数与工艺判定

`mapping.normal_mode=scan` 用同一鲁棒局部高度曲面的导数生成刀轴；`cad` 保持旧模式。
两者均保存 CAD/扫描法向。曲率通过第一/第二基本形式计算，包含非零坡度修正。
缺测、多层、外推、噪声或姿态不连续会拒绝或拆段，邻域只允许已归属到同一 CAD 面的点。

`scan_sigma_mm`、`registration_sigma_mm` 是外部提供的沿 CAD 法向标准不确定度，
与局部预测项按独立条件 RSS 合成。默认 null，不会从很小拟合残差推断精度。
该预算没有姿态协方差、杆臂效应、相关误差和模型偏差，不能证明真实绝对精度。

法向不确定度不等于切向边界误差。`process.boundary_position_bound_mm` 需独立提供
边界空间误差上界；默认 null。边界保护裕度为配置接触半径、保护间隙、该上界及
跟随误差上界之和。接触半径是工艺输入，与 `example_ball_radius_mm` 的示例球半径不同。
该保护只是几何预览，尚无完整磨头/刀杆包络和接触变形证明。

只有外部法向误差、边界误差及 `material_side_confirmed=true` 明确时，偏差下界超过
`allowance_threshold_mm` 且通过保护检查的样本才标成 allowance_candidate。
否则保留 uncertain_metrology、material_side_unconfirmed、boundary_uncertainty_unknown
等状态。禁止把字符串 "false" 当材料侧确认。

`protected_uv_boxes` 可配置 `{face_id, bounds:[umin,umax,vmin,vmax]}`，
保护区域覆盖余量判定。区域按同面、位置距离、法向一致性及精确 UV 裁剪可见性连接；
小于 `minimum_region_samples` 的孤立区域保留诊断，不自动加工。
区域拓扑精度受路径采样和连接半径限制，不能识别采样间未观测的小缺陷。

`layer_depths_mm` 必须从 0 严格递增，0 层表示当前拟合基体。正层仅在 scan 法向模式
及材料侧确认后生成沿实际法向的常量偏置，并要求偏差下界覆盖深度、边界增加深度裕度、
局部 `depth*max_abs_principal_curvature<0.5`。这只是局部平行曲面正则性检查；
未定义实际单次去除能力、力、进给和工具整体碰撞，不能直接用于设备。

## 真实运行验收

先检查 `manifest.json` 的 STEP、扫描及配准矩阵哈希，确认所有文件来自同一轮输入。
再检查 `run_state.json` 为 completed，而不是 running/failed；中断结果不能当作全量输出。

验收主要看三方面：

- 分割：全体原始点都有对应记录，报告确定归属、歧义、边界和离群数量；三维图按型面着色。
- 路径：每段都在有支持的范围内；孔洞、缺测和不连续处断开，保留未规划区域。
- 位姿：接触点贴合局部实际型面，刀轴与所选表面法向一致，坐标往返和 CSV/矩阵检查通过。

未经独立参考测量，不把“成功归属比例”当作分割正确率，也不把局部拟合残差当成设备精度。
工具接触半径、正式工艺区域和去除深度须在实际工艺条件下确认。

人工平板/圆柱工件生成器和相关端到端演示已删除。只保留点序/属性、孔洞/薄壁、
法向/坐标和分块一致性等必要检查，开发重点放在真实数据的处理质量与可视化结果。
依赖仍使用现有环境，不引入 ROS、机器人模型或新训练网络。
