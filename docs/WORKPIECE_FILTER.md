# 配准后工件筛选与 CAD 型面分割

`11_filter_and_segment.py` 复用已通过筛选的粗细配准结果，先排除远离 CAD 的扫描点，
再对保留点重新计算 CAD 型面归属。流程在型面分割后结束，不执行覆盖排线、曲面路径映射或刀轴规划。

## 开源方法选择

- [Open3D](https://github.com/isl-org/Open3D)：MIT，提供直接可用的 Python 点到三角网格距离查询，
  无需训练数据，适用于配准已知、形状复杂的 CAD。本入口采用这一实现。
- [CloudCompare](https://github.com/CloudCompare/CloudCompare)：支持
  [Cloud-to-Mesh 距离](https://cloudcompare.org/doc/wiki/index.php?title=Cloud-to-Mesh_Distance)
  和按距离标量范围提取点云，可作为独立复核工具。
- [PCL](https://github.com/PointCloudLibrary/pcl)：提供
  [ModelOutlierRemoval](https://pointclouds.org/documentation/classpcl_1_1_model_outlier_removal.html)，
  其标准 SAC 模型过滤适合平面、球、圆柱等模型；当前任意 STEP 曲面优先使用网格距离。

距离提取属于 CAD 条件前景筛选，不能将通用离群点滤波等同于夹具语义识别。

## 方法与阈值

- 使用 [Open3D RaycastingScene](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)
  的 `compute_distance` 计算扫描点到 CAD **三角面**的最近无符号距离，不采用 CAD 采样点距离。
- 距离查询只在内部将扫描坐标转至 CAD 工作区；导出的扫描 XYZ 始终保持原始机械臂基坐标。
- `--distance-mm` 显式设置工件邻域阈值。超过阈值的点写入剔除点云，便于复核。
  边界点在此阶段不因型面归属未知而删除；未网格化 CAD 面的近邻包围盒内采取保守保留。
- 对保留点逐块重新运行既有型面归属算法，检查最近面、物理边界、邻面竞争及未离散面附近的歧义。
  型面分割的距离阈值仍来自 `surface-config`，与工件筛选距离分别记录，不自动放宽。
- 无需训练数据和新增依赖。当前方法属于 CAD 几何邻域提取，无法凭距离区分贴近工件、
  处于阈值内的夹具点；不把所有剔除点自动解释为夹具，也不把保留比例解释为准确率。

## 运行

输入必须是原始 `ROBOT_BASE` 扫描，并与已有配准的 STEP、扫描哈希一致。
保存矩阵必须和 `registration_quality.json` 一致，且配准筛选状态必须为通过。
原配准目录由本机 `config.json` 的 `output_dir` 指定。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  11_filter_and_segment.py --config config.json \
  --surface-config surface_workflow.example.json \
  --cad-cache-dir output/cad_cache \
  --distance-mm 3 --run-dir output/filtered_segmentation_run
```

示例中的 `3 mm` 仅用于演示显式参数，实际值应按该批数据的距离分布和空间分布复核。
缓存目录需包含与输入 STEP 哈希、面号顺序和离散参数一致的 `face_topology.json`、
`cad_face_mesh.npz` 及可选边界缓存。省略缓存参数时，采用型面配置中的 `cad_cache_dir`；
两者均未设置时从 STEP 构造。运行目录必须为空，不能覆盖之前的分割或路径结果。

## 文件与逐点对应关系

| 文件 | 内容与行号含义 |
|---|---|
| `scan_workpiece_BASE.ply` | 距离筛选后保留点；原始 XYZ 和标量属性原样保留，按原行号顺序排列 |
| `scan_removed_BASE.ply` | 被筛选规则剔除的原始点；同样保留原属性，供检查是否误删 |
| `scan_segmented_BASE.ply` | 与保留点云逐行相同，增加 `cad_face_id` 和 RGB；所有已归属面按面号分色，未知点灰色 |
| `source_indices.npy` | 每个保留点在**原始未筛选扫描**中的零起始行号 |
| `removed_source_indices.npy` | 每个剔除点在原始扫描中的零起始行号 |
| `filter_keep_mask.npy` | 与**原始全部扫描点**等长的保留布尔掩码 |
| `cad_distance_mm.npy` | 与原始扫描等长的 CAD 网格距离，不是型面归属置信度 |
| `filter_protected_unmeshed.npy` | 与原始扫描等长的未网格化面近邻保守保护标记 |
| `face_labels.npy` | 与**新保留点云**逐行对应的 CAD 面号；`-1` 未确定归属，非负值为面号 |
| `point_diagnostics.npy` | 与新保留点云逐行对应；其中 `source_index` 仍是原始全扫描行号，保留候选面、UV、距离和拒绝原因 |
| `point_diagnostics_schema.json` | 诊断状态码、坐标含义和逐点对应规则 |
| `face_counts.csv` / `segmentation_summary.json` | 各已归属面点数、几何类型、面积和筛选统计 |
| `face_topology.json` | CAD 面号、相邻关系及边界信息；几何范围字段为 CAD 工作区，另附 CAD→BASE 变换 |
| `T_BASE_to_CAD.*` / `T_CAD_to_BASE.*` | 原有配准矩阵及其逆矩阵，未重新拟合 |
| `filter_summary.json` / `summary.json` | 原始、保留、剔除、已归属和保留未知点的数量；阈值及处理范围 |
| `verification.json` | 全部导出行的原始字段、标签、索引及点数一致性检查 |
| `manifest.json` / `resolved_config.json` | 输入及代码哈希、Git 版本、本次实际参数和停止阶段 |

例如 `source_indices[j] = i`，则新点云第 `j` 个点来自原始扫描第 `i` 个点，
其型面标签为 `face_labels[j]`。不要继续把新标签的 `j` 当成原始全扫描行号。

保持原始输入文件不变；“原始点序保留”表示每个子集内部的相对顺序不变，
不表示新点云仍具有原始点数。新入口不输出 `planned_surface`，因为本流程没有选择任何加工型面。

## 验证与限制

程序逐点核对三个 PLY 的原始字段与源点完全一致，检查保留/剔除索引完整划分原扫描，
并检查 `face_labels`、诊断和 PLY 面号一致。运行状态为 `completed` 且
`verification.json` 的 `passed=true` 才表示导出完成。
以上证明软件数据与坐标的一致性，不认证工件/夹具语义分类正确率或实物绝对精度。
