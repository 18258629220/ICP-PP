# 本地输入数据

将你有权处理的 STEP 模型和扫描点云放在这里，并在根目录 `config.json`
中设置路径。示例文件名为 `model.step` 和 `scan.ply`。仓库不附带实际工件数据。

本目录除本说明外默认全部忽略。原始 CAD、点云以及包含真实几何的截图、
HTML、报告和衍生产物均不随代码发布。不要使用 `git add -f` 绕过忽略规则。

输入长度单位为 mm。STEP 必须显式声明毫米；完整管线要求 vertex-only
二进制 PLY，包含 float XYZ，以及 intensity、event_id、camera_column
标量字段。验证报告目前假设至少 10,000 个扫描点，且输入 XYZ 均有限。
不同点云格式需先调整读取与验证逻辑，不能仅改扩展名。
