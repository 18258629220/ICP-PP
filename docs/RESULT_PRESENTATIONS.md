# 使用现有模板制作阶段成果 PPT

`scripts/build_template_presentation.mjs` 读取本地 PPTX 模板和私有 JSON 配置，
保留所选页面的尺寸、主题与背景，通过形状名称更新可编辑文字，并插入实际
结果图片。它使用桌面环境提供的 `@oai/artifact-tool` 与 `sharp`，不修改模板原件。

模板、配置、实际测量数字、工件图片、演示文稿和校验记录放在已忽略的
`output/` 目录。本仓库只保存通用生成代码，不公开这些材料。

## 配置与运行

JSON 顶层字段包括 `template`、`workspaceDirectory`、`buildDirectory`、
`finalPath`、`slideSizeEmu`、`fontFamily` 和 `slides`。路径相对于 JSON 所在目录
解析，或使用绝对路径。每页的 `templateSlide` 使用从 1 开始的源页码。

每页可包含：

- `removeNames`：删除不再使用的原页面元素名称。
- `edits`：按唯一形状名称 `name` 更新 `text`、`position`、`style`。
- `texts`：添加可编辑图注，包含 `name`、`text`、`position` 与可选 `style`。
- `images`：嵌入本地图片，包含 `path`、`alt`、`position` 和可选像素裁剪
  `crop: {left, top, width, height}`。裁剪时保留解释图像所需的信息。
- `notes`：记录每页数据来源、统计口径、图片来源及重要适用范围。

位置和字体数字使用 96 DPI 像素。字体跟随模板；统计表述须区分最近距离、
法向偏差、质量筛选比例及独立测量精度。不要将候选毛刺或几何预览写成
已确认的缺陷或可直接执行的机器人程序。

先使用工作区依赖工具提供的 Node 路径和模块目录构建：

```bash
RUNTIME_NODE_MODULES=/path/to/bundled/node_modules /path/to/bundled/node \
  scripts/build_template_presentation.mjs output/my_deck/deck_spec.json
```

检查生成目录内每一页的预览，与原模板逐页比较。必要时修改私有配置并重新生成。
随后设置 `PRESENTATION_SKILL_DIR` 和 `RUNTIME_PYTHON` 为当前演示文稿技能及
捆绑 Python 的实际路径，使用同一命令附加 `--finalize`。最终文件必须使用尚未
存在的路径。程序执行包结构、页数、字体、尺寸及可再次导入检查，并渲染最终文件。

该脚本适用于以普通命名形状为主的模板。本次没有实现复杂嵌入图表与跨页链接的
复制适配；其他模板使用前仍应检查结果。自动检查不能替代图片、统计和排版审阅。
