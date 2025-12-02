# PromptLab · 本地 Prompt 工作台
Tauri + React 桌面端，内置 SQLite + 本地 NLP（jieba-rs），默认离线运行，可选接入 Qwen 剪贴板判定。

## 功能概览
- 剪贴板监听：自动入库，qwen-max 判定 `metadata.qwen_clipboard_pred`，仅在判定为非提示词且置信度≥阈值时拦截。
- 提示词管理：编辑/保存/删除，分页、当页全选、批量删除；正文保留原文。
- 本地分析：`summarize_prompt` 抽取摘要/主题/标签/角色/面向对象，写入 `analyses`。
- 星级与质量：列表/详情可设星级；标注 Prompt/Not prompt + 原因，写入 `metadata.is_prompt_label/label_reason`，可筛选。
- 导出/导入：CSV（UTF-8 BOM）导出最新分析/metadata，导入 CSV 按 body 去重，附带 summary/tags/classification 可写入一条分析。
- 自动阈值优化：累计 N 次判定（默认 20，可调）自动计算建议阈值并更新；阈值可手动设置。
- 词库：增删关键词，分析时加权。
- 托盘 & 单实例：关闭隐藏到托盘，单实例防重复启动。

## 目录
```
apps/
  desktop/   # Tauri + React 桌面端
  web/       # Vite 示例（未完备）
crates/
  core/      # Rust：SQLite 存储 + NLP
```

## 数据模型（简）
```sql
table prompts (
  id TEXT PRIMARY KEY,
  title TEXT,
  body TEXT,
  language TEXT,
  model_hint TEXT,
  metadata JSON,
  created_at DATETIME,
  updated_at DATETIME
);

table analyses (
  id TEXT PRIMARY KEY,
  prompt_id TEXT REFERENCES prompts(id) ON DELETE CASCADE,
  summary TEXT,
  tags JSON,
  classification JSON,
  qwen_model TEXT,
  created_at DATETIME
);
```

## 运行 / 打包
- Node 20.19+ 或 22.12+，Rust/Cargo。
```bash
cd apps/desktop
npm install
# 开发
npm run tauri:dev
# 打包
npm run tauri:build
```

## 环境变量（可选）
- `DASHSCOPE_API_KEY`: 剪贴板判定使用 qwen-max。
- `QWEN_PROMPT_THRESHOLD`: 判定拦截阈值，默认 0.1。
- `QWEN_OPTIMIZE_INTERVAL`: 自动优化阈值的判定次数间隔，默认 20。

## 前端操作要点
- 标注：详情面板标记 Prompt/Not prompt + 原因；列表可按标注状态筛选。
- 星级：列表右侧 1~5 星按钮写入 `metadata.star`。
- 自动优化次数：底部 Settings 可调整间隔，应用后生效。
- 导出/导入：留空路径使用默认目录；导入时填写 CSV 路径。

## 日志
- Windows：`%APPDATA%/com.promptlab.desktop/promptlab.log`
