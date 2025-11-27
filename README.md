# PromptLab · 本地 Prompt 工作台

Tauri + React 的本地 Prompt 管理工具，内置 SQLite 与本地 NLP（jieba-rs），默认离线运行，无需云端 API。

## 功能概览
- 提示词管理：编辑/保存/删除，历史分页（5~100 条/页），当页全选与批量删除；提示词正文始终保留原文。
- 本地分析：`summarize_prompt` 提取摘要、主题/角色/对象、标签，只展示最新一次分析（历史记录仍保存在 `analyses` 表）。
- 剪贴板监听：自动过滤过短/过长/多行聊天/大量 URL 文本；原文写入 `prompts.body`，分析写入 `analyses`，metadata 记录 `raw/structured/tags/theme/role/targets` 便于溯源。
- 词库：增删自定义词条（ASCII 自动小写），分析时加权关键词。
- 导出：CSV 写入 UTF-8 BOM，Excel 直接打开不乱码，包含最新分析的摘要/标签/分类字段。
- 托盘 & 单实例：窗口关闭仅隐藏到托盘；托盘右键确认退出；左键恢复窗口；启用单实例，重复启动只激活已运行实例。

## 目录结构
```
apps/
  desktop/   # Tauri + React 桌面端
  web/       # Vite 示例（未完备）
crates/
  core/      # Rust 底层：SQLite 存储 + NLP
```

## 数据模型
```sql
table prompts (
  id TEXT PRIMARY KEY,
  title TEXT,
  body TEXT,              -- 原始提示词正文
  language TEXT,
  model_hint TEXT,
  metadata JSON,          -- 可能包含 {source, raw, structured, tags, theme, role, targets}
  created_at DATETIME,
  updated_at DATETIME
);

table analyses (
  id TEXT PRIMARY KEY,
  prompt_id TEXT REFERENCES prompts(id) ON DELETE CASCADE,
  summary TEXT,
  tags JSON,
  classification JSON,    -- 含 topic/theme/role/targets/keywords/length/source
  qwen_model TEXT,
  created_at DATETIME
);

table attachments (
  id TEXT PRIMARY KEY,
  prompt_id TEXT REFERENCES prompts(id) ON DELETE CASCADE,
  filename TEXT,
  bytes BLOB
);
```
索引：`idx_prompts_updated_at`、`idx_prompts_created_at`、`idx_analyses_prompt_id_created_at`、`idx_attachments_prompt_id`。

## 运行与构建
环境要求：Node 20.19+ 或 22.12+，已安装 Rust/Cargo。

```bash
cd apps/desktop
npm install

# 开发
npm run tauri:dev
# DevTools: 窗口按 Ctrl+Shift+I

# 打包
npm run tauri:build
```

## 使用要点
- 剪贴板监听：过滤过短/过长/多行聊天/大量 URL；原文存 `prompts.body`，分析存 `analyses`，编辑器总是显示原文。
- 分析展示：历史列表按时间倒序，仅回填最新一条分析；完整历史仍在 `analyses` 可追溯。
- 导出：CSV 自带 BOM，Excel 中文不乱码。
- 历史操作：分页可调、当页全选、批量删除。
- 托盘/关闭：关闭按钮仅隐藏到托盘；托盘右键确认退出；左键恢复窗口；单实例防重复启动。
- 日志路径（Windows）：`%APPDATA%/com.promptlab.desktop/promptlab.log`，便于排查。

## 后续规划
- 按主题/标签筛选与搜索。
- 剪贴板监听开关与“仅分析不落库”模式。
- Web/CLI/HTTP API 衔接。
