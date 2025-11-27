# PromptLab · 本地 Prompt 工作台

Tauri + React 的本地 Prompt 管理工具，内置 SQLite 与本地 NLP（jieba-rs）分析；默认离线，无需云端 API。

## 功能概览
- 提示词管理：编辑/保存/删除，历史分页、当页全选和批量删除；提示词正文始终保留原文。
- 本地分析：summarize_prompt 提取摘要、主题/角色/对象、标签，显示最新一次分析（历史记录仍保存在 analyses 表）。
- 剪贴板监听：自动抓取文本（长度/行数/URL 过滤）→ 原文存入 prompts，分析结果写入 analyses，metadata 保存 raw/structured 便于溯源。
- 词库：增删自定义词条（ASCII 自动小写），分析时加权关键词。
- 导出：CSV 写入 UTF-8 BOM，兼容 Excel；包含最新分析摘要/标签/分类字段。
- UI：历史分页、复选框批量删；分析面板展示最新分析；支持角色/主题/对象/标签。

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
环境要求：Node 20.19+ 或 22.12+，Rust/Cargo 可用。

```bash
# 安装依赖
cd apps/desktop
npm install

# 开发
npm run tauri:dev
# 打开 DevTools: 窗口中按 Ctrl+Shift+I

# 打包
npm run tauri:build
```

## 使用要点
- 剪贴板监听：自动过滤过短/过长/多行聊天/大量 URL；原文写入 prompts，分析写入 analyses；提示词编辑器显示原文，不混入分析内容。
- 分析展示：历史列表仅显示最新一条分析（按时间倒序），仍保留完整记录供追溯。
- CSV 导出：写入 BOM，确保 Excel 正常显示中文。
- 历史操作：分页（可调 5~100 条/页）、当页全选、批量删除。
- 日志路径：`%APPDATA%/com.promptlab.desktop/promptlab.log`。

## 计划与扩展
- 支持按主题/标签筛选/搜索。
- 更多剪贴板策略与开关（仅分析不落库、前台时才监听）。
- Web/CLI/HTTP API 衔接。
