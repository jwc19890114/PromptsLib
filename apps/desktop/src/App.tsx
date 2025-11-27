import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

type PromptAnalysis = {
  summary: string;
  suggestedTags: string[];
  length: number;
  topic: string | null;
  theme: string | null;
  role: string;
  targetEntities: string[];
};

type StoredPrompt = {
  id: string;
  title: string;
  body: string;
  language: string | null;
  model_hint: string | null;
  metadata: unknown;
  created_at: string;
  updated_at: string;
};

type AnalysisClassification = {
  topic?: string;
  targets?: string[];
  keywords?: string[];
  token_count?: number;
  length?: number;
  source?: string;
  [key: string]: unknown;
};

type AnalysisRecord = {
  id: string;
  prompt_id: string;
  summary: string;
  tags: string[];
  classification: AnalysisClassification | null;
  qwen_model: string | null;
  created_at: string;
};

const deriveTitle = (body: string) => {
  const firstLine = body.split("\n")[0]?.trim() ?? "";
  if (!firstLine) return "新建 Prompt";
  return firstLine.slice(0, 80);
};

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [analysis, setAnalysis] = useState<PromptAnalysis | null>(null);
  const [history, setHistory] = useState<StoredPrompt[]>([]);
  const [analysisLog, setAnalysisLog] = useState<AnalysisRecord[]>([]);
  const [activePromptId, setActivePromptId] = useState<string | null>(null);
  const [status, setStatus] = useState("就绪");
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [topicOverride, setTopicOverride] = useState("");
  const [tagInput, setTagInput] = useState("");
  const [vocabulary, setVocabulary] = useState<string[]>([]);
  const [newVocab, setNewVocab] = useState("");
  const [isVocabBusy, setIsVocabBusy] = useState(false);
  const [exportPath, setExportPath] = useState("");
  const [lastExportPath, setLastExportPath] = useState("");
  const [historyPage, setHistoryPage] = useState(1);
  const [pageSize, setPageSize] = useState(8);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const tokenCount = useMemo(() => prompt.trim().split(/\s+/).filter(Boolean).length, [prompt]);

  const parseTags = (value: string) =>
    Array.from(
      new Set(
        value
          .split(/[,\uFF0C\u3001;\uFF1B\s]+/)
          .map((token) => token.trim())
          .filter(Boolean)
          .map((token) => (/[A-Za-z]/.test(token) ? token.toLowerCase() : token)),
      ),
    );

  const currentTags = useMemo(() => parseTags(tagInput), [tagInput]);
  const displayTags = analysis ? (currentTags.length > 0 ? currentTags : analysis.suggestedTags) : [];
  const topicDisplay = topicOverride.trim() || analysis?.theme || analysis?.topic || "";
  const totalPages = Math.max(1, Math.ceil(history.length / pageSize));
  const pagedHistory = useMemo(() => {
    const start = (historyPage - 1) * pageSize;
    return history.slice(start, start + pageSize);
  }, [history, historyPage]);

  useEffect(() => {
    if (historyPage > totalPages) {
      setHistoryPage(totalPages);
    }
  }, [historyPage, totalPages]);

  useEffect(() => {
    setSelectedIds((prev) => prev.filter((id) => history.some((item) => item.id === id)));
  }, [history]);

  const persistPrompt = async () => {
    if (!prompt.trim()) {
      throw new Error("请输入描述词后再进行保存或分析");
    }
    const payload = {
      title: deriveTitle(prompt),
      body: prompt.trim(),
      language: null,
      model_hint: null,
      metadata: null,
    };
    let record: StoredPrompt;
    if (activePromptId) {
      record = await invoke<StoredPrompt>("update_prompt", { id: activePromptId, payload });
    } else {
      record = await invoke<StoredPrompt>("save_prompt", { payload });
    }
    setActivePromptId(record.id);
    setHistory((prev) => {
      const filtered = prev.filter((item) => item.id !== record.id);
      return [record, ...filtered];
    });
    return record;
  };

  const refreshHistory = async () => {
    try {
      const records = await invoke<StoredPrompt[]>("list_prompts");
      setHistory(records);
      if (activePromptId) {
        const active = records.find((item) => item.id === activePromptId);
        if (!active) {
          setActivePromptId(null);
          setAnalysisLog([]);
        }
      }
    } catch (error) {
      console.error(error);
      setStatus("读取历史失败，请查看日志");
    }
  };

  const fetchVocabulary = async () => {
    try {
      const list = await invoke<string[]>("list_vocabulary");
      setVocabulary(list);
    } catch (error) {
      console.error("vocabulary", error);
    }
  };

  useEffect(() => {
    refreshHistory();
    fetchVocabulary();
  }, []);

  const loadAnalyses = async (promptId: string) => {
    try {
      const items = await invoke<AnalysisRecord[]>("list_analyses", { promptId });
      console.log("list_analyses", promptId, "rows", items.length, items[0]);
      setAnalysisLog(items);
      const latest = items[0] ?? null;
      if (latest) {
        const cls = (latest.classification || {}) as AnalysisClassification;
        const targets = Array.isArray(cls.targets)
          ? (cls.targets as string[]).filter((t) => typeof t === "string" && t.trim().length > 0)
          : [];
        setAnalysis({
          summary: latest.summary,
          suggestedTags: latest.tags,
          length: cls.length ?? latest.summary.length,
          topic: cls.topic ?? null,
          theme: (cls as any).theme ?? cls.topic ?? null,
          role: (cls as any).role ?? "",
          targetEntities: targets,
        });
      } else {
        setAnalysis(null);
      }
    } catch (error) {
      console.error("list_analyses", error);
      setStatus("获取分析记录失败");
    }
  };

  const handleNewDraft = () => {
    setPrompt("");
    setActivePromptId(null);
    setAnalysis(null);
    setAnalysisLog([]);
    setTopicOverride("");
    setTagInput("");
    setHistoryPage(1);
    setSelectedIds([]);
    setStatus("已切换到新草稿");
  };

  const handleAnalyze = async () => {
    if (!prompt.trim()) {
      setStatus("请先写下提示词");
      return;
    }

    setIsLoading(true);
    setStatus("正在提炼主题...");

    try {
      const result = await invoke<PromptAnalysis>("summarize_prompt", { body: prompt });
      setAnalysis(result);
      setTopicOverride(result.theme ?? result.topic ?? "");
      setTagInput(result.suggestedTags.join(", "));
      setStatus("分析完成，可手动调整主题/标签后保存分析");

      let promptId = activePromptId;
      if (!promptId) {
        try {
          const saved = await persistPrompt();
          promptId = saved.id;
          setStatus("已保存并记录本次分析");
        } catch (error) {
          const message = error instanceof Error ? error.message : "自动保存失败";
          setStatus(message);
          return;
        }
      }

    } catch (err) {
      console.error(err);
      setStatus("分析失败，请查看日志");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    if (!prompt.trim()) {
      setStatus("无法保存空内容");
      return;
    }

    setStatus(activePromptId ? "更新中..." : "写入数据库...");
    try {
      const record = await persistPrompt();
      await loadAnalyses(record.id);
      setStatus("已保存");
    } catch (error) {
      console.error(error);
      setStatus("保存失败");
    }
  };

  const handleSelectHistory = (entry: StoredPrompt) => {
    setPrompt(entry.body);
    setActivePromptId(entry.id);
    setAnalysis(null);
    setTopicOverride("");
    setTagInput("");
    setHistoryPage(1);
    setSelectedIds([]);
    setStatus(`已载入 ${new Date(entry.updated_at).toLocaleTimeString()}`);
    loadAnalyses(entry.id);
  };

  const handleDelete = async () => {
    if (!activePromptId) {
      setStatus("未选择记录");
      return;
    }

    setStatus("删除中...");
    try {
      await invoke<boolean>("delete_prompt", { id: activePromptId });
      setHistory((prev) => prev.filter((item) => item.id !== activePromptId));
      handleNewDraft();
      setStatus("已删除");
    } catch (error) {
      console.error(error);
      setStatus("删除失败");
    }
  };

  const handleCommitAnalysis = async () => {
    if (!analysis) {
      setStatus("暂无可保存的分析");
      return;
    }

    try {
      let promptId = activePromptId;
      if (!promptId) {
        const record = await persistPrompt();
        promptId = record.id;
      }

      const trimmedTopic = topicOverride.trim();
      const topic = trimmedTopic || analysis.theme || analysis.topic || currentTags[0] || null;
      const tags = currentTags.length > 0 ? currentTags : analysis.suggestedTags;

      await invoke("record_analysis", {
        payload: {
          prompt_id: promptId,
          summary: analysis.summary,
          tags,
          classification: {
            topic,
            targets: analysis.targetEntities,
            keywords: tags,
            token_count: tokenCount,
            length: analysis.length,
            source: "local-nlp",
          },
          qwen_model: "local-nlp",
        },
      });
      await loadAnalyses(promptId);
      setStatus("分析结果已保存");
    } catch (error) {
      console.error(error);
      const message = error instanceof Error ? error.message : "写入分析失败";
      setStatus(message);
    }
  };

  const handleAddVocabulary = async () => {
    if (!newVocab.trim()) return;
    setIsVocabBusy(true);
    try {
      const list = await invoke<string[]>("add_vocabulary_entry", { term: newVocab });
      setVocabulary(list);
      setNewVocab("");
    } catch (error) {
      console.error(error);
    } finally {
      setIsVocabBusy(false);
    }
  };

  const handleRemoveVocabulary = async (term: string) => {
    setIsVocabBusy(true);
    try {
      const list = await invoke<string[]>("remove_vocabulary_entry", { term });
      setVocabulary(list);
    } catch (error) {
      console.error(error);
    } finally {
      setIsVocabBusy(false);
    }
  };

  const handleExportCsv = async () => {
    setIsExporting(true);
    setStatus("导出 CSV 中...");
    try {
      const path = await invoke<string>("export_prompts_csv", {
        target_path: exportPath.trim() || null,
      });
      setLastExportPath(path);
      setStatus(`已导出到 ${path}`);
    } catch (error) {
      console.error(error);
      const message = error instanceof Error ? error.message : "导出失败";
      setStatus(message);
    } finally {
      setIsExporting(false);
    }
  };

  const clampPageSize = (value: number) => Math.min(100, Math.max(5, value));

  const handlePageSizeChange = (value: number) => {
    const next = clampPageSize(value);
    setPageSize(next);
    setHistoryPage(1);
    setSelectedIds([]);
  };

  const handleToggleSelect = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  };

  const handleSelectAllPage = () => {
    const ids = pagedHistory.map((entry) => entry.id);
    const allSelected = ids.every((id) => selectedIds.includes(id));
    if (allSelected) {
      setSelectedIds((prev) => prev.filter((id) => !ids.includes(id)));
    } else {
      setSelectedIds((prev) => Array.from(new Set([...prev, ...ids])));
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) {
      setStatus("δѡ��Ҫɾ���ļ�¼");
      return;
    }
    setStatus("����ɾ����...");
    try {
      await Promise.all(selectedIds.map((id) => invoke<boolean>("delete_prompt", { id })));
      setHistory((prev) => prev.filter((item) => !selectedIds.includes(item.id)));
      if (activePromptId && selectedIds.includes(activePromptId)) {
        handleNewDraft();
      }
      setSelectedIds([]);
      setStatus("����ɾ�����");
    } catch (error) {
      console.error(error);
      setStatus("����ɾ��ʧ��");
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <h1>PromptLab · 本地提示词工作台</h1>
          <p>离线保存、自动抽取主题/对象，轻量而高效</p>
        </div>
        <div className="top-bar-actions">
          <div className="stats">
            <div>
              <span className="label">当前字数</span>
              <strong>{prompt.length}</strong>
            </div>
            <div>
              <span className="label">估算 tokens</span>
              <strong>{tokenCount}</strong>
            </div>
            <div>
              <span className="label">提示词总数</span>
              <strong>{history.length}</strong>
            </div>
          </div>
          <button type="button" className="ghost" onClick={handleExportCsv} disabled={isExporting}>
            {isExporting ? "导出中..." : "导出 CSV"}
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="panel editor">
          <div className="panel-head">
            <h2>提示词编辑器</h2>
            <span>{status}</span>
          </div>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            spellCheck={false}
            placeholder="使用 Markdown、注释或变量占位符描述你的想法"
          />
          <div className="command-row">
            <button type="button" onClick={handleAnalyze} disabled={isLoading}>
              {isLoading ? "分析中..." : "提取主题"}
            </button>
            <button type="button" className="ghost" onClick={handleSave}>
              {activePromptId ? "更新" : "保存到数据库"}
            </button>
            <button type="button" className="ghost" onClick={handleNewDraft}>
              新建草稿
            </button>
            <button type="button" className="ghost danger" onClick={handleDelete} disabled={!activePromptId}>
              删除
            </button>
          </div>
        </section>

        <section className="panel analysis">
          <div className="panel-head">
            <h2>分析结果</h2>
            <span>{analysis ? "最近执行" : "暂无结果"}</span>
          </div>

          {analysis ? (
            <>
              <pre className="summary-block">{analysis.summary}</pre>
              <div className="tag-shelf">
                {displayTags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
              {analysis.role && analysis.role !== "空" && <div className="meta-line">角色设定：{analysis.role}</div>}
              {topicDisplay && <div className="meta-line">主题：{topicDisplay}</div>}
              {analysis.targetEntities.length > 0 && (
                <div className="meta-line">对象：{analysis.targetEntities.join("、")}</div>
              )}
              <div className="meta-line">提示词长度：{analysis.length} 字符</div>
              <div className="analysis-controls">
                <label>
                  自定义主题
                  <input
                    value={topicOverride}
                    onChange={(event) => setTopicOverride(event.target.value)}
                    placeholder="例如：B2B 邮件营销策略"
                  />
                </label>
                <label>
                  标签（逗号 / 空格分隔）
                  <input
                    value={tagInput}
                    onChange={(event) => setTagInput(event.target.value)}
                    placeholder="品牌, 运营, 邮件"
                  />
                </label>
                <div className="analysis-actions">
                  <button type="button" className="ghost" onClick={handleCommitAnalysis}>
                    保存分析
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="placeholder">尚未执行本地分析</div>
          )}

          <div className="analysis-log">
            <div className="panel-head">
              <h3>历史分析</h3>
              <span>{analysisLog.length}</span>
            </div>
            <ul>
              {analysisLog.length === 0 && <li className="placeholder">暂无分析记录</li>}
              {analysisLog.map((item) => {
                const targets = Array.isArray(item.classification?.targets)
                  ? (item.classification?.targets as string[]).filter(
                      (target) => typeof target === "string" && target.trim().length > 0,
                    )
                  : [];
                return (
                  <li key={item.id}>
                    <div>
                      <strong>{new Date(item.created_at).toLocaleTimeString()}</strong>
                      <span>{item.classification?.topic ?? "未识别"}</span>
                    </div>
                    <code>{item.summary.slice(0, 120)}</code>
                    {targets.length > 0 && <small className="meta-line">对象：{targets.join("、")}</small>}
                    {item.tags.length > 0 && (
                      <div className="tag-shelf compact">
                        {item.tags.slice(0, 4).map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        </section>
        <section className="panel history">
          <div className="panel-head">
            <h2>提示词仓库</h2>
            <div className="history-actions">
              <label className="page-size">
                每页显示
                <input
                  type="number"
                  min={5}
                  max={100}
                  value={pageSize}
                  onChange={(event) => handlePageSizeChange(Number(event.target.value) || pageSize)}
                />
              </label>
              <button type="button" className="ghost" onClick={handleSelectAllPage}>
                当页全选
              </button>
              <button type="button" className="ghost danger" onClick={handleDeleteSelected} disabled={selectedIds.length === 0}>
                删除选中
              </button>
              <button type="button" className="ghost" onClick={refreshHistory}>
                刷新
              </button>
            </div>
          </div>
          <ul>
            {history.length === 0 && <li className="placeholder">暂无记录</li>}
            {pagedHistory.map((entry) => (
              <li
                key={entry.id}
                className={entry.id === activePromptId ? "active" : ""}
                onClick={() => handleSelectHistory(entry)}
              >
                <label className="history-select">
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(entry.id)}
                    onChange={() => handleToggleSelect(entry.id)}
                    onClick={(event) => event.stopPropagation()}
                  />
                  <span>{new Date(entry.updated_at).toLocaleString()}</span>
                </label>
                <strong title={entry.title || "Untitled"}>{entry.title || "Untitled"}</strong>
                <code>{entry.body.slice(0, 120)}</code>
              </li>
            ))}
          </ul>
          {history.length > pageSize && (
            <div className="pagination">
              <button
                type="button"
                className="ghost"
                onClick={() => setHistoryPage((page) => Math.max(1, page - 1))}
                disabled={historyPage <= 1}
              >
                上一页
              </button>
              <span>
                {historyPage} / {totalPages}
              </span>
              <button
                type="button"
                className="ghost"
                onClick={() => setHistoryPage((page) => Math.min(totalPages, page + 1))}
                disabled={historyPage >= totalPages}
              >
                下一页
              </button>
            </div>
          )}
<div className="export-block">
            <div className="panel-head">
              <h3>数据导出</h3>
            </div>
            <p>一键导出所有提示词及最新分析，用于 Excel / Pandas 深入研究。</p>
            <label>
              保存路径（可选）
              <input
                value={exportPath}
                onChange={(event) => setExportPath(event.target.value)}
                placeholder="留空则导出到默认目录"
              />
            </label>
            {lastExportPath && <small className="meta-line">上次导出：{lastExportPath}</small>}
            <button type="button" onClick={handleExportCsv} disabled={isExporting}>
              {isExporting ? "导出中..." : "生成 CSV"}
            </button>
          </div>

          <div className="vocab-block">
            <div className="panel-head">
              <h3>词表</h3>
              <button type="button" className="ghost" onClick={fetchVocabulary} disabled={isVocabBusy}>
                刷新
              </button>
            </div>
            <p>词表中的词将被优先识别为主题或标签，可随时增删。</p>
            <div className="vocab-add">
              <input
                value={newVocab}
                onChange={(event) => setNewVocab(event.target.value)}
                placeholder="输入关键主题词"
              />
              <button
                type="button"
                onClick={handleAddVocabulary}
                disabled={isVocabBusy || !newVocab.trim()}
              >
                添加
              </button>
            </div>
            <ul className="vocab-list">
              {vocabulary.length === 0 && <li className="placeholder">暂无词条</li>}
              {vocabulary.map((term) => (
                <li key={term}>
                  <span>{term}</span>
                  <button
                    type="button"
                    className="ghost danger"
                    onClick={() => handleRemoveVocabulary(term)}
                    disabled={isVocabBusy}
                  >
                    删除
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}
