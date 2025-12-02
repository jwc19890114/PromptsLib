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
  quality?: QualityInfo;
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

type QualityInfo = {
  star?: number;
  score?: number;
  tags: string[];
  reason?: string;
  suggestions?: string[];
};

type ParsedMeta = {
  star?: unknown;
  quality_score?: unknown;
  quality_tags?: unknown;
  quality_reason?: unknown;
  is_prompt_label?: unknown;
  label_reason?: unknown;
  [key: string]: unknown;
};

const deriveTitle = (body: string) => {
  const firstLine = body.split("\n")[0]?.trim() ?? "";
  if (!firstLine) return "新建 Prompt";
  return firstLine.slice(0, 80);
};

const ensureArray = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item.trim() : ""))
      .filter(Boolean);
  }
  return [];
};

const coerceNumber = (value: unknown): number | undefined => {
  const num = typeof value === "string" ? Number(value) : (value as number);
  if (Number.isFinite(num)) return num as number;
  return undefined;
};

const parseMetadata = (metadata: unknown): ParsedMeta => {
  if (!metadata) return {};
  if (typeof metadata === "string") {
    try {
      return JSON.parse(metadata) as ParsedMeta;
    } catch {
      return {};
    }
  }
  if (typeof metadata === "object") {
    return metadata as ParsedMeta;
  }
  return {};
};

const deriveQualityFromMeta = (metadata: unknown): QualityInfo | null => {
  const meta = parseMetadata(metadata);
  const star = coerceNumber(meta.star);
  const score = coerceNumber(meta.quality_score);
  const tags = ensureArray(meta.quality_tags);
  const reason = typeof meta.quality_reason === "string" ? meta.quality_reason : undefined;
  if (!star && !score && tags.length === 0 && !reason) return null;
  return {
    star,
    score,
    tags,
    reason,
    suggestions: [],
  };
};

const deriveQualityFromClassification = (cls: AnalysisClassification | null): QualityInfo | null => {
  if (!cls || typeof cls.quality !== "object" || cls.quality === null) return null;
  const quality = cls.quality as Record<string, unknown>;
  return {
    star: coerceNumber(quality.star),
    score: coerceNumber(quality.score),
    tags: ensureArray(quality.tags),
    reason: typeof quality.reason === "string" ? quality.reason : undefined,
    suggestions: ensureArray(quality.suggestions),
  };
};

const renderStars = (star?: number) => {
  const value = Math.max(0, Math.min(5, Math.round(star ?? 0)));
  return "★".repeat(value) + "☆".repeat(5 - value);
};

const deriveLabel = (metadata: unknown): { label: boolean | null; reason: string } => {
  const meta = parseMetadata(metadata);
  const raw = meta.is_prompt_label;
  const reason = typeof meta.label_reason === "string" ? meta.label_reason : "";
  if (raw == null) return { label: null, reason };
  if (typeof raw === "boolean") return { label: raw, reason };
  if (typeof raw === "number") return { label: raw !== 0, reason };
  if (typeof raw === "string") {
    const lowered = raw.toLowerCase();
    if (["true", "yes", "1"].includes(lowered)) return { label: true, reason };
    if (["false", "no", "0"].includes(lowered)) return { label: false, reason };
  }
  return { label: null, reason };
};

const upsertStarToMetadata = (metadata: unknown, star: number): ParsedMeta => {
  const meta = parseMetadata(metadata);
  return {
    ...meta,
    star,
  };
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
  const [importPath, setImportPath] = useState("");
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
  const [quality, setQuality] = useState<QualityInfo | null>(null);
  const [minStar, setMinStar] = useState(0);
  const [isStarUpdating, setIsStarUpdating] = useState<string | null>(null);
  const [labelChoice, setLabelChoice] = useState<boolean | null>(null);
  const [labelReason, setLabelReason] = useState("");
  const [labelFilter, setLabelFilter] = useState<"all" | "prompt" | "non" | "unlabeled">("all");
  const [optimizeInterval, setOptimizeInterval] = useState(20);
  const [isOptUpdating, setIsOptUpdating] = useState(false);

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
  const filteredHistory = useMemo(() => {
    return history.filter((item) => {
      const quality = deriveQualityFromMeta(item.metadata);
      const star = quality?.star ?? 0;
      if (star < minStar) return false;
      const labelInfo = deriveLabel(item.metadata);
      if (labelFilter === "prompt") return labelInfo.label === true;
      if (labelFilter === "non") return labelInfo.label === false;
      if (labelFilter === "unlabeled") return labelInfo.label === null;
      return true;
    });
  }, [history, minStar, labelFilter]);
  const totalPages = Math.max(1, Math.ceil(filteredHistory.length / pageSize));
  const pagedHistory = useMemo(() => {
    const start = (historyPage - 1) * pageSize;
    return filteredHistory.slice(start, start + pageSize);
  }, [filteredHistory, historyPage, pageSize]);

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

  useEffect(() => {
    const fetchInterval = async () => {
      try {
        const val = await invoke<number>("get_optimize_interval");
        setOptimizeInterval(val);
      } catch (error) {
        console.error("get_optimize_interval", error);
      }
    };
    fetchInterval();
  }, []);

  const loadAnalyses = async (promptId: string) => {
    try {
      const items = await invoke<AnalysisRecord[]>("list_analyses", { promptId });
      console.log("list_analyses", promptId, "rows", items.length, items[0]);
      setAnalysisLog(items);
      const promptRow = history.find((item) => item.id === promptId);
      const qualityFromMeta = deriveQualityFromMeta(promptRow?.metadata);
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
        const q = deriveQualityFromClassification(cls) || qualityFromMeta;
        setQuality(q);
      } else {
        setAnalysis(null);
        setQuality(qualityFromMeta || null);
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
    setSelectedIds([]);
    setQuality(null);
    setLabelChoice(null);
    setLabelReason("");
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
    setQuality(deriveQualityFromMeta(entry.metadata));
    const labelInfo = deriveLabel(entry.metadata);
    setLabelChoice(labelInfo.label);
    setLabelReason(labelInfo.reason);
    setTopicOverride("");
    setTagInput("");
    
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

  const handleSetStar = async (entry: StoredPrompt, targetStar: number) => {
    if (targetStar < 1 || targetStar > 5) return;
    setIsStarUpdating(entry.id);
    try {
      const meta = upsertStarToMetadata(entry.metadata, targetStar);
      const payload = {
        title: entry.title,
        body: entry.body,
        language: entry.language,
        model_hint: entry.model_hint,
        metadata: meta,
      };
      const updated = await invoke<StoredPrompt>("update_prompt", { id: entry.id, payload });
      setHistory((prev) =>
        prev.map((item) => (item.id === entry.id ? { ...item, metadata: updated.metadata } : item)),
      );
      if (activePromptId === entry.id) {
        setQuality(deriveQualityFromMeta(updated.metadata));
      }
      setStatus(`Star set to ${targetStar} for ${entry.title || "Untitled"}`);
    } catch (error) {
      console.error("toggle star", error);
      setStatus("Star update failed");
    } finally {
      setIsStarUpdating(null);
    }
  };

  const updatePromptMetadata = async (entry: StoredPrompt, metaPatch: Record<string, unknown>) => {
    const currentMeta = parseMetadata(entry.metadata);
    const nextMeta = { ...currentMeta, ...metaPatch };
    const payload = {
      title: entry.title,
      body: entry.body,
      language: entry.language,
      model_hint: entry.model_hint,
      metadata: nextMeta,
    };
    const updated = await invoke<StoredPrompt>("update_prompt", { id: entry.id, payload });
    setHistory((prev) => prev.map((item) => (item.id === entry.id ? { ...item, metadata: updated.metadata } : item)));
    if (activePromptId === entry.id) {
      setQuality(deriveQualityFromMeta(updated.metadata));
      const labelInfo = deriveLabel(updated.metadata);
      setLabelChoice(labelInfo.label);
      setLabelReason(labelInfo.reason);
    }
  };

  const handleSetLabel = async (entry: StoredPrompt, label: boolean | null, reason: string) => {
    setStatus("Saving label...");
    try {
      await updatePromptMetadata(entry, {
        is_prompt_label: label,
        label_reason: reason,
      });
      if (label === null) {
        setLabelChoice(null);
      } else {
        setLabelChoice(label);
      }
      setStatus("Label saved");
    } catch (error) {
      console.error("set label", error);
      setStatus("Label save failed");
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
      const trimmed = exportPath.trim();
      const args = trimmed ? { targetPath: trimmed } : {};
      const path = await invoke<string>("export_prompts_csv", args);
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

  const handleImportCsv = async () => {
    if (!importPath.trim()) {
      setStatus("请输入导入路径");
      return;
    }
    setStatus("导入中...");
    try {
      const count = await invoke<number>("import_prompts_csv", { path: importPath.trim() });
      setStatus(`导入完成：${count} 条`);
      await refreshHistory();
    } catch (error) {
      console.error(error);
      const message = error instanceof Error ? error.message : "导入失败";
      setStatus(message);
    }
  };

  const handleSetOptimizeInterval = async () => {
    if (!Number.isFinite(optimizeInterval) || optimizeInterval <= 0) {
      setStatus("请输入有效的次数（>=1）");
      return;
    }
    setIsOptUpdating(true);
    try {
      const val = await invoke<number>("set_optimize_interval", { value: optimizeInterval });
      setOptimizeInterval(val);
      setStatus(`已更新：每 ${val} 条判定自动优化阈值`);
    } catch (error) {
      console.error(error);
      setStatus("更新优化次数失败");
    } finally {
      setIsOptUpdating(false);
    }
  };

  const clampPageSize = (value: number) => Math.min(100, Math.max(5, value));

  const handlePageSizeChange = (value: number) => {
    const next = clampPageSize(value);
    setPageSize(next);
    
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
      setStatus("δѡ**Ҫɾ***ļ*¼");
      return;
    }
    setStatus("****ɾ****...");
    try {
      await Promise.all(selectedIds.map((id) => invoke<boolean>("delete_prompt", { id })));
      setHistory((prev) => prev.filter((item) => !selectedIds.includes(item.id)));
      if (activePromptId && selectedIds.includes(activePromptId)) {
        handleNewDraft();
      }
      setSelectedIds([]);
      setStatus("****ɾ*****");
    } catch (error) {
      console.error(error);
      setStatus("****ɾ**ʧ**");
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
            <h2>Analysis</h2>
            <span>{analysis ? "Ready" : "No result"}</span>
          </div>

          {analysis ? (
            <>
              <pre className="summary-block">{analysis.summary}</pre>
              <div className="tag-shelf">
                {displayTags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
              {quality && (
                <div className="quality-block">
                  <div className="quality-header">
                    {typeof quality.star === "number" && quality.star > 0 && (
                      <span className="star-badge">{renderStars(quality.star)}</span>
                    )}
                    {typeof quality.score === "number" && (
                      <span className="quality-score">Score {quality.score.toFixed(2)}</span>
                    )}
                  </div>
                  {quality.tags.length > 0 && (
                    <div className="tag-shelf compact">
                      {quality.tags.slice(0, 6).map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </div>
                  )}
                  {quality.reason && <p className="quality-reason">{quality.reason}</p>}
                  {quality.suggestions && quality.suggestions.length > 0 && (
                    <ul className="quality-suggestions">
                      {quality.suggestions.slice(0, 4).map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              {analysis.role && analysis.role.trim() !== "" && <div className="meta-line">Role: {analysis.role}</div>}
              {topicDisplay && <div className="meta-line">Topic: {topicDisplay}</div>}
              {analysis.targetEntities.length > 0 && (
                <div className="meta-line">Targets: {analysis.targetEntities.join(', ')}</div>
              )}
              <div className="meta-line">Length: {analysis.length} chars</div>
              <div className="analysis-controls">
                <div className="label-controls">
                  <span>Manual label</span>
                  <div className="label-buttons">
                    <button
                      type="button"
                      className={`chip-button ${labelChoice === true ? "active" : ""}`}
                      onClick={() => {
                        if (activePromptId) {
                          const entry = history.find((p) => p.id === activePromptId);
                          if (entry) handleSetLabel(entry, true, labelReason);
                        }
                      }}
                    >
                      Prompt
                    </button>
                    <button
                      type="button"
                      className={`chip-button ${labelChoice === false ? "active" : ""}`}
                      onClick={() => {
                        if (activePromptId) {
                          const entry = history.find((p) => p.id === activePromptId);
                          if (entry) handleSetLabel(entry, false, labelReason);
                        }
                      }}
                    >
                      Not prompt
                    </button>
                    <button
                      type="button"
                      className="chip-button"
                      onClick={() => {
                        if (activePromptId) {
                          const entry = history.find((p) => p.id === activePromptId);
                          if (entry) handleSetLabel(entry, null, labelReason);
                        }
                      }}
                    >
                      Clear
                    </button>
                  </div>
                  <textarea
                    value={labelReason}
                    onChange={(event) => setLabelReason(event.target.value)}
                    placeholder="Reason / note"
                    rows={2}
                  />
                </div>
                <label>
                  Topic / scenario (override)
                  <input
                    value={topicOverride}
                    onChange={(event) => setTopicOverride(event.target.value)}
                    placeholder="e.g., B2B email marketing recap"
                  />
                </label>
                <label>
                  Tags (comma / space separated)
                  <input
                    value={tagInput}
                    onChange={(event) => setTagInput(event.target.value)}
                    placeholder="brand, marketing, email"
                  />
                </label>
                <div className="analysis-actions">
                  <button type="button" className="ghost" onClick={handleCommitAnalysis}>
                    Save analysis
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="placeholder">No local analysis yet</div>
          )}
          <div className="analysis-log">
            <div className="panel-head">
              <h3>Analysis history</h3>
              <span>{analysisLog.length}</span>
            </div>
            <ul>
              {analysisLog.length === 0 && <li className="placeholder">No analysis records</li>}
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
                      <span>{item.classification?.topic ?? "Unknown"}</span>
                    </div>
                    <code>{item.summary.slice(0, 120)}</code>
                    {targets.length > 0 && <small className="meta-line">Targets: {targets.join(", ")}</small>}
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
            <h2>Prompt Library</h2>
            <div className="history-actions">
              <label className="page-size">
                Page size
                <input
                  type="number"
                  min={5}
                  max={100}
                  value={pageSize}
                  onChange={(event) => handlePageSizeChange(Number(event.target.value) || pageSize)}
                />
              </label>
              <label className="page-size">
                Star filter
                <select value={minStar} onChange={(event) => setMinStar(Number(event.target.value))}>
                  <option value={0}>All</option>
                  <option value={3}>3+ stars</option>
                  <option value={4}>4+ stars</option>
                  <option value={5}>5 stars</option>
                </select>
              </label>
              <label className="page-size">
                Label filter
                <select value={labelFilter} onChange={(event) => setLabelFilter(event.target.value as any)}>
                  <option value="all">All</option>
                  <option value="prompt">Labeled prompt</option>
                  <option value="non">Labeled non-prompt</option>
                  <option value="unlabeled">Unlabeled</option>
                </select>
              </label>
              <button type="button" className="ghost" onClick={handleSelectAllPage}>
                Select page
              </button>
              <button type="button" className="ghost danger" onClick={handleDeleteSelected} disabled={selectedIds.length === 0}>
                Delete selected
              </button>
              <button type="button" className="ghost" onClick={refreshHistory}>
                Refresh
              </button>
            </div>
          </div>
          <ul>
            {history.length === 0 && <li className="placeholder">No records</li>}
            {pagedHistory.map((entry) => {
              const qualityMeta = deriveQualityFromMeta(entry.metadata);
              const starValue = qualityMeta?.star;
              const qualityTags = qualityMeta?.tags ?? [];
              return (
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
                  <div className="history-title-row">
                    <strong title={entry.title || "Untitled"}>{entry.title || "Untitled"}</strong>
                    <div className="history-actions-inline">
                      {typeof starValue === "number" && starValue > 0 && (
                        <span className="star-badge small">{renderStars(starValue)}</span>
                      )}
                      <div className="label-chips">
                        {(() => {
                          const labelInfo = deriveLabel(entry.metadata);
                          if (labelInfo.label === true) return <span className="chip chip-good">Prompt</span>;
                          if (labelInfo.label === false) return <span className="chip chip-bad">Not prompt</span>;
                          return <span className="chip chip-neutral">Unlabeled</span>;
                        })()}
                      </div>
                      <div className="star-control">
                        {[1, 2, 3, 4, 5].map((level) => (
                          <button
                            key={level}
                            type="button"
                            className={`star-toggle ${starValue === level ? "active" : ""}`}
                            onClick={(event) => {
                              event.stopPropagation();
                              handleSetStar(entry, level);
                            }}
                            disabled={isStarUpdating === entry.id}
                            title={`Set ${level} star${level > 1 ? "s" : ""}`}
                          >
                            ★
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                  {qualityTags.length > 0 && (
                    <div className="tag-shelf compact">
                      {qualityTags.slice(0, 3).map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </div>
                  )}
                  <code>{entry.body.slice(0, 120)}</code>
                </li>
              );
            })}
          </ul>
          {history.length > pageSize && (
            <div className="pagination">
              <button
                type="button"
                className="ghost"
                onClick={() => setHistoryPage((page) => Math.max(1, page - 1))}
                disabled={historyPage <= 1}
              >
                Prev
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
                Next
              </button>
            </div>
          )}

          <div className="export-block">
            <div className="panel-head">
              <h3>Export</h3>
            </div>
            <p>Export current prompts and analyses for Excel / Pandas review</p>
            <label>
              Export path (optional)
              <input
                value={exportPath}
                onChange={(event) => setExportPath(event.target.value)}
                placeholder="Leave empty to use default folder"
              />
            </label>
            {lastExportPath && <small className="meta-line">Last export: {lastExportPath}</small>}
            <button type="button" onClick={handleExportCsv} disabled={isExporting}>
              {isExporting ? "Exporting..." : "Export CSV"}
            </button>
            <label>
              Import path
              <input
                value={importPath}
                onChange={(event) => setImportPath(event.target.value)}
                placeholder="Path to CSV to import"
              />
            </label>
            <button type="button" onClick={handleImportCsv} disabled={isExporting}>
              Import CSV
            </button>
          </div>

          <div className="vocab-block">
            <div className="panel-head">
              <h3>Vocabulary</h3>
              <button type="button" className="ghost" onClick={fetchVocabulary} disabled={isVocabBusy}>
                Refresh
              </button>
            </div>
            <p>Terms will be treated as keywords during analysis; remove to stop weighting.</p>
            <div className="vocab-add">
              <input
                value={newVocab}
                onChange={(event) => setNewVocab(event.target.value)}
                placeholder="Add keyword"
              />
              <button
                type="button"
                onClick={handleAddVocabulary}
                disabled={isVocabBusy || !newVocab.trim()}
              >
                Add
              </button>
            </div>
            <ul className="vocab-list">
              {vocabulary.length === 0 && <li className="placeholder">No entries</li>}
              {vocabulary.map((term) => (
                <li key={term}>
                  <span>{term}</span>
                  <button
                    type="button"
                    className="ghost danger"
                    onClick={() => handleRemoveVocabulary(term)}
                    disabled={isVocabBusy}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div className="settings-block">
            <div className="panel-head">
              <h3>Settings</h3>
            </div>
            <div className="settings-row">
              <label className="page-size">
                Auto-optimize interval (judgments)
                <input
                  type="number"
                  min={1}
                  max={10000}
                  value={optimizeInterval}
                  onChange={(event) => setOptimizeInterval(Number(event.target.value) || optimizeInterval)}
                />
              </label>
              <button type="button" onClick={handleSetOptimizeInterval} disabled={isOptUpdating}>
                Apply
              </button>
              <small className="meta-line">Default 20; adjustable</small>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
