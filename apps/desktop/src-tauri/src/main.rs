#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
  fs::OpenOptions,
  io::Write,
  path::{Path, PathBuf},
  sync::{Arc, Mutex},
  thread,
  time::Duration,
};

use chrono::Local;
use promptlab_core::analysis::{summarize_prompt_with_vocab, PromptAnalysis};
use promptlab_core::storage::{Analysis, NewAnalysis, NewPrompt, Prompt, Storage, UpdatePrompt};
use serde::Deserialize;
use serde_json::{json, Value};
use tauri::{tray::TrayIconBuilder, Builder, Manager, State, WindowEvent};

struct AppState {
  storage: Storage,
  log_path: PathBuf,
  export_dir: PathBuf,
  vocabulary_path: PathBuf,
  vocabulary: Arc<Mutex<Vec<String>>>,
}

impl AppState {
  fn log(&self, message: &str) {
    if let Err(error) = append_log(&self.log_path, message) {
      eprintln!("failed to write log: {error}");
    }
  }
}

fn append_log(path: &PathBuf, message: &str) -> std::io::Result<()> {
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)?;
  }
  let mut file = OpenOptions::new().create(true).append(true).open(path)?;
  writeln!(file, "[{}] {message}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
  Ok(())
}

#[derive(Debug, Deserialize)]
struct PromptPayload {
  title: String,
  body: String,
  language: Option<String>,
  model_hint: Option<String>,
  metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct UpdatePromptPayload {
  title: Option<String>,
  body: Option<String>,
  language: Option<Option<String>>,
  model_hint: Option<Option<String>>,
  metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnalysisPayload {
  prompt_id: String,
  summary: String,
  tags: Vec<String>,
  classification: Value,
  qwen_model: Option<String>,
}

#[tauri::command]
fn summarize_prompt(state: State<AppState>, body: &str) -> PromptAnalysis {
  let vocabulary = state.vocabulary.lock().unwrap().clone();
  summarize_prompt_with_vocab(body, &vocabulary)
}

#[tauri::command]
fn save_prompt(state: State<AppState>, payload: PromptPayload) -> Result<Prompt, String> {
  let PromptPayload {
    title,
    body,
    language,
    model_hint,
    metadata,
  } = payload;
  let mut record = NewPrompt::new(title, body);
  record.language = language;
  record.model_hint = model_hint;
  record.metadata = metadata.unwrap_or(Value::Null);

  state
    .storage
    .create_prompt(record)
    .map(|prompt| {
      state.log(&format!("创建 Prompt 成功: {}", prompt.id));
      prompt
    })
    .map_err(|error| {
      state.log(&format!("创建 Prompt 失败: {error}"));
      error.to_string()
    })
}

#[tauri::command]
fn list_prompts(state: State<AppState>) -> Result<Vec<Prompt>, String> {
  state.storage.list_prompts().map_err(|error| {
    state.log(&format!("获取 Prompt 列表失败: {error}"));
    error.to_string()
  })
}

#[tauri::command]
fn get_prompt(state: State<AppState>, id: String) -> Result<Option<Prompt>, String> {
  state.storage.get_prompt(&id).map_err(|error| {
    state.log(&format!("获取 Prompt {id} 失败: {error}"));
    error.to_string()
  })
}

#[tauri::command]
fn update_prompt(state: State<AppState>, id: String, payload: UpdatePromptPayload) -> Result<Prompt, String> {
  let UpdatePromptPayload {
    title,
    body,
    language,
    model_hint,
    metadata,
  } = payload;

  let mut patch = UpdatePrompt::default();
  patch.title = title;
  patch.body = body;
  patch.language = language;
  patch.model_hint = model_hint;
  patch.metadata = metadata;

  state
    .storage
    .update_prompt(&id, patch)
    .map_err(|error| {
      state.log(&format!("更新 Prompt {id} 失败: {error}"));
      error.to_string()
    })?
    .ok_or_else(|| {
      state.log(&format!("更新 Prompt {id} 失败: 未找到"));
      "Prompt not found".to_string()
    })
}

#[tauri::command]
fn delete_prompt(state: State<AppState>, id: String) -> Result<bool, String> {
  state
    .storage
    .delete_prompt(&id)
    .map(|result| {
      state.log(&format!("删除 Prompt {id} => {result}"));
      result
    })
    .map_err(|error| {
      state.log(&format!("删除 Prompt {id} 失败: {error}"));
      error.to_string()
    })
}

#[tauri::command]
fn record_analysis(state: State<AppState>, payload: AnalysisPayload) -> Result<Analysis, String> {
  let AnalysisPayload {
    prompt_id,
    summary,
    tags,
    classification,
    qwen_model,
  } = payload;
  let entry = NewAnalysis {
    prompt_id,
    summary,
    tags,
    classification,
    qwen_model,
  };

  state.storage.create_analysis(entry).map_err(|error| {
    state.log(&format!("写入分析失败: {error}"));
    error.to_string()
  })
}

#[tauri::command]
fn list_analyses(state: State<AppState>, prompt_id: String) -> Result<Vec<Analysis>, String> {
  state.log(&format!("list_analyses called with prompt_id={prompt_id}"));
  match state.storage.list_analyses_for_prompt(&prompt_id) {
    Ok(list) => {
      state.log(&format!("list_analyses prompt_id={prompt_id} -> {} rows", list.len()));
      Ok(list)
    }
    Err(error) => {
      state.log(&format!("获取 Prompt {prompt_id} 分析失败: {error}"));
      Err(error.to_string())
    }
  }
}

#[tauri::command]
fn latest_analysis(state: State<AppState>, prompt_id: String) -> Result<Option<Analysis>, String> {
  state
    .storage
    .latest_analysis_for_prompt(&prompt_id)
    .map_err(|error| {
      state.log(&format!("获取 Prompt {prompt_id} 最新分析失败: {error}"));
      error.to_string()
    })
}

#[tauri::command]
fn export_prompts_csv(state: State<AppState>, target_path: Option<String>) -> Result<String, String> {
  let file_path = if let Some(custom_path) = target_path {
    let path = PathBuf::from(custom_path);
    if let Some(parent) = path.parent() {
      std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    path
  } else {
    std::fs::create_dir_all(&state.export_dir).map_err(|error| error.to_string())?;
    let file_name = format!("prompts-{}.csv", Local::now().format("%Y%m%d-%H%M%S"));
    state.export_dir.join(file_name)
  };
  let prompts = state
    .storage
    .list_prompts()
    .map_err(|error| {
      state.log(&format!("导出 prompts 失败: {error}"));
      error.to_string()
    })?;

  let mut file = std::fs::File::create(&file_path).map_err(|error| {
    state.log(&format!("创建导出文件失败: {error}"));
    error.to_string()
  })?;
  // Write UTF-8 BOM to improve compatibility with Excel
  if let Err(err) = file.write_all(&[0xEF, 0xBB, 0xBF]) {
    state.log(&format!("写入 BOM 失败: {err}"));
  }
  let mut writer = csv::Writer::from_writer(file);
  writer
    .write_record([
      "id",
      "title",
      "body",
      "language",
      "model_hint",
      "metadata",
      "created_at",
      "updated_at",
      "latest_summary",
      "latest_tags",
      "classification",
    ])
    .map_err(|error| error.to_string())?;

  for prompt in prompts {
    let analyses = state
      .storage
      .list_analyses_for_prompt(&prompt.id)
      .map_err(|error| error.to_string())?;
    let latest = analyses.first();
    let summary = latest.map(|entry| entry.summary.as_str()).unwrap_or_default();
    let tags = latest.map(|entry| entry.tags.join("|")).unwrap_or_default();
    let classification = latest
      .map(|entry| entry.classification.to_string())
      .unwrap_or_else(|| "null".into());

    writer
      .write_record([
        prompt.id.clone(),
        prompt.title.clone(),
        prompt.body.replace('\n', " ").replace('\r', " "),
        prompt.language.clone().unwrap_or_default(),
        prompt.model_hint.clone().unwrap_or_default(),
        prompt.metadata.to_string(),
        prompt.created_at.to_rfc3339(),
        prompt.updated_at.to_rfc3339(),
        summary.to_string(),
        tags,
        classification,
      ])
      .map_err(|error| error.to_string())?;
  }
  writer.flush().map_err(|error| error.to_string())?;
  Ok(file_path.to_string_lossy().to_string())
}

#[tauri::command]
fn list_vocabulary(state: State<AppState>) -> Vec<String> {
  let mut vocab = state.vocabulary.lock().unwrap().clone();
  vocab.sort();
  vocab
}

#[tauri::command]
fn add_vocabulary_entry(state: State<AppState>, term: String) -> Result<Vec<String>, String> {
  let normalized = normalize_vocab_term(&term);
  if normalized.is_empty() {
    return Err("请输入有效的词条".into());
  }
  let mut vocab = state.vocabulary.lock().unwrap();
  if !vocab.iter().any(|item| normalize_vocab_term(item) == normalized) {
    vocab.push(normalized.clone());
    persist_vocabulary(&state.vocabulary_path, &vocab).map_err(|error| error.to_string())?;
    state.log(&format!("新增词条: {normalized}"));
  }
  let mut list = vocab.clone();
  list.sort();
  Ok(list)
}

#[tauri::command]
fn remove_vocabulary_entry(state: State<AppState>, term: String) -> Result<Vec<String>, String> {
  let cleaned = normalize_vocab_term(&term);
  let mut vocab = state.vocabulary.lock().unwrap();
  let before = vocab.len();
  vocab.retain(|item| *item != cleaned);
  if vocab.len() != before {
    persist_vocabulary(&state.vocabulary_path, &vocab).map_err(|error| error.to_string())?;
    state.log(&format!("删除词条: {cleaned}"));
  }
  let mut list = vocab.clone();
  list.sort();
  Ok(list)
}

fn main() {
  Builder::default()
    .plugin(tauri_plugin_shell::init())
    .on_window_event(|window, event| {
      if let WindowEvent::CloseRequested { api, .. } = event {
        // Hide to tray instead of quitting.
        api.prevent_close();
        let _ = window.hide();
      }
    })
    .setup(|app| {
      let app_handle = app.handle();
      let path_api = app_handle.path();
      let data_dir = path_api
        .app_data_dir()
        .or_else(|_| path_api.app_config_dir())
        .or_else(|_| path_api.resource_dir())
        .or_else(|_| std::env::current_dir())?;
      std::fs::create_dir_all(&data_dir)?;
      let db_path = data_dir.join("promptlab.db");
      let storage = Storage::new(db_path)?;
      let log_path = data_dir.join("promptlab.log");
      let export_dir = data_dir.join("exports");
      let vocabulary_path = data_dir.join("vocabulary.json");
      let vocabulary = Arc::new(Mutex::new(load_vocabulary(&vocabulary_path)));

      app.manage(AppState {
        storage,
        log_path,
        export_dir,
        vocabulary_path,
        vocabulary,
      });

      let _tray = TrayIconBuilder::new()
        .on_tray_icon_event(|tray, event| {
          if let tauri::tray::TrayIconEvent::Click { .. } = event {
            if let Some(window) = tray.app_handle().get_webview_window("main") {
              let _ = window.show();
              let _ = window.set_focus();
            }
          }
        })
        .build(app)?;

      start_clipboard_watcher(app_handle.clone());

      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      summarize_prompt,
      save_prompt,
      list_prompts,
      get_prompt,
      update_prompt,
      delete_prompt,
      record_analysis,
      list_analyses,
      latest_analysis,
      export_prompts_csv,
      list_vocabulary,
      add_vocabulary_entry,
      remove_vocabulary_entry
    ])
    .run(tauri::generate_context!())
    .expect("error while running PromptLab desktop app");
}

fn start_clipboard_watcher(app_handle: tauri::AppHandle) {
  let state = app_handle.state::<AppState>();
  let storage = state.storage.clone();
  let vocab = state.vocabulary.clone();
  let log_path = state.log_path.clone();

  thread::spawn(move || {
    let mut clipboard = match arboard::Clipboard::new() {
      Ok(cb) => cb,
      Err(err) => {
        let _ = append_log(&log_path, &format!("clipboard init failed: {err}"));
        return;
      }
    };

    let mut last = String::new();
    loop {
      thread::sleep(Duration::from_millis(3500));
      let Ok(text) = clipboard.get_text() else {
        continue;
      };
      let candidate = text.trim();
      if !is_potential_prompt(candidate) {
        continue;
      }
      if candidate == last {
        continue;
      }
      last = candidate.to_string();

      match storage.find_prompt_by_body(candidate) {
        Ok(Some(_)) => continue,
        Ok(None) => {}
        Err(err) => {
          let _ = append_log(&log_path, &format!("clipboard lookup failed: {err}"));
          continue;
        }
      }

      let vocab_guard = vocab.lock().unwrap().clone();
      let analysis = summarize_prompt_with_vocab(candidate, &vocab_guard);
      let title = derive_title(candidate);

      let new_prompt = NewPrompt {
        title: title.to_string(),
        body: candidate.to_string(), // keep original text in prompt body
        language: None,
        model_hint: None,
        metadata: json!({
          "source": "clipboard",
          "raw": candidate,
          "structured": build_structured_body(&analysis, candidate),
          "tags": analysis.suggested_tags,
          "theme": analysis.theme,
          "topic": analysis.topic,
          "role": analysis.role,
          "targets": analysis.target_entities
        }),
      };

      match storage.create_prompt(new_prompt) {
        Ok(prompt) => {
          let _ = append_log(&log_path, &format!("clipboard saved prompt {}", prompt.id));
          let classification = json!({
            "topic": analysis.theme.clone().or(analysis.topic.clone()).unwrap_or_default(),
            "theme": analysis.theme,
            "targets": analysis.target_entities,
            "keywords": analysis.suggested_tags,
            "length": analysis.length,
            "role": analysis.role,
            "source": "clipboard"
          });

          let record = NewAnalysis {
            prompt_id: prompt.id,
            summary: analysis.summary,
            tags: analysis.suggested_tags,
            classification,
            qwen_model: Some("local-nlp".into()),
          };

          if let Err(err) = storage.create_analysis(record) {
            let _ = append_log(&log_path, &format!("clipboard analysis save failed: {err}"));
          }
        }
        Err(err) => {
          let _ = append_log(&log_path, &format!("clipboard save prompt failed: {err}"));
        }
      }
    }
  });
}

fn normalize_vocab_term(term: &str) -> String {
  let cleaned = term.trim();
  if cleaned.chars().all(|c| c.is_ascii()) {
    cleaned.to_lowercase()
  } else {
    cleaned.to_string()
  }
}

fn derive_title(body: &str) -> String {
  let first_line = body.split('\n').next().unwrap_or("").trim();
  if first_line.is_empty() {
    "剪贴板导入".to_string()
  } else {
    first_line.chars().take(80).collect()
  }
}

fn is_potential_prompt(text: &str) -> bool {
  let trimmed = text.trim();
  if trimmed.is_empty() {
    return false;
  }
  let len = trimmed.chars().count();
  if len < 8 || len > 600 {
    return false;
  }
  let lines: Vec<&str> = trimmed.lines().collect();
  if lines.len() > 12 {
    return false;
  }
  let chat_like = lines.iter().filter(|line| line.contains(':') || line.contains('：')).count();
  if chat_like >= 6 {
    return false;
  }
  let url_like = ["http://", "https://", ".com", ".net", ".org"];
  let url_hits = url_like.iter().filter(|pat| trimmed.contains(*pat)).count();
  if url_hits >= 3 {
    return false;
  }
  true
}

fn build_structured_body(analysis: &PromptAnalysis, original: &str) -> String {
  let mut parts = Vec::new();
  if !analysis.role.is_empty() {
    parts.push(format!("角色：{}", analysis.role));
  }
  if let Some(theme) = analysis.theme.as_ref().or(analysis.topic.as_ref()) {
    parts.push(format!("主题：{}", theme));
  }
  if !analysis.target_entities.is_empty() {
    parts.push(format!("对象：{}", analysis.target_entities.join("、")));
  }
  parts.push(format!("关键词：{}", analysis.suggested_tags.join("、")));
  parts.push(format!("摘要：{}", analysis.summary));
  parts.push("原文：".to_string());
  parts.push(original.trim().to_string());
  parts.join("\n")
}

fn load_vocabulary(path: &Path) -> Vec<String> {
  if path.exists() {
    if let Ok(data) = std::fs::read_to_string(path) {
      if let Ok(entries) = serde_json::from_str::<Vec<String>>(&data) {
        let mut cleaned: Vec<String> = entries
          .into_iter()
          .map(|item| normalize_vocab_term(&item))
          .filter(|item| !item.is_empty())
          .collect();
        cleaned.sort();
        cleaned.dedup();
        return cleaned;
      }
    }
  }
  Vec::new()
}

fn persist_vocabulary(path: &Path, vocab: &[String]) -> std::io::Result<()> {
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)?;
  }
  let payload = serde_json::to_string_pretty(vocab)
    .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))?;
  std::fs::write(path, payload)
}
