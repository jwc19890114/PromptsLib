#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
  fs::OpenOptions,
  io::{Read, Write},
  path::{Path, PathBuf},
  sync::{Arc, Mutex},
  thread,
  time::Duration,
};

use chrono::Local;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use promptlab_core::analysis::{summarize_prompt_with_vocab, PromptAnalysis};
use promptlab_core::storage::{Analysis, NewAnalysis, NewPrompt, Prompt, Storage, UpdatePrompt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{
  tray::{MouseButton, TrayIcon, TrayIconBuilder, TrayIconEvent},
  Builder, Manager, State, WindowEvent,
};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_single_instance::init as single_instance;

struct AppState {
  storage: Storage,
  log_path: PathBuf,
  export_dir: PathBuf,
  vocabulary_path: PathBuf,
  vocabulary: Arc<Mutex<Vec<String>>>,
  http_client: Client,
  dashscope_key: Option<String>,
  dashscope_base: String,
  prompt_conf_threshold: Arc<Mutex<f64>>,
  optimize_interval: Arc<Mutex<usize>>,
  optimize_counter: Arc<Mutex<usize>>,
}

#[derive(Clone)]
struct QwenCtx {
  client: Client,
  api_key: Option<String>,
  base_url: String,
  log_path: PathBuf,
  prompt_conf_threshold: Arc<Mutex<f64>>,
  optimize_interval: Arc<Mutex<usize>>,
  optimize_counter: Arc<Mutex<usize>>,
}

#[derive(Deserialize)]
struct QwenChoiceMessage {
  content: Option<String>,
}

#[derive(Deserialize)]
struct QwenChoice {
  message: QwenChoiceMessage,
}

#[derive(Deserialize)]
struct QwenResponse {
  choices: Vec<QwenChoice>,
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

fn call_qwen_chat(qwen: &QwenCtx, messages: Vec<Value>, model: &str, max_tokens: u32) -> Result<Value, String> {
  let api_key = qwen
    .api_key
    .as_ref()
    .ok_or_else(|| "DASHSCOPE_API_KEY missing".to_string())?;
  let url = format!("{}/chat/completions", qwen.base_url.trim_end_matches('/'));

  let mut headers = HeaderMap::new();
  headers.insert(
    AUTHORIZATION,
    HeaderValue::from_str(&format!("Bearer {}", api_key)).map_err(|e| e.to_string())?,
  );
  headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

  let payload = json!({
    "model": model,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": 0.2,
    "response_format": { "type": "json_object" }
  });

  let resp = qwen
    .client
    .post(url)
    .headers(headers)
    .json(&payload)
    .send()
    .map_err(|e| e.to_string())?
    .error_for_status()
    .map_err(|e| e.to_string())?;

  let parsed: QwenResponse = resp.json().map_err(|e| e.to_string())?;
  let content = parsed
    .choices
    .get(0)
    .and_then(|c| c.message.content.as_ref())
    .ok_or_else(|| "empty qwen response".to_string())?;
  serde_json::from_str(content).map_err(|e| e.to_string())
}

fn classify_prompt_with_qwen(qwen: &QwenCtx, text: &str) -> Option<(bool, f64)> {
  let system = "判断输入是否为大模型提示词（prompt）。Prompt 特征：指令/角色/格式要求/步骤/输出约束/占位符。非 prompt：叙事、论文、新闻、无指令。只输出 JSON: {\"is_prompt\": bool, \"confidence\": 0-1}. 示例：长篇论文段落 -> false；“请你扮演产品经理，输出PRD模板” -> true。";
  let messages = vec![
    json!({"role": "system", "content": system}),
    json!({"role": "user", "content": text}),
  ];
  match call_qwen_chat(qwen, messages, "qwen-max", 200) {
    Ok(value) => {
      let is_prompt = value.get("is_prompt").and_then(|v| v.as_bool());
      let confidence = value.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
      is_prompt.map(|p| (p, confidence))
    }
    Err(err) => {
      let _ = append_log(&qwen.log_path, &format!("qwen classify failed: {err}"));
      None
    }
  }
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
fn export_prompts_csv(state: State<AppState>, #[allow(non_snake_case)] targetPath: Option<String>) -> Result<String, String> {
  let target_path = targetPath;
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
fn import_prompts_csv(state: State<AppState>, path: String) -> Result<usize, String> {
  let path = path;
  let mut file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
  let mut data = String::new();
  file.read_to_string(&mut data).map_err(|e| e.to_string())?;

  let mut reader = csv::Reader::from_reader(data.as_bytes());
  let headers = reader
    .headers()
    .map_err(|e| e.to_string())?
    .iter()
    .map(|h| h.to_string())
    .collect::<Vec<_>>();
  let idx = |name: &str| headers.iter().position(|h| h == name);

  let mut imported = 0usize;
  for result in reader.records() {
    let record = result.map_err(|e| e.to_string())?;
    let body = idx("body")
      .and_then(|i| record.get(i))
      .map(|s| s.trim().to_string())
      .unwrap_or_default();
    if body.is_empty() {
      continue;
    }
    if let Ok(Some(_)) = state.storage.find_prompt_by_body(&body) {
      continue;
    }

    let title = idx("title")
      .and_then(|i| record.get(i))
      .map(|s| s.to_string())
      .unwrap_or_else(|| derive_title(&body));
    let language = idx("language").and_then(|i| record.get(i)).map(|s| s.to_string());
    let model_hint = idx("model_hint").and_then(|i| record.get(i)).map(|s| s.to_string());
    let metadata_raw = idx("metadata").and_then(|i| record.get(i)).unwrap_or("");
    let metadata = serde_json::from_str::<Value>(metadata_raw).unwrap_or(Value::Null);

    let mut new_prompt = NewPrompt::new(title, body.clone());
    new_prompt.language = language;
    new_prompt.model_hint = model_hint;
    new_prompt.metadata = metadata;

    let prompt = match state.storage.create_prompt(new_prompt) {
      Ok(p) => p,
      Err(err) => {
        let _ = append_log(&state.log_path, &format!("import prompt failed: {err}"));
        continue;
      }
    };

    let summary = idx("latest_summary")
      .and_then(|i| record.get(i))
      .map(|s| s.trim().to_string())
      .unwrap_or_default();
    let tags_raw = idx("latest_tags")
      .and_then(|i| record.get(i))
      .map(|s| s.to_string())
      .unwrap_or_default();
    let tags: Vec<String> = if tags_raw.is_empty() {
      Vec::new()
    } else {
      tags_raw
        .split('|')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
    };
    let classification_raw = idx("classification")
      .and_then(|i| record.get(i))
      .map(|s| s.to_string())
      .unwrap_or_else(|| "null".into());
    let classification: Value = serde_json::from_str(&classification_raw).unwrap_or(Value::Null);

    if !summary.is_empty() || !tags.is_empty() || !classification.is_null() {
      let new_analysis = NewAnalysis {
        prompt_id: prompt.id.clone(),
        summary: if summary.is_empty() { "Imported".into() } else { summary },
        tags,
        classification,
        qwen_model: None,
      };
      if let Err(err) = state.storage.create_analysis(new_analysis) {
        let _ = append_log(&state.log_path, &format!("import analysis failed: {err}"));
      }
    }

    imported += 1;
  }

  Ok(imported)
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
    .plugin(tauri_plugin_dialog::init())
    .plugin(single_instance(|app, _argv, _cwd| {
      if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
      }
    }))
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
      let dashscope_key = std::env::var("DASHSCOPE_API_KEY").ok();
      let dashscope_base =
        std::env::var("DASHSCOPE_BASE_URL").unwrap_or_else(|_| "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string());
      let prompt_conf_threshold_val = std::env::var("QWEN_PROMPT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.1);
      let prompt_conf_threshold = Arc::new(Mutex::new(prompt_conf_threshold_val));
      let optimize_interval_val = std::env::var("QWEN_OPTIMIZE_INTERVAL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(20);
      let optimize_interval = Arc::new(Mutex::new(optimize_interval_val));
      let optimize_counter = Arc::new(Mutex::new(0usize));
      let http_client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .expect("http client");

      app.manage(AppState {
        storage,
        log_path,
        export_dir,
        vocabulary_path,
        vocabulary,
        http_client,
        dashscope_key,
        dashscope_base,
        prompt_conf_threshold,
        optimize_interval,
        optimize_counter,
      });

      let _tray: TrayIcon = TrayIconBuilder::new()
        .on_tray_icon_event(|tray, event| match event {
          TrayIconEvent::Click { button, .. } => {
            if button == MouseButton::Left {
              if let Some(window) = tray.app_handle().get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
              }
            } else if button == MouseButton::Right {
              let app_handle = tray.app_handle().clone();
              app_handle
                .dialog()
                .message("确定要退出 PromptLab 吗？")
                .title("退出应用")
                .kind(MessageDialogKind::Warning)
                .buttons(MessageDialogButtons::OkCancel)
                .show(move |ok| {
                  if ok {
                    app_handle.exit(0);
                  }
                });
            }
          }
          _ => {}
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
      remove_vocabulary_entry,
      import_prompts_csv,
      set_prompt_threshold,
      get_prompt_threshold,
      set_optimize_interval,
      get_optimize_interval,
      optimize_threshold
    ])
    .run(tauri::generate_context!())
    .expect("error while running PromptLab desktop app");
}

fn start_clipboard_watcher(app_handle: tauri::AppHandle) {
  let (storage, vocab, log_path, http_client, dashscope_key, dashscope_base, prompt_conf_threshold, optimize_interval, optimize_counter) = {
    let state = app_handle.state::<AppState>();
    (
      state.storage.clone(),
      state.vocabulary.clone(),
      state.log_path.clone(),
      state.http_client.clone(),
      state.dashscope_key.clone(),
      state.dashscope_base.clone(),
      state.prompt_conf_threshold.clone(),
      state.optimize_interval.clone(),
      state.optimize_counter.clone(),
    )
  };

  thread::spawn(move || {
    let mut clipboard = match arboard::Clipboard::new() {
      Ok(cb) => cb,
      Err(err) => {
        let _ = append_log(&log_path, &format!("clipboard init failed: {err}"));
        return;
      }
    };

    let mut last = String::new();
  let qwen_state = QwenCtx {
    client: http_client,
    api_key: dashscope_key,
    base_url: dashscope_base,
    log_path: log_path.clone(),
    prompt_conf_threshold,
    optimize_interval,
    optimize_counter,
  };

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

  let mut qwen_pred = None;
      match classify_prompt_with_qwen(&qwen_state, candidate) {
        Some((flag, conf)) => {
          qwen_pred = Some((flag, conf));
          let threshold = *qwen_state
            .prompt_conf_threshold
            .lock()
            .unwrap_or_else(|e| e.into_inner());
          if !flag && conf >= threshold {
            let _ = append_log(&log_path, "qwen: skipped non-prompt clipboard text");
            continue;
          }
          // auto-optimize threshold every N samples
          if let Ok(mut counter) = qwen_state.optimize_counter.lock() {
            *counter += 1;
            let interval_guard = qwen_state
              .optimize_interval
              .lock()
              .unwrap_or_else(|e| e.into_inner());
            let interval_val = (*interval_guard).max(1);
            if *counter >= interval_val {
              *counter = 0;
              if let Ok(suggestion) = optimize_threshold_internal(&storage) {
                if let Ok(mut guard) = qwen_state.prompt_conf_threshold.lock() {
                  *guard = suggestion.best_threshold;
                }
            let _ = append_log(
              &log_path,
              &format!(
                "auto-optimized threshold -> {:.2} (acc {:.2}, total {})",
                suggestion.best_threshold, suggestion.accuracy, suggestion.total
              ),
            );
          }
        }
      }
    }
    None => {
      // fallback: accept
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
          "targets": analysis.target_entities,
          "qwen_clipboard_pred": qwen_pred.map(|(flag, conf)| json!({"is_prompt": flag, "confidence": conf})),
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

#[tauri::command]
fn set_prompt_threshold(state: State<AppState>, value: f64) -> Result<f64, String> {
  if !value.is_finite() || value < 0.0 || value > 1.0 {
    return Err("threshold must be between 0 and 1".into());
  }
  if let Ok(mut guard) = state.prompt_conf_threshold.lock() {
    *guard = value;
    return Ok(value);
  }
  Err("failed to set threshold".into())
}

#[tauri::command]
fn get_prompt_threshold(state: State<AppState>) -> f64 {
  *state
    .prompt_conf_threshold
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

#[tauri::command]
fn set_optimize_interval(state: State<AppState>, value: usize) -> Result<usize, String> {
  let clamped = value.max(1).min(10_000);
  if let Ok(mut guard) = state.optimize_interval.lock() {
    *guard = clamped;
    return Ok(clamped);
  }
  Err("failed to set interval".into())
}

#[tauri::command]
fn get_optimize_interval(state: State<AppState>) -> usize {
  *state
    .optimize_interval
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

#[derive(Serialize)]
struct ThresholdSuggestion {
  best_threshold: f64,
  accuracy: f64,
  total: usize,
  positive: usize,
  negative: usize,
}

#[tauri::command]
fn optimize_threshold(state: State<AppState>) -> Result<ThresholdSuggestion, String> {
  let prompts = state.storage.list_prompts().map_err(|e| e.to_string())?;
  compute_threshold(&prompts)
}

fn optimize_threshold_internal(storage: &Storage) -> Result<ThresholdSuggestion, String> {
  let prompts = storage.list_prompts().map_err(|e| e.to_string())?;
  compute_threshold(&prompts)
}

fn compute_threshold(prompts: &[Prompt]) -> Result<ThresholdSuggestion, String> {
  let mut samples: Vec<(bool, bool, f64)> = Vec::new(); // (label, model_flag, model_conf)
  for p in prompts {
    let meta = &p.metadata;
    let label = meta
      .get("is_prompt_label")
      .and_then(|v| v.as_bool())
      .or_else(|| meta.get("is_prompt_label").and_then(|v| v.as_i64().map(|i| i != 0)));
    let pred_obj = meta.get("qwen_clipboard_pred").and_then(|v| v.as_object());
    let model_flag = pred_obj.and_then(|o| o.get("is_prompt")).and_then(|v| v.as_bool());
    let model_conf = pred_obj.and_then(|o| o.get("confidence")).and_then(|v| v.as_f64());
    if let (Some(lbl), Some(flag), Some(conf)) = (label, model_flag, model_conf) {
      samples.push((lbl, flag, conf));
    }
  }
  if samples.is_empty() {
    return Err("no labeled samples with model prediction".into());
  }
  let thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  let mut best = (0.0, 0.0);
  for t in thresholds {
    let mut correct = 0usize;
    for (lbl, flag, conf) in &samples {
      let predicted_prompt = if !flag && *conf >= t { false } else { true };
      if predicted_prompt == *lbl {
        correct += 1;
      }
    }
    let acc = correct as f64 / samples.len() as f64;
    if acc > best.1 {
      best = (t, acc);
    }
  }
  Ok(ThresholdSuggestion {
    best_threshold: best.0,
    accuracy: best.1,
    total: samples.len(),
    positive: samples.iter().filter(|(l, _, _)| *l).count(),
    negative: samples.iter().filter(|(l, _, _)| !*l).count(),
  })
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
