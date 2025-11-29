#!/usr/bin/env python3
"""
Standalone Qwen-based analyzer aligned with PromptLib tables.
- Detects whether text is an LLM prompt.
- Segments text into pragmatic triples.
- Returns a payload shaped for the `analyses` table (summary/tags/classification/qwen_model)
  plus a `prompt_stub` matching `prompts` columns for easy future insertion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
ALLOWED_SEGMENT_TYPES = {"goal", "action", "constraint", "context", "data"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CLASSIFIER_SYSTEM_PROMPT = """你是 PromptLab 的判定器，需要判断文本是否属于“大模型提示词（prompt）”。
Prompt 常见信号：角色扮演（“你是/请扮演”）、分步执行、格式/JSON 约束、链式推理、调用工具/函数、特定模型/温度、强调准确/简洁/拒绝情绪化、输入输出槽位、上下文示例。
非 prompt：日常聊天、感叹、单句事实、广告口号、无任务指令的段落。
输出严格 JSON 对象，字段：
- is_prompt: bool
- confidence: 0-1 浮点数
- reason: 30~80 字中文理由
- signals: 2~6 个关键词或片段，概括判断依据
- language: 语言代码（如 zh/en/ja）
- recommended_title: 10~30 字标题，便于写入 prompts.title
- length: 原文字符数（含空格）
不要输出多余文本。"""

SEGMENT_SYSTEM_PROMPT = """你是 PromptLab 的“语用-三元组切分器”。请结合语用学（意图/约束/角色/输入输出）与知识图谱主谓宾，将文本拆为语义原子。
输出严格 JSON 对象：
{
  "summary": "不超过120字，概括核心意图",
  "tags": ["3~8 个短标签"],
  "segments": [
    {
      "type": "goal|action|constraint|context|data",
      "subject": "主语，可为空字符串",
      "predicate": "谓词/动作/属性",
      "object": "宾语/结果，可为空字符串",
      "context": "前置条件/输入/范围/示例等，可为空字符串",
      "modality": "must|should|can|forbid|ask",
      "priority": 1-5
    }
  ]
}
规则：保留原意，4~12 条即可；缺省字段用空字符串；去重相似切分；不要自然语言解释。"""


def normalize_text(text: str) -> str:
    """Trim outer whitespace while keeping paragraph structure."""
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def build_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def call_chat_json(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format or {"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise RuntimeError("Unexpected response content type")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse model JSON: {exc}") from exc


def classify_text(client: OpenAI, text: str, model: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return call_chat_json(client, model=model, messages=messages, max_tokens=400)


def segment_text(client: OpenAI, text: str, model: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return call_chat_json(client, model=model, messages=messages, max_tokens=1100)


def normalize_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        seg_type = str(seg.get("type", "")).lower().strip()
        if seg_type not in ALLOWED_SEGMENT_TYPES:
            seg_type = "action" if seg_type else "action"
        normalized.append(
            {
                "type": seg_type,
                "subject": str(seg.get("subject", "") or ""),
                "predicate": str(seg.get("predicate", "") or ""),
                "object": str(seg.get("object", "") or ""),
                "context": str(seg.get("context", "") or ""),
                "modality": str(seg.get("modality", "") or ""),
                "priority": int(seg.get("priority", 3)) if str(seg.get("priority", "")).strip() else 3,
            }
        )
    return normalized


def analyze_text(client: OpenAI, text: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    normalized = normalize_text(text)
    classifier = classify_text(client, normalized, model=model)
    segments = segment_text(client, normalized, model=model)
    cleaned_segments = normalize_segments(segments.get("segments"))

    summary = segments.get("summary") or normalized[:160]
    tags = [t for t in (segments.get("tags") or classifier.get("signals") or []) if str(t).strip()]
    deduped_tags: List[str] = []
    for tag in tags:
        if tag not in deduped_tags:
            deduped_tags.append(tag)
    tags = deduped_tags

    classification = {
        "is_prompt": classifier.get("is_prompt"),
        "confidence": classifier.get("confidence"),
        "reason": classifier.get("reason"),
        "signals": classifier.get("signals", []),
        "language": classifier.get("language"),
        "length": classifier.get("length"),
        "recommended_title": classifier.get("recommended_title"),
        "segments": cleaned_segments,
    }

    prompt_stub = {
        "title": classifier.get("recommended_title") or "Untitled",
        "body": normalized,
        "language": classifier.get("language"),
        "model_hint": None,
        "metadata": {"source": "standalone_qwen_script"},
    }

    analysis_record = {
        "summary": summary,
        "tags": tags,
        "classification": classification,
        "qwen_model": model,
    }

    return {
        "prompt_stub": prompt_stub,
        "analysis_record": analysis_record,
        "raw_text": normalized,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Qwen analyzer for PromptLib schema.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze. If omitted, read from --file or stdin.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to a UTF-8 text file to analyze.",
    )
    parser.add_argument(
        "--api-key",
        help="DashScope API key. Defaults to env DASHSCOPE_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for Qwen compatible API (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Qwen model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write JSON output.",
    )
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Please provide text via argument, --file, or stdin.")


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set DASHSCOPE_API_KEY or --api-key.")

    text = load_text(args)
    client = build_client(api_key=api_key, base_url=args.base_url)
    result = analyze_text(client, text, model=args.model)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote analysis to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
