#!/usr/bin/env python3
"""
Demo helper for integrating Qwen with PromptLib data.

- Copy production SQLite to a demo file.
- Batch analyze prompts into `analyses`.
- Optimize a single prompt and optionally write a new prompt row.

This avoids touching the main app; use the demo DB path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow importing sibling helper.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_prompt_analyzer import (  # type: ignore
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_client,
    call_chat_json,
    analyze_text,
    normalize_text,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

OPTIMIZER_SYSTEM_PROMPT = """你是 PromptLab 的提示词优化器，请对输入的提示词做质量评估与优化。
要求输出 JSON（不得有多余文本），字段：
- optimized: 优化后的提示词正文（保留任务、格式、安全约束，补充缺失约束，精简啰嗦）
- improvements: 3~6 条改进要点（短语）
- risks: 1~3 条风险或模糊点
- star: 1-5 的整数评分
- tags: 3~8 个短标签
- recommended_title: 10~30 字标题
- variants: 可选 2~4 个风格或场景变体（数组，内容为提示词正文）
优化重点：角色清晰、输入输出槽位清晰、格式/JSON 约束、边界/拒答安全、简洁可执行。"""

QUALITY_SYSTEM_PROMPT = """你是 PromptLab 的提示词质量评估器，对给定提示词打星并给出标签和理由。
输出严格 JSON：
{
  "star": 1-5,              // 整数星级
  "score": 0-1,             // 质量分（精度/安全/可执行性综合）
  "tags": ["3~8标签"],       // 主题/用途/风格
  "reason": "40~80字理由",   // 为什么这么打分
  "suggestions": ["可选改进点1", "可选改进点2"]
}
评估维度：清晰度、可执行性、输入输出格式、约束与安全、长度/成本、领域适配。保持简洁。"""

TEMPLATE_SYSTEM_PROMPT = """你是 PromptLab 的提示词合并生成器。给你一组相似提示词，请输出融合版模板。
输出严格 JSON：
{
  "template": "融合后的提示词正文，包含占位符和必要约束",
  "title": "10~30字标题",
  "tags": ["3~8标签"],
  "rationale": "50~120字，解释融合要点与改进",
  "placeholders": ["列出占位符（如{input},{style}）"]
}
规则：保留核心意图，统一格式/安全约束，补全输入槽位与输出要求，精简冗余。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PromptLib Qwen demo CLI (uses a demo SQLite).")
    sub = parser.add_subparsers(dest="command", required=True)

    copy_p = sub.add_parser("copy-db", help="Copy production SQLite to a demo file.")
    copy_p.add_argument("--source", required=True, type=Path, help="Source SQLite path (prod).")
    copy_p.add_argument("--dest", required=True, type=Path, help="Destination demo SQLite path.")
    copy_p.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists.")

    analyze_p = sub.add_parser("analyze-db", help="Batch analyze prompts into analyses table.")
    analyze_p.add_argument("--db", required=True, type=Path, help="Demo SQLite path.")
    analyze_p.add_argument("--limit", type=int, default=20, help="Max prompts to process.")
    analyze_p.add_argument("--skip-has-analysis", action="store_true", help="Skip prompts already having analyses.")
    analyze_p.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name.")
    analyze_p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DashScope compatible base URL.")
    analyze_p.add_argument("--api-key", help="DashScope API key (env DASHSCOPE_API_KEY if omitted).")

    opt_p = sub.add_parser("optimize", help="Optimize a prompt text or a prompt row.")
    source_grp = opt_p.add_mutually_exclusive_group(required=True)
    source_grp.add_argument("--text", help="Raw prompt text to optimize.")
    source_grp.add_argument("--prompt-id", help="Prompt id from the DB to load and optimize.")
    opt_p.add_argument("--db", type=Path, help="Demo SQLite path (needed if --prompt-id or --write).")
    opt_p.add_argument("--write", action="store_true", help="Write optimized prompt as a new row into DB.")
    opt_p.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name.")
    opt_p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DashScope compatible base URL.")
    opt_p.add_argument("--api-key", help="DashScope API key (env DASHSCOPE_API_KEY if omitted).")
    opt_p.add_argument("--variants", type=int, default=2, help="Max variants to keep (if returned).")

    quality_p = sub.add_parser("quality", help="Batch quality eval: star + tags + reason, write into metadata and analyses.")
    quality_p.add_argument("--db", required=True, type=Path, help="Demo SQLite path.")
    quality_p.add_argument("--limit", type=int, default=20, help="Max prompts to process.")
    quality_p.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name.")
    quality_p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DashScope compatible base URL.")
    quality_p.add_argument("--api-key", help="DashScope API key (env DASHSCOPE_API_KEY if omitted).")

    merge_p = sub.add_parser("merge-template", help="Generate a merged template from a seed prompt and its similars.")
    merge_p.add_argument("--db", required=True, type=Path, help="Demo SQLite path.")
    merge_p.add_argument("--seed-id", required=True, help="Seed prompt id.")
    merge_p.add_argument("--top-k", type=int, default=3, help="Number of similar prompts to merge with seed.")
    merge_p.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name.")
    merge_p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DashScope compatible base URL.")
    merge_p.add_argument("--api-key", help="DashScope API key (env DASHSCOPE_API_KEY if omitted).")

    return parser.parse_args()


def get_api_key(args: argparse.Namespace) -> str:
    env_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHCOPE_API_KEY")
    key = args.api_key or env_key
    if not key:
        raise SystemExit("Missing API key: set DASHSCOPE_API_KEY or pass --api-key.")
    return key


def copy_db(source: Path, dest: Path, overwrite: bool) -> None:
    if not source.exists():
        raise SystemExit(f"Source DB not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise SystemExit(f"Destination already exists: {dest}. Use --overwrite to replace.")
    shutil.copy2(source, dest)
    print(f"Copied DB from {source} to {dest}")


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 8000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            language TEXT,
            model_hint TEXT,
            metadata TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            tags TEXT NOT NULL,
            classification TEXT NOT NULL,
            qwen_model TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
        );
        """
    )


def fetch_prompts(conn: sqlite3.Connection, limit: int, skip_has_analysis: bool) -> List[sqlite3.Row]:
    if skip_has_analysis:
        rows = conn.execute(
            """
            SELECT p.*
            FROM prompts p
            LEFT JOIN analyses a ON a.prompt_id = p.id
            WHERE a.id IS NULL
            ORDER BY datetime(p.updated_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prompts ORDER BY datetime(updated_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows


def insert_analysis(conn: sqlite3.Connection, prompt_id: str, analysis: Dict[str, Any], model: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    analysis_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO analyses (id, prompt_id, summary, tags, classification, qwen_model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_id,
            prompt_id,
            analysis["summary"],
            json.dumps(analysis.get("tags") or []),
            json.dumps(analysis.get("classification") or {}),
            model,
            now,
        ),
    )


def batch_analyze(db_path: Path, api_key: str, base_url: str, model: str, limit: int, skip_has_analysis: bool) -> None:
    client = build_client(api_key=api_key, base_url=base_url)
    conn = connect_db(db_path)
    prompts = fetch_prompts(conn, limit=limit, skip_has_analysis=skip_has_analysis)
    if not prompts:
        print("No prompts to process.")
        return

    processed = 0
    for row in prompts:
        prompt_id = row["id"]
        body = row["body"]
        print(f"Analyzing prompt {prompt_id} ...")
        result = analyze_text(client, body, model=model)
        insert_analysis(conn, prompt_id, result["analysis_record"], model)
        processed += 1
    conn.commit()
    print(f"Done. Inserted analyses for {processed} prompts.")


def quality_evaluate(client, text: str, model: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    data = call_chat_json(
        client,
        model=model,
        messages=messages,
        max_tokens=400,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return {
        "star": int(data.get("star", 0) or 0),
        "score": float(data.get("score", 0) or 0),
        "tags": data.get("tags") or [],
        "reason": data.get("reason") or "",
        "suggestions": data.get("suggestions") or [],
    }


def update_prompt_with_quality(
    conn: sqlite3.Connection,
    prompt_row: sqlite3.Row,
    quality: Dict[str, Any],
    model: str,
) -> None:
    meta = parse_metadata(prompt_row["metadata"])
    meta.update(
        {
            "star": quality["star"],
            "quality_score": quality["score"],
            "quality_reason": quality["reason"],
            "quality_tags": quality["tags"],
        }
    )
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "UPDATE prompts SET metadata = ?, updated_at = ? WHERE id = ?",
        (serialize_metadata(meta), now, prompt_row["id"]),
    )

    classification = {"quality": quality}
    analysis_record = {
        "summary": quality["reason"] or "Quality evaluation",
        "tags": quality["tags"],
        "classification": classification,
        "qwen_model": model,
    }
    insert_analysis(conn, prompt_row["id"], analysis_record, model)


def batch_quality(db_path: Path, api_key: str, base_url: str, model: str, limit: int) -> None:
    client = build_client(api_key=api_key, base_url=base_url)
    conn = connect_db(db_path)
    prompts = fetch_prompts(conn, limit=limit, skip_has_analysis=False)
    if not prompts:
        print("No prompts to process.")
        return

    processed = 0
    for row in prompts:
        print(f"Quality evaluating prompt {row['id']} ...")
        quality = quality_evaluate(client, row["body"], model=model)
        update_prompt_with_quality(conn, row, quality, model)
        processed += 1
    conn.commit()
    print(f"Done. Quality-evaluated {processed} prompts.")


def tokenize(text: str) -> List[str]:
    buf = []
    current = []
    for ch in text:
        if ch.isalnum():
            current.append(ch.lower())
        else:
            if current:
                buf.append("".join(current))
                current = []
    if current:
        buf.append("".join(current))
    return buf


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_similar_prompts(conn: sqlite3.Connection, seed_id: str, top_k: int = 3) -> List[sqlite3.Row]:
    seed = conn.execute("SELECT * FROM prompts WHERE id = ?", (seed_id,)).fetchone()
    if not seed:
        raise SystemExit(f"Seed prompt not found: {seed_id}")
    seed_tokens = set(tokenize(seed["body"]))
    rows = conn.execute("SELECT * FROM prompts WHERE id != ?", (seed_id,)).fetchall()
    scored: List[Tuple[float, sqlite3.Row]] = []
    for row in rows:
        tokens = set(tokenize(row["body"]))
        score = jaccard(seed_tokens, tokens)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [seed] + [row for _, row in scored[:top_k]]


def generate_template(client, prompts: List[sqlite3.Row], model: str) -> Dict[str, Any]:
    numbered = []
    for idx, row in enumerate(prompts, 1):
        numbered.append(f"{idx}. {row['title']}\n{row['body']}")
    content = "以下是相似提示词，请合并生成模板：\n\n" + "\n\n".join(numbered)
    messages = [
        {"role": "system", "content": TEMPLATE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    data = call_chat_json(
        client,
        model=model,
        messages=messages,
        max_tokens=900,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return {
        "template": data.get("template", ""),
        "title": data.get("title", ""),
        "tags": data.get("tags") or [],
        "rationale": data.get("rationale", ""),
        "placeholders": data.get("placeholders") or [],
    }


def merge_similar(
    db_path: Path,
    seed_id: str,
    api_key: str,
    base_url: str,
    model: str,
    top_k: int,
) -> str:
    client = build_client(api_key=api_key, base_url=base_url)
    conn = connect_db(db_path)
    cluster = find_similar_prompts(conn, seed_id=seed_id, top_k=top_k)
    result = generate_template(client, cluster, model=model)
    cluster_id = str(uuid.uuid4())
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    prompt_id = str(uuid.uuid4())
    metadata = {
        "source": "merge_template",
        "cluster_id": cluster_id,
        "from_prompt_ids": [row["id"] for row in cluster],
        "placeholders": result["placeholders"],
        "rationale": result["rationale"],
        "tags": result["tags"],
    }
    conn.execute(
        """
        INSERT INTO prompts (id, title, body, language, model_hint, metadata, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt_id,
            result["title"] or "Merged prompt template",
            result["template"],
            None,
            None,
            serialize_metadata(metadata),
            now,
            now,
        ),
    )
    conn.commit()
    return prompt_id


def load_prompt_by_id(conn: sqlite3.Connection, prompt_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()


def parse_metadata(raw: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def serialize_metadata(meta: Dict[str, Any]) -> str:
    try:
        return json.dumps(meta)
    except Exception:
        return "{}"


def optimize_prompt(client, text: str, model: str, max_variants: int) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    data = call_chat_json(
        client,
        model=model,
        messages=messages,
        max_tokens=900,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    variants = data.get("variants") or []
    if isinstance(variants, list) and max_variants > 0:
        variants = variants[:max_variants]
    return {
        "optimized": data.get("optimized") or text,
        "improvements": data.get("improvements") or [],
        "risks": data.get("risks") or [],
        "star": data.get("star") or 0,
        "tags": data.get("tags") or [],
        "recommended_title": data.get("recommended_title") or "",
        "variants": variants,
    }


def write_new_prompt(
    conn: sqlite3.Connection,
    optimized: Dict[str, Any],
    base_prompt: Optional[sqlite3.Row],
    source_id: Optional[str],
) -> str:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    prompt_id = str(uuid.uuid4())
    title = optimized["recommended_title"] or (base_prompt["title"] if base_prompt else "Optimized prompt")
    language = base_prompt["language"] if base_prompt else None
    metadata = parse_metadata(base_prompt["metadata"]) if base_prompt else {}
    metadata.update(
        {
            "source": "demo_optimizer",
            "from_prompt_id": source_id,
            "star": optimized["star"],
            "tags": optimized["tags"],
            "risks": optimized["risks"],
        }
    )
    conn.execute(
        """
        INSERT INTO prompts (id, title, body, language, model_hint, metadata, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt_id,
            title,
            optimized["optimized"],
            language,
            None,
            serialize_metadata(metadata),
            now,
            now,
        ),
    )
    return prompt_id


def run_optimize(
    db_path: Optional[Path],
    prompt_id: Optional[str],
    text: Optional[str],
    api_key: str,
    base_url: str,
    model: str,
    write: bool,
    max_variants: int,
) -> None:
    client = build_client(api_key=api_key, base_url=base_url)

    base_prompt_row = None
    if prompt_id:
        if not db_path:
            raise SystemExit("--db is required when using --prompt-id.")
        conn = connect_db(db_path)
        base_prompt_row = load_prompt_by_id(conn, prompt_id)
        if not base_prompt_row:
            raise SystemExit(f"Prompt not found: {prompt_id}")
        text = base_prompt_row["body"]
    elif text is None:
        raise SystemExit("Either --text or --prompt-id must be provided.")

    assert text is not None
    normalized = normalize_text(text)
    result = optimize_prompt(client, normalized, model=model, max_variants=max_variants)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if write:
        if not db_path:
            raise SystemExit("--db is required to write optimized prompt.")
        conn = connect_db(db_path)
        new_id = write_new_prompt(conn, result, base_prompt_row, prompt_id)
        conn.commit()
        print(f"Written optimized prompt as new id: {new_id}")


def main() -> None:
    args = parse_args()
    if args.command == "copy-db":
        copy_db(args.source, args.dest, args.overwrite)
        return

    api_key = get_api_key(args)
    if args.command == "analyze-db":
        batch_analyze(
            db_path=args.db,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            limit=args.limit,
            skip_has_analysis=args.skip_has_analysis,
        )
        return

    if args.command == "optimize":
        run_optimize(
            db_path=args.db,
            prompt_id=args.prompt_id,
            text=args.text,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            write=args.write,
            max_variants=args.variants,
        )
        return

    if args.command == "quality":
        batch_quality(
            db_path=args.db,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            limit=args.limit,
        )
        return

    if args.command == "merge-template":
        new_id = merge_similar(
            db_path=args.db,
            seed_id=args.seed_id,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            top_k=args.top_k,
        )
        print(f"Written merged template prompt id: {new_id}")
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
