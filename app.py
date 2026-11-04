# app.py
from flask import Flask, jsonify, abort, request, send_file, render_template_string
import os
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL (or DATABASE_URL) environment variable.")
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
    )
    return _engine

def parse_natural_language_query(q_str):
    if not q_str:
        return None

    negations = [
        r"\bbut\s+not\b", r"\bwithout\b", r"\bexcept\b", r"\bexcluding\b", r"\bnot\b",
        r"但不要", r"不要", r"沒有", r"排除", r"非"
    ]
    pattern = r"|".join(negations)
    match = re.search(pattern, q_str, flags=re.IGNORECASE)

    coord_pattern = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*\)?")

    def parse_entity(text_segment):
        text_segment = text_segment.strip()
        m = coord_pattern.search(text_segment)
        if m:
            coords = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            fmt = f"{int(coords[0])}_{int(coords[1])}_{int(coords[2])}"
            return {"type": "location", "val": fmt, "display": f"({coords[0]}, {coords[1]}, {coords[2]})"}
        cleaned = re.sub(r"^[我想要請幫我找\s]+", "", text_segment).strip()
        return {"type": "term", "val": cleaned, "display": cleaned}

    if match:
        a_raw = q_str[:match.start()]
        b_raw = q_str[match.end():]
        part_a = parse_entity(a_raw)
        part_b = parse_entity(b_raw)
        return {"ok": True, "target": part_a, "exclude": part_b}
    else:
        part_a = parse_entity(q_str)
        return {"ok": True, "target": part_a, "exclude": None}


# 前端 HTML + 互動腳本（純黑白設計、無 Emoji、Google 自動補全、3D 散佈圖、PubMed 連結）
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Neurosynth 腦功能分離查詢系統</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {
      --bg: #000000;
      --panel: #111111;
      --panel-hover: #1f1f1f;
      --text: #ffffff;
      --text-muted: #888888;
      --border: #2e2e2e;
      --accent: #ffffff;
    }
    body {
      margin: 0; padding: 32px 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg); color: var(--text);
      line-height: 1.5;
    }
    .container { max-width: 900px; margin: 0 auto; }
    h1 {
      font-size: 1.6rem; font-weight: 700; margin-bottom: 6px;
      letter-spacing: -0.02em; color: var(--text);
    }
    p.desc { color: var(--text-muted); margin-bottom: 24px; font-size: 0.95rem; }
    .search-box {
      position: relative; background: var(--panel); padding: 24px;
      border-radius: 8px; border: 1px solid var(--border);
    }
    .input-wrapper { position: relative; width: 100%; }
    input[type="text"] {
      width: 100%; box-sizing: border-box; padding: 14px 16px; font-size: 1rem;
      border-radius: 6px; border: 1px solid var(--border); background: #000000;
      color: #ffffff; outline: none; transition: border-color 0.2s;
    }
    input[type="text"]:focus { border-color: #ffffff; }
    
    /* 黑色與白色下拉選單 */
    .suggestions-list {
      position: absolute; top: calc(100% + 4px); left: 0; right: 0;
      background: #111111; border: 1px solid var(--border); border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.8); z-index: 50;
      max-height: 240px; overflow-y: auto; display: none; padding: 4px 0;
    }
    .suggestion-item {
      padding: 10px 16px; cursor: pointer; font-size: 0.95rem; color: #cccccc;
      transition: background 0.15s, color 0.15s;
    }
    .suggestion-item:hover, .suggestion-item.active {
      background: #222222; color: #ffffff;
    }

    .preview {
      margin-top: 14px; font-size: 0.9rem; color: #cccccc;
      min-height: 22px; font-family: monospace;
    }
    .examples { margin-top: 14px; font-size: 0.85rem; color: var(--text-muted); }
    .chip {
      display: inline-block; background: #1c1c1c; padding: 4px 10px;
      border-radius: 4px; margin: 4px 4px 4px 0; cursor: pointer; color: #dddddd;
      border: 1px solid var(--border); font-size: 0.8rem;
    }
    .chip:hover { background: #2a2a2a; color: #ffffff; }
    button {
      margin-top: 18px; background: #ffffff; color: #000000;
      border: 1px solid #ffffff; padding: 10px 20px; border-radius: 6px;
      font-weight: 600; font-size: 0.95rem; cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    button:hover { background: #000000; color: #ffffff; }
    #status { margin-top: 20px; color: var(--text-muted); font-size: 0.9rem; }
    #plot-container {
      margin-top: 24px; background: var(--panel); border-radius: 8px;
      border: 1px solid var(--border); padding: 16px; display: none;
    }
    .plot-title {
      font-size: 0.95rem; font-weight: 600; margin-bottom: 12px; color: #ffffff;
    }
    #plot { width: 100%; height: 500px; }
    .results { margin-top: 24px; }
    .paper-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 6px; padding: 18px; margin-bottom: 12px;
    }
    .paper-title { font-size: 1rem; font-weight: 600; margin-bottom: 6px; color: #ffffff; }
    .paper-meta { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 10px; }
    .pubmed-link {
      display: inline-block; font-size: 0.85rem; color: #ffffff;
      text-decoration: underline; text-underline-offset: 3px;
    }
    .pubmed-link:hover { color: #888888; }
  </style>
</head>
<body>
<div class="container">
  <h1>Neurosynth 功能分離查詢系統</h1>
  <p class="desc">輸入腦區名稱或 MNI 座標，系統支援關鍵字自動補全與功能分離文獻檢索。</p>

  <div class="search-box">
    <div class="input-wrapper">
      <input type="text" id="query-input" placeholder="輸入搜尋語句，例如：posterior cingulate but not ventromedial prefrontal" autocomplete="off" autofocus>
      <div id="suggestions" class="suggestions-list"></div>
    </div>
    <div class="preview" id="preview-box">解析預覽：等待輸入...</div>
    
    <div class="examples">
      範例查詢：
      <span class="chip" onclick="setQuery('posterior cingulate but not ventromedial prefrontal')">詞彙分離 (DMN)</span>
      <span class="chip" onclick="
