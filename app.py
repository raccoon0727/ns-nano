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


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Neurosynth 功能分離查詢系統</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {
      --bg: #000000;
      --panel: #111111;
      --text: #ffffff;
      --text-muted: #888888;
      --border: #2e2e2e;
    }
    body {
      margin: 0; padding: 32px 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text);
      line-height: 1.5;
    }
    .container { max-width: 900px; margin: 0 auto; }
    h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; }
    p.desc { color: var(--text-muted); margin-bottom: 24px; font-size: 0.95rem; }
    .search-box {
      position: relative; background: var(--panel); padding: 24px;
      border-radius: 8px; border: 1px solid var(--border);
    }
    .input-wrapper { position: relative; width: 100%; }
    input[type="text"] {
      width: 100%; box-sizing: border-box; padding: 14px 16px; font-size: 1rem;
      border-radius: 6px; border: 1px solid var(--border); background: #000000;
      color: #ffffff; outline: none;
    }
    input[type="text"]:focus { border-color: #ffffff; }
    .suggestions-list {
      position: absolute; top: calc(100% + 4px); left: 0; right: 0;
      background: #111111; border: 1px solid var(--border); border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.8); z-index: 50;
      max-height: 240px; overflow-y: auto; display: none; padding: 4px 0;
    }
    .suggestion-item {
      padding: 10px 16px; cursor: pointer; font-size: 0.95rem; color: #cccccc;
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
    }
    button:hover { background: #000000; color: #ffffff; }
    #status { margin-top: 20px; color: var(--text-muted); font-size: 0.9rem; }
    #plot-container {
      margin-top: 24px; background: var(--panel); border-radius: 8px;
      border: 1px solid var(--border); padding: 16px; display: none;
    }
    .plot-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 12px; }
    #plot { width: 100%; height: 500px; }
    .results { margin-top: 24px; }
    .paper-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 6px; padding: 18px; margin-bottom: 12px;
    }
    .paper-title { font-size: 1rem; font-weight: 600; margin-bottom: 6px; }
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
      <span class="chip" onclick="setQuery('(0,-52,26) but not (-2,50,-6)')">座標分離 (0,-52,26) not (-2,50,-6)</span>
      <span class="chip" onclick="setQuery('pain but not reward')">pain but not reward</span>
    </div>

    <button onclick="performSearch()">開始查詢</button>
  </div>

  <div id="status"></div>

  <div id="plot-container">
    <div class="plot-title">3D MNI 腦部座標分佈圖</div>
    <div id="plot"></div>
  </div>

  <div class="results" id="results"></div>
</div>

<script>
  let debounceTimer = null;
  let activeIndex = -1;
  const input = document.getElementById('query-input');
  const preview = document.getElementById('preview-box');
  const suggestionsBox = document.getElementById('suggestions');

  function setQuery(text) {
    input.value = text;
    hideSuggestions();
    triggerParse();
  }

  function getCurrentWordContext(fullText) {
    const delimiters = /(but not|without|except|excluding|not|但不要|不要|沒有|排除|非|\s+)/i;
    const parts = fullText.split(delimiters);
    return parts[parts.length - 1].trim();
  }

  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      const fullText = input.value;
      triggerParse();

      const currentWord = getCurrentWordContext(fullText);
      if (currentWord.length < 2) {
        hideSuggestions();
        return;
      }

      try {
        const res = await fetch(`/suggest?q=${encodeURIComponent(currentWord)}`);
        const suggestions = await res.json();
        showSuggestions(suggestions, currentWord);
      } catch (e) {
        console.error(e);
      }
    }, 200);
  });

  input.addEventListener('keydown', (e) => {
    const items = suggestionsBox.querySelectorAll('.suggestion-item');
    if (!items.length || suggestionsBox.style.display === 'none') {
      if (e.key === 'Enter') performSearch();
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = (activeIndex + 1) % items.length;
      updateActiveItem(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = (activeIndex - 1 + items.length) % items.length;
      updateActiveItem(items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && items[activeIndex]) {
        items[activeIndex].click();
      } else {
        performSearch();
      }
    } else if (e.key === 'Escape') {
      hideSuggestions();
    }
  });

  function updateActiveItem(items) {
    items.forEach((item, idx) => {
      item.classList.toggle('active', idx === activeIndex);
    });
  }

  function showSuggestions(list, currentWord) {
    if (!list || list.length === 0) {
      hideSuggestions();
      return;
    }
    activeIndex = -1;
    suggestionsBox.innerHTML = '';
    list.forEach(item => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.innerText = item;
      div.onclick = () => {
        applySuggestion(item, currentWord);
      };
      suggestionsBox.appendChild(div);
    });
    suggestionsBox.style.display = 'block';
  }

  function hideSuggestions() {
    suggestionsBox.style.display = 'none';
    activeIndex = -1;
  }

  function applySuggestion(chosenTerm, currentWord) {
    const fullText = input.value;
    const lastIndex = fullText.lastIndexOf(currentWord);
    if (lastIndex !== -1) {
      input.value = fullText.substring(0, lastIndex) + chosenTerm + ' ';
    } else {
      input.value = chosenTerm + ' ';
    }
    input.focus();
    hideSuggestions();
    triggerParse();
  }

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box')) {
      hideSuggestions();
    }
  });

  async function triggerParse() {
    const text = input.value.trim();
    if (!text) {
      preview.innerText = "解析預覽：等待輸入...";
      return;
    }
    try {
      const res = await fetch(`/parse?q=${encodeURIComponent(text)}`);
      const data = await res.json();
      if (data.ok) {
        let msg = `理解為：包含 [${data.target.display}]`;
        if (data.exclude) {
          msg += `，排除 [${data.exclude.display}]`;
        }
        preview.innerText = msg;
      }
    } catch(e) {
      console.error(e);
    }
  }

  async function performSearch() {
    hideSuggestions();
    const text = input.value.trim();
    if (!text) return;

    const status = document.getElementById('status');
    const results = document.getElementById('results');
    const plotContainer = document.getElementById('plot-container');

    status.innerText = "正在檢索資料庫...";
    results.innerHTML = "";
    plotContainer.style.display = "none";

    try {
      const parseRes = await fetch(`/parse?q=${encodeURIComponent(text)}`);
      const parsed = await parseRes.json();
      if (!parsed.ok || !parsed.target) {
        status.innerText = "無法解析輸入格式，請檢視輸入語句。";
        return;
      }

      let apiUrl = "";
      if (parsed.target.type === "location" && parsed.exclude && parsed.exclude.type === "location") {
        apiUrl = `/dissociate/locations/${parsed.target.val}/${parsed.exclude.val}`;
      } else {
        const targetVal = parsed.target.val;
        const excludeVal = parsed.exclude ? parsed.exclude.val : "none";
        apiUrl = `/dissociate/terms/${encodeURIComponent(targetVal)}/${encodeURIComponent(excludeVal)}`;
      }

      const res = await fetch(apiUrl);
      const data = await res.json();

      if (!data.ok) {
        status.innerText = `查詢失敗：${data.error || '資料庫錯誤'}`;
        return;
      }

      status.innerText = `找到 ${data.count} 篇符合條件之文獻（顯示前 ${data.returned} 筆）：`;

      if (data.type === "locations" && data.in) {
        plotContainer.style.display = "block";
        const traceTarget = {
          x: [data.in[0]], y: [data.in[1]], z: [data.in[2]],
          mode: 'markers',
          name: '目標座標 (A)',
          marker: { size: 12, color: '#ffffff', symbol: 'circle' },
          type: 'scatter3d'
        };
        const traceExclude = {
          x: [data.not_in[0]], y: [data.not_in[1]], z: [data.not_in[2]],
          mode: 'markers',
          name: '排除座標 (B)',
          marker: { size: 12, color: '#666666', symbol: 'circle-open' },
          type: 'scatter3d'
        };
        const layout = {
          paper_bgcolor: '#111111',
          plot_bgcolor: '#111111',
          scene: {
            xaxis: { title: 'X (Sagittal)', color: '#888888', gridcolor: '#222222' },
            yaxis: { title: 'Y (Coronal)', color: '#888888', gridcolor: '#222222' },
            zaxis: { title: 'Z (Axial)', color: '#888888', gridcolor: '#222222' }
          },
          margin: { l: 0, r: 0, b: 0, t: 0 },
          legend: { font: { color: '#ffffff' } }
        };
        Plotly.newPlot('plot', [traceTarget, traceExclude], layout);
      }

      let html = "";
      data.studies.forEach(item => {
        const meta = item.metadata || {};
        const title = meta.title || "無標題資訊";
        const authors = meta.authors || "未知作者";
        const journal = meta.journal || "";
        const year = meta.year || "";
        const studyId = item.study_id;

        html += `
          <div class="paper-card">
            <div class="paper-title">${title}</div>
            <div class="paper-meta">${authors} (${year}) - <em>${journal}</em></div>
            <div>
              <a class="pubmed-link" href="https://pubmed.ncbi.nlm.nih.gov/${studyId}/" target="_blank" rel="noopener">
                查看 PubMed 文獻 (ID: ${studyId})
              </a>
            </div>
          </div>
        `;
      });
      results.innerHTML = html;

    } catch(err) {
      status.innerText = "連線失敗，請檢查網路狀態或伺服器回應。";
      console.error(err);
    }
  }
</script>
</body>
</html>"""

def create_app():
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.get("/", endpoint="home")
    def home():
        return render_template_string(HTML_PAGE)

    @app.get("/suggest", endpoint="suggest_terms")
    def suggest_terms():
        q = request.args.get("q", "").strip().lower()
        if not q or len(q) < 2:
            return jsonify([])

        raw_token = q.replace(" ", "_")
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(text("SET search_path TO ns, public;"))
            sql = """
                SELECT DISTINCT 
                    REPLACE(REPLACE(term, 'terms_abstract_tfidf__', ''), '_', ' ') AS clean_term
                FROM ns.annotations_terms
                WHERE term LIKE :prefix_query OR term LIKE :like_query
                ORDER BY clean_term
                LIMIT 8
            """
            rows = conn.execute(text(sql), {
                "prefix_query": f"terms_abstract_tfidf__{raw_token}%",
                "like_query": f"%_{raw_token}%"
            }).scalars().all()

        results = [r for r in rows if not r.isdigit()]
        return jsonify(results)

    @app.get("/parse", endpoint="parse_sentence")
    def parse_sentence():
        q = request.args.get("q", "")
        parsed = parse_natural_language_query(q)
        if not parsed:
            return jsonify({"ok": False, "error": "query is empty"}), 400
        return jsonify(parsed)

    @app.get("/img", endpoint="show_img")
    def show_img():
        return send_file("amygdala.gif", mimetype="image/gif")

    @app.get("/terms/<term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        return term

    @app.get("/locations/<coords>/studies", endpoint="locations_studies")
    def get_studies_by_coordinates(coords):
        x, y, z = map(int, coords.split("_"))
        return jsonify([x, y, z])

    def _limit():
        try:
            return max(1, min(int(request.args.get("limit", 1000)), 10000))
        except ValueError:
            abort(400, "limit must be an integer")

    def _metadata_key(conn):
        cols = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'ns' AND table_name = 'metadata'"))}
        return "study_id" if "study_id" in cols else "id"

    def _studies_response(ids_sql, params, query_info):
        eng = get_engine()
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))
                key = _metadata_key(conn)
                rows = conn.execute(text(f"""
                    WITH hits AS ({ids_sql})
                    SELECT DISTINCT ON (h.study_id) h.study_id,
                           (to_jsonb(m) - 'fts') AS metadata
                    FROM hits h
                    LEFT JOIN ns.metadata m ON m.{key}::text = h.study_id
                    ORDER BY h.study_id
                    LIMIT :lim
                """), {**params, "lim": _limit()}).mappings().all()
                total = conn.execute(text(
                    f"SELECT COUNT(*) FROM ({ids_sql}) t"), params).scalar()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({
            "ok": True,
            **query_info,
            "count": total,
            "returned
