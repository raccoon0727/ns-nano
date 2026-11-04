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
    """
    自然語言句子解析器：
    識別否定詞（but not, without, except, excluding, not, 但不要, 不要, 沒有, 排除, 非）
    分割出 A（想要的）與 B（排除的），並判斷是座標還是詞彙。
    """
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
        # 清理多餘符號作為 term
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


# 前端 HTML + 互動腳本（包含防抖即時預覽、Plotly 3D 視覺化、PubMed 整合）
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Neurosynth 腦功能分離查詢系統</title>
  <!-- 載入 Plotly.js 進行 3D 腦部座標散佈圖視覺化 -->
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {
      --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8;
      --text: #f8fafc; --text-muted: #94a3b8; --border: #334155;
    }
    body {
      margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text);
    }
    .container { max-width: 1000px; margin: 0 auto; }
    h1 { font-size: 1.8rem; margin-bottom: 8px; color: var(--accent); }
    p.desc { color: var(--text-muted); margin-bottom: 24px; }
    .search-box {
      background: var(--panel); padding: 20px; border-radius: 12px;
      border: 1px solid var(--border); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    input[type="text"] {
      width: 100%; box-sizing: border-box; padding: 14px 16px; font-size: 1.05rem;
      border-radius: 8px; border: 1px solid var(--border); background: #0b1120;
      color: #fff; outline: none; transition: border-color 0.2s;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    .preview {
      margin-top: 12px; font-size: 0.95rem; color: #38bdf8;
      min-height: 24px; font-family: monospace;
    }
    .examples { margin-top: 14px; font-size: 0.85rem; color: var(--text-muted); }
    .chip {
      display: inline-block; background: #334155; padding: 4px 10px;
      border-radius: 6px; margin: 4px; cursor: pointer; color: #e2e8f0;
    }
    .chip:hover { background: #475569; }
    button {
      margin-top: 16px; background: var(--accent); color: #0f172a;
      border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold;
      font-size: 1rem; cursor: pointer; transition: opacity 0.2s;
    }
    button:hover { opacity: 0.9; }
    #status { margin-top: 20px; color: var(--text-muted); font-size: 0.95rem; }
    #plot-container {
      margin-top: 24px; background: var(--panel); border-radius: 12px;
      border: 1px solid var(--border); padding: 16px; display: none;
    }
    #plot { width: 100%; height: 500px; }
    .results { margin-top: 24px; }
    .paper-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px; margin-bottom: 12px;
    }
    .paper-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 6px; }
    .paper-meta { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px; }
    .pubmed-link {
      display: inline-block; font-size: 0.85rem; color: var(--accent);
      text-decoration: none; font-weight: 500;
    }
    .pubmed-link:hover { text-decoration: underline; }
  </style>
</head>
<body>
<div class="container">
  <h1>🧠 Neurosynth 功能分離自然語言搜尋系統</h1>
  <p class="desc">輸入自然語言句子查詢腦影像研究，支援「要 A，但排除 B」的概念與座標分離。</p>

  <div class="search-box">
    <input type="text" id="query-input" placeholder="試試輸入：posterior cingulate but not ventromedial prefrontal 或 (0,-52,26) but not (-2,50,-6)" autofocus>
    <div class="preview" id="preview-box">💡 即時解析：等待輸入...</div>
    
    <div class="examples">
      範例點擊帶入：
      <span class="chip" onclick="setQuery('posterior cingulate but not ventromedial prefrontal')">詞彙分離 (DMN)</span>
      <span class="chip" onclick="setQuery('(0,-52,26) but not (-2,50,-6)')">座標分離 (0,-52,26) not (-2,50,-6)</span>
      <span class="chip" onclick="setQuery('pain but not reward')">pain but not reward</span>
    </div>

    <button onclick="performSearch()">開始查詢</button>
  </div>

  <div id="status"></div>

  <!-- 功能 7：腦部座標視覺化散佈圖 -->
  <div id="plot-container">
    <div style="font-weight: 600; margin-bottom: 8px;">📍 3D MNI 腦部座標分佈圖</div>
    <div id="plot"></div>
  </div>

  <div class="results" id="results"></div>
</div>

<script>
  let debounceTimer = null;
  const input = document.getElementById('query-input');
  const preview = document.getElementById('preview-box');

  function setQuery(text) {
    input.value = text;
    triggerParse();
  }

  // 功能 2：即時預覽理解結果（防抖 0.3 秒觸發）
  input.addEventListener('keyup', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerParse, 300);
  });

  async function triggerParse() {
    const text = input.value.trim();
    if (!text) {
      preview.innerText = "💡 即時解析：等待輸入...";
      return;
    }
    try {
      const res = await fetch(`/parse?q=${encodeURIComponent(text)}`);
      const data = await res.json();
      if (data.ok) {
        let msg = `💡 理解為：要 【${data.target.display}】`;
        if (data.exclude) {
          msg += `，但不要 【${data.exclude.display}】`;
        }
        preview.innerText = msg;
      }
    } catch(e) {
      console.error(e);
    }
  }

  // 執行搜尋
  async function performSearch() {
    const text = input.value.trim();
    if (!text) return;

    const status = document.getElementById('status');
    const results = document.getElementById('results');
    const plotContainer = document.getElementById('plot-container');

    status.innerText = "🔍 搜尋中，請稍候...";
    results.innerHTML = "";
    plotContainer.style.display = "none";

    try {
      const parseRes = await fetch(`/parse?q=${encodeURIComponent(text)}`);
      const parsed = await parseRes.json();
      if (!parsed.ok || !parsed.target) {
        status.innerText = "❌ 無法理解查詢內容，請確認輸入格式。";
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
        status.innerText = `❌ 查詢失敗：${data.error || '未知錯誤'}`;
        return;
      }

      status.innerText = `✅ 找到 ${data.count} 篇符合條件之文獻（顯示前 ${data.returned} 筆）：`;

      // 功能 7：若為座標查詢，繪製 3D 空間分佈散佈圖
      if (data.type === "locations" && data.in) {
        plotContainer.style.display = "block";
        const traceTarget = {
          x: [data.in[0]], y: [data.in[1]], z: [data.in[2]],
          mode: 'markers',
          name: '目標座標 (A)',
          marker: { size: 14, color: '#38bdf8', symbol: 'circle' },
          type: 'scatter3d'
        };
        const traceExclude = {
          x: [data.not_in[0]], y: [data.not_in[1]], z: [data.not_in[2]],
          mode: 'markers',
          name: '排除座標 (B)',
          marker: { size: 14, color: '#ef4444', symbol: 'x' },
          type: 'scatter3d'
        };
        const layout = {
          paper_bgcolor: '#1e293b',
          plot_bgcolor: '#1e293b',
          scene: {
            xaxis: { title: 'X (Sagittal)', color: '#94a3b8' },
            yaxis: { title: 'Y (Coronal)', color: '#94a3b8' },
            zaxis: { title: 'Z (Axial)', color: '#94a3b8' }
          },
          margin: { l: 0, r: 0, b: 0, t: 0 },
          legend: { font: { color: '#f8fafc' } }
        };
        Plotly.newPlot('plot', [traceTarget, traceExclude], layout);
      }

      // 功能 3：每筆文獻附 PubMed 連結
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
                🔗 在 PubMed 上查看 (ID: ${studyId}) ↗
              </a>
            </div>
          </div>
        `;
      });
      results.innerHTML = html;

    } catch(err) {
      status.innerText = "❌ 連線失敗，請稍後重試。";
      console.error(err);
    }
  }
</script>
</body>
</html>
"""

def create_app():
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    # 功能 1：讓首頁 / 直接提供精美前端網頁
    @app.get("/", endpoint="home")
    def home():
        return render_template_string(HTML_PAGE)

    # 功能 2：自然語言快速解析端點（純比對、不查 DB，反應速度極快）
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
            "returned": len(rows),
            "studies": [{"study_id": r["study_id"], "metadata": r["metadata"]} for r in rows],
        })

    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        clean_a = term_a.strip().lower()
        clean_b = term_b.strip().lower()
        if not clean_a or not clean_b:
            abort(400, "both terms are required")

        # 支援前綴與底線/空格容錯
        prefix = "terms_abstract_tfidf__"
        vars_a = list({clean_a, clean_a.replace(" ", "_"), f"{prefix}{clean_a}", f"{prefix}{clean_a.replace(' ', '_')}"})
        vars_b = list({clean_b, clean_b.replace(" ", "_"), f"{prefix}{clean_b}", f"{prefix}{clean_b.replace(' ', '_')}"})

        ids_sql = """
            SELECT DISTINCT a.study_id 
            FROM ns.annotations_terms a
            WHERE a.term = ANY(:a_list)
              AND NOT EXISTS (
                  SELECT 1 FROM ns.annotations_terms b 
                  WHERE b.study_id = a.study_id AND b.term = ANY(:b_list)
              )
        """
        return _studies_response(
            ids_sql,
            {"a_list": vars_a, "b_list": vars_b},
            {"type": "terms", "in": clean_a, "not_in": clean_b}
        )

    @app.get("/dissociate/locations/<loc_a>/<loc_b>", endpoint="dissociate_locations")
    def dissociate_locations(loc_a, loc_b):
        def parse(s):
            try:
                x, y, z = (float(v) for v in s.split("_"))
            except ValueError:
                abort(400, f"bad coordinate '{s}', expected x_y_z")
            return x, y, z
        try:
            r = float(request.args.get("r", 10))
        except ValueError:
            abort(400, "r must be a number")
        ax, ay, az = parse(loc_a)
        bx, by, bz = parse(loc_b)
        ids_sql = """
            SELECT DISTINCT study_id FROM ns.coordinates
            WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:ax,:ay,:az), 4326), :r)
              AND study_id NOT IN (
                  SELECT study_id FROM ns.coordinates
                  WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:bx,:by,:bz), 4326), :r))
        """
        return _studies_response(
            ids_sql,
            {"ax": ax, "ay": ay, "az": az, "bx": bx, "by": by, "bz": bz, "r": r},
            {"type": "locations", "in": [ax, ay, az], "not_in": [bx, by, bz], "radius_mm": r})

    @app.get("/version", endpoint="version")
    def version():
        return "dissociate-v4 (NLP sentence preview + 3D brain plot + PubMed)"

    @app.get("/test_db", endpoint="test_db")
    def test_db():
        eng = get_engine()
        payload = {"ok": False, "dialect": eng.dialect.name}
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))
                payload["version"] = conn.exec_driver_sql("SELECT version()").scalar()
                payload["coordinates_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.coordinates")).scalar()
                payload["metadata_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.metadata")).scalar()
                payload["annotations_terms_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.annotations_terms")).scalar()
            payload["ok"] = True
            return jsonify(payload), 200
        except Exception as e:
            payload["error"] = str(e)
            return jsonify(payload), 500

    return app

app = create_app()
