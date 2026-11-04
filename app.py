# app.py
from flask import Flask, jsonify, abort, request, send_file, render_template
import os
import re
import json
from sqlalchemy import create_engine, text

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL environment variable.")
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine

# ------------------------------------------------------------------
# 神經科學詞彙與別名對照表
# ------------------------------------------------------------------
SYNONYM_DICT = {
    "vmpfc": "ventromedial prefrontal",
    "pcc": "posterior cingulate",
    "acc": "anterior cingulate",
    "dlpfc": "dorsolateral prefrontal",
    "dacc": "dorsal anterior cingulate",
    "ofc": "orbitofrontal cortex",
    "dmn": "default mode",
    "amy": "amygdala",
    "前扣帶迴": "anterior cingulate",
    "後扣帶迴": "posterior cingulate",
    "杏仁核": "amygdala",
    "腹內側前額葉": "ventromedial prefrontal",
    "背外側前額葉": "dorsolateral prefrontal",
    "預設模式網路": "default mode",
    "工作記憶": "working memory",
    "注意力": "attention",
    "疼痛": "pain",
    "獎勵": "reward",
}

GEMINI_SYSTEM_INSTRUCTION = """
你是一個專精於神經科學資料庫 Neurosynth 的嚴格語意解析器。
你的職責是將使用者的任意口語、中英夾雜或倒裝句，轉換為符合後端 API 規格的結構化 JSON 物件。

【輸出架構約束】
1. 只能輸出純 JSON 物件，嚴禁包含 Markdown 標記（如 ```json）、思考過程或額外對話文字。
2. JSON 物件必須嚴格包含以下三個鍵：
   - "type": 字串，只能是 "terms" 或 "locations"。
   - "target": 字串，代表【要包含、正向選取】的大腦區域、概念或座標。
   - "exclude": 字串或 null，代表【要排除、否定、去除】的大腦區域、概念或座標；若無排除項目則必須為 null。

【解析規則】
1. 正負向識別：
   - 找出代表排除的詞（but not, without, except, excluding, not, 但不要, 不要, 排除, 去除, 除了...之外）。
   - 被排除詞修飾的實體必須歸入 "exclude"。主要欲查詢的主體歸入 "target"。
   - 倒裝句（例：「除了杏仁核之外，我想看後扣帶迴」）中，"target" 是 posterior cingulate，"exclude" 是 amygdala。
2. 座標規格：
   - MNI 座標（如 (0, -52, 26)）轉換為 "x_y_z" 底線格式（"0_-52_26"），此時 "type" 為 "locations"。
3. 詞彙標準化：
   - 中文必須翻譯為標準英文小寫詞彙。
   - 縮寫轉全稱（例：PCC -> posterior cingulate、vmPFC -> ventromedial prefrontal、amy -> amygdala）。

【範例】
輸入: "除了杏仁核之外，我想看後扣帶迴"
輸出: {"type": "terms", "target": "posterior cingulate", "exclude": "amygdala"}

輸入: "PCC but not vmPFC"
輸出: {"type": "terms", "target": "posterior cingulate", "exclude": "ventromedial prefrontal"}
"""

def clean_term_entity(raw_text):
    if not raw_text:
        return ""
    t = raw_text.strip().lower()
    t = re.sub(r"^[我想要請幫我找查尋看包含\s]+", "", t)
    t = re.sub(r"[之以外，,。\s]+$", "", t)
    t = re.sub(r"^[之以外，,。\s]+", "", t)
    t = t.strip()
    return SYNONYM_DICT.get(t, t)

def sanitize_and_lock_output(data):
    q_type = data.get("type", "terms")
    if q_type not in ["terms", "locations"]:
        q_type = "terms"

    target_val = clean_term_entity(str(data.get("target", "")))
    exclude_val = data.get("exclude")
    if exclude_val is not None:
        exclude_val = clean_term_entity(str(exclude_val))
        if exclude_val in ["", "none", "null"]:
            exclude_val = None

    if q_type == "locations":
        target_val = re.sub(r"[(),\s]+", "_", target_val).strip("_")
        if exclude_val:
            exclude_val = re.sub(r"[(),\s]+", "_", exclude_val).strip("_")

    return {
        "ok": True,
        "type": q_type,
        "target": {"type": "location" if q_type == "locations" else "term", "val": target_val, "display": target_val},
        "exclude": {"type": "location" if q_type == "locations" else "term", "val": exclude_val, "display": exclude_val} if exclude_val else None
    }

def robust_rule_fallback(text_input):
    """強固型本地倒裝與否定句解析器"""
    s = text_input.strip()
    coord_pattern = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*\)?")

    # 1. 精準匹配「除了 <排除項> 之外/以外/外，(我想看) <目標項>」
    inv_match = re.search(r"^除了\s*(.+?)\s*(?:之外|以外|外)\s*[,，]?\s*(.*)$", s)
    if inv_match:
        exclude_part = inv_match.group(1).strip()
        target_part = inv_match.group(2).strip()
        return sanitize_and_lock_output({"type": "terms", "target": target_part, "exclude": exclude_part})

    # 2. 匹配標準「<目標> but not / 不要 / 排除 <排除項>」
    neg_pattern = r"\bbut\s+not\b|\bwithout\b|\bexcept\b|\bexcluding\b|\bnot\b|但不要|不要|沒有|排除|非"
    match = re.search(neg_pattern, s, flags=re.IGNORECASE)

    def extract_entity(chunk):
        chunk = chunk.strip()
        m = coord_pattern.search(chunk)
        if m:
            coords = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            return "locations", f"{int(coords[0])}_{int(coords[1])}_{int(coords[2])}"
        return "terms", chunk

    if match:
        t_type, target_str = extract_entity(s[:match.start()])
        _, exclude_str = extract_entity(s[match.end():])
    else:
        t_type, target_str = extract_entity(s)
        exclude_str = None

    return sanitize_and_lock_output({"type": t_type, "target": target_str, "exclude": exclude_str})

def parse_with_gemini(user_input):
    api_key = os.getenv("GEMINI_API_KEY")

    if genai and api_key:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=GEMINI_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            parsed = json.loads(response.text.strip())
            print(f"[Gemini Success] Parsed: {parsed}")
            return sanitize_and_lock_output(parsed)
        except Exception as e:
            print(f"[Gemini API Failed]: {e}, switching to fallback rule.")
    else:
        print("[Notice] GEMINI_API_KEY not found or google-genai not loaded. Using fallback rule.")

    return robust_rule_fallback(user_input)

# ------------------------------------------------------------------
# Flask App Factory
# ------------------------------------------------------------------
def create_app():
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.get("/", endpoint="home")
    def home():
        return render_template("index.html")

    @app.get("/parse", endpoint="parse_sentence")
    def parse_sentence():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"ok": False, "error": "query is empty"}), 400
        return jsonify(parse_with_gemini(q))

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

    def _studies_response(ids_sql, params, query_info, fetch_points=False):
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
                total = conn.execute(text(f"SELECT COUNT(*) FROM ({ids_sql}) t"), params).scalar()

                points = []
                if fetch_points and rows:
                    study_ids = [r["study_id"] for r in rows[:100]]
                    pt_rows = conn.execute(text("""
                        SELECT ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z
                        FROM ns.coordinates
                        WHERE study_id = ANY(:sids)
                        LIMIT 500
                    """), {"sids": study_ids}).fetchall()
                    points = [[float(r[0]), float(r[1]), float(r[2])] for r in pt_rows]

        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        res_payload = {
            "ok": True,
            **query_info,
            "count": total,
            "returned": len(rows),
            "studies": [{"study_id": r["study_id"], "metadata": r["metadata"]} for r in rows],
        }
        if fetch_points:
            res_payload["points"] = points
        return jsonify(res_payload)

    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        clean_a = term_a.strip().lower()
        clean_b = term_b.strip().lower()
        if not clean_a or not clean_b:
            abort(400, "both terms are required")

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
            {"type": "locations", "in": [ax, ay, az], "not_in": [bx, by, bz], "radius_mm": r},
            fetch_points=True)

    @app.get("/version", endpoint="version")
    def version():
        return "dissociate-v12 (Fixed Inverted Regex + Gemini Fallback Logging)"

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
