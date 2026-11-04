# app.py
from flask import Flask, jsonify, abort, request, send_file, render_template
import os
import re
import json
from sqlalchemy import create_engine, text

# 引用 Google GenAI SDK
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
    db_url = os.getenv("DB_URL")[cite: 2]
    if not db_url:
        raise RuntimeError("Missing DB_URL environment variable.")[cite: 2]
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):][cite: 2]
    _engine = create_engine(db_url, pool_pre_ping=True)[cite: 2]
    return _engine

# ------------------------------------------------------------------
# Gemini 語意解析系統提示詞（嚴格格式鎖定與神經科學詞彙標準化）
# ------------------------------------------------------------------
GEMINI_SYSTEM_INSTRUCTION = """
你是一個專精於神經科學資料庫 Neurosynth 的嚴格語意解析器。
你的職責是將使用者的任意口語、中英夾雜或倒裝句，轉換為符合後端 API 規格的結構化 JSON 物件。

【輸出架構約束】
1. 只能輸出純 JSON 物件，嚴禁包含 Markdown 標記（如 ```json）、思考過程或額外對話文字。
2. JSON 物件必須嚴格包含以下三個鍵：
   - "type": 字串，只能是 "terms" 或 "locations"。
   - "target": 字串，代表【要包含、正向選取】的大腦區域、概念或座標。
   - "exclude": 字串或 null，代表【要排除、否定、去除】的大腦區域、概念或座標；若無排除項目則必須為 null。

【解析規則與執行順序】
Step 1. 識別正負向（排除邏輯）：
   - 找出句子中代表排除的詞（如：but not, without, except, excluding, not, 但不要, 不要, 排除, 去除, 除了...之外）。
   - 被排除詞修飾的實體【必須】歸入 "exclude"。
   - 使用者想要觀察、主要查詢的主體【必須】歸入 "target"。
   - 即使使用者倒裝（如：「除了杏仁核之外，我想看後扣帶迴」），"target" 依然是 posterior cingulate，"exclude" 依然是 amygdala。

Step 2. 判斷實體型態 (type)：
   - 若輸入為 MNI 空間座標（如 "(0, -52, 26)" 或 "0 -52 26"），"type" 設為 "locations"，並轉為底線相連格式："x_y_z"（例如 "0_-52_26"）。
   - 若為解剖結構、腦網絡或認知詞彙，"type" 設為 "terms"。

Step 3. 詞彙標準化（Neurosynth 規範）：
   - 中文一律翻譯為英文標準小寫詞彙。
   - 去除 "cortex"、"area" 等多餘後綴（例如 "anterior cingulate cortex" -> "anterior cingulate"）。
   - 常見縮寫與別名強制轉換為全稱：
     * vmpfc / 腹內側前額葉 -> "ventromedial prefrontal"
     * pcc / 後扣帶迴 -> "posterior cingulate"
     * acc / 前扣帶迴 -> "anterior cingulate"
     * dlpfc / 背外側前額葉 -> "dorsolateral prefrontal"
     * ofc / 眶額葉 -> "orbitofrontal"
     * dmn / 預設模式網絡 -> "default mode"
     * amy / 杏仁核 -> "amygdala"

【範例基準】
輸入: "除了杏仁核之外，我想看後扣帶迴"
輸出: {"type": "terms", "target": "posterior cingulate", "exclude": "amygdala"}

輸入: "PCC but not vmPFC"
輸出: {"type": "terms", "target": "posterior cingulate", "exclude": "ventromedial prefrontal"}

輸入: "只留 (0, -52, 26) 排除 (-2, 50, -6)"
輸出: {"type": "locations", "target": "0_-52_26", "exclude": "-2_50_-6"}

輸入: "pain except reward"
輸出: {"type": "terms", "target": "pain", "exclude": "reward"}

輸入: "只看 working memory"
輸出: {"type": "terms", "target": "working memory", "exclude": null}
"""

def sanitize_and_lock_output(data):
    """防禦性校驗：確保無論 Gemini 怎麼回傳，資料型態與鍵值完全鎖定"""
    q_type = data.get("type", "terms")
    if q_type not in ["terms", "locations"]:
        q_type = "terms"

    target = str(data.get("target", "")).strip().lower()
    exclude = data.get("exclude")
    if exclude is not None:
        exclude = str(exclude).strip().lower()
        if exclude in ["", "none", "null"]:
            exclude = None

    if q_type == "locations":
        target = re.sub(r"[(),\s]+", "_", target).strip("_")
        if exclude:
            exclude = re.sub(r"[(),\s]+", "_", exclude).strip("_")

    return {
        "ok": True,
        "type": q_type,
        "target": {"type": "location" if q_type == "locations" else "term", "val": target, "display": target},
        "exclude": {"type": "location" if q_type == "locations" else "term", "val": exclude, "display": exclude} if exclude else None
    }

def parse_with_gemini(user_input):
    """使用 Google Gemini API 進行語意理解與參數提取"""
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
            parsed = json.loads(response.text)
            return sanitize_and_lock_output(parsed)
        except Exception as e:
            print(f"Gemini API 解析異常，切換至備援本機規則: {e}")

    # 備援規則解析（未提供 Key、網路異常或 API 超時時使用）
    negations = [r"\bbut\s+not\b", r"\bwithout\b", r"\bexcept\b", r"\bexcluding\b", r"\bnot\b", r"但不要", r"不要", r"沒有", r"排除", r"非", r"除了"]
    match = re.search(r"|".join(negations), user_input, flags=re.IGNORECASE)
    coord_pattern = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*\)?")

    def parse_chunk(s):
        s = s.strip()
        m = coord_pattern.search(s)
        if m:
            c = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            fmt = f"{int(c[0])}_{int(c[1])}_{int(c[2])}"
            return "location", fmt
        cleaned = re.sub(r"^[我想要請幫我找查尋\s]+", "", s).strip().lower()
        return "terms", cleaned

    if match:
        t_type, target = parse_chunk(user_input[:match.start()])
        _, exclude = parse_chunk(user_input[match.end():])
    else:
        t_type, target = parse_chunk(user_input)
        exclude = None

    return sanitize_and_lock_output({"type": t_type, "target": target, "exclude": exclude})

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

    # 串接 Gemini 語意解析
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
            conn.execute(text("SET search_path TO ns, public;"))[cite: 2]
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
        return send_file("amygdala.gif", mimetype="image/gif")[cite: 2]

    @app.get("/terms/<term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        return term[cite: 2]

    @app.get("/locations/<coords>/studies", endpoint="locations_studies")
    def get_studies_by_coordinates(coords):
        x, y, z = map(int, coords.split("_"))[cite: 2]
        return jsonify([x, y, z])[cite: 2]

    def _limit():
        try:
            return max(1, min(int(request.args.get("limit", 1000)), 10000))[cite: 2]
        except ValueError:
            abort(400, "limit must be an integer")[cite: 2]

    def _metadata_key(conn):
        cols = {r[0] for r in conn.execute(text([cite: 2]
            "SELECT column_name FROM information_schema.columns "[cite: 2]
            "WHERE table_schema = 'ns' AND table_name = 'metadata'"))}[cite: 2]
        return "study_id" if "study_id" in cols else "id"[cite: 2]

    def _studies_response(ids_sql, params, query_info, fetch_points=False):
        eng = get_engine()
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))[cite: 2]
                key = _metadata_key(conn)[cite: 2]
                rows = conn.execute(text(f"""
                    WITH hits AS ({ids_sql})
                    SELECT DISTINCT ON (h.study_id) h.study_id,
                           (to_jsonb(m) - 'fts') AS metadata
                    FROM hits h
                    LEFT JOIN ns.metadata m ON m.{key}::text = h.study_id
                    ORDER BY h.study_id
                    LIMIT :lim
                """), {**params, "lim": _limit()}).mappings().all()[cite: 2]
                total = conn.execute(text(f"SELECT COUNT(*) FROM ({ids_sql}) t"), params).scalar()[cite: 2]

                points = []
                if fetch_points and rows:
                    study_ids = [r["study_id"] for r in rows[:100]][cite: 2]
                    pt_rows = conn.execute(text("""
                        SELECT ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z
                        FROM ns.coordinates
                        WHERE study_id = ANY(:sids)
                        LIMIT 500
                    """), {"sids": study_ids}).fetchall()
                    points = [[float(r[0]), float(r[1]), float(r[2])] for r in pt_rows]

        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500[cite: 2]

        res_payload = {
            "ok": True,[cite: 2]
            **query_info,[cite: 2]
            "count": total,[cite: 2]
            "returned": len(rows),[cite: 2]
            "studies": [{"study_id": r["study_id"], "metadata": r["metadata"]} for r in rows],[cite: 2]
        }
        if fetch_points:
            res_payload["points"] = points
        return jsonify(res_payload)

    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        clean_a = term_a.strip().lower()
        clean_b = term_b.strip().lower()
        if not clean_a or not clean_b:
            abort(400, "both terms are required")[cite: 2]

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
                x, y, z = (float(v) for v in s.split("_"))[cite: 2]
            except ValueError:
                abort(400, f"bad coordinate '{s}', expected x_y_z")[cite: 2]
            return x, y, z[cite: 2]
        try:
            r = float(request.args.get("r", 10))[cite: 2]
        except ValueError:
            abort(400, "r must be a number")[cite: 2]
        ax, ay, az = parse(loc_a)[cite: 2]
        bx, by, bz = parse(loc_b)[cite: 2]
        ids_sql = """
            SELECT DISTINCT study_id FROM ns.coordinates
            WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:ax,:ay,:az), 4326), :r)
              AND study_id NOT IN (
                  SELECT study_id FROM ns.coordinates
                  WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:bx,:by,:bz), 4326), :r))
        """[cite: 2]
        return _studies_response(
            ids_sql,
            {"ax": ax, "ay": ay, "az": az, "bx": bx, "by": by, "bz": bz, "r": r},[cite: 2]
            {"type": "locations", "in": [ax, ay, az], "not_in": [bx, by, bz], "radius_mm": r},[cite: 2]
            fetch_points=True)

    @app.get("/version", endpoint="version")
    def version():
        return "dissociate-v10 (Powered by Google Gemini 2.5 Flash API)"

    @app.get("/test_db", endpoint="test_db")
    def test_db():
        eng = get_engine()[cite: 2]
        payload = {"ok": False, "dialect": eng.dialect.name}[cite: 2]
        try:
            with eng.begin() as conn:[cite: 2]
                conn.execute(text("SET search_path TO ns, public;"))[cite: 2]
                payload["version"] = conn.exec_driver_sql("SELECT version()").scalar()[cite: 2]
                payload["coordinates_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.coordinates")).scalar()[cite: 2]
                payload["metadata_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.metadata")).scalar()[cite: 2]
                payload["annotations_terms_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.annotations_terms")).scalar()[cite: 2]
            payload["ok"] = True[cite: 2]
            return jsonify(payload), 200[cite: 2]
        except Exception as e:
            payload["error"] = str(e)[cite: 2]
            return jsonify(payload), 500[cite: 2]

    return app

app = create_app()[cite: 2]
