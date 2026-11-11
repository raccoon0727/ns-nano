# app.py
from flask import Flask, jsonify, abort, request, send_file, render_template
import os
import re
import json
import time
import urllib.request
import urllib.error
from sqlalchemy import create_engine, text

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
# Local Model 設定 (對接 http://localhost:1234/v1 - Gemma 3 4B Instruct QAT)
# ------------------------------------------------------------------
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1")
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "gemma-3-4b-it-qat")

# ------------------------------------------------------------------
# 神經科學詞彙與別名對照表 (本地保底字典)
# ------------------------------------------------------------------
SYNONYM_DICT = {
    "vmpfc": "ventromedial prefrontal",
    "pcc": "posterior cingulate",
    "acc": "anterior cingulate",
    "dlpfc": "dorsolateral prefrontal",
    "dacc": "dorsal anterior cingulate",
    "ofc": "orbitofrontal",
    "dmn": "default mode",
    "amy": "amygdala",
    "sma": "supplementary motor",
    "pmc": "premotor",
    "ffa": "fusiform face",
    "v1": "visual",
    "m1": "motor",
    "前扣帶迴": "anterior cingulate",
    "後扣帶迴": "posterior cingulate",
    "杏仁核": "amygdala",
    "腹內側前額葉": "ventromedial prefrontal",
    "背外側前額葉": "dorsolateral prefrontal",
    "眼窩前額葉": "orbitofrontal",
    "預設模式網路": "default mode",
    "前額葉皮質": "prefrontal",
    "前額葉": "prefrontal",
    "prefrontal cortex": "prefrontal",
    "島葉": "insula",
    "腦島": "insula",
    "視覺": "visual",
    "視覺皮質": "visual",
    "visual cortex": "visual",
    "聽覺": "auditory",
    "感覺": "sensory",
    "感覺皮質": "sensory",
    "海馬迴": "hippocampus",
    "海馬體": "hippocampus",
    "海馬旁回": "parahippocampal",
    "紋狀體": "striatum",
    "腹側紋狀體": "ventral striatum",
    "ventral striatum": "ventral striatum",
    "視丘": "thalamus",
    "基底核": "basal ganglia",
    "小腦": "cerebellum",
    "工作記憶": "working memory",
    "注意力": "attention",
    "語言": "language",
    "運動": "motor",
    "運動皮質": "motor",
    "motor cortex": "motor",
    "疼痛": "pain",
    "獎勵": "reward",
    "reward system": "reward",
    "情緒": "emotion",
    "臉部辨識": "face recognition",
    "梭狀迴": "fusiform face",
}

# ------------------------------------------------------------------
# 腦區標準 MNI 座標對照表 (供三切面平面圖與 3D 腦區標示使用)
# ------------------------------------------------------------------
BRAIN_MNI_MAP = {
    "amygdala": {"mni": [-22, -4, -18], "zh": "杏仁核 (Amygdala)"},
    "emotion": {"mni": [-22, -4, -18], "zh": "情緒相關區 / 杏仁核 (Emotion / Amygdala)"},
    "anterior cingulate": {"mni": [0, 28, 24], "zh": "前扣帶迴 (ACC)"},
    "dorsal anterior cingulate": {"mni": [0, 20, 36], "zh": "背側前扣帶迴 (dACC)"},
    "posterior cingulate": {"mni": [0, -52, 26], "zh": "後扣帶迴 (PCC)"},
    "default mode": {"mni": [0, -52, 26], "zh": "預設模式網路核心 (DMN / PCC)"},
    "ventromedial prefrontal": {"mni": [0, 46, -8], "zh": "腹內側前額葉 (vmPFC)"},
    "dorsolateral prefrontal": {"mni": [-42, 34, 28], "zh": "背外側前額葉 (dlPFC)"},
    "working memory": {"mni": [-42, 34, 28], "zh": "工作記憶網路 (dlPFC)"},
    "prefrontal": {"mni": [-36, 44, 20], "zh": "前額葉皮質 (Prefrontal)"},
    "orbitofrontal": {"mni": [0, 38, -16], "zh": "眼窩前額葉 (OFC)"},
    "insula": {"mni": [-36, 14, 0], "zh": "島葉 / 腦島 (Insula)"},
    "pain": {"mni": [-36, 14, 4], "zh": "疼痛網路 / 島葉 (Pain / Insula)"},
    "hippocampus": {"mni": [-26, -20, -14], "zh": "海馬迴 (Hippocampus)"},
    "parahippocampal": {"mni": [-24, -36, -12], "zh": "海馬旁回 (Parahippocampal)"},
    "striatum": {"mni": [-14, 10, -2], "zh": "紋狀體 (Striatum)"},
    "ventral striatum": {"mni": [-12, 12, -8], "zh": "腹側紋狀體 / 伏隔核 (Ventral Striatum)"},
    "reward": {"mni": [-12, 12, -8], "zh": "獎勵系統 / 腹側紋狀體 (Reward)"},
    "thalamus": {"mni": [-10, -18, 8], "zh": "視丘 (Thalamus)"},
    "basal ganglia": {"mni": [-18, 4, 4], "zh": "基底核 (Basal Ganglia)"},
    "cerebellum": {"mni": [-24, -60, -28], "zh": "小腦 (Cerebellum)"},
    "visual": {"mni": [0, -84, 6], "zh": "初級視覺皮質 (V1 / Visual)"},
    "auditory": {"mni": [-48, -22, 8], "zh": "聽覺皮質 (Auditory)"},
    "sensory": {"mni": [-40, -26, 54], "zh": "感覺皮質 (S1 / Sensory)"},
    "motor": {"mni": [-38, -22, 56], "zh": "初級運動皮質 (M1 / Motor)"},
    "supplementary motor": {"mni": [-4, -4, 58], "zh": "輔助運動區 (SMA)"},
    "premotor": {"mni": [-28, -10, 56], "zh": "前運動皮質 (Premotor)"},
    "fusiform face": {"mni": [40, -52, -18], "zh": "梭狀迴臉孔區 (FFA)"},
    "face recognition": {"mni": [40, -52, -18], "zh": "臉部辨識區 (FFA)"},
    "language": {"mni": [-48, 20, 18], "zh": "語言區 / 布洛卡區 (Broca / Language)"},
    "attention": {"mni": [30, -56, 48], "zh": "背側注意力網路 (Attention / IPS)"},
}

SYSTEM_INSTRUCTION = """
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
   - 識別所有否定排除語義（包括但不限於：拿掉、濾掉、不需要、別包含、去掉、去除、排除、不要、沒有、非、but not, without, except, excluding, not, minus, ignore, pass掉, 剔除, 扣除, 避開）。
   - 被否定詞修飾的實體歸入 "exclude"，想要保留或主要研究的實體歸入 "target"。
   - 倒裝句（例：「除了杏仁核之外，我想看後扣帶迴」）中，"target" 是 posterior cingulate，"exclude" 是 amygdala。
2. 座標規格：
   - MNI 座標轉換為 "x_y_z" 底線格式，此時 "type" 為 "locations"。
3. 詞彙標準化：
   - 中文必須翻譯為標準英文小寫詞彙。
   - 縮寫轉全稱（例：SMA -> supplementary motor, FFA -> fusiform face, dlpfc -> dorsolateral prefrontal, acc -> anterior cingulate, ofc -> orbitofrontal）。
"""

def resolve_mni_and_label(entity_type, val_str):
    if not val_str:
        return None, None
    if entity_type == "location":
        try:
            parts = [int(round(float(v))) for v in val_str.split("_")[:3]]
            if len(parts) == 3:
                best_zh = f"MNI 座標 ({parts[0]}, {parts[1]}, {parts[2]})"
                min_d = 999999
                for k, info in BRAIN_MNI_MAP.items():
                    cx, cy, cz = info["mni"]
                    d = ((parts[0] - cx) ** 2 + (parts[1] - cy) ** 2 + (parts[2] - cz) ** 2) ** 0.5
                    if d < min_d:
                        min_d = d
                        if d <= 24:
                            best_zh = f"{info['zh']} 附近"
                return parts, best_zh
        except Exception:
            pass
    clean_v = val_str.strip().lower()
    if clean_v in BRAIN_MNI_MAP:
        return BRAIN_MNI_MAP[clean_v]["mni"], BRAIN_MNI_MAP[clean_v]["zh"]
    for key, info in BRAIN_MNI_MAP.items():
        if key in clean_v or clean_v in key:
            return info["mni"], info["zh"]
    return None, None

def clean_term_entity(raw_text):
    if not raw_text:
        return ""
    t = raw_text.strip().lower()
    t = re.sub(r"^[我想要請幫找查尋看包含研究只保留留關注對\s]+", "", t)
    t = re.sub(r"(?:的部分|區域|相關|活化|介入|很有興趣|就免了)?$", "", t)
    t = re.sub(r"^(?:這次的\s*analysis\s*)", "", t)
    t = re.sub(r"[之以外，,。\s優先]+$", "", t)
    t = re.sub(r"^[之以外，,。\s]+", "", t)
    t = t.strip()
    return SYNONYM_DICT.get(t, t)

def sanitize_and_lock_output(data, engine="Rule Fallback"):
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
        target_val = re.sub(r"[()\[\],\s]+", "_", target_val).strip("_")
        if exclude_val:
            exclude_val = re.sub(r"[()\[\],\s]+", "_", exclude_val).strip("_")

    t_item_type = "location" if q_type == "locations" else "term"
    t_mni, t_region_zh = resolve_mni_and_label(t_item_type, target_val)

    target_obj = {
        "type": t_item_type,
        "val": target_val,
        "display": target_val,
        "mni": t_mni,
        "region_zh": t_region_zh
    }

    exclude_obj = None
    if exclude_val:
        e_mni, e_region_zh = resolve_mni_and_label(t_item_type, exclude_val)
        exclude_obj = {
            "type": t_item_type,
            "val": exclude_val,
            "display": exclude_val,
            "mni": e_mni,
            "region_zh": e_region_zh
        }

    return {
        "ok": True,
        "engine": engine,
        "type": q_type,
        "target": target_obj,
        "exclude": exclude_obj
    }

def robust_rule_fallback(text_input):
    s = text_input.strip()
    coord_pattern = re.compile(r"\(?\[?\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*[,_\s]\s*(-?\d+(?:\.\d+)?)\s*\]?\)?")

    inv_match = re.search(r"^(?:除了|避開|ignore)\s*(.+?)\s*(?:之外|以外|外)?\s*[,，]?\s*(.*)$", s, flags=re.IGNORECASE)
    if inv_match:
        return sanitize_and_lock_output({"type": "terms", "target": inv_match.group(2), "exclude": inv_match.group(1)}, "Rule Fallback")

    colloquial_match = re.search(r"^(.+?)(?:優先)?\s*[,，]?\s*(?:但把|但是把|不過|但|請|至於)?\s*(.+?)\s*(?:的部分|區域)?\s*(?:拿掉|濾掉|不需要|去掉|別包含|排除|不要|就免了)$", s)
    if colloquial_match:
        return sanitize_and_lock_output({"type": "terms", "target": colloquial_match.group(1), "exclude": colloquial_match.group(2)}, "Rule Fallback")

    neg_pattern = r"\bbut\s+not\b|\bwithout\b|\bexcept\b|\bexcluding\b|\bnot\b|\bminus\b|\bno\b|\bignore\b|但先pass\s*掉|pass\s*掉|剔除(?:掉)?|扣除(?:掉)?|不包含|避開|但不要|不要|沒有|排除|別包含|拿掉|濾掉|去掉|非"
    match = re.search(neg_pattern, s, flags=re.IGNORECASE)

    def extract_entity(chunk):
        chunk = chunk.strip()
        chunk = re.sub(r"^(?:target\s+is|給我看|這(?:個)?點)\s*", "", chunk, flags=re.IGNORECASE)
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

    return sanitize_and_lock_output({"type": t_type, "target": target_str, "exclude": exclude_str}, "Rule Fallback")

# ------------------------------------------------------------------
# Local Model 解讀器：呼叫 http://localhost:1234/v1 (Gemma 3 4B Instruct QAT)
# ------------------------------------------------------------------
def parse_with_local_gemma(user_input):
    endpoint = f"{LOCAL_API_BASE.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": LOCAL_MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.0,
        "max_tokens": 256
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            res_body = json.loads(resp.read().decode("utf-8"))
            content = res_body["choices"][0]["message"]["content"].strip()
            # 清除模型可能夾帶的 ```json 與 ``` 標記
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
            # 擷取第一個完整的 {...} JSON 區塊
            json_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if json_match:
                content = json_match.group(0)
            parsed = json.loads(content)
            if parsed.get("target"):
                return sanitize_and_lock_output(parsed, "Gemma 3 4B Instruct QAT")
    except Exception as e:
        print(f"[Local Gemma 3 4B QAT Failed -> Using Rule Fallback]: {e}")

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
        return jsonify(parse_with_local_gemma(q))

    @app.get("/atlas", endpoint="get_atlas")
    def get_atlas():
        atlas_list = [
            {"term": k, "zh": v["zh"], "mni": v["mni"]}
            for k, v in BRAIN_MNI_MAP.items()
        ]
        return jsonify(atlas_list)

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
                total = conn.execute(text(f"SELECT COUNT(*) FROM ({ids_sql}) t"), params).scalar()

                study_ids = [r["study_id"] for r in rows[:40]]
                peaks = []
                if study_ids:
                    try:
                        peak_rows = conn.execute(text("""
                            SELECT ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z, study_id
                            FROM ns.coordinates
                            WHERE study_id = ANY(:s_ids)
                            LIMIT 150
                        """), {"s_ids": study_ids}).mappings().all()
                        peaks = [
                            {"x": round(float(p["x"]), 1), "y": round(float(p["y"]), 1), "z": round(float(p["z"]), 1), "study_id": p["study_id"]}
                            for p in peak_rows
                        ]
                    except Exception:
                        peaks = []
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        return jsonify({
            "ok": True,
            **query_info,
            "count": total,
            "returned": len(rows),
            "peaks": peaks,
            "studies": [{"study_id": r["study_id"], "metadata": r["metadata"]} for r in rows],
        })

    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        if re.match(r"^-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?$", term_a):
            return dissociate_locations(term_a, term_b)

        clean_a = term_a.strip().lower()
        clean_b = term_b.strip().lower()
        if not clean_a or not clean_b:
            abort(400, "both terms are required")

        prefix = "terms_abstract_tfidf__"
        vars_a = list({clean_a, clean_a.replace(" ", "_"), f"{prefix}{clean_a}", f"{prefix}{clean_a.replace(' ', '_')}"})
        
        if clean_b == "none":
            ids_sql = """
                SELECT DISTINCT a.study_id 
                FROM ns.annotations_terms a
                WHERE a.term = ANY(:a_list)
            """
            return _studies_response(
                ids_sql,
                {"a_list": vars_a},
                {"type": "terms", "in": clean_a, "not_in": None}
            )
        else:
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

        if loc_b.lower() == "none":
            ids_sql = """
                SELECT DISTINCT study_id FROM ns.coordinates
                WHERE ST_3DDWithin(geom, ST_SetSRID(ST_MakePoint(:ax,:ay,:az), 4326), :r)
            """
            return _studies_response(
                ids_sql,
                {"ax": ax, "ay": ay, "az": az, "r": r},
                {"type": "locations", "in": [ax, ay, az], "not_in": None, "radius_mm": r}
            )
        else:
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
                {"type": "locations", "in": [ax, ay, az], "not_in": [bx, by, bz], "radius_mm": r}
            )

    @app.get("/version", endpoint="version")
    def version():
        return "dissociate-v22 (Gemma 3 4B Instruct QAT @ localhost:1234/v1)"

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
