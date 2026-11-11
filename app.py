import os
import re
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, static_folder="public/static")
CORS(app)

# ==========================================
# Local LLM 設定 (支援 Ollama 或 HuggingFace 本地模型)
# ==========================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "llama3.2:latest")  # 可改為 qwen2.5:7b 或 mistral

# 常見腦區與認知功能的中英對照與 MNI 中心座標 (作為快速標示與 Local Model Fallback)
BRAIN_ATLAS_DICT = {
    "amygdala": {"zh": "杏仁核 (Amygdala)", "mni": [-22, -4, -18], "terms": ["amygdala", "emotional", "fear"]},
    "杏仁核": {"zh": "杏仁核 (Amygdala)", "mni": [-22, -4, -18], "terms": ["amygdala", "emotion"]},
    "visual": {"zh": "初級視覺皮層 (V1 / Visual Cortex)", "mni": [0, -84, 6], "terms": ["visual", "occipital", "v1"]},
    "視覺": {"zh": "初級視覺皮層 (V1 / Visual Cortex)", "mni": [0, -84, 6], "terms": ["visual", "occipital", "v1"]},
    "hippocampus": {"zh": "海馬迴 (Hippocampus)", "mni": [-26, -20, -14], "terms": ["hippocampus", "memory", "episodic"]},
    "海馬迴": {"zh": "海馬迴 (Hippocampus)", "mni": [-26, -20, -14], "terms": ["hippocampus", "memory"]},
    "記憶": {"zh": "海馬迴 (Hippocampus)", "mni": [-26, -20, -14], "terms": ["memory", "hippocampus"]},
    "prefrontal": {"zh": "前額葉皮層 (Prefrontal Cortex)", "mni": [-36, 44, 20], "terms": ["prefrontal", "dlPFC", "executive"]},
    "前額葉": {"zh": "前額葉皮層 (Prefrontal Cortex)", "mni": [-36, 44, 20], "terms": ["prefrontal", "executive"]},
    "motor": {"zh": "初級運動皮層 (M1 / Motor Cortex)", "mni": [-38, -22, 56], "terms": ["motor", "precentral", "movement"]},
    "運動": {"zh": "初級運動皮層 (M1 / Motor Cortex)", "mni": [-38, -22, 56], "terms": ["motor", "movement"]},
    "insula": {"zh": "腦島 (Insula)", "mni": [-36, 14, 0], "terms": ["insula", "pain", "interoception"]},
    "腦島": {"zh": "腦島 (Insula)", "mni": [-36, 14, 0], "terms": ["insula", "pain"]},
    "anterior cingulate": {"zh": "前扣帶迴 (ACC)", "mni": [0, 28, 24], "terms": ["anterior cingulate", "acc", "conflict"]},
    "扣帶迴": {"zh": "前扣帶迴 (ACC)", "mni": [0, 28, 24], "terms": ["anterior cingulate", "cingulate"]},
}


def fallback_rule_based_interpreter(user_text: str):
    """當本地 Ollama 未啟動時，使用規則與字典進行自然語言解讀"""
    text_lower = user_text.lower()
    matched_terms = []
    excluded_terms = []
    detected_regions = []
    target_mni = [0, 0, 0]

    # 檢查是否有排除語氣 (例如：不要、排除、not, without)
    neg_split = re.split(r"不要|排除|除了|not |without |excluding ", text_lower)
    pos_part = neg_split[0]
    neg_part = " ".join(neg_split[1:]) if len(neg_split) > 1 else ""

    for key, info in BRAIN_ATLAS_DICT.items():
        if key in pos_part:
            term = info["terms"][0]
            if term not in matched_terms:
                matched_terms.append(term)
            detected_regions.append({
                "name": info["zh"],
                "mni": info["mni"],
                "term": term
            })
            target_mni = info["mni"]
        if neg_part and key in neg_part:
            term = info["terms"][0]
            if term not in excluded_terms:
                excluded_terms.append(term)

    # 若未匹配到字典，嘗試直接提取英文單字
    if not matched_terms:
        eng_words = re.findall(r"[a-zA-Z]{3,}", pos_part)
        stop_words = {"and", "the", "for", "with", "about", "find", "studies", "brain", "region"}
        matched_terms = [w for w in eng_words if w not in stop_words][:3]

    query_str = " AND ".join(matched_terms) if matched_terms else user_text.strip()
    if excluded_terms:
        query_str += " NOT " + " NOT ".join(excluded_terms)

    return {
        "parsed_query": query_str,
        "mni": target_mni,
        "regions": detected_regions,
        "explanation": f"已透過本地端解讀器將輸入解析為檢索條件：`{query_str}`" + (
            f"，並定位至 {detected_regions[0]['name']} {target_mni}" if detected_regions else ""
        ),
        "model_used": "Local Rule-Atlas Hybrid (Fallback)"
    }


@app.route("/api/parse_query", methods=["POST"])
def parse_natural_language_query():
    """
    使用 Local Model (Ollama / Local LLM) 將使用者輸入的自然語言解讀為：
    1. LoTUS-BF 支援的 Boolean Query (AND / OR / NOT)
    2. 對應的 MNI 座標 [x, y, z]
    3. 3D 腦區標示資訊
    """
    data = request.get_json(silent=True) or {}
    user_text = data.get("text", "").strip()
    model_choice = data.get("model", LOCAL_MODEL_NAME)

    if not user_text:
        return jsonify({"error": "請輸入欲解讀的文字描述"}), 400

    system_prompt = (
        "You are a neuroscience query interpreter for the LoTUS-BF neuroimaging database. "
        "Convert the user's natural language input (in Chinese or English) into a structured JSON object. "
        "Output ONLY valid JSON with these exact keys:\n"
        "- \"parsed_query\": A boolean query string using English neuroscience terms and operators (AND, OR, NOT), e.g., \"amygdala AND emotion NOT motor\".\n"
        "- \"mni\": A list of 3 integers [x, y, z] representing the primary MNI coordinate mentioned or implied (e.g. [-22, -4, -18] for amygdala, [0, -84, 6] for V1/visual cortex, [-26, -20, -14] for hippocampus).\n"
        "- \"regions\": A list of objects [{\"name\": \"Region Name\", \"mni\": [x, y, z], \"term\": \"english_term\"}].\n"
        "- \"explanation\": A brief explanation in Traditional Chinese of how you interpreted the input."
    )

    try:
        # 嘗試呼叫本地 Ollama 模型
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model_choice,
                "prompt": f"{system_prompt}\n\nUser Input: {user_text}\nJSON Output:",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1}
            },
            timeout=5
        )
        if resp.status_code == 200:
            res_json = resp.json()
            raw_output = res_json.get("response", "{}")
            parsed_data = json.loads(raw_output)
            return jsonify({
                "parsed_query": parsed_data.get("parsed_query", user_text),
                "mni": parsed_data.get("mni", [0, 0, 0]),
                "regions": parsed_data.get("regions", []),
                "explanation": parsed_data.get("explanation", "由本地 LLM 解讀完成。"),
                "model_used": f"Ollama ({model_choice})"
            })
    except Exception as e:
        # 如果本地端未開 Ollama，自動無縫切換至本地神經科學字典與規則解讀器
        pass

    fallback_res = fallback_rule_based_interpreter(user_text)
    return jsonify(fallback_res)


if __name__ == "__main__":
    # 請確保 port 與你的前端設定一致 (例如 5000)
    app.run(host="0.0.0.0", port=5000, debug=True)
