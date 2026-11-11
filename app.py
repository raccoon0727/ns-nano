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
    "reward": {"mni": [-12, 12
