# ============================================================
#  ボートレース AI シミュレーター v6.0  ─  app.py
#  Part 1/4: インポート・データクラス・シミュレーター
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from itertools import permutations
import warnings
warnings.filterwarnings("ignore")

# ── 日本語フォント設定 ──
def setup_japanese_font():
    candidates = [
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic",
        "IPAPGothic", "Hiragino Sans", "Yu Gothic", "Meiryo",
        "TakaoPGothic", "VL PGothic", "sans-serif"
    ]
    for font in candidates:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = font
            return
        except Exception:
            continue
    matplotlib.rcParams["font.family"] = "sans-serif"

# ── 会場プロファイル（全24場） ──
VENUE_PROFILES = {
    "桐生": {
        "code": "01", "water": "淡水", "tide": "なし",
        "course_win_rate": [54.0, 13.5, 12.5, 11.5, 5.5, 3.0],
        "course_top2": [72.0, 36.0, 28.0, 26.0, 20.0, 18.0],
        "course_top3": [82.0, 52.0, 44.0, 42.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.54, "差し": 0.12, "捲り": 0.16, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "ナイター・淡水",
        "seasons": {"spring": [53, 14, 13, 12, 5, 3], "summer": [52, 14, 13, 12, 6, 3],
                    "autumn": [55, 13, 12, 11, 5, 4], "winter": [56, 13, 12, 11, 5, 3]}
    },
    "戸田": {
        "code": "02", "water": "淡水", "tide": "なし",
        "course_win_rate": [45.0, 15.0, 13.0, 14.0, 8.0, 5.0],
        "course_top2": [65.0, 38.0, 30.0, 30.0, 22.0, 20.0],
        "course_top3": [78.0, 54.0, 46.0, 46.0, 40.0, 36.0],
        "kimarite_prob": {"逃げ": 0.45, "差し": 0.14, "捲り": 0.18, "捲差": 0.13, "抜き": 0.07, "恵まれ": 0.03},
        "wind_effect": "大", "memo": "日本一狭い水面",
        "seasons": {"spring": [44, 15, 14, 14, 8, 5], "summer": [43, 16, 14, 14, 8, 5],
                    "autumn": [46, 15, 13, 13, 8, 5], "winter": [47, 14, 13, 13, 8, 5]}
    },
    "江戸川": {
        "code": "03", "water": "汽水", "tide": "あり",
        "course_win_rate": [44.0, 15.0, 13.0, 14.0, 8.0, 6.0],
        "course_top2": [64.0, 37.0, 30.0, 30.0, 22.0, 22.0],
        "course_top3": [76.0, 53.0, 46.0, 46.0, 40.0, 38.0],
        "kimarite_prob": {"逃げ": 0.44, "差し": 0.14, "捲り": 0.18, "捲差": 0.13, "抜き": 0.07, "恵まれ": 0.04},
        "wind_effect": "大", "memo": "河川・波高い",
        "seasons": {"spring": [43, 15, 14, 14, 8, 6], "summer": [42, 16, 14, 14, 8, 6],
                    "autumn": [45, 15, 13, 14, 7, 6], "winter": [46, 14, 13, 14, 7, 6]}
    },
    "平和島": {
        "code": "04", "water": "海水", "tide": "あり",
        "course_win_rate": [43.0, 16.0, 13.0, 14.0, 8.0, 6.0],
        "course_top2": [63.0, 38.0, 30.0, 30.0, 22.0, 22.0],
        "course_top3": [76.0, 54.0, 46.0, 46.0, 40.0, 38.0],
        "kimarite_prob": {"逃げ": 0.43, "差し": 0.16, "捲り": 0.17, "捲差": 0.13, "抜き": 0.07, "恵まれ": 0.04},
        "wind_effect": "大", "memo": "インが弱い",
        "seasons": {"spring": [42, 16, 14, 14, 8, 6], "summer": [41, 17, 14, 14, 8, 6],
                    "autumn": [44, 16, 13, 14, 7, 6], "winter": [45, 15, 13, 14, 7, 6]}
    },
    "多摩川": {
        "code": "05", "water": "淡水", "tide": "なし",
        "course_win_rate": [55.0, 13.0, 12.0, 11.0, 5.5, 3.5],
        "course_top2": [73.0, 35.0, 28.0, 25.0, 20.0, 18.0],
        "course_top3": [83.0, 51.0, 44.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.55, "差し": 0.12, "捲り": 0.15, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "小", "memo": "静水面",
        "seasons": {"spring": [54, 13, 13, 11, 6, 3], "summer": [53, 14, 13, 11, 6, 3],
                    "autumn": [56, 13, 12, 11, 5, 3], "winter": [57, 12, 12, 11, 5, 3]}
    },
    "浜名湖": {
        "code": "06", "water": "汽水", "tide": "あり",
        "course_win_rate": [54.0, 13.0, 12.0, 12.0, 5.5, 3.5],
        "course_top2": [72.0, 36.0, 28.0, 26.0, 20.0, 18.0],
        "course_top3": [82.0, 52.0, 44.0, 42.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.54, "差し": 0.13, "捲り": 0.15, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "広い水面",
        "seasons": {"spring": [53, 14, 12, 12, 6, 3], "summer": [52, 14, 13, 12, 6, 3],
                    "autumn": [55, 13, 12, 11, 5, 4], "winter": [56, 12, 12, 11, 5, 4]}
    },
    "蒲郡": {
        "code": "07", "water": "汽水", "tide": "あり",
        "course_win_rate": [56.0, 13.0, 11.0, 11.0, 5.5, 3.5],
        "course_top2": [74.0, 35.0, 27.0, 25.0, 20.0, 18.0],
        "course_top3": [84.0, 51.0, 43.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.56, "差し": 0.12, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "ナイター",
        "seasons": {"spring": [55, 13, 12, 11, 6, 3], "summer": [54, 14, 12, 11, 6, 3],
                    "autumn": [57, 13, 11, 11, 5, 3], "winter": [58, 12, 11, 11, 5, 3]}
    },
    "常滑": {
        "code": "08", "water": "海水", "tide": "あり",
        "course_win_rate": [56.0, 13.0, 11.0, 11.0, 5.5, 3.5],
        "course_top2": [74.0, 35.0, 27.0, 25.0, 20.0, 18.0],
        "course_top3": [84.0, 51.0, 43.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.56, "差し": 0.12, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "風影響あり",
        "seasons": {"spring": [55, 13, 12, 11, 6, 3], "summer": [54, 14, 12, 11, 6, 3],
                    "autumn": [57, 13, 11, 11, 5, 3], "winter": [58, 12, 11, 11, 5, 3]}
    },
    "津": {
        "code": "09", "water": "海水", "tide": "あり",
        "course_win_rate": [55.0, 13.0, 12.0, 11.0, 5.5, 3.5],
        "course_top2": [73.0, 36.0, 28.0, 26.0, 20.0, 18.0],
        "course_top3": [83.0, 52.0, 44.0, 42.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.55, "差し": 0.12, "捲り": 0.15, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "ナイター",
        "seasons": {"spring": [54, 14, 12, 11, 6, 3], "summer": [53, 14, 13, 12, 6, 3],
                    "autumn": [56, 13, 12, 11, 5, 3], "winter": [57, 12, 12, 11, 5, 3]}
    },
    "三国": {
        "code": "10", "water": "淡水", "tide": "なし",
        "course_win_rate": [53.0, 13.5, 12.5, 12.0, 5.5, 3.5],
        "course_top2": [71.0, 36.0, 29.0, 27.0, 20.0, 18.0],
        "course_top3": [81.0, 52.0, 45.0, 43.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.53, "差し": 0.13, "捲り": 0.15, "捲差": 0.11, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "大", "memo": "北風強い",
        "seasons": {"spring": [52, 14, 13, 12, 6, 3], "summer": [51, 14, 13, 12, 6, 4],
                    "autumn": [54, 13, 12, 12, 5, 4], "winter": [55, 13, 12, 12, 5, 3]}
    },
    "びわこ": {
        "code": "11", "water": "淡水", "tide": "なし",
        "course_win_rate": [50.0, 14.0, 13.0, 12.0, 7.0, 4.0],
        "course_top2": [68.0, 37.0, 30.0, 28.0, 22.0, 20.0],
        "course_top3": [80.0, 53.0, 46.0, 44.0, 40.0, 36.0],
        "kimarite_prob": {"逃げ": 0.50, "差し": 0.14, "捲り": 0.16, "捲差": 0.11, "抜き": 0.07, "恵まれ": 0.02},
        "wind_effect": "大", "memo": "うねり注意",
        "seasons": {"spring": [49, 14, 14, 12, 7, 4], "summer": [48, 15, 14, 12, 7, 4],
                    "autumn": [51, 14, 13, 12, 6, 4], "winter": [52, 14, 12, 12, 6, 4]}
    },
    "住之江": {
        "code": "12", "water": "淡水", "tide": "なし",
        "course_win_rate": [56.0, 13.0, 11.0, 11.0, 5.5, 3.5],
        "course_top2": [74.0, 35.0, 27.0, 25.0, 20.0, 18.0],
        "course_top3": [84.0, 51.0, 43.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.56, "差し": 0.12, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "小", "memo": "インが強い",
        "seasons": {"spring": [55, 13, 12, 11, 6, 3], "summer": [54, 14, 12, 11, 6, 3],
                    "autumn": [57, 13, 11, 11, 5, 3], "winter": [58, 12, 11, 11, 5, 3]}
    },
    "尼崎": {
        "code": "13", "water": "淡水", "tide": "なし",
        "course_win_rate": [55.0, 14.0, 12.0, 11.0, 5.0, 3.0],
        "course_top2": [73.0, 36.0, 28.0, 26.0, 20.0, 18.0],
        "course_top3": [83.0, 52.0, 44.0, 42.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.55, "差し": 0.13, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "小", "memo": "センターポール",
        "seasons": {"spring": [54, 14, 13, 11, 5, 3], "summer": [53, 15, 13, 11, 5, 3],
                    "autumn": [56, 14, 12, 11, 5, 2], "winter": [57, 13, 12, 11, 5, 2]}
    },
    "鳴門": {
        "code": "14", "water": "海水", "tide": "あり",
        "course_win_rate": [54.0, 14.0, 12.0, 11.0, 5.5, 3.5],
        "course_top2": [72.0, 36.0, 28.0, 26.0, 20.0, 18.0],
        "course_top3": [82.0, 52.0, 44.0, 42.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.54, "差し": 0.13, "捲り": 0.15, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "潮流影響",
        "seasons": {"spring": [53, 14, 13, 11, 6, 3], "summer": [52, 15, 13, 11, 6, 3],
                    "autumn": [55, 14, 12, 11, 5, 3], "winter": [56, 13, 12, 11, 5, 3]}
    },
    "丸亀": {
        "code": "15", "water": "海水", "tide": "あり",
        "course_win_rate": [56.0, 13.0, 11.0, 11.0, 5.5, 3.5],
        "course_top2": [74.0, 35.0, 27.0, 25.0, 20.0, 18.0],
        "course_top3": [84.0, 51.0, 43.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.56, "差し": 0.12, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "ナイター",
        "seasons": {"spring": [55, 13, 12, 11, 6, 3], "summer": [54, 14, 12, 11, 6, 3],
                    "autumn": [57, 13, 11, 11, 5, 3], "winter": [58, 12, 11, 11, 5, 3]}
    },
    "児島": {
        "code": "16", "water": "海水", "tide": "あり",
        "course_win_rate": [53.0, 14.0, 12.0, 12.0, 6.0, 3.0],
        "course_top2": [71.0, 36.0, 29.0, 27.0, 21.0, 18.0],
        "course_top3": [81.0, 52.0, 45.0, 43.0, 39.0, 34.0],
        "kimarite_prob": {"逃げ": 0.53, "差し": 0.13, "捲り": 0.15, "捲差": 0.11, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "潮位差大",
        "seasons": {"spring": [52, 14, 13, 12, 6, 3], "summer": [51, 15, 13, 12, 6, 3],
                    "autumn": [54, 14, 12, 12, 5, 3], "winter": [55, 13, 12, 12, 5, 3]}
    },
    "宮島": {
        "code": "17", "water": "海水", "tide": "あり",
        "course_win_rate": [53.0, 14.0, 12.0, 12.0, 5.5, 3.5],
        "course_top2": [71.0, 36.0, 29.0, 27.0, 20.0, 18.0],
        "course_top3": [81.0, 52.0, 45.0, 43.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.53, "差し": 0.14, "捲り": 0.15, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "潮・風影響",
        "seasons": {"spring": [52, 14, 13, 12, 6, 3], "summer": [51, 15, 13, 12, 6, 3],
                    "autumn": [54, 14, 12, 12, 5, 3], "winter": [55, 13, 12, 12, 5, 3]}
    },
    "徳山": {
        "code": "18", "water": "海水", "tide": "あり",
        "course_win_rate": [63.4, 11.7, 12.8, 9.7, 3.5, 1.1],
        "course_top2": [78.0, 33.0, 28.0, 24.0, 18.0, 14.0],
        "course_top3": [86.0, 50.0, 44.0, 40.0, 36.0, 30.0],
        "kimarite_prob": {"逃げ": 0.63, "差し": 0.10, "捲り": 0.12, "捲差": 0.08, "抜き": 0.05, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "インが非常に強い",
        "seasons": {"spring": [62, 12, 13, 10, 4, 1], "summer": [61, 13, 13, 10, 4, 1],
                    "autumn": [64, 12, 12, 9, 3, 1], "winter": [65, 11, 12, 9, 3, 1]}
    },
    "下関": {
        "code": "19", "water": "海水", "tide": "あり",
        "course_win_rate": [56.0, 13.0, 12.0, 11.0, 5.0, 3.0],
        "course_top2": [74.0, 35.0, 28.0, 25.0, 20.0, 18.0],
        "course_top3": [84.0, 51.0, 44.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.56, "差し": 0.12, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "ナイター",
        "seasons": {"spring": [55, 13, 13, 11, 5, 3], "summer": [54, 14, 13, 11, 5, 3],
                    "autumn": [57, 13, 12, 11, 5, 2], "winter": [58, 12, 12, 11, 5, 2]}
    },
    "若松": {
        "code": "20", "water": "海水", "tide": "あり",
        "course_win_rate": [55.0, 13.0, 12.0, 11.0, 5.5, 3.5],
        "course_top2": [73.0, 36.0, 28.0, 26.0, 20.0, 18.0],
        "course_top3": [83.0, 52.0, 44.0, 42.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.55, "差し": 0.13, "捲り": 0.15, "捲差": 0.10, "抜き": 0.05, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "ナイター",
        "seasons": {"spring": [54, 13, 13, 11, 6, 3], "summer": [53, 14, 13, 11, 6, 3],
                    "autumn": [56, 13, 12, 11, 5, 3], "winter": [57, 12, 12, 11, 5, 3]}
    },
    "芦屋": {
        "code": "21", "water": "淡水", "tide": "なし",
        "course_win_rate": [58.0, 12.0, 11.0, 10.0, 5.5, 3.5],
        "course_top2": [76.0, 34.0, 27.0, 24.0, 20.0, 18.0],
        "course_top3": [86.0, 50.0, 43.0, 40.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.58, "差し": 0.11, "捲り": 0.14, "捲差": 0.09, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "小", "memo": "モーニング",
        "seasons": {"spring": [57, 12, 12, 10, 6, 3], "summer": [56, 13, 12, 10, 6, 3],
                    "autumn": [59, 12, 11, 10, 5, 3], "winter": [60, 11, 11, 10, 5, 3]}
    },
    "福岡": {
        "code": "22", "water": "海水", "tide": "あり",
        "course_win_rate": [50.0, 14.0, 13.0, 12.0, 7.0, 4.0],
        "course_top2": [68.0, 37.0, 30.0, 28.0, 22.0, 20.0],
        "course_top3": [80.0, 53.0, 46.0, 44.0, 40.0, 36.0],
        "kimarite_prob": {"逃げ": 0.50, "差し": 0.14, "捲り": 0.16, "捲差": 0.11, "抜き": 0.06, "恵まれ": 0.03},
        "wind_effect": "大", "memo": "うねり・インが弱い",
        "seasons": {"spring": [49, 14, 14, 12, 7, 4], "summer": [48, 15, 14, 13, 7, 3],
                    "autumn": [51, 14, 13, 12, 6, 4], "winter": [52, 14, 13, 12, 6, 3]}
    },
    "唐津": {
        "code": "23", "water": "海水", "tide": "あり",
        "course_win_rate": [56.0, 13.0, 11.0, 11.0, 5.5, 3.5],
        "course_top2": [74.0, 35.0, 27.0, 25.0, 20.0, 18.0],
        "course_top3": [84.0, 51.0, 43.0, 41.0, 38.0, 34.0],
        "kimarite_prob": {"逃げ": 0.56, "差し": 0.12, "捲り": 0.14, "捲差": 0.10, "抜き": 0.06, "恵まれ": 0.02},
        "wind_effect": "中", "memo": "風影響あり",
        "seasons": {"spring": [55, 13, 12, 11, 6, 3], "summer": [54, 14, 12, 11, 6, 3],
                    "autumn": [57, 13, 11, 11, 5, 3], "winter": [58, 12, 11, 11, 5, 3]}
    },
    "大村": {
        "code": "24", "water": "海水", "tide": "あり",
        "course_win_rate": [65.0, 11.0, 10.0, 8.0, 3.5, 2.5],
        "course_top2": [80.0, 32.0, 26.0, 22.0, 18.0, 14.0],
        "course_top3": [88.0, 48.0, 42.0, 38.0, 36.0, 30.0],
        "kimarite_prob": {"逃げ": 0.65, "差し": 0.09, "捲り": 0.11, "捲差": 0.08, "抜き": 0.05, "恵まれ": 0.02},
        "wind_effect": "小", "memo": "インが最も強い",
        "seasons": {"spring": [64, 11, 11, 8, 4, 2], "summer": [63, 12, 11, 8, 4, 2],
                    "autumn": [66, 11, 10, 8, 3, 2], "winter": [67, 10, 10, 8, 3, 2]}
    }
}

DEFAULT_VENUE_PROFILE = VENUE_PROFILES["徳山"]

def get_venue_profile(name):
    for key in VENUE_PROFILES:
        if key in name:
            return VENUE_PROFILES[key]
    return DEFAULT_VENUE_PROFILE

def get_season(month):
    if month in [3, 4, 5]: return "spring"
    if month in [6, 7, 8]: return "summer"
    if month in [9, 10, 11]: return "autumn"
    return "winter"

# ── BoatAgent データクラス ──
@dataclass
class BoatAgent:
    lane: int = 1
    number: int = 0
    name: str = "選手"
    rank: str = "B1"
    age: int = 30
    weight: float = 52.0
    avg_st: float = 0.15
    national_win_rate: float = 5.0
    national_top2_rate: float = 30.0
    national_top3_rate: float = 45.0
    local_win_rate: float = 4.5
    lane_win_rate: float = 10.0
    ability: int = 50
    motor_contribution: float = 0.0
    motor_top2_rate: float = 30.0
    exhibition_time: float = 6.80
    exhibition_rank: int = 3
    turn_time: float = 6.70
    tilt: float = -0.5
    accident_rate: float = 0.0
    flying_count: int = 0
    kimarite_nige: int = 0
    kimarite_sashi: int = 0
    kimarite_makuri: int = 0
    kimarite_makuzashi: int = 0
    kimarite_nuki: int = 0
    kimarite_megumre: int = 0
    straight_time: float = 7.50
    lap_time: float = 37.0
    branch: str = ""

    def calculate_start_timing(self):
        base = self.avg_st
        variation = np.random.normal(0, 0.02)
        st = base + variation
        if self.flying_count > 0:
            st += 0.01 * self.flying_count
        return max(0.01, st)

    def get_power_score(self):
        score = self.ability / 100.0
        rank_bonus = {"A1": 0.15, "A2": 0.08, "B1": 0.0, "B2": -0.08}
        score += rank_bonus.get(self.rank, 0.0)
        return score

    def get_machine_score(self):
        score = 0.5
        score += self.motor_contribution * 0.1
        score += (self.motor_top2_rate - 30.0) / 100.0
        score += (6.80 - self.exhibition_time) * 2.0
        score += self.tilt * 0.02
        return max(0.1, min(1.0, score))

    def get_turn_score(self):
        score = 0.5
        score += (6.70 - self.turn_time) * 1.5
        score += (3 - self.exhibition_rank) * 0.03
        score -= (self.weight - 52.0) * 0.01
        return max(0.1, min(1.0, score))

    def get_kimarite_tendency(self):
        total = (self.kimarite_nige + self.kimarite_sashi +
                 self.kimarite_makuri + self.kimarite_makuzashi +
                 self.kimarite_nuki + self.kimarite_megumre)
        if total == 0:
            return {"逃げ": 0.3, "差し": 0.2, "捲り": 0.2, "捲差": 0.15, "抜き": 0.1, "恵まれ": 0.05}
        return {
            "逃げ": self.kimarite_nige / total,
            "差し": self.kimarite_sashi / total,
            "捲り": self.kimarite_makuri / total,
            "捲差": self.kimarite_makuzashi / total,
            "抜き": self.kimarite_nuki / total,
            "恵まれ": self.kimarite_megumre / total
        }

# ── RaceCondition データクラス ──
@dataclass
class RaceCondition:
    weather: str = "晴"
    temperature: float = 20.0
    wind_speed: float = 1.0
    water_temperature: float = 18.0
    wave_height: int = 1
    wind_direction: str = ""
    tide: str = ""

# ── RaceSimulator ──
class RaceSimulator:
    def __init__(self, agents, conditions, venue_profile=None, month=2):
        self.agents = agents
        self.conditions = conditions
        self.venue = venue_profile or DEFAULT_VENUE_PROFILE
        self.season = get_season(month)

    def _compute_race_weights(self):
        weights = np.zeros(6)
        season_rates = self.venue.get("seasons", {}).get(self.season,
                       self.venue["course_win_rate"])

        for i, a in enumerate(self.agents):
            w = 0.0
            w += season_rates[i] * 0.20
            w += a.lane_win_rate * 0.08
            w += a.national_win_rate * 2.5 * 0.12
            w += a.local_win_rate * 2.0 * 0.06
            w += a.get_power_score() * 40 * 0.12
            w += a.get_machine_score() * 40 * 0.10
            w += a.get_turn_score() * 40 * 0.08
            st_score = max(0, (0.20 - a.avg_st) * 200)
            w += st_score * 0.08
            weight_score = max(0, (55 - a.weight) * 0.5)
            w += weight_score * 0.03
            wind_effect = self.venue.get("wind_effect", "中")
            if wind_effect == "大" and self.conditions.wind_speed >= 4:
                if i == 0:
                    w *= 0.90
                elif i >= 3:
                    w *= 1.05
            if self.conditions.wave_height >= 5:
                if i == 0:
                    w *= 0.92
                elif i >= 3:
                    w *= 1.03
            if a.accident_rate > 0.5:
                w *= 0.92
            if a.flying_count > 0:
                w *= 0.95
            w += a.national_top2_rate * 0.02
            w += a.national_top3_rate * 0.01
            weights[i] = max(w, 0.1)

        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    def simulate_race(self):
        st_times = [a.calculate_start_timing() for a in self.agents]
        weights = self._compute_race_weights()
        adjusted = weights.copy()

        best_st = min(st_times)
        for i in range(6):
            st_diff = st_times[i] - best_st
            adjusted[i] *= max(0.5, 1.0 - st_diff * 2.0)
            ex_bonus = (6.90 - self.agents[i].exhibition_time) * 0.5
            adjusted[i] *= max(0.7, 1.0 + ex_bonus)

        weather_noise = 0.02
        if self.conditions.weather in ["雨", "雪"]:
            weather_noise = 0.05
        if self.conditions.wave_height >= 5:
            weather_noise += 0.03
        if self.conditions.wind_speed >= 5:
            weather_noise += 0.02

        noise = np.random.normal(0, weather_noise, 6)
        adjusted = adjusted + noise
        adjusted = np.maximum(adjusted, 0.01)
        adjusted = adjusted / adjusted.sum()

        order = []
        remaining = list(range(6))
        temp_w = adjusted.copy()
        for _ in range(6):
            probs = np.array([temp_w[j] for j in remaining])
            probs = np.maximum(probs, 0.001)
            probs = probs / probs.sum()
            idx = np.random.choice(len(remaining), p=probs)
            chosen = remaining.pop(idx)
            order.append(chosen + 1)
            temp_w[chosen] = 0

        positions = {lane: [] for lane in range(1, 7)}
        n_points = 8
        for pt in range(n_points):
            progress = pt / (n_points - 1)
            for lane in range(1, 7):
                final_pos = order.index(lane) + 1
                start_pos = lane
                current = start_pos + (final_pos - start_pos) * progress
                current += np.random.normal(0, 0.3) * (1 - progress)
                positions[lane].append(current)

        kimarite = self._determine_kimarite(order, st_times)

        return {
            "finish_order": order,
            "start_times": st_times,
            "kimarite": kimarite,
            "positions": positions,
            "weights": weights
        }

    def _determine_kimarite(self, order, st_times):
        winner = order[0]
        winner_agent = self.agents[winner - 1]
        tendency = winner_agent.get_kimarite_tendency()
        venue_prob = self.venue.get("kimarite_prob", {})

        combined = {}
        for k in ["逃げ", "差し", "捲り", "捲差", "抜き", "恵まれ"]:
            combined[k] = tendency.get(k, 0.1) * 0.6 + venue_prob.get(k, 0.1) * 0.4

        if winner == 1:
            combined["逃げ"] *= 3.0
            combined["差し"] *= 0.1
            combined["捲り"] *= 0.1
        elif winner == 2:
            combined["差し"] *= 2.5
            combined["捲り"] *= 1.5
            combined["逃げ"] *= 0.1
        elif winner >= 3:
            combined["捲り"] *= 2.0
            combined["捲差"] *= 2.0
            combined["逃げ"] *= 0.1

        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        keys = list(combined.keys())
        probs = [combined[k] for k in keys]
        return np.random.choice(keys, p=probs)
# ============================================================
#  ボートレース AI シミュレーター v6.1  ─  app.py
#  Part 2/4: パーサー（公式サイトコピー完全対応・修正版）
# ============================================================

# ── ユーティリティ ──
def safe_float(s, default=0.0):
    if s is None: return default
    s = str(s).strip().replace(",", "").replace("％", "").replace("%", "")
    if s in ("", "-", "−", "—", "―"): return default
    try: return float(s)
    except ValueError: return default

def safe_int(s, default=0):
    return int(safe_float(s, default))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  公式サイトパーサー v6.1（行単位マッチング方式）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_official_site_text(text):
    """
    公式サイトからのコピーテキストを解析する。
    
    方針: テキストを行に分割し、各行から数値を抽出。
    「ラベル行」の直後（または同一行）に6つの数値がある構造を利用。
    セクション境界を正確に特定するために、全てのセクション見出しの
    出現位置を記録し、範囲を限定して検索する。
    """
    raw_lines = text.split("\n")
    lines = [l.rstrip() for l in raw_lines]
    N = len(lines)
    
    agents = [BoatAgent(lane=i+1) for i in range(6)]
    conditions = RaceCondition()

    # ────────────────────────────────────
    # ヘルパー関数
    # ────────────────────────────────────
    def nums_in(s):
        """文字列から全ての数値（負の小数含む）を抽出"""
        return [safe_float(x) for x in re.findall(r'-?\d+\.?\d*', s)]

    def pcts_in(s):
        """文字列から %の前の数値を抽出"""
        return [safe_float(x) for x in re.findall(r'(\d+\.?\d*)%', s)]

    def find_all(keyword, start=0, end=None):
        """キーワードを含む全行のインデックスを返す"""
        end = end or N
        result = []
        for i in range(start, min(end, N)):
            if keyword in lines[i]:
                result.append(i)
        return result

    def find_first(keyword, start=0, end=None):
        """キーワードを含む最初の行のインデックスを返す"""
        hits = find_all(keyword, start, end)
        return hits[0] if hits else -1

    def get_six(line_idx, is_pct=False):
        """指定行（とその次行）から6つの数値を取得"""
        if line_idx < 0 or line_idx >= N:
            return None
        text1 = lines[line_idx]
        text2 = lines[line_idx + 1] if line_idx + 1 < N else ""
        
        if is_pct:
            vals = pcts_in(text1)
            if len(vals) >= 6: return vals[:6]
            vals = pcts_in(text1 + " " + text2)
            if len(vals) >= 6: return vals[:6]
        else:
            vals = nums_in(text1)
            if len(vals) >= 6: return vals[:6]
            vals = nums_in(text1 + " " + text2)
            if len(vals) >= 6: return vals[:6]
        return None

    def find_label_then_six(label, start, end, is_pct=False):
        """
        start~end の範囲で label を含む行を見つけ、
        その行（ラベル部分を除去した残り）から6値を取得。
        同一行に6値なければ次行も合わせて取得。
        """
        for i in range(max(0, start), min(end, N)):
            if label not in lines[i]:
                continue
            # ラベル以降の文字列
            after = lines[i].split(label, 1)[-1]
            if is_pct:
                vals = pcts_in(after)
                if len(vals) >= 6: return vals[:6]
                if i + 1 < N:
                    vals = pcts_in(after + " " + lines[i+1])
                    if len(vals) >= 6: return vals[:6]
            else:
                vals = nums_in(after)
                if len(vals) >= 6: return vals[:6]
                if i + 1 < N:
                    vals = nums_in(after + " " + lines[i+1])
                    if len(vals) >= 6: return vals[:6]
        return None

    # ────────────────────────────────────
    # セクション境界の特定
    # ────────────────────────────────────
    # メニュー項目としての「基本情報」と実データの「基本情報」を区別するため
    # 「平均ST」の出現位置を基準にする（平均STは基本情報セクション内にのみある）
    
    avg_st_positions = find_all("平均ST")
    # 最初の平均STの位置が基本情報データセクションの開始付近
    data_start = avg_st_positions[0] - 2 if avg_st_positions else 0
    
    # 各セクション見出しをdata_start以降で探す
    sec_positions = {}
    for sec_name in ["枠別情報", "モーター情報", "モータ情報", "直前情報"]:
        idx = find_first(sec_name, data_start)
        if idx >= 0:
            sec_positions[sec_name] = idx
    
    motor_idx = sec_positions.get("モーター情報", sec_positions.get("モータ情報", -1))
    waku_idx = sec_positions.get("枠別情報", -1)
    direct_idx = sec_positions.get("直前情報", -1)
    
    # セクション範囲
    base_start = data_start
    base_end = min(x for x in [waku_idx, motor_idx, direct_idx, N] if x > 0)
    waku_start = waku_idx if waku_idx > 0 else base_end
    waku_end = min(x for x in [motor_idx, direct_idx, N] if x > waku_start) if waku_idx > 0 else base_end
    motor_start = motor_idx if motor_idx > 0 else waku_end
    motor_end = direct_idx if direct_idx > motor_start else N
    direct_start = direct_idx if direct_idx > 0 else N
    direct_end = N

    # ────────────────────────────────────
    # 1. 登録番号・名前・級
    # ────────────────────────────────────
    skip_names = {"号艇", "情報", "選手", "コース", "名前", "登録", "モーター",
                  "1号艇", "2号艇", "3号艇", "4号艇", "5号艇", "6号艇",
                  "基本情報", "枠別情報", "直前情報"}

    found_numbers = False
    for idx in range(N):
        line = lines[idx]
        four_digit = re.findall(r'\b(\d{4})\b', line)
        # 4桁数字が6個（登録番号行）
        candidates = [int(n) for n in four_digit if 1000 <= int(n) <= 5999]
        if len(candidates) >= 6 and not found_numbers:
            for i in range(6):
                agents[i].number = candidates[i]
            found_numbers = True
            
            # 次の行 = 名前行
            if idx + 1 < N:
                name_line = lines[idx + 1].strip()
                parts = re.split(r'[\t\s　]+', name_line)
                name_candidates = [
                    p for p in parts
                    if len(p) >= 2
                    and p not in skip_names
                    and not re.match(r'^[\d.%]+$', p)
                    and re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', p)
                ]
                if len(name_candidates) >= 6:
                    for i in range(6):
                        agents[i].name = name_candidates[i]
                    
                    # その次 = 級行
                    if idx + 2 < N:
                        rank_line = lines[idx + 2]
                        ranks = re.findall(r'[AB][12]', rank_line)
                        if len(ranks) >= 6:
                            for i in range(6):
                                agents[i].rank = ranks[i]
            break

    # ────────────────────────────────────
    # 2. 基本情報セクション (base_start ~ base_end)
    # ────────────────────────────────────
    
    # --- 平均ST ---
    # 「平均ST」セクション内の「直近6ヶ月」
    st_header = find_first("平均ST", base_start, base_end)
    if st_header >= 0:
        vals = find_label_then_six("直近6ヶ月", st_header, st_header + 12)
        if vals and all(0.05 <= v <= 0.30 for v in vals):
            for i in range(6):
                agents[i].avg_st = vals[i]

    # --- 勝率 ---
    # 「勝率」がbase_start~base_end内にあるはず
    # 「ST順位」と「勝率」の間の「勝率」を探す
    wr_positions = find_all("勝率", base_start, base_end)
    # ST順位の後の勝率を探す
    st_rank_pos = find_first("ST順位", base_start, base_end)
    wr_header = -1
    for pos in wr_positions:
        # 「勝率」が単独行（行頭に「勝率」）かつST順位より後
        line_stripped = lines[pos].strip()
        if line_stripped == "勝率" or line_stripped.startswith("勝率\n") or line_stripped.startswith("勝率"):
            if st_rank_pos < 0 or pos > st_rank_pos:
                wr_header = pos
                break
    
    if wr_header >= 0:
        # 勝率セクションの終わり = 次の大見出し（2連対率）
        wr_end_pos = find_first("2連対率", wr_header + 1, base_end)
        if wr_end_pos < 0: wr_end_pos = base_end
        
        vals = find_label_then_six("直近6ヶ月", wr_header, wr_end_pos)
        if vals and all(1.0 <= v <= 12.0 for v in vals):
            for i in range(6):
                agents[i].national_win_rate = vals[i]
        
        vals = find_label_then_six("当地", wr_header, wr_end_pos)
        if vals:
            for i in range(6):
                agents[i].local_win_rate = vals[i]

    # --- 2連対率 ---
    r2_positions = find_all("2連対率", base_start, base_end)
    r2_header = -1
    for pos in r2_positions:
        stripped = lines[pos].strip()
        if stripped.startswith("2連対率"):
            r2_header = pos
            break
    
    if r2_header >= 0:
        r2_end_pos = find_first("3連対率", r2_header + 1, base_end)
        if r2_end_pos < 0: r2_end_pos = base_end
        
        vals = find_label_then_six("直近6ヶ月", r2_header, r2_end_pos, is_pct=True)
        if vals:
            for i in range(6):
                agents[i].national_top2_rate = vals[i]

    # --- 3連対率 ---
    r3_positions = find_all("3連対率", base_start, base_end)
    r3_header = -1
    for pos in r3_positions:
        stripped = lines[pos].strip()
        if stripped.startswith("3連対率"):
            r3_header = pos
            break
    
    if r3_header >= 0:
        r3_end_pos = find_first("決", r3_header + 1, base_end)  # 決り手数 or 決まり手
        if r3_end_pos < 0: r3_end_pos = base_end
        
        vals = find_label_then_six("直近6ヶ月", r3_header, r3_end_pos, is_pct=True)
        if vals:
            for i in range(6):
                agents[i].national_top3_rate = vals[i]

    # --- 決り手数 / 決まり手数 ---
    kima_header = find_first("決り手数", base_start, base_end)
    if kima_header < 0:
        kima_header = find_first("決まり手数", base_start, base_end)
    if kima_header >= 0:
        kima_end = min(kima_header + 20, base_end)
        
        for kw, attr in [("逃げ", "kimarite_nige"), ("差し", "kimarite_sashi"),
                          ("捲り", "kimarite_makuri"), ("捲差", "kimarite_makuzashi"),
                          ("抜き", "kimarite_nuki"), ("恵まれ", "kimarite_megumre")]:
            vals = find_label_then_six(kw, kima_header, kima_end)
            if vals:
                for i in range(6):
                    setattr(agents[i], attr, safe_int(vals[i]))

    # --- 能力値 ---
    ability_pos = find_first("能力値", base_start, base_end)
    if ability_pos >= 0:
        vals = find_label_then_six("今期", ability_pos, ability_pos + 5)
        if vals and all(20 <= v <= 100 for v in vals):
            for i in range(6):
                agents[i].ability = safe_int(vals[i])

    # --- 事故率 ---
    # 「事故率」行から直接数値を取る（「事故率」の後に6つの小数）
    acc_positions = find_all("事故率", base_start, base_end)
    for pos in acc_positions:
        after = lines[pos].split("事故率")[-1]
        vals = nums_in(after)
        if len(vals) >= 6 and all(0.0 <= v <= 5.0 for v in vals):
            for i in range(6):
                agents[i].accident_rate = vals[i]
            break

    # --- フライング ---
    f_header = find_first("フライング", base_start, base_end)
    if f_header >= 0:
        vals = find_label_then_six("未消化", f_header, f_header + 8)
        if vals:
            for i in range(6):
                agents[i].flying_count = safe_int(vals[i])

    # --- 年齢 ---
    age_pos = find_first("年齢", base_start, base_end)
    if age_pos >= 0:
        after = lines[age_pos].split("年齢")[-1]
        ages = re.findall(r'(\d{2})歳?', after)
        if len(ages) < 6 and age_pos + 1 < N:
            ages = re.findall(r'(\d{2})歳?', after + " " + lines[age_pos + 1])
        if len(ages) >= 6:
            for i in range(6):
                agents[i].age = int(ages[i])

    # --- 支部 ---
    branch_pos = find_first("支部", base_start, base_end)
    if branch_pos >= 0:
        after = lines[branch_pos].split("支部")[-1]
        parts = re.split(r'[\t\s　]+', after.strip())
        branches = [p for p in parts if len(p) >= 2 and re.match(r'^[\u4e00-\u9fff]+$', p)]
        if len(branches) >= 6:
            for i in range(6):
                agents[i].branch = branches[i]

    # ────────────────────────────────────
    # 3. 枠別情報セクション (waku_start ~ waku_end)
    # ────────────────────────────────────
    if waku_idx > 0:
        # 1着率(総合) → 直近6ヶ月
        r1_waku = find_first("1着率", waku_start, waku_end)
        if r1_waku >= 0:
            # 1着率セクション内の直近6ヶ月（%値）
            r1_sub_end = find_first("2連対率", r1_waku + 1, waku_end)
            if r1_sub_end < 0: r1_sub_end = waku_end
            vals = find_label_then_six("直近6ヶ月", r1_waku, r1_sub_end, is_pct=True)
            if vals:
                for i in range(6):
                    agents[i].lane_win_rate = vals[i]

    # ────────────────────────────────────
    # 4. モーター情報セクション (motor_start ~ motor_end)
    # ────────────────────────────────────
    if motor_idx > 0:
        # 貢献P
        cp_pos = find_first("貢献P", motor_start, motor_end)
        if cp_pos >= 0:
            after = lines[cp_pos].split("貢献P")[-1]
            vals = nums_in(after)
            if len(vals) < 6 and cp_pos + 1 < N:
                vals = nums_in(after + " " + lines[cp_pos + 1])
            if len(vals) >= 6:
                for i in range(6):
                    agents[i].motor_contribution = vals[i]
        
        # モーター2連率はデータにないのでデフォルトのまま
        
        # 展示順位（通算の展示順位）
        ex_rank_pos = find_first("展示順位", motor_start, motor_end)
        if ex_rank_pos >= 0:
            after = lines[ex_rank_pos].split("展示順位")[-1]
            vals = nums_in(after)
            if len(vals) < 6 and ex_rank_pos + 1 < N:
                vals = nums_in(after + " " + lines[ex_rank_pos + 1])
            if len(vals) >= 6:
                for i in range(6):
                    agents[i].exhibition_rank = max(1, safe_int(round(vals[i])))

    # ────────────────────────────────────
    # 5. 直前情報セクション (direct_start ~ direct_end)
    # ────────────────────────────────────
    if direct_idx > 0:
        # 展示タイム
        # 「展示」で始まる行で、6.xx の値が6つある行を探す
        # ただし「展示情報」「展示タイム1位」「展示順位」は除外
        exclude_exhibition = ["展示情報", "展示タイム1位", "展示順位", "平均展示", "前走展示"]
        for i in range(direct_start, direct_end):
            line = lines[i].strip()
            if not line.startswith("展示"):
                continue
            if any(ex in line for ex in exclude_exhibition):
                continue
            after = line.split("展示")[-1]
            vals = nums_in(after)
            if len(vals) >= 6 and all(6.0 <= v <= 7.5 for v in vals):
                for j in range(6):
                    agents[j].exhibition_time = vals[j]
                break

        # 体重
        vals = find_label_then_six("体重", direct_start, direct_end)
        if vals and all(40 <= v <= 70 for v in vals):
            for i in range(6):
                agents[i].weight = vals[i]

        # チルト
        vals = find_label_then_six("チルト", direct_start, direct_end)
        if vals:
            for i in range(6):
                agents[i].tilt = vals[i]

        # 周り足
        vals = find_label_then_six("周り足", direct_start, direct_end)
        if vals and all(4.0 <= v <= 8.0 for v in vals):
            for i in range(6):
                agents[i].turn_time = vals[i]

        # 直線
        # 「直線」が複数あるので、直前情報内の最初の「直線」行で値がある行
        for i in range(direct_start, direct_end):
            line = lines[i].strip()
            if line.startswith("直線"):
                after = line.split("直線")[-1]
                vals = nums_in(after)
                if len(vals) >= 6 and all(6.5 <= v <= 9.0 for v in vals):
                    for j in range(6):
                        agents[j].straight_time = vals[j]
                    break

        # 周回
        for i in range(direct_start, direct_end):
            line = lines[i].strip()
            if line.startswith("周回") and "周り足" not in line:
                after = line.split("周回")[-1]
                vals = nums_in(after)
                if len(vals) >= 6 and all(30.0 <= v <= 45.0 for v in vals):
                    for j in range(6):
                        agents[j].lap_time = vals[j]
                    break

    # ────────────────────────────────────
    # 6. 天候
    # ────────────────────────────────────
    # パターン1: "13.0℃ 雨 3m 風向き 12.0℃ 3cm" 形式の行
    for i in range(N):
        line = lines[i]
        if "℃" in line and "cm" in line:
            weather_words = ["晴", "曇", "雨", "雪", "霧"]
            has_weather = any(w in line for w in weather_words)
            if has_weather:
                for w in weather_words:
                    if w in line:
                        conditions.weather = w
                        break
                temps = re.findall(r'(\d+\.?\d*)℃', line)
                if temps:
                    conditions.temperature = safe_float(temps[0])
                if len(temps) >= 2:
                    conditions.water_temperature = safe_float(temps[1])
                wind_m = re.search(r'(\d+)m', line)
                if wind_m:
                    conditions.wind_speed = safe_float(wind_m.group(1))
                wave_m = re.search(r'(\d+)cm', line)
                if wave_m:
                    conditions.wave_height = safe_int(wave_m.group(1))
                break

    # パターン2: ヘッダー行（気温 天気 風速...）の次行にデータがある
    if conditions.temperature == 20.0:
        header_pos = find_first("気温")
        if header_pos >= 0 and header_pos + 1 < N:
            # ヘッダー行に「天気」も含まれているか確認
            if "天気" in lines[header_pos] or "天候" in lines[header_pos]:
                data_line = lines[header_pos + 1]
                temps = re.findall(r'(\d+\.?\d*)℃', data_line)
                if temps:
                    conditions.temperature = safe_float(temps[0])
                if len(temps) >= 2:
                    conditions.water_temperature = safe_float(temps[1])
                for w in ["晴", "曇", "雨", "雪", "霧"]:
                    if w in data_line:
                        conditions.weather = w
                        break
                wind_m = re.search(r'(\d+)m', data_line)
                if wind_m:
                    conditions.wind_speed = safe_float(wind_m.group(1))
                wave_m = re.search(r'(\d+)cm', data_line)
                if wave_m:
                    conditions.wave_height = safe_int(wave_m.group(1))

    return agents, conditions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1行1艇フォーマット パーサー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_manual_format(text):
    agents = []
    conditions = RaceCondition()

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("天候"):
            temp_m = re.search(r'気温(\d+\.?\d*)℃?', line)
            if temp_m: conditions.temperature = safe_float(temp_m.group(1))
            for w in ["雨", "晴", "曇", "雪", "霧"]:
                if w in line:
                    conditions.weather = w
                    break
            wind_m = re.search(r'風速(\d+\.?\d*)m', line)
            if wind_m: conditions.wind_speed = safe_float(wind_m.group(1))
            wt_m = re.search(r'水温(\d+\.?\d*)℃?', line)
            if wt_m: conditions.water_temperature = safe_float(wt_m.group(1))
            wave_m = re.search(r'波高(\d+)', line)
            if wave_m: conditions.wave_height = safe_int(wave_m.group(1))
            continue

        m = re.match(r'(\d)号艇[:：]?\s*(.*)', line)
        if m:
            lane = int(m.group(1))
            rest = m.group(2).strip()
            parts = re.split(r'[\s\t]+', rest)

            if len(parts) >= 2:
                number = safe_int(parts[0])
                name = parts[1]
                rank = parts[2] if len(parts) > 2 and parts[2] in ["A1","A2","B1","B2"] else "B1"
                offset = 3 if rank in ["A1","A2","B1","B2"] else 2

                def gp(idx, default=0.0):
                    return safe_float(parts[idx]) if idx < len(parts) else default
                def gi(idx, default=0):
                    return safe_int(parts[idx]) if idx < len(parts) else default

                agents.append(BoatAgent(
                    lane=lane, number=number, name=name, rank=rank,
                    age=gi(offset, 30), weight=gp(offset+1, 52.0),
                    avg_st=gp(offset+2, 0.15), national_win_rate=gp(offset+3, 5.0),
                    national_top2_rate=gp(offset+4, 30.0), national_top3_rate=gp(offset+5, 45.0),
                    local_win_rate=gp(offset+6, 4.5), lane_win_rate=gp(offset+7, 10.0),
                    ability=gi(offset+8, 50), motor_contribution=gp(offset+9, 0.0),
                    motor_top2_rate=gp(offset+10, 30.0), exhibition_time=gp(offset+11, 6.80),
                    exhibition_rank=gi(offset+12, 3), turn_time=gp(offset+13, 6.70),
                    tilt=gp(offset+14, -0.5), accident_rate=gp(offset+15, 0.0),
                    kimarite_nige=gi(offset+16, 0), kimarite_sashi=gi(offset+17, 0),
                    kimarite_makuri=gi(offset+18, 0), kimarite_makuzashi=gi(offset+19, 0),
                    kimarite_nuki=gi(offset+20, 0),
                ))

    agents.sort(key=lambda a: a.lane)
    return agents, conditions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  自動判定パーサー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_any_text(text):
    if not text or not text.strip():
        return [], RaceCondition()
    if re.search(r'[1-6]号艇[:：]', text):
        return parse_manual_format(text)
    return parse_official_site_text(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  オッズ取得
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_trifecta_odds(venue_code, date_str, race_no):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={race_no}&jcd={venue_code}&hd={date_str}"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.select_one("table.is-w495")
        if not table:
            for t in soup.find_all("table"):
                if t.find("td", class_="oddsPoint"):
                    table = t
                    break

        all_values = []
        if table:
            for row in table.find_all("tr"):
                for c in row.find_all("td", class_="oddsPoint"):
                    txt = c.get_text(strip=True).replace(",", "")
                    all_values.append(safe_float(txt))
        
        if len(all_values) < 120:
            cells = soup.select("td.oddsPoint")
            all_values = [safe_float(c.get_text(strip=True).replace(",", "")) for c in cells]

        if len(all_values) < 120:
            all_text = soup.get_text()
            nums = re.findall(r'\b\d{1,5}\.\d\b', all_text)
            if len(nums) >= 120:
                all_values = [safe_float(n) for n in nums[:120]]

        if len(all_values) < 120:
            return {}

        matrix = np.array(all_values[:120]).reshape(20, 6)
        transposed = matrix.T.flatten()

        odds_dict = {}
        boats = [1, 2, 3, 4, 5, 6]
        idx = 0
        for first in boats:
            others = [b for b in boats if b != first]
            for second in others:
                for third in [b for b in others if b != second]:
                    if idx < len(transposed):
                        val = transposed[idx]
                        if val > 0:
                            odds_dict[f"{first}-{second}-{third}"] = val
                        idx += 1

        return odds_dict
    except Exception as e:
        st.warning(f"オッズ取得エラー: {e}")
        return {}


def parse_pasted_odds(text):
    if not text or not text.strip():
        return {}
    odds = {}

    pattern1 = re.findall(r'(\d)[--](\d)[--](\d)\s+([\d,.]+)', text)
    if pattern1:
        for m in pattern1:
            odds[f"{m[0]}-{m[1]}-{m[2]}"] = safe_float(m[3].replace(",", ""))
        if len(odds) >= 10:
            return odds

    all_nums = re.findall(r'\d+\.?\d*', text.replace(",", ""))
    float_nums = [safe_float(n) for n in all_nums]
    odds_nums = [n for n in float_nums if n >= 1.0]
    if len(odds_nums) >= 120:
        matrix = np.array(odds_nums[:120]).reshape(20, 6)
        transposed = matrix.T.flatten()
        boats = [1, 2, 3, 4, 5, 6]
        idx = 0
        for first in boats:
            others = [b for b in boats if b != first]
            for second in others:
                for third in [b for b in others if b != second]:
                    if idx < len(transposed):
                        val = transposed[idx]
                        if val > 0:
                            odds[f"{first}-{second}-{third}"] = val
                        idx += 1
        return odds

    return odds


def compute_synthetic_odds(trifecta_odds):
    if not trifecta_odds:
        return {}

    trio_odds = {}
    exacta_odds = {}
    quinella_odds = {}
    wide_odds = {}

    trio_groups = {}
    for key, val in trifecta_odds.items():
        parts = key.split("-")
        if len(parts) == 3:
            sorted_key = "-".join(sorted(parts))
            trio_groups.setdefault(sorted_key, []).append(val)
    for key, vals in trio_groups.items():
        inv_sum = sum(1/v for v in vals if v > 0)
        if inv_sum > 0: trio_odds[key] = round(1 / inv_sum, 1)

    exacta_groups = {}
    for key, val in trifecta_odds.items():
        parts = key.split("-")
        if len(parts) == 3:
            ex_key = f"{parts[0]}-{parts[1]}"
            exacta_groups.setdefault(ex_key, []).append(val)
    for key, vals in exacta_groups.items():
        inv_sum = sum(1/v for v in vals if v > 0)
        if inv_sum > 0: exacta_odds[key] = round(1 / inv_sum, 1)

    quinella_groups = {}
    for key, val in trifecta_odds.items():
        parts = key.split("-")
        if len(parts) == 3:
            q_key = "-".join(sorted(parts[:2]))
            quinella_groups.setdefault(q_key, []).append(val)
    for key, vals in quinella_groups.items():
        inv_sum = sum(1/v for v in vals if v > 0)
        if inv_sum > 0: quinella_odds[key] = round(1 / inv_sum, 1)

    wide_groups = {}
    for key, val in trifecta_odds.items():
        parts = key.split("-")
        if len(parts) == 3:
            for combo in [(parts[0], parts[1]), (parts[0], parts[2]), (parts[1], parts[2])]:
                w_key = "-".join(sorted(combo))
                wide_groups.setdefault(w_key, []).append(val)
    for key, vals in wide_groups.items():
        inv_sum = sum(1/v for v in vals if v > 0)
        if inv_sum > 0: wide_odds[key] = round(1 / inv_sum, 1)

    return {"trio": trio_odds, "exacta": exacta_odds,
            "quinella": quinella_odds, "wide": wide_odds}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  モンテカルロ EV計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_ev_simulation(agents, conditions, venue_profile, month, n_sim):
    sim = RaceSimulator(agents, conditions, venue_profile, month)
    trifecta_counts = {}
    trio_counts = {}
    exacta_counts = {}
    quinella_counts = {}
    wide_counts = {}

    progress = st.progress(0)
    for i in range(n_sim):
        if i % 200 == 0:
            progress.progress(i / n_sim)
        result = sim.simulate_race()
        order = result["finish_order"]

        tri_key = f"{order[0]}-{order[1]}-{order[2]}"
        trifecta_counts[tri_key] = trifecta_counts.get(tri_key, 0) + 1

        trio_key = "-".join(str(x) for x in sorted(order[:3]))
        trio_counts[trio_key] = trio_counts.get(trio_key, 0) + 1

        ex_key = f"{order[0]}-{order[1]}"
        exacta_counts[ex_key] = exacta_counts.get(ex_key, 0) + 1

        q_key = "-".join(str(x) for x in sorted(order[:2]))
        quinella_counts[q_key] = quinella_counts.get(q_key, 0) + 1

        top3 = order[:3]
        for ci in range(len(top3)):
            for cj in range(ci+1, len(top3)):
                w_key = "-".join(str(x) for x in sorted([top3[ci], top3[cj]]))
                wide_counts[w_key] = wide_counts.get(w_key, 0) + 1

    progress.progress(1.0)
    return {"trifecta": trifecta_counts, "trio": trio_counts,
            "exacta": exacta_counts, "quinella": quinella_counts,
            "wide": wide_counts}


def compute_expected_values(mc_results, all_odds, n_sim):
    ev_results = {}
    for bet_type, counts in mc_results.items():
        odds_dict = all_odds.get(bet_type, {})
        if not odds_dict: continue
        ev_list = []
        for combo, count in counts.items():
            if combo in odds_dict:
                prob = count / n_sim
                odds_val = odds_dict[combo]
                ev = prob * odds_val
                flag = "◎" if ev >= 1.5 else "○" if ev >= 1.0 else "△" if ev >= 0.8 else "×"
                ev_list.append({
                    "combination": combo, "probability": round(prob, 4),
                    "odds": odds_val, "ev": round(ev, 3),
                    "flag": flag, "count": count
                })
        ev_results[bet_type] = sorted(ev_list, key=lambda x: -x["ev"])
    return ev_results

# ============================================================
#  ボートレース AI シミュレーター v6.0  ─  app.py
#  Part 3/4: Streamlit UI（タブ1〜3）
# ============================================================

# ── ページ設定 ──
st.set_page_config(page_title="ボートレース AI v6.0", layout="wide")
setup_japanese_font()
st.title("🚤 ボートレース AI シミュレーター v6.0")
st.caption("30項目エージェント × 会場別特性(全24場) × 季節補正 × モンテカルロ × 合成オッズ × 期待値計算")

BOAT_COLORS = {1:"#e74c3c", 2:"#000000", 3:"#2ecc71",
               4:"#3498db", 5:"#f1c40f", 6:"#9b59b6"}

VENUE_CODE_MAP = {
    "桐生":"01","戸田":"02","江戸川":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24"
}

# ── サイドバー ──
st.sidebar.header("⚙️ レース設定")
venue_name = st.sidebar.selectbox(
    "会場", list(VENUE_PROFILES.keys()),
    index=list(VENUE_PROFILES.keys()).index("徳山"))
venue_profile = get_venue_profile(venue_name)
venue_code = VENUE_CODE_MAP.get(venue_name, "18")
race_date = st.sidebar.date_input("日付", value=pd.Timestamp("2026-02-27"))
race_no = st.sidebar.number_input("レース番号", 1, 12, 3)

st.sidebar.subheader(f"📍 {venue_name}の特徴")
st.sidebar.write(f"水面: {venue_profile.get('water','—')}　潮: {venue_profile.get('tide','—')}")
st.sidebar.write(f"風影響: {venue_profile.get('wind_effect','—')}")
st.sidebar.write(f"メモ: {venue_profile.get('memo','—')}")

fig_sb, ax_sb = plt.subplots(figsize=(4, 2.2))
bars_sb = ax_sb.bar(range(1, 7), venue_profile["course_win_rate"],
                    color=[BOAT_COLORS[i] for i in range(1, 7)])
ax_sb.set_xlabel("コース"); ax_sb.set_ylabel("1着率(%)")
ax_sb.set_title(f"{venue_name} コース別1着率"); ax_sb.set_xticks(range(1, 7))
for b in bars_sb:
    ax_sb.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
               f"{b.get_height():.1f}", ha="center", fontsize=7)
plt.tight_layout(); st.sidebar.pyplot(fig_sb); plt.close(fig_sb)

# ── メインタブ ──
tab_input, tab_sim, tab_mc, tab_odds, tab_ev = st.tabs(
    ["📝 データ入力", "🏁 単発シミュレーション",
     "📊 モンテカルロ", "💰 オッズ取得", "📈 期待値計算"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 1 ─ データ入力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_input:
    st.subheader("📝 出走データ入力")
    input_method = st.radio("入力方式",
                            ["テキスト貼り付け（公式サイト / 1行形式）", "フォーム入力"],
                            horizontal=True)

    if input_method == "テキスト貼り付け（公式サイト / 1行形式）":
        st.info("💡 公式サイトのデータをそのままコピー＆貼り付けできます。"
                "または「1号艇: 4753 森照夫 B1 ...」形式でも入力できます。")
        sample_text = """1号艇: 4753 森照夫 B1 37 53.1 0.15 4.92 34.5 44.8 3.95 50.0 48 1.13 30.0 6.78 3 5.58 -0.5 0.34 7 1 4 0 0
2号艇: 5090 生方靖亜 B1 25 53.0 0.16 5.41 33.0 53.0 5.00 18.8 49 -0.16 28.0 6.84 4 5.67 -0.5 0.15 11 1 8 1 1
3号艇: 4226 村田浩司 B1 43 53.1 0.16 4.77 29.9 42.3 4.00 15.0 48 0.78 26.0 6.80 2 5.46 -0.5 0.00 6 0 5 3 1
4号艇: 5121 定松勇樹 A1 24 52.0 0.14 7.61 62.3 74.7 6.77 22.2 64 0.26 38.0 6.79 1 5.17 -0.5 0.17 25 5 10 4 3
5号艇: 4486 野村誠 B1 39 52.2 0.19 5.09 37.9 48.3 5.09 5.3 47 -0.22 24.0 6.85 5 5.50 -0.5 0.69 5 5 2 1 0
6号艇: 4262 馬場貴也 A1 41 52.6 0.14 7.26 48.5 62.5 6.29 4.8 70 0.54 36.0 6.83 6 5.53 -0.5 0.00 23 4 2 7 3
天候: 気温13.0℃ 雨 風速3m 水温12.0℃ 波高3cm"""
        raw_text = st.text_area("出走データを貼り付け", value=sample_text, height=350,
                                help="公式サイトからのコピー or 1行1艇フォーマット")
    else:
        raw_text = None
        form_agents = []
        for i in range(1, 7):
            with st.expander(f"🚤 {i}号艇", expanded=(i <= 2)):
                c1 = st.columns(5)
                reg  = c1[0].number_input(f"登録番号#{i}", value=0, key=f"reg_{i}")
                name = c1[1].text_input(f"名前#{i}", value=f"選手{i}", key=f"name_{i}")
                rank = c1[2].selectbox(f"級#{i}", ["A1","A2","B1","B2"], index=2, key=f"rank_{i}")
                age  = c1[3].number_input(f"年齢#{i}", value=30, key=f"age_{i}")
                wt   = c1[4].number_input(f"体重#{i}", value=52.0, step=0.5, key=f"wt_{i}")

                c2 = st.columns(5)
                avg_st   = c2[0].number_input(f"平均ST#{i}", value=0.15, format="%.2f", key=f"st_{i}")
                win_r    = c2[1].number_input(f"勝率#{i}", value=5.00, format="%.2f", key=f"wr_{i}")
                top2     = c2[2].number_input(f"2連対率#{i}", value=30.0, format="%.1f", key=f"t2_{i}")
                top3     = c2[3].number_input(f"3連対率#{i}", value=45.0, format="%.1f", key=f"t3_{i}")
                local_wr = c2[4].number_input(f"当地勝率#{i}", value=4.50, format="%.2f", key=f"lw_{i}")

                c3 = st.columns(5)
                lane1 = c3[0].number_input(f"枠別1着率#{i}", value=10.0, format="%.1f", key=f"l1_{i}")
                abil  = c3[1].number_input(f"能力#{i}", value=50, key=f"ab_{i}")
                mp    = c3[2].number_input(f"モーターP#{i}", value=0.0, format="%.2f", key=f"mp_{i}")
                m2r   = c3[3].number_input(f"モ2連率#{i}", value=30.0, format="%.1f", key=f"m2_{i}")
                ext   = c3[4].number_input(f"展示T#{i}", value=6.80, format="%.2f", key=f"ex_{i}")

                c4 = st.columns(5)
                exr  = c4[0].number_input(f"展示順#{i}", value=i, key=f"exr_{i}")
                turn = c4[1].number_input(f"周り足#{i}", value=5.50, format="%.2f", key=f"tu_{i}")
                tilt = c4[2].number_input(f"チルト#{i}", value=-0.5, format="%.1f", key=f"ti_{i}")
                acc  = c4[3].number_input(f"事故率#{i}", value=0.0, format="%.2f", key=f"ac_{i}")
                f_cnt = c4[4].number_input(f"F数#{i}", value=0, key=f"fc_{i}")

                c5 = st.columns(6)
                k_nige   = c5[0].number_input(f"逃げ#{i}", value=0, key=f"kn_{i}")
                k_sashi  = c5[1].number_input(f"差し#{i}", value=0, key=f"ks_{i}")
                k_makuri = c5[2].number_input(f"捲り#{i}", value=0, key=f"km_{i}")
                k_makusa = c5[3].number_input(f"捲差#{i}", value=0, key=f"kms_{i}")
                k_nuki   = c5[4].number_input(f"抜き#{i}", value=0, key=f"knk_{i}")
                k_megum  = c5[5].number_input(f"恵まれ#{i}", value=0, key=f"kme_{i}")

                form_agents.append(BoatAgent(
                    lane=i, number=reg, name=name, rank=rank, age=age,
                    weight=wt, avg_st=avg_st, national_win_rate=win_r,
                    national_top2_rate=top2, national_top3_rate=top3,
                    local_win_rate=local_wr, lane_win_rate=lane1,
                    ability=abil, motor_contribution=mp, motor_top2_rate=m2r,
                    exhibition_time=ext, exhibition_rank=exr,
                    turn_time=turn, tilt=tilt, accident_rate=acc,
                    flying_count=f_cnt,
                    kimarite_nige=k_nige, kimarite_sashi=k_sashi,
                    kimarite_makuri=k_makuri, kimarite_makuzashi=k_makusa,
                    kimarite_nuki=k_nuki, kimarite_megumre=k_megum
                ))

        st.markdown("---")
        st.subheader("🌤️ 天候条件")
        wc = st.columns(5)
        w_weather = wc[0].selectbox("天候", ["晴","曇","雨","雪","霧"], index=2)
        w_temp    = wc[1].number_input("気温(℃)", value=13.0, format="%.1f")
        w_wind    = wc[2].number_input("風速(m/s)", value=3.0, format="%.1f")
        w_wtemp   = wc[3].number_input("水温(℃)", value=12.0, format="%.1f")
        w_wave    = wc[4].number_input("波高(cm)", value=3, min_value=0)

    # ── データ確定 ──
    if st.button("✅ データ確定", type="primary", use_container_width=True):
        if input_method == "テキスト貼り付け（公式サイト / 1行形式）":
            agents, conditions = parse_any_text(raw_text)
        else:
            agents = form_agents
            conditions = RaceCondition(
                weather=w_weather, temperature=w_temp,
                wind_speed=w_wind, water_temperature=w_wtemp,
                wave_height=w_wave)

        if agents and len(agents) == 6:
            st.session_state["agents"] = agents
            st.session_state["conditions"] = conditions
            st.success(f"✅ {len(agents)}艇のデータを確定しました")
        else:
            st.error(f"❌ 6艇のデータが必要です（検出: {len(agents) if agents else 0}艇）")

    # ── 確定データ表示 ──
    if "agents" in st.session_state:
        agents_disp = st.session_state["agents"]
        cond_disp = st.session_state["conditions"]
        st.markdown("---")
        st.subheader("📋 確定データ")

        # 基本情報テーブル
        st.markdown("**基本情報・成績**")
        df_basic = pd.DataFrame([{
            "枠": a.lane, "登番": a.number, "名前": a.name, "級": a.rank,
            "年齢": a.age, "体重": a.weight, "平均ST": f"{a.avg_st:.2f}",
            "勝率": f"{a.national_win_rate:.2f}",
            "2連対": f"{a.national_top2_rate:.1f}%",
            "3連対": f"{a.national_top3_rate:.1f}%",
            "当地勝率": f"{a.local_win_rate:.2f}",
            "枠別1着": f"{a.lane_win_rate:.1f}%",
            "能力": a.ability
        } for a in agents_disp])
        st.dataframe(df_basic, use_container_width=True, hide_index=True)

        # 機力・展示
        st.markdown("**機力・展示・直前情報**")
        df_machine = pd.DataFrame([{
            "枠": a.lane, "名前": a.name,
            "モーターP": f"{a.motor_contribution:+.2f}",
            "モ2連率": f"{a.motor_top2_rate:.1f}%",
            "展示T": f"{a.exhibition_time:.2f}",
            "展示順": a.exhibition_rank,
            "周り足": f"{a.turn_time:.2f}",
            "直線T": f"{a.straight_time:.2f}",
            "周回T": f"{a.lap_time:.2f}",
            "チルト": a.tilt,
            "事故率": f"{a.accident_rate:.2f}",
            "F数": a.flying_count
        } for a in agents_disp])
        st.dataframe(df_machine, use_container_width=True, hide_index=True)

        # 決まり手
        st.markdown("**決まり手傾向**")
        df_kima = pd.DataFrame([{
            "枠": a.lane, "名前": a.name,
            "逃げ": a.kimarite_nige, "差し": a.kimarite_sashi,
            "捲り": a.kimarite_makuri, "捲差": a.kimarite_makuzashi,
            "抜き": a.kimarite_nuki, "恵まれ": a.kimarite_megumre
        } for a in agents_disp])
        st.dataframe(df_kima, use_container_width=True, hide_index=True)

        # 天候
        st.info(f"🌤️ {cond_disp.weather}　気温{cond_disp.temperature}℃　"
                f"風速{cond_disp.wind_speed}m/s　水温{cond_disp.water_temperature}℃　"
                f"波高{cond_disp.wave_height}cm")

        # デバッグ
        with st.expander("🔍 デバッグ: パース結果チェック"):
            check_fields = {
                "name": ("名前", "選手", lambda a, d: a.name == d),
                "number": ("登録番号", 0, lambda a, d: a.number == d),
                "avg_st": ("平均ST", 0.15, lambda a, d: abs(a.avg_st - d) < 0.001),
                "national_win_rate": ("勝率", 5.0, lambda a, d: abs(a.national_win_rate - d) < 0.01),
                "national_top2_rate": ("2連対率", 30.0, lambda a, d: abs(a.national_top2_rate - d) < 0.1),
                "national_top3_rate": ("3連対率", 45.0, lambda a, d: abs(a.national_top3_rate - d) < 0.1),
                "local_win_rate": ("当地勝率", 4.5, lambda a, d: abs(a.local_win_rate - d) < 0.01),
                "lane_win_rate": ("枠別1着率", 10.0, lambda a, d: abs(a.lane_win_rate - d) < 0.1),
                "ability": ("能力", 50, lambda a, d: a.ability == d),
                "motor_contribution": ("モーターP", 0.0, lambda a, d: abs(a.motor_contribution - d) < 0.001),
                "exhibition_time": ("展示T", 6.80, lambda a, d: abs(a.exhibition_time - d) < 0.001),
                "turn_time": ("周り足", 6.70, lambda a, d: abs(a.turn_time - d) < 0.001),
                "weight": ("体重", 52.0, lambda a, d: abs(a.weight - d) < 0.01),
            }
            warn_count = 0
            for field_key, (label, default_val, check_fn) in check_fields.items():
                for a in agents_disp:
                    if check_fn(a, default_val):
                        st.warning(f"⚠️ {a.lane}号艇 {a.name}: {label} = {getattr(a, field_key)}（デフォルト値の可能性）")
                        warn_count += 1
            if warn_count == 0:
                st.success("✅ すべてのフィールドが正常にパースされています")
            else:
                st.info(f"⚠️ {warn_count}件のデフォルト値警告があります。公式サイト貼り付けの場合、一部項目はデータが無い場合があります。")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 2 ─ 単発シミュレーション
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_sim:
    st.subheader("🏁 単発シミュレーション")
    if "agents" not in st.session_state:
        st.warning("⬅️ 先に「データ入力」タブでデータを確定してください")
    else:
        n_trials = st.slider("試行回数", 1, 10, 3, key="sim_trials")
        if st.button("🏁 シミュレーション実行", type="primary", key="run_sim"):
            agents = st.session_state["agents"]
            conditions = st.session_state["conditions"]
            month = race_date.month if race_date else 2
            sim = RaceSimulator(agents, conditions, venue_profile, month)

            for trial in range(n_trials):
                st.markdown(f"### 🏁 試行 {trial + 1}")
                result = sim.simulate_race()
                order = result["finish_order"]
                kimarite = result["kimarite"]
                st_times = result["start_times"]

                # 着順表示
                cols_finish = st.columns(6)
                for pos, lane in enumerate(order):
                    a = agents[lane - 1]
                    cols_finish[pos].markdown(
                        f"<div style='text-align:center; padding:10px; "
                        f"background:{BOAT_COLORS[lane]}; color:white; "
                        f"border-radius:10px; margin:2px;'>"
                        f"<b>{pos+1}着</b><br>"
                        f"<span style='font-size:1.3em;'>{lane}号艇</span><br>"
                        f"{a.name}</div>",
                        unsafe_allow_html=True)

                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**決まり手:** {kimarite}")
                    tri = f"{order[0]}-{order[1]}-{order[2]}"
                    trio_s = sorted(order[:3])
                    trio_str = f"{trio_s[0]}-{trio_s[1]}-{trio_s[2]}"
                    exacta = f"{order[0]}-{order[1]}"
                    st.write(f"**3連単:** {tri}　**3連複:** {trio_str}　**2連単:** {exacta}")

                with col_info2:
                    st_df = pd.DataFrame([{
                        "枠": i+1, "名前": agents[i].name,
                        "ST": f"{st_times[i]:.2f}"
                    } for i in range(6)])
                    st.dataframe(st_df, use_container_width=True, hide_index=True)

                # レース展開グラフ
                positions = result["positions"]
                fig_race, ax_race = plt.subplots(figsize=(10, 4))
                for lane in range(1, 7):
                    y = positions[lane]
                    x = list(range(len(y)))
                    ax_race.plot(x, y, color=BOAT_COLORS[lane], linewidth=2.5,
                                 label=f"{lane}号 {agents[lane-1].name}", marker="o", markersize=4)
                ax_race.set_xlabel("区間")
                ax_race.set_ylabel("順位")
                ax_race.set_yticks(range(1, 7))
                ax_race.invert_yaxis()
                ax_race.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
                ax_race.set_title(f"試行{trial+1} レース展開")
                ax_race.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_race)
                plt.close(fig_race)

                # 重み詳細
                with st.expander(f"📊 重み詳細（試行{trial+1}）"):
                    weights = sim._compute_race_weights()
                    w_df = pd.DataFrame([{
                        "枠": i+1, "名前": agents[i].name,
                        "重み": f"{weights[i]:.4f}",
                        "勝率予測": f"{weights[i]*100:.1f}%"
                    } for i in range(6)])
                    st.dataframe(w_df, use_container_width=True, hide_index=True)

                    # 重みバーチャート
                    fig_w, ax_w = plt.subplots(figsize=(6, 3))
                    bars_w = ax_w.bar(range(1, 7), weights * 100,
                                      color=[BOAT_COLORS[i] for i in range(1, 7)])
                    ax_w.set_xlabel("枠"); ax_w.set_ylabel("予測勝率(%)")
                    ax_w.set_xticks(range(1, 7))
                    for b in bars_w:
                        ax_w.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                                  f"{b.get_height():.1f}%", ha="center", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_w)
                    plt.close(fig_w)

                st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 3 ─ モンテカルロ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_mc:
    st.subheader("📊 モンテカルロシミュレーション")
    if "agents" not in st.session_state:
        st.warning("⬅️ 先に「データ入力」タブでデータを確定してください")
    else:
        n_mc = st.slider("シミュレーション回数", 1000, 50000, 10000, step=1000, key="mc_n")
        if st.button("📊 モンテカルロ実行", type="primary", key="run_mc"):
            agents = st.session_state["agents"]
            conditions = st.session_state["conditions"]
            month = race_date.month if race_date else 2
            sim = RaceSimulator(agents, conditions, venue_profile, month)

            win_counts = np.zeros(6)
            top2_counts = np.zeros(6)
            top3_counts = np.zeros(6)
            kima_counts = {}
            trifecta_counts = {}

            progress = st.progress(0)
            status = st.empty()
            for i in range(n_mc):
                if i % 200 == 0:
                    progress.progress(i / n_mc)
                    status.text(f"シミュレーション中... {i:,}/{n_mc:,}")
                result = sim.simulate_race()
                order = result["finish_order"]
                kima = result["kimarite"]

                win_counts[order[0]-1] += 1
                top2_counts[order[0]-1] += 1
                top2_counts[order[1]-1] += 1
                top3_counts[order[0]-1] += 1
                top3_counts[order[1]-1] += 1
                top3_counts[order[2]-1] += 1

                kima_counts[kima] = kima_counts.get(kima, 0) + 1
                tri_key = f"{order[0]}-{order[1]}-{order[2]}"
                trifecta_counts[tri_key] = trifecta_counts.get(tri_key, 0) + 1

            progress.progress(1.0)
            status.text(f"✅ {n_mc:,}回完了")
            st.session_state["mc_trifecta_counts"] = trifecta_counts
            st.session_state["mc_n"] = n_mc

            # ── 結果表示 ──
            st.markdown("### 🏆 各艇の成績")
            rate_df = pd.DataFrame([{
                "枠": i+1, "名前": agents[i].name,
                "1着率": f"{win_counts[i]/n_mc*100:.1f}%",
                "2連対率": f"{top2_counts[i]/n_mc*100:.1f}%",
                "3連対率": f"{top3_counts[i]/n_mc*100:.1f}%",
                "1着回数": int(win_counts[i])
            } for i in range(6)])
            st.dataframe(rate_df, use_container_width=True, hide_index=True)

            # 1着率バーチャート
            fig_mc1, ax_mc1 = plt.subplots(figsize=(8, 4))
            bars_mc = ax_mc1.bar(range(1, 7), win_counts/n_mc*100,
                                 color=[BOAT_COLORS[i] for i in range(1, 7)])
            ax_mc1.set_xlabel("枠番"); ax_mc1.set_ylabel("1着率(%)")
            ax_mc1.set_title(f"モンテカルロ 1着率（{n_mc:,}回）")
            ax_mc1.set_xticks(range(1, 7))
            for b in bars_mc:
                ax_mc1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                            f"{b.get_height():.1f}%", ha="center", fontsize=9)
            plt.tight_layout(); st.pyplot(fig_mc1); plt.close(fig_mc1)

            # 決まり手
            col_mc1, col_mc2 = st.columns(2)
            with col_mc1:
                st.markdown("### 🥊 決まり手分布")
                kima_df = pd.DataFrame([
                    {"決まり手": k, "回数": v, "割合": f"{v/n_mc*100:.1f}%"}
                    for k, v in sorted(kima_counts.items(), key=lambda x: -x[1])
                ])
                st.dataframe(kima_df, use_container_width=True, hide_index=True)

            with col_mc2:
                # 決まり手パイチャート
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                labels = [k for k, v in sorted(kima_counts.items(), key=lambda x: -x[1])]
                sizes = [v for k, v in sorted(kima_counts.items(), key=lambda x: -x[1])]
                ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax_pie.set_title("決まり手分布")
                plt.tight_layout(); st.pyplot(fig_pie); plt.close(fig_pie)

            # 3連単 Top20
            st.markdown("### 🎯 3連単 出現頻度 Top20")
            sorted_tri = sorted(trifecta_counts.items(), key=lambda x: -x[1])[:20]
            tri_df = pd.DataFrame([
                {"順位": idx+1, "3連単": k, "回数": v,
                 "確率": f"{v/n_mc*100:.2f}%"}
                for idx, (k, v) in enumerate(sorted_tri)
            ])
            st.dataframe(tri_df, use_container_width=True, hide_index=True)

            # 3連単頻度バーチャート
            fig_tri, ax_tri = plt.subplots(figsize=(10, 5))
            top10 = sorted_tri[:10]
            ax_tri.barh([k for k, v in top10][::-1],
                        [v/n_mc*100 for k, v in top10][::-1],
                        color="#3498db")
            ax_tri.set_xlabel("出現率(%)")
            ax_tri.set_title("3連単 出現率 Top10")
            plt.tight_layout(); st.pyplot(fig_tri); plt.close(fig_tri)
# ============================================================
#  ボートレース AI シミュレーター v6.0  ─  app.py
#  Part 4/4: タブ4（オッズ）・タブ5（期待値）・フッター
# ============================================================

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 4 ─ オッズ取得
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_odds:
    st.subheader("💰 オッズ取得")
    odds_method = st.radio(
        "取得方法",
        ["自動取得（公式サイト）", "テキスト貼り付け", "手動入力"],
        horizontal=True, key="odds_method")

    date_str = race_date.strftime("%Y%m%d") if race_date else "20260227"

    if odds_method == "自動取得（公式サイト）":
        st.info(f"🌐 {venue_name} {date_str} {race_no}R の3連単オッズを取得します")
        st.caption(f"URL: https://www.boatrace.jp/owpc/pc/race/odds3t?rno={race_no}&jcd={venue_code}&hd={date_str}")
        if st.button("🔄 オッズ自動取得", key="fetch_odds"):
            with st.spinner("公式サイトからオッズを取得中…"):
                odds = fetch_trifecta_odds(venue_code, date_str, race_no)
            if odds and len(odds) > 0:
                st.success(f"✅ {len(odds)}件の3連単オッズを取得しました")
                st.session_state["trifecta_odds"] = odds
            else:
                st.error("❌ オッズの自動取得に失敗しました。\n"
                         "レースが締め切り前、またはサイト構造の変更の可能性があります。\n"
                         "「テキスト貼り付け」をお試しください。")

    elif odds_method == "テキスト貼り付け":
        st.info("💡 公式サイトのオッズ表をそのままコピー＆貼り付け、\n"
                "または「1-2-3 27.7」形式で入力できます。")
        odds_sample = """1-2-3 27.7
1-3-2 27.1
1-4-2 45.7
1-5-2 14.0
4-1-2 539.5
4-1-3 42.5"""
        odds_text = st.text_area("オッズテキストを貼り付け",
                                  value="", height=300,
                                  placeholder=odds_sample,
                                  help="「1-2-3 27.7」形式 or 公式テーブルコピー",
                                  key="odds_paste")
        if st.button("📋 オッズ解析", key="parse_odds"):
            if odds_text.strip():
                odds = parse_pasted_odds(odds_text)
                if odds and len(odds) > 0:
                    st.success(f"✅ {len(odds)}件のオッズを解析しました")
                    st.session_state["trifecta_odds"] = odds
                else:
                    st.error("❌ オッズの解析に失敗しました。形式を確認してください。\n"
                             "対応形式: 「1-2-3 27.7」（1行1組）")
            else:
                st.warning("テキストを入力してください")

    else:  # 手動入力
        st.info("💡 1行1組で「1-2-3 27.7」形式で入力してください")
        odds_manual = st.text_area(
            "オッズを入力",
            height=250,
            placeholder="1-2-3 27.7\n1-3-2 27.1\n1-4-2 45.7\n...",
            key="odds_manual")
        if st.button("✏️ オッズ登録", key="register_odds"):
            if odds_manual.strip():
                odds = {}
                error_lines = []
                for line_no, line in enumerate(odds_manual.strip().split("\n"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r'(\d)[--](\d)[--](\d)\s+([\d,.]+)', line)
                    if m:
                        key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        odds[key] = float(m.group(4).replace(",", ""))
                    else:
                        error_lines.append(f"行{line_no}: {line}")
                if odds:
                    st.success(f"✅ {len(odds)}件登録しました")
                    st.session_state["trifecta_odds"] = odds
                    if error_lines:
                        with st.expander(f"⚠️ 解析できなかった行（{len(error_lines)}件）"):
                            for el in error_lines:
                                st.text(el)
                else:
                    st.error("❌ 有効なオッズ行がありません")
            else:
                st.warning("テキストを入力してください")

    # ── オッズ表示 ──
    if "trifecta_odds" in st.session_state:
        odds = st.session_state["trifecta_odds"]
        st.markdown("---")

        # 概要
        sorted_odds = sorted(odds.items(), key=lambda x: x[1])
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        col_o1.metric("取得件数", f"{len(odds)}件")
        col_o2.metric("最低オッズ", f"{sorted_odds[0][1]:.1f}" if sorted_odds else "—")
        col_o3.metric("最高オッズ", f"{sorted_odds[-1][1]:.1f}" if sorted_odds else "—")
        median_val = sorted_odds[len(sorted_odds)//2][1] if sorted_odds else 0
        col_o4.metric("中央値", f"{median_val:.1f}")

        # 低オッズ Top30
        st.markdown("### 📊 3連単オッズ（低い順 Top30）")
        odds_df = pd.DataFrame([
            {"順位": idx+1, "3連単": k, "オッズ": f"{v:.1f}"}
            for idx, (k, v) in enumerate(sorted_odds[:30])
        ])
        st.dataframe(odds_df, use_container_width=True, hide_index=True)

        # 1着別オッズ分布
        with st.expander("📈 1着別オッズ分布"):
            for first_boat in range(1, 7):
                filtered = {k: v for k, v in odds.items() if k.startswith(f"{first_boat}-")}
                if filtered:
                    sorted_f = sorted(filtered.items(), key=lambda x: x[1])[:5]
                    items_str = "　".join([f"{k}({v:.1f})" for k, v in sorted_f])
                    st.write(f"**{first_boat}号艇1着:** {items_str} …")

        # ── 合成オッズ ──
        st.markdown("### 🔧 合成オッズ")
        synthetic = compute_synthetic_odds(odds)
        st.session_state["synthetic_odds"] = synthetic

        label_map = {"trio": "3連複", "exacta": "2連単",
                     "quinella": "2連複", "wide": "ワイド"}

        syn_tabs = st.tabs([f"{label_map[k]}（{len(v)}件）"
                            for k, v in synthetic.items()])

        for syn_tab, (bet_type, bet_odds) in zip(syn_tabs, synthetic.items()):
            with syn_tab:
                if not bet_odds:
                    st.info("該当データなし")
                    continue
                sorted_bo = sorted(bet_odds.items(), key=lambda x: x[1])[:25]
                bo_df = pd.DataFrame([
                    {"順位": idx+1, "組合せ": k, "オッズ": f"{v:.1f}"}
                    for idx, (k, v) in enumerate(sorted_bo)
                ])
                st.dataframe(bo_df, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 5 ─ 期待値計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ev:
    st.subheader("📈 期待値計算")

    has_agents = "agents" in st.session_state
    has_odds = "trifecta_odds" in st.session_state

    if not has_agents:
        st.warning("⬅️ 先に「データ入力」タブでデータを確定してください")
    if not has_odds:
        st.warning("⬅️ 先に「オッズ取得」タブでオッズを取得してください")

    if has_agents and has_odds:
        n_ev = st.slider("シミュレーション回数", 1000, 50000, 10000, step=1000, key="ev_n")

        st.caption("モンテカルロシミュレーションで各買い目の出現確率を計算し、"
                   "オッズと掛け合わせて期待値(EV)を算出します。EV ≥ 1.0 が理論上プラスの買い目です。")

        if st.button("📈 期待値計算 実行", type="primary", key="run_ev"):
            agents = st.session_state["agents"]
            conditions = st.session_state["conditions"]
            odds = st.session_state["trifecta_odds"]
            synthetic = st.session_state.get("synthetic_odds", {})
            month = race_date.month if race_date else 2

            all_odds = {"trifecta": odds}
            all_odds.update(synthetic)

            # モンテカルロ実行
            with st.spinner("モンテカルロシミュレーション実行中…"):
                mc_results = run_ev_simulation(
                    agents, conditions, venue_profile, month, n_ev)

            # 期待値計算
            ev_results = compute_expected_values(mc_results, all_odds, n_ev)

            if not ev_results:
                st.error("期待値の計算結果がありません。オッズデータを確認してください。")
            else:
                # 券種別タブ
                label_map_ev = {"trifecta": "3連単", "trio": "3連複",
                                "exacta": "2連単", "quinella": "2連複", "wide": "ワイド"}

                available_types = [k for k in ev_results if ev_results[k]]
                if not available_types:
                    st.info("オッズとシミュレーション結果が一致する買い目がありません")
                else:
                    ev_tabs = st.tabs([
                        f"{label_map_ev.get(k, k)}（{len(ev_results[k])}件）"
                        for k in available_types])

                    all_recommended = []

                    for ev_tab, bet_type in zip(ev_tabs, available_types):
                        with ev_tab:
                            ev_data = ev_results[bet_type]
                            top20 = ev_data[:20]

                            # テーブル
                            ev_df = pd.DataFrame([{
                                "組合せ": d["combination"],
                                "確率": f"{d['probability']*100:.2f}%",
                                "オッズ": f"{d['odds']:.1f}",
                                "期待値": f"{d['ev']:.3f}",
                                "評価": d["flag"],
                                "出現回数": d["count"]
                            } for d in top20])
                            st.dataframe(ev_df, use_container_width=True, hide_index=True)

                            # EVバーチャート
                            top15 = top20[:15]
                            if top15:
                                fig_ev, ax_ev = plt.subplots(figsize=(10, 5))
                                colors = []
                                for d in top15:
                                    if d["ev"] >= 1.5:
                                        colors.append("#e74c3c")  # 赤（超推奨）
                                    elif d["ev"] >= 1.0:
                                        colors.append("#2ecc71")  # 緑（推奨）
                                    elif d["ev"] >= 0.8:
                                        colors.append("#f39c12")  # オレンジ（準推奨）
                                    else:
                                        colors.append("#95a5a6")  # グレー
                                ax_ev.barh(
                                    [d["combination"] for d in top15][::-1],
                                    [d["ev"] for d in top15][::-1],
                                    color=colors[::-1])
                                ax_ev.axvline(x=1.0, color="red", linestyle="--",
                                              alpha=0.7, label="EV=1.0（損益分岐）")
                                ax_ev.axvline(x=0.8, color="orange", linestyle=":",
                                              alpha=0.5, label="EV=0.8")
                                ax_ev.set_xlabel("期待値 (EV)")
                                type_label = label_map_ev.get(bet_type, bet_type)
                                ax_ev.set_title(f"{type_label} 期待値 Top15")
                                ax_ev.legend(fontsize=8)
                                plt.tight_layout()
                                st.pyplot(fig_ev)
                                plt.close(fig_ev)

                            # 推奨買い目をリストに追加
                            for d in ev_data:
                                if d["ev"] >= 1.0:
                                    d["bet_type"] = label_map_ev.get(bet_type, bet_type)
                                    all_recommended.append(d)

                    # ── おすすめ買い目まとめ ──
                    st.markdown("---")
                    st.markdown("### 🎯 おすすめ買い目（EV ≥ 1.0）")

                    if all_recommended:
                        rec_sorted = sorted(all_recommended, key=lambda x: -x["ev"])
                        rec_df = pd.DataFrame([{
                            "券種": d["bet_type"],
                            "組合せ": d["combination"],
                            "確率": f"{d['probability']*100:.2f}%",
                            "オッズ": f"{d['odds']:.1f}",
                            "期待値": f"{d['ev']:.3f}",
                            "評価": d["flag"]
                        } for d in rec_sorted])
                        st.dataframe(rec_df, use_container_width=True, hide_index=True)

                        # サマリー
                        n_super = len([d for d in rec_sorted if d["ev"] >= 1.5])
                        n_rec = len([d for d in rec_sorted if 1.0 <= d["ev"] < 1.5])
                        col_s1, col_s2, col_s3 = st.columns(3)
                        col_s1.metric("🎯 推奨買い目数", f"{len(rec_sorted)}件")
                        col_s2.metric("🔥 超推奨(EV≥1.5)", f"{n_super}件")
                        col_s3.metric("✅ 推奨(EV≥1.0)", f"{n_rec}件")

                        # 最高EV
                        best = rec_sorted[0]
                        st.success(
                            f"🏆 最高期待値: **{best['bet_type']} {best['combination']}** "
                            f"（EV={best['ev']:.3f}, オッズ={best['odds']:.1f}, "
                            f"確率={best['probability']*100:.2f}%）")
                    else:
                        st.info("EV ≥ 1.0 の買い目はありませんでした。\n"
                                "シミュレーション回数を増やすか、準推奨を確認してください。")

                    # 準推奨
                    with st.expander("📋 準推奨（EV 0.8〜1.0）"):
                        semi = []
                        for bet_type, ev_data in ev_results.items():
                            for d in ev_data:
                                if 0.8 <= d["ev"] < 1.0:
                                    d_copy = d.copy()
                                    d_copy["bet_type"] = label_map_ev.get(bet_type, bet_type)
                                    semi.append(d_copy)
                        if semi:
                            semi_sorted = sorted(semi, key=lambda x: -x["ev"])[:30]
                            semi_df = pd.DataFrame([{
                                "券種": d["bet_type"],
                                "組合せ": d["combination"],
                                "確率": f"{d['probability']*100:.2f}%",
                                "オッズ": f"{d['odds']:.1f}",
                                "期待値": f"{d['ev']:.3f}",
                                "評価": d["flag"]
                            } for d in semi_sorted])
                            st.dataframe(semi_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("EV 0.8〜1.0 の買い目もありません")

                    # 全体統計
                    with st.expander("📊 全体統計"):
                        for bet_type, ev_data in ev_results.items():
                            if ev_data:
                                evs = [d["ev"] for d in ev_data]
                                type_label = label_map_ev.get(bet_type, bet_type)
                                st.write(f"**{type_label}:** "
                                         f"件数={len(evs)}, "
                                         f"最大EV={max(evs):.3f}, "
                                         f"平均EV={np.mean(evs):.3f}, "
                                         f"EV≥1.0={len([e for e in evs if e >= 1.0])}件")

# ── フッター ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em;'>"
    "🚤 ボートレース AI シミュレーター v6.0<br>"
    "30項目完全エージェント × 会場別特性(全24場) × 季節補正 × "
    "モンテカルロ × 合成オッズ × 期待値計算<br>"
    "© 2026 Boat Race AI Simulator"
    "</div>", unsafe_allow_html=True)
