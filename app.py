# ============================================================
# 🚤 ボートレース AI シミュレーター v3.1
# 場別プロファイル + オッズ自動取得 + 期待値計算 統合版
# オッズパーサー修正版
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import itertools
import re
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from urllib.request import urlopen, Request

# ============================================================
# 0. フォント設定
# ============================================================
def setup_japanese_font():
    font_path = None
    for p in fm.findSystemFonts():
        lower = p.lower()
        if 'notosanscjk' in lower or 'notosansjp' in lower or 'ipagothic' in lower or 'ipaexgothic' in lower:
            font_path = p
            break
    if font_path is None:
        try:
            subprocess.run(['fc-list', ':lang=ja'], capture_output=True, timeout=5)
            for p in fm.findSystemFonts():
                if 'noto' in p.lower() and 'cjk' in p.lower():
                    font_path = p
                    break
        except Exception:
            pass
    if font_path is None:
        candidates = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        ]
        for c in candidates:
            if os.path.exists(c):
                font_path = c
                break
    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = prop.get_name()
    else:
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
    matplotlib.rcParams['axes.unicode_minus'] = False

setup_japanese_font()

# ============================================================
# 1. 場別プロファイル (全24場 + デフォルト)
# ============================================================
VENUE_CODES = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05",
    "浜名湖": "06", "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10",
    "びわこ": "11", "住之江": "12", "尼崎": "13", "鳴門": "14", "丸亀": "15",
    "児島": "16", "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

DEFAULT_VENUE_PROFILE = {
    "code": "00", "water": "淡水", "tide": False,
    "course_win_rate": [55.0, 12.5, 12.0, 10.5, 6.0, 2.5],
    "course_top2":     [72.0, 44.0, 34.0, 30.0, 20.0, 9.0],
    "course_top3":     [82.0, 60.0, 54.0, 50.0, 38.0, 22.0],
    "kimarite": {
        1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
        3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
        5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
    },
    "wind_effect": 1.0, "memo": "全国平均プロファイル"
}

VENUE_PROFILES = {
    "桐生": {
        "code": "01", "water": "淡水", "tide": False,
        "course_win_rate": [49.4, 11.7, 12.8, 11.5, 8.5, 6.1],
        "course_top2": [68.5, 48.6, 32.8, 29.0, 21.6, 11.8],
        "course_top3": [77.7, 62.7, 52.1, 50.8, 42.6, 26.1],
        "kimarite": {
            1: [92.0, 0, 0, 0, 8.0, 0], 2: [0, 27, 55, 0, 15, 3],
            3: [0, 40, 18, 30, 10, 2], 4: [0, 42, 20, 28, 8, 2],
            5: [0, 18, 10, 55, 15, 2], 6: [0, 35, 15, 35, 10, 5],
        },
        "wind_effect": 1.2, "memo": "冬季の赤城おろし(向かい風)でインが弱体化"
    },
    "戸田": {
        "code": "02", "water": "淡水", "tide": False,
        "course_win_rate": [44.0, 13.0, 13.5, 14.0, 9.0, 6.5],
        "course_top2": [60.4, 48.0, 36.0, 32.0, 24.0, 13.0],
        "course_top3": [71.9, 62.0, 55.0, 52.0, 44.0, 28.0],
        "kimarite": {
            1: [94.5, 0, 0, 0, 5.5, 0], 2: [0, 25, 60, 0, 12, 3],
            3: [0, 45, 15, 25, 12, 3], 4: [0, 50, 18, 22, 8, 2],
            5: [0, 15, 8, 60, 12, 5], 6: [0, 30, 15, 40, 10, 5],
        },
        "wind_effect": 0.8, "memo": "全国最狭水面。イン最弱。まくり多発"
    },
    "江戸川": {
        "code": "03", "water": "汽水", "tide": True,
        "course_win_rate": [47.6, 13.0, 12.0, 12.0, 9.0, 5.5],
        "course_top2": [67.4, 46.0, 34.0, 30.0, 22.0, 12.0],
        "course_top3": [78.4, 62.0, 54.0, 50.0, 40.0, 26.0],
        "kimarite": {
            1: [93.0, 0, 0, 0, 7.0, 0], 2: [0, 30, 52, 0, 14, 4],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.5, "memo": "風・波の影響が全国一大きい。荒天時は大波乱"
    },
    "平和島": {
        "code": "04", "water": "海水", "tide": True,
        "course_win_rate": [46.2, 14.5, 12.0, 12.0, 9.0, 5.5],
        "course_top2": [65.0, 48.0, 34.0, 30.0, 22.0, 12.0],
        "course_top3": [76.6, 62.0, 54.0, 50.0, 40.0, 26.0],
        "kimarite": {
            1: [93.0, 0, 0, 0, 7.0, 0], 2: [0, 25, 60, 0, 12, 3],
            3: [0, 40, 18, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 35, 15, 35, 10, 5],
        },
        "wind_effect": 1.1, "memo": "バックが広く6コースの1着率高め"
    },
    "多摩川": {
        "code": "05", "water": "淡水", "tide": False,
        "course_win_rate": [52.3, 14.0, 12.0, 10.5, 7.0, 3.5],
        "course_top2": [70.3, 46.0, 34.0, 30.0, 20.0, 10.0],
        "course_top3": [79.4, 62.0, 54.0, 50.0, 38.0, 24.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 0.7, "memo": "静水面でクセが少ない。実力通り決まりやすい"
    },
    "浜名湖": {
        "code": "06", "water": "汽水", "tide": True,
        "course_win_rate": [51.8, 13.5, 12.5, 10.5, 7.0, 3.5],
        "course_top2": [69.6, 45.0, 34.0, 30.0, 20.0, 10.0],
        "course_top3": [78.9, 61.0, 54.0, 50.0, 38.0, 24.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.0, "memo": "広い水面。風向き次第で荒れることも"
    },
    "蒲郡": {
        "code": "07", "water": "汽水", "tide": True,
        "course_win_rate": [57.6, 11.8, 10.5, 10.0, 6.5, 3.0],
        "course_top2": [74.2, 42.0, 32.0, 28.0, 18.0, 8.0],
        "course_top3": [83.2, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.9, "memo": "イン有利。ナイター開催多い"
    },
    "常滑": {
        "code": "08", "water": "海水", "tide": True,
        "course_win_rate": [58.0, 12.8, 10.0, 10.0, 6.0, 3.0],
        "course_top2": [73.9, 44.0, 32.0, 28.0, 18.0, 8.0],
        "course_top3": [82.1, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 1.1, "memo": "伊勢湾沿い。冬場の北西風でイン不安定"
    },
    "津": {
        "code": "09", "water": "海水", "tide": True,
        "course_win_rate": [56.9, 12.5, 11.5, 10.5, 6.0, 2.5],
        "course_top2": [73.1, 44.0, 34.0, 28.0, 18.0, 8.0],
        "course_top3": [81.6, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.0, "memo": "イン安定。企画レースで1号艇にA級配置多い"
    },
    "三国": {
        "code": "10", "water": "淡水", "tide": False,
        "course_win_rate": [53.2, 13.0, 12.0, 10.5, 7.0, 3.5],
        "course_top2": [72.1, 44.0, 34.0, 30.0, 20.0, 9.0],
        "course_top3": [80.9, 60.0, 54.0, 50.0, 38.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.3, "memo": "北陸特有の強風。冬場は荒れやすい"
    },
    "びわこ": {
        "code": "11", "water": "淡水", "tide": False,
        "course_win_rate": [52.0, 13.5, 12.0, 10.5, 7.0, 3.5],
        "course_top2": [71.1, 45.0, 34.0, 30.0, 20.0, 9.0],
        "course_top3": [80.9, 61.0, 54.0, 50.0, 38.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.2, "memo": "標高85m、気圧低くモーター出力に影響"
    },
    "住之江": {
        "code": "12", "water": "淡水", "tide": False,
        "course_win_rate": [58.0, 13.0, 10.5, 10.0, 5.5, 2.5],
        "course_top2": [73.7, 44.0, 32.0, 28.0, 18.0, 8.0],
        "course_top3": [82.3, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.6, "memo": "静水面のイン天国。SG開催多数"
    },
    "尼崎": {
        "code": "13", "water": "淡水", "tide": False,
        "course_win_rate": [57.6, 13.0, 11.0, 10.5, 5.5, 2.5],
        "course_top2": [74.5, 44.0, 32.0, 28.0, 18.0, 8.0],
        "course_top3": [83.3, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.7, "memo": "静水面。センターポールが遠くST難しい"
    },
    "鳴門": {
        "code": "14", "water": "海水", "tide": True,
        "course_win_rate": [46.5, 14.5, 12.5, 12.0, 8.5, 5.5],
        "course_top2": [64.5, 48.0, 35.0, 30.0, 22.0, 12.0],
        "course_top3": [75.8, 62.0, 55.0, 52.0, 42.0, 28.0],
        "kimarite": {
            1: [93.0, 0, 0, 0, 7.0, 0], 2: [0, 26, 58, 0, 13, 3],
            3: [0, 40, 18, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 35, 15, 35, 10, 5],
        },
        "wind_effect": 1.3, "memo": "うねりが大きい。干満差激しくイン不安定"
    },
    "丸亀": {
        "code": "15", "water": "海水", "tide": True,
        "course_win_rate": [56.8, 13.0, 11.0, 10.5, 6.0, 2.5],
        "course_top2": [73.8, 44.0, 34.0, 28.0, 18.0, 8.0],
        "course_top3": [82.3, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.0, "memo": "海水で干満差あり。向かい風でセンター有利"
    },
    "児島": {
        "code": "16", "water": "海水", "tide": True,
        "course_win_rate": [57.4, 12.0, 11.0, 10.5, 6.0, 2.5],
        "course_top2": [75.0, 44.0, 34.0, 28.0, 18.0, 8.0],
        "course_top3": [83.6, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.9, "memo": "瀬戸内海。干満差大きいが比較的穏やか"
    },
    "宮島": {
        "code": "17", "water": "海水", "tide": True,
        "course_win_rate": [56.2, 13.0, 11.0, 10.5, 6.5, 3.0],
        "course_top2": [73.1, 44.0, 34.0, 28.0, 18.0, 8.0],
        "course_top3": [80.9, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.1, "memo": "干満差が全国屈指。潮の影響大"
    },
    "徳山": {
        "code": "18", "water": "海水", "tide": True,
        "course_win_rate": [63.4, 11.7, 12.8, 9.7, 3.5, 1.1],
        "course_top2": [80.4, 42.2, 32.8, 27.2, 15.7, 6.3],
        "course_top3": [87.5, 59.4, 53.0, 48.8, 32.7, 23.1],
        "kimarite": {
            1: [95.7, 0, 0, 0, 4.2, 0],
            2: [0, 30.0, 58.7, 0, 11.2, 0],
            3: [0, 44.8, 14.9, 26.4, 11.4, 2.2],
            4: [0, 45.4, 21.2, 27.2, 4.5, 1.5],
            5: [0, 20.8, 8.3, 58.3, 12.5, 0],
            6: [0, 50.0, 12.5, 37.5, 0, 0],
        },
        "seasonal": {"春": 64.6, "夏": 66.1, "秋": 61.1, "冬": 62.9},
        "wind_effect": 0.6,
        "memo": "全国1位のイン天国。追い風安定、海水で干満差あり"
    },
    "下関": {
        "code": "19", "water": "海水", "tide": True,
        "course_win_rate": [62.0, 12.0, 10.0, 9.0, 4.5, 2.0],
        "course_top2": [77.4, 44.0, 32.0, 28.0, 16.0, 7.0],
        "course_top3": [85.4, 58.0, 50.0, 46.0, 32.0, 20.0],
        "kimarite": {
            1: [95.5, 0, 0, 0, 4.5, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 44, 16, 28, 10, 2], 4: [0, 46, 20, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.7, "memo": "ナイター。徳山に次ぐ1コース1着率"
    },
    "若松": {
        "code": "20", "water": "海水", "tide": True,
        "course_win_rate": [57.0, 13.0, 11.0, 10.0, 6.0, 3.0],
        "course_top2": [74.1, 44.0, 34.0, 28.0, 18.0, 8.0],
        "course_top3": [83.4, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.0, "memo": "ナイター。海水で干満差あり"
    },
    "芦屋": {
        "code": "21", "water": "淡水", "tide": False,
        "course_win_rate": [57.8, 12.5, 11.0, 10.0, 5.5, 2.5],
        "course_top2": [74.1, 44.0, 32.0, 28.0, 18.0, 8.0],
        "course_top3": [83.5, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.7, "memo": "モーニング開催。企画レースでイン有利"
    },
    "福岡": {
        "code": "22", "water": "海水", "tide": True,
        "course_win_rate": [57.3, 13.0, 11.0, 10.5, 5.5, 2.5],
        "course_top2": [74.5, 44.0, 34.0, 28.0, 18.0, 8.0],
        "course_top3": [82.4, 58.0, 52.0, 48.0, 34.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.2, "memo": "博多湾沿い。うねり注意"
    },
    "唐津": {
        "code": "23", "water": "海水", "tide": True,
        "course_win_rate": [55.2, 13.0, 12.0, 10.5, 6.0, 3.0],
        "course_top2": [73.3, 44.0, 34.0, 30.0, 20.0, 9.0],
        "course_top3": [82.3, 60.0, 54.0, 50.0, 38.0, 22.0],
        "kimarite": {
            1: [94.0, 0, 0, 0, 6.0, 0], 2: [0, 28, 56, 0, 13, 3],
            3: [0, 42, 16, 28, 11, 3], 4: [0, 44, 20, 26, 7, 3],
            5: [0, 20, 8, 56, 13, 3], 6: [0, 40, 15, 35, 5, 5],
        },
        "wind_effect": 1.1, "memo": "向かい風時は4コースのまくりが有力"
    },
    "大村": {
        "code": "24", "water": "海水", "tide": True,
        "course_win_rate": [62.6, 11.5, 11.0, 9.5, 4.0, 1.8],
        "course_top2": [79.3, 43.0, 33.0, 28.0, 17.0, 7.0],
        "course_top3": [87.3, 60.0, 54.0, 49.0, 34.0, 24.0],
        "kimarite": {
            1: [95.0, 0, 0, 0, 5.0, 0], 2: [0, 28, 58, 0, 12, 2],
            3: [0, 42, 16, 28, 12, 2], 4: [0, 44, 22, 26, 6, 2],
            5: [0, 22, 8, 56, 12, 2], 6: [0, 45, 15, 30, 5, 5],
        },
        "wind_effect": 0.7, "memo": "徳山に次ぐイン天国。ナイター開催"
    },
}

def get_venue_profile(venue_name: str) -> dict:
    return VENUE_PROFILES.get(venue_name, DEFAULT_VENUE_PROFILE)

def get_season(month: int) -> str:
    if month in [3, 4, 5]: return "春"
    elif month in [6, 7, 8]: return "夏"
    elif month in [9, 10, 11]: return "秋"
    else: return "冬"

# ============================================================
# 2. エージェント & レース条件
# ============================================================
@dataclass
class BoatAgent:
    lane: int
    number: int = 0
    name: str = ""
    rank: str = "B1"
    age: int = 35
    branch: str = ""
    period: int = 0
    avg_st: float = 0.18
    st_stability: float = 50.0
    win_rate: float = 4.0
    lane_win_rate: float = 0.0
    top2_rate: float = 30.0
    top3_rate: float = 45.0
    ability: int = 45
    motor_contribution: float = 0.0
    exhibition_time: float = 6.90
    lap_time: float = 0.0
    turn_time: float = 0.0
    weight: float = 52.0
    wins_escape: int = 0
    wins_makuri: int = 0
    wins_sashi: int = 0
    wins_makuri_sashi: int = 0
    wins_nuki: int = 0
    comment: str = ""
    actual_st: float = 0.0

    def calculate_start_timing(self):
        base = self.avg_st
        stability_noise = (100 - self.st_stability) / 100.0 * 0.05
        noise = np.random.normal(0, max(0.001, stability_noise))
        self.actual_st = max(0.01, base + noise)
        return self.actual_st

    def get_power_score(self) -> float:
        rank_scores = {"A1": 1.0, "A2": 0.8, "B1": 0.5, "B2": 0.2}
        rank_s = rank_scores.get(self.rank, 0.5)
        wr = min(self.win_rate / 8.0, 1.0)
        motor = 0.5 + self.motor_contribution * 0.15
        motor = max(0.1, min(1.0, motor))
        return rank_s * 0.35 + wr * 0.40 + motor * 0.25

@dataclass
class RaceCondition:
    temperature: float = 15.0
    wind_speed: float = 3.0
    wave_height: float = 3.0
    weather: str = "曇り"
    wind_direction: str = "追い風"
    tide: str = ""

# ============================================================
# 3. レースシミュレーター (場別プロファイル対応版)
# ============================================================
class RaceSimulator:
    def __init__(self, agents: List[BoatAgent], conditions: RaceCondition,
                 venue_name: str = "徳山", race_month: int = 2):
        self.agents = sorted(agents, key=lambda a: a.lane)
        self.conditions = conditions
        self.venue_name = venue_name
        self.month = race_month
        self.profile = get_venue_profile(venue_name)

    def _compute_race_weights(self) -> Dict[int, float]:
        profile = self.profile
        season = get_season(self.month)
        venue_cwr = list(profile['course_win_rate'])
        if 'seasonal' in profile:
            s1c = profile['seasonal'].get(season, venue_cwr[0])
            remaining = 100.0 - s1c
            orig_remaining = 100.0 - venue_cwr[0]
            venue_cwr[0] = s1c
            if orig_remaining > 0:
                for i in range(1, 6):
                    venue_cwr[i] = profile['course_win_rate'][i] * (remaining / orig_remaining)
        weights = {}
        for agent in self.agents:
            idx = agent.lane - 1
            venue_base = venue_cwr[idx]
            player_lw = agent.lane_win_rate if agent.lane_win_rate > 0 else venue_base
            base_prob = venue_base * 0.55 + player_lw * 0.45
            power = agent.get_power_score()
            power_adj = 0.85 + power * 0.30
            st_q = max(0, (0.22 - agent.avg_st) / 0.10)
            st_adj = 0.92 + st_q * 0.08
            motor_adj = 1.0 + agent.motor_contribution * 0.05
            ws = self.conditions.wind_speed
            w_eff = profile.get('wind_effect', 1.0)
            if agent.lane <= 2:
                wind_adj = 1.0 + (ws - 3) * 0.008 * w_eff
            else:
                wind_adj = 1.0 + (ws - 3) * 0.004 * w_eff
            tide_adj = 1.0
            if profile.get('tide') and self.conditions.wave_height >= 5:
                if agent.lane == 1: tide_adj = 0.95
                elif agent.lane == 2: tide_adj = 1.06
            w = max(0.3, base_prob * power_adj * st_adj * motor_adj * wind_adj * tide_adj)
            weights[agent.lane] = w
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] = weights[k] / total * 100
        return weights

    def simulate_race(self) -> dict:
        for a in self.agents:
            a.calculate_start_timing()
        base_w = self._compute_race_weights()
        sts = {a.lane: a.actual_st for a in self.agents}
        best_st = min(sts.values())
        adjusted = dict(base_w)
        for lane, st_val in sts.items():
            diff = st_val - best_st
            adjusted[lane] *= max(0.7, 1.0 - diff * 2.5)
        profile = self.profile
        for a in self.agents:
            kim = profile.get('kimarite', {}).get(a.lane, [0]*6)
            if a.lane == 1:
                escape_r = kim[0] / 100.0 if len(kim) > 0 else 0.95
                adjusted[a.lane] *= (0.90 + escape_r * 0.15)
            else:
                attack = (kim[1] + kim[3]) / 100.0 if len(kim) > 3 else 0.3
                adjusted[a.lane] *= (0.95 + attack * 0.15)
        for lane in adjusted:
            adjusted[lane] *= np.random.uniform(0.85, 1.15)
            adjusted[lane] = max(0.1, adjusted[lane])
        lanes = list(adjusted.keys())
        ws_arr = np.array([adjusted[l] for l in lanes])
        ws_arr = ws_arr / ws_arr.sum()
        order = []
        remaining = list(range(len(lanes)))
        rem_w = ws_arr.copy()
        for _ in range(len(lanes)):
            rem_w_norm = rem_w[remaining] / rem_w[remaining].sum()
            chosen_idx = np.random.choice(remaining, p=rem_w_norm)
            order.append(lanes[chosen_idx])
            remaining.remove(chosen_idx)
        n_steps = 300
        history = {a.lane: [] for a in self.agents}
        positions = {a.lane: 0.0 for a in self.agents}
        final_rank = {lane: rank for rank, lane in enumerate(order)}
        for step in range(n_steps):
            for a in self.agents:
                target = 1800 * (1 - final_rank[a.lane] / 6.0)
                current = positions[a.lane]
                speed = (target - current) * 0.03 + np.random.normal(0, 2)
                positions[a.lane] = current + speed
                history[a.lane].append(positions[a.lane])
        return {'finish_order': order, 'history': history, 'start_timings': sts, 'weights': adjusted}

    def determine_kimarite(self, order, sts) -> str:
        winner = order[0]
        if winner == 1:
            return "逃げ"
        elif winner in [2, 3]:
            w_st = sts.get(winner, 0.18)
            in_st = sts.get(1, 0.18)
            if w_st < in_st - 0.03:
                return "差し" if winner == 2 else "捲り"
            return "差し" if np.random.random() < 0.55 else "捲り"
        elif winner == 4:
            r = np.random.random()
            if r < 0.45: return "捲り"
            elif r < 0.70: return "捲り差し"
            else: return "差し"
        else:
            r = np.random.random()
            if r < 0.25: return "捲り"
            elif r < 0.75: return "捲り差し"
            else: return "抜き"

# ============================================================
# 4. データパーサー
# ============================================================
def parse_race_data(text: str) -> Tuple[List[BoatAgent], Optional[RaceCondition]]:
    agents = []
    lines = text.strip().split('\n')
    full = text
    name_patterns = [
        re.compile(r'(\d)号艇\s*(\d{3,5})\s+(.+?)\s+(A1|A2|B1|B2)'),
        re.compile(r'(\d)\s+(\d{3,5})\s+(\S+)\s+(A1|A2|B1|B2)'),
    ]
    found = []
    for line in lines:
        for pat in name_patterns:
            m = pat.search(line)
            if m:
                found.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)))
                break
    def find_numbers_after(keyword, count=6):
        for i, line in enumerate(lines):
            if keyword in line:
                nums = re.findall(r'[\d]+\.[\d]+|[\d]+', line)
                nums = [float(n) for n in nums if '.' in n or (n.isdigit() and float(n) < 1000)]
                if len(nums) >= count:
                    return nums[:count]
                for j in range(i+1, min(i+5, len(lines))):
                    nums += re.findall(r'[\d]+\.[\d]+', lines[j])
                    nums = [float(n) for n in nums]
                    if len(nums) >= count:
                        return nums[:count]
        return None
    def find_pcts_after(keyword, count=6):
        for i, line in enumerate(lines):
            if keyword in line:
                combined = line
                for j in range(i+1, min(i+15, len(lines))):
                    combined += ' ' + lines[j]
                pcts = re.findall(r'([\d]+\.[\d]+)\s*%', combined)
                if len(pcts) >= count:
                    return [float(p) for p in pcts[:count]]
        return None
    avg_sts = find_numbers_after('平均ST', 6) or find_numbers_after('ST', 6)
    win_rates = find_pcts_after('勝率', 6) or find_numbers_after('勝率', 6)
    top2 = find_pcts_after('2連対', 6)
    top3 = find_pcts_after('3連対', 6)
    lane_win = find_pcts_after('枠別1着', 6) or find_pcts_after('枠別', 6)
    cond = RaceCondition()
    temp_m = re.search(r'(\d+)\s*[°℃]', full)
    if temp_m: cond.temperature = float(temp_m.group(1))
    wind_m = re.search(r'風[速\s]*(\d+)\s*m', full)
    if wind_m: cond.wind_speed = float(wind_m.group(1))
    wave_m = re.search(r'波[高\s]*(\d+)\s*cm', full)
    if wave_m: cond.wave_height = float(wave_m.group(1))
    if '雨' in full: cond.weather = "雨"
    elif '晴' in full: cond.weather = "晴れ"
    elif '曇' in full: cond.weather = "曇り"
    for i in range(6):
        lane = i + 1
        if i < len(found):
            _, num, name, rank = found[i]
        else:
            num, name, rank = 0, f"選手{lane}", "B1"
        default_lw = {1: 55.0, 2: 12.0, 3: 12.0, 4: 10.0, 5: 6.0, 6: 2.5}
        agent = BoatAgent(
            lane=lane, number=num, name=name, rank=rank,
            avg_st=avg_sts[i] if avg_sts and i < len(avg_sts) else 0.18,
            win_rate=win_rates[i] if win_rates and i < len(win_rates) else 4.0,
            lane_win_rate=lane_win[i] if lane_win and i < len(lane_win) else default_lw.get(lane, 10.0),
            top2_rate=top2[i] if top2 and i < len(top2) else 30.0,
            top3_rate=top3[i] if top3 and i < len(top3) else 45.0,
        )
        agents.append(agent)
    return agents, cond
# ============================================================
# 5. オッズ取得 & 合成オッズ & 期待値 (修正版)
# ============================================================
def fetch_trifecta_odds(venue_code: str, date_str: str, race_no: int) -> dict:
    """
    boatrace.jp の3連単オッズを取得し、正確な買い目に対応付ける。

    公式テーブルの構造:
      - 20行 × 6列 = 120セル (行優先で格納)
      - 各列が「1着=X号艇」(X=1..6)
      - 各列内の20セルは 2着(小→大) × 3着(小→大) の順
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={race_no}&jcd={venue_code}&hd={date_str}"
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req, timeout=15).read()
    except Exception as e:
        st.error(f"取得失敗: {e}")
        return {}
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
    except ImportError:
        st.error("beautifulsoup4 が必要です")
        return {}

    odds_vals = []
    for table in soup.find_all('table'):
        cells = table.find_all('td', class_='oddsPoint')
        if len(cells) >= 120:
            for c in cells:
                txt = c.get_text(strip=True)
                try:
                    odds_vals.append(float(txt.replace(',', '')))
                except ValueError:
                    odds_vals.append(0.0)
            break

    if len(odds_vals) < 120:
        st.warning(f"⚠️ oddsPointセルが{len(odds_vals)}個しか見つかりません")
        return {}

    boats = [1, 2, 3, 4, 5, 6]

    def get_column_order(first):
        others = sorted([b for b in boats if b != first])
        order = []
        for second in others:
            thirds = sorted([b for b in others if b != second])
            for third in thirds:
                order.append((first, second, third))
        return order

    column_orders = []
    for first in boats:
        column_orders.append(get_column_order(first))

    odds_dict = {}
    for row_idx in range(20):
        for col_idx in range(6):
            cell_idx = row_idx * 6 + col_idx
            if cell_idx < len(odds_vals):
                f, s, t = column_orders[col_idx][row_idx]
                key = f"{f}-{s}-{t}"
                odds_dict[key] = odds_vals[cell_idx]

    return odds_dict


def parse_pasted_odds(text: str) -> dict:
    odds_dict = {}
    for m in re.finditer(r'(\d)\s*-\s*(\d)\s*-\s*(\d)\s+([\d,.]+)', text):
        key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        try:
            odds_dict[key] = float(m.group(4).replace(',', ''))
        except ValueError:
            pass
    if len(odds_dict) >= 60:
        return odds_dict
    nums = re.findall(r'(?<!\d)([\d]{1,5}(?:\.[\d]+))(?!\d)', text)
    candidates = [float(n) for n in nums if 1.0 <= float(n) <= 99999]
    if len(candidates) >= 120:
        boats = [1, 2, 3, 4, 5, 6]
        def get_column_order(first):
            others = sorted([b for b in boats if b != first])
            order = []
            for second in others:
                thirds = sorted([b for b in others if b != second])
                for third in thirds:
                    order.append((first, second, third))
            return order
        column_orders = [get_column_order(f) for f in boats]
        for row_idx in range(20):
            for col_idx in range(6):
                cell_idx = row_idx * 6 + col_idx
                if cell_idx < len(candidates):
                    f, s, t = column_orders[col_idx][row_idx]
                    odds_dict[f"{f}-{s}-{t}"] = candidates[cell_idx]
    return odds_dict


def compute_synthetic_odds(tri_odds: dict) -> dict:
    boats = [1, 2, 3, 4, 5, 6]
    result = {'trifecta': dict(tri_odds), 'trio': {}, 'exacta': {}, 'quinella': {}, 'wide': {}}

    for combo in itertools.combinations(boats, 3):
        inv = 0.0
        for p in itertools.permutations(combo):
            k = f"{p[0]}-{p[1]}-{p[2]}"
            if k in tri_odds and tri_odds[k] > 0:
                inv += 1.0 / tri_odds[k]
        result['trio'][f"{combo[0]}={combo[1]}={combo[2]}"] = round(1.0 / inv, 1) if inv > 0 else 0

    for f in boats:
        for s in boats:
            if f == s:
                continue
            inv = 0.0
            for t in boats:
                if t == f or t == s:
                    continue
                k = f"{f}-{s}-{t}"
                if k in tri_odds and tri_odds[k] > 0:
                    inv += 1.0 / tri_odds[k]
            result['exacta'][f"{f}-{s}"] = round(1.0 / inv, 1) if inv > 0 else 0

    for combo in itertools.combinations(boats, 2):
        a, b = combo
        inv_q = 0.0
        for p in itertools.permutations(combo):
            for t in boats:
                if t in combo:
                    continue
                k = f"{p[0]}-{p[1]}-{t}"
                if k in tri_odds and tri_odds[k] > 0:
                    inv_q += 1.0 / tri_odds[k]
        result['quinella'][f"{a}={b}"] = round(1.0 / inv_q, 1) if inv_q > 0 else 0

        inv_w = 0.0
        for third in boats:
            if third in combo:
                continue
            for p in itertools.permutations([a, b, third]):
                k = f"{p[0]}-{p[1]}-{p[2]}"
                if k in tri_odds and tri_odds[k] > 0:
                    inv_w += 1.0 / tri_odds[k]
        result['wide'][f"{a}={b}"] = round(1.0 / inv_w, 1) if inv_w > 0 else 0

    return result


def run_ev_simulation(agents, conditions, venue_name, month, n_sims=10000):
    boats = [1, 2, 3, 4, 5, 6]
    counts = {
        'trifecta': {}, 'trio': {}, 'exacta': {}, 'quinella': {}, 'wide': {}
    }
    for p in itertools.permutations(boats, 3):
        counts['trifecta'][f"{p[0]}-{p[1]}-{p[2]}"] = 0
    for c in itertools.combinations(boats, 3):
        counts['trio'][f"{c[0]}={c[1]}={c[2]}"] = 0
    for p in itertools.permutations(boats, 2):
        counts['exacta'][f"{p[0]}-{p[1]}"] = 0
    for c in itertools.combinations(boats, 2):
        counts['quinella'][f"{c[0]}={c[1]}"] = 0
        counts['wide'][f"{c[0]}={c[1]}"] = 0

    sim = RaceSimulator(agents, conditions, venue_name, month)
    bar = st.progress(0)
    step = max(1, n_sims // 20)

    for i in range(n_sims):
        result = sim.simulate_race()
        order = result['finish_order']
        top3 = order[:3]

        tkey = f"{top3[0]}-{top3[1]}-{top3[2]}"
        if tkey in counts['trifecta']:
            counts['trifecta'][tkey] += 1

        ts = sorted(top3)
        trio_key = f"{ts[0]}={ts[1]}={ts[2]}"
        if trio_key in counts['trio']:
            counts['trio'][trio_key] += 1

        ekey = f"{top3[0]}-{top3[1]}"
        if ekey in counts['exacta']:
            counts['exacta'][ekey] += 1

        qs = sorted(top3[:2])
        qkey = f"{qs[0]}={qs[1]}"
        if qkey in counts['quinella']:
            counts['quinella'][qkey] += 1

        for combo in itertools.combinations(top3, 2):
            wsorted = sorted(combo)
            wkey = f"{wsorted[0]}={wsorted[1]}"
            if wkey in counts['wide']:
                counts['wide'][wkey] += 1

        if (i + 1) % step == 0:
            bar.progress((i + 1) / n_sims)

    bar.progress(1.0)

    probs = {}
    for ticket_type in counts:
        probs[ticket_type] = {}
        for key, cnt in counts[ticket_type].items():
            probs[ticket_type][key] = cnt / n_sims
    return probs


def compute_expected_values(synth_odds: dict, sim_probs: dict) -> dict:
    ev = {}
    for tt in synth_odds:
        ev[tt] = {}
        for key, odds in synth_odds[tt].items():
            prob = sim_probs.get(tt, {}).get(key, 0.0)
            val = prob * odds
            ev[tt][key] = {
                'odds': odds,
                'probability': prob,
                'expected_value': round(val, 4),
                'profitable': val > 1.0
            }
    return ev


# ============================================================
# 6. Streamlit UI
# ============================================================
st.set_page_config(page_title="🚤 ボートレース AI シミュレーター", layout="wide")

st.markdown("""
<div style='background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            padding: 25px; border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: #00d2ff; text-align: center;'>🚤 ボートレース AI シミュレーター v3.1</h1>
    <p style='color: #ccc; text-align: center;'>
        場別プロファイル対応 ｜ オッズ自動取得(修正版) ｜ 全券種期待値計算
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 6-A. サイドバー: レース場 & 日付設定
# ----------------------------------------------------------
st.sidebar.markdown("## ⚙️ レース設定")

venue_list = list(VENUE_PROFILES.keys())
selected_venue = st.sidebar.selectbox(
    "🏟️ レース場", venue_list,
    index=venue_list.index("徳山"), key="sel_venue")

from datetime import date, datetime
race_date = st.sidebar.date_input("📅 日付", value=date(2026, 2, 27), key="sel_date")
race_no = st.sidebar.number_input("🏁 レース番号", 1, 12, 1, key="sel_race_no")
race_month = race_date.month
season = get_season(race_month)
profile = get_venue_profile(selected_venue)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 {selected_venue}の特徴")
st.sidebar.write(f"**水質:** {profile['water']}　**干満差:** {'あり' if profile.get('tide') else 'なし'}")
st.sidebar.write(f"**風の影響度:** {profile.get('wind_effect', 1.0)}　**季節:** {season}")
st.sidebar.write(f"💡 {profile.get('memo', '')}")

fig_sb, ax_sb = plt.subplots(figsize=(5, 2.5))
cwr = profile['course_win_rate']
colors_sb = ['#e74c3c', '#000000', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6']
ax_sb.bar([f'{i+1}コース' for i in range(6)], cwr, color=colors_sb)
for i, v in enumerate(cwr):
    ax_sb.text(i, v + 0.5, f'{v}%', ha='center', fontsize=8)
ax_sb.set_ylabel('1着率(%)', fontsize=8)
ax_sb.set_title(f'{selected_venue} コース別1着率', fontsize=10)
ax_sb.tick_params(labelsize=7)
plt.tight_layout()
st.sidebar.pyplot(fig_sb)
plt.close()

# ----------------------------------------------------------
# 6-B. データ入力
# ----------------------------------------------------------
st.markdown("## 📝 レースデータ入力")

input_mode = st.radio(
    "入力方式", ["📋 テキスト貼り付け", "✏️ フォーム入力"],
    horizontal=True, key="input_mode")

agents = []
conditions = RaceCondition()

if input_mode == "📋 テキスト貼り付け":
    sample_text = """1号艇 4704 河野大 A2 37歳 徳島
2号艇 3614 谷勝幸 B1 53歳 広島
3号艇 3875 廣中良一 B1 49歳 山口
4号艇 3519 冨田秀幸 B1 57歳 愛知
5号艇 4782 浜崎準也 B1 39歳 岡山
6号艇 5284 嘉手苅徹哉 B2 27歳 福岡
平均ST: 0.13 0.16 0.18 0.16 0.20 0.17
勝率: 6.45% 4.45% 4.38% 4.58% 4.43% 2.66%
2連対率: 76.0% 29.4% 26.7% 11.1% 25.0% 0.0%
3連対率: 88.0% 47.1% 26.7% 38.9% 37.5% 0.0%
気温12℃ 風速3m 波高3cm 曇り"""

    race_text = st.text_area(
        "レースデータを貼り付け", value="", height=300,
        placeholder="出走表のデータをここに貼り付けてください...", key="race_text")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📝 サンプルデータを挿入", key="btn_sample"):
            st.session_state['race_text'] = sample_text
            st.rerun()
    with col_s2:
        parse_btn = st.button("🔍 データを解析", key="btn_parse", type="primary")

    if parse_btn and race_text.strip():
        agents, conditions = parse_race_data(race_text)
        st.session_state['agents'] = agents
        st.session_state['conditions'] = conditions
        st.success(f"✅ {len(agents)}艇のエージェントを作成しました！")

elif input_mode == "✏️ フォーム入力":
    st.markdown("#### 各艇のデータを入力")
    form_agents = []
    for i in range(6):
        lane = i + 1
        with st.expander(f"🚤 {lane}号艇", expanded=(i == 0)):
            cols = st.columns(4)
            with cols[0]:
                name = st.text_input("選手名", key=f"name_{lane}")
                rank = st.selectbox("級別", ["A1", "A2", "B1", "B2"], index=2, key=f"rank_{lane}")
            with cols[1]:
                avg_st = st.number_input("平均ST", 0.05, 0.30, 0.18, 0.01, key=f"st_{lane}")
                win_rate = st.number_input("勝率", 1.0, 10.0, 4.0, 0.1, key=f"wr_{lane}")
            with cols[2]:
                lane_wr = st.number_input("枠別1着率(%)", 0.0, 100.0, 0.0, 1.0, key=f"lwr_{lane}")
                motor = st.number_input("モータ貢献", -2.0, 2.0, 0.0, 0.1, key=f"motor_{lane}")
            with cols[3]:
                top2 = st.number_input("2連対率(%)", 0.0, 100.0, 30.0, 1.0, key=f"t2_{lane}")
                top3 = st.number_input("3連対率(%)", 0.0, 100.0, 45.0, 1.0, key=f"t3_{lane}")
            default_lw = {1: 55.0, 2: 12.0, 3: 12.0, 4: 10.0, 5: 6.0, 6: 2.5}
            form_agents.append(BoatAgent(
                lane=lane, name=name or f"選手{lane}", rank=rank,
                avg_st=avg_st, win_rate=win_rate,
                lane_win_rate=lane_wr if lane_wr > 0 else default_lw.get(lane, 10.0),
                top2_rate=top2, top3_rate=top3, motor_contribution=motor
            ))

    st.markdown("#### 🌤️ 気象条件")
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        temp = st.number_input("気温(℃)", -5.0, 40.0, 15.0, key="f_temp")
    with wc2:
        wind = st.number_input("風速(m)", 0.0, 15.0, 3.0, key="f_wind")
    with wc3:
        wave = st.number_input("波高(cm)", 0.0, 30.0, 3.0, key="f_wave")
    with wc4:
        weather = st.selectbox("天候", ["晴れ", "曇り", "雨", "雪"], index=1, key="f_weather")

    if st.button("✅ エージェント作成", key="btn_form", type="primary"):
        agents = form_agents
        conditions = RaceCondition(
            temperature=temp, wind_speed=wind,
            wave_height=wave, weather=weather)
        st.session_state['agents'] = agents
        st.session_state['conditions'] = conditions
        st.success(f"✅ {len(agents)}艇のエージェントを作成しました！")

# ----------------------------------------------------------
# 6-C. エージェント一覧表示
# ----------------------------------------------------------
if 'agents' in st.session_state and st.session_state['agents']:
    agents = st.session_state['agents']
    conditions = st.session_state.get('conditions', RaceCondition())

    st.markdown("### 🚤 エージェント一覧")
    agent_rows = []
    for a in agents:
        agent_rows.append({
            '枠': f"{a.lane}号艇", '選手名': a.name, '級別': a.rank,
            '平均ST': f"{a.avg_st:.2f}", '勝率': f"{a.win_rate:.2f}",
            '枠別1着率': f"{a.lane_win_rate:.1f}%",
            '2連対率': f"{a.top2_rate:.1f}%", '3連対率': f"{a.top3_rate:.1f}%",
            'モータ': f"{a.motor_contribution:+.2f}",
            'パワー': f"{a.get_power_score():.3f}"
        })
    st.dataframe(pd.DataFrame(agent_rows), use_container_width=True)

    # ----------------------------------------------------------
    # 6-D. シミュレーション実行
    # ----------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🎲 レースシミュレーション")

    n_sims = st.slider("モンテカルロ回数", 1000, 50000, 5000, 1000, key="mc_sims")

    if st.button("🚀 シミュレーション実行", key="btn_sim", type="primary"):

        st.markdown("### 🏁 単発シミュレーション (3回)")
        sim = RaceSimulator(agents, conditions, selected_venue, race_month)
        for trial in range(3):
            result = sim.simulate_race()
            order = result['finish_order']
            sts = result['start_timings']
            kim = sim.determine_kimarite(order, sts)
            col_r, col_g = st.columns([1, 2])
            with col_r:
                st.write(f"**第{trial+1}回** — 決まり手: **{kim}**")
                for rank_i, lane in enumerate(order):
                    a = next(ag for ag in agents if ag.lane == lane)
                    st.write(f"{rank_i+1}着: {lane}号艇 {a.name} (ST {sts[lane]:.3f})")
            with col_g:
                fig_t, ax_t = plt.subplots(figsize=(8, 3))
                for lane_h, hist in result['history'].items():
                    a = next(ag for ag in agents if ag.lane == lane_h)
                    ax_t.plot(hist, label=f'{lane_h}号艇 {a.name}', linewidth=1.5)
                ax_t.set_xlabel('ステップ')
                ax_t.set_ylabel('位置')
                ax_t.set_title(f'レース展開 #{trial+1}')
                ax_t.legend(fontsize=7, ncol=3)
                plt.tight_layout()
                st.pyplot(fig_t)
                plt.close()

        st.markdown(f"### 📊 モンテカルロシミュレーション ({n_sims}回)")
        st.write(f"**レース場:** {selected_venue}　**季節:** {season}　"
                 f"**1コース場別1着率:** {profile['course_win_rate'][0]}%")

        mc_sim = RaceSimulator(agents, conditions, selected_venue, race_month)
        win_counts = {a.lane: 0 for a in agents}
        top2_counts = {a.lane: 0 for a in agents}
        top3_counts = {a.lane: 0 for a in agents}
        kim_counts = {}

        mc_bar = st.progress(0)
        mc_step = max(1, n_sims // 20)
        for i in range(n_sims):
            res = mc_sim.simulate_race()
            o = res['finish_order']
            win_counts[o[0]] += 1
            top2_counts[o[0]] += 1
            top2_counts[o[1]] += 1
            for l_idx in o[:3]:
                top3_counts[l_idx] += 1
            k = mc_sim.determine_kimarite(o, res['start_timings'])
            kim_counts[k] = kim_counts.get(k, 0) + 1
            if (i + 1) % mc_step == 0:
                mc_bar.progress((i + 1) / n_sims)
        mc_bar.progress(1.0)

        mc_rows = []
        for a in agents:
            mc_rows.append({
                '枠': f"{a.lane}号艇", '選手名': a.name,
                '1着率': f"{win_counts[a.lane]/n_sims*100:.1f}%",
                '2連対率': f"{top2_counts[a.lane]/n_sims*100:.1f}%",
                '3連対率': f"{top3_counts[a.lane]/n_sims*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(mc_rows), use_container_width=True)

        fig_mc, axes_mc = plt.subplots(1, 3, figsize=(14, 4))
        labels = [f"{a.lane}号艇\n{a.name}" for a in agents]
        colors_mc = ['#e74c3c', '#000000', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6']
        for ax, data, title in zip(axes_mc,
                [[win_counts[a.lane]/n_sims*100 for a in agents],
                 [top2_counts[a.lane]/n_sims*100 for a in agents],
                 [top3_counts[a.lane]/n_sims*100 for a in agents]],
                ['1着率 (%)', '2連対率 (%)', '3連対率 (%)']):
            ax.bar(labels, data, color=colors_mc)
            ax.set_title(title)
            for j, v in enumerate(data):
                ax.text(j, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_mc)
        plt.close()

        if kim_counts:
            st.markdown("#### 🎯 決まり手分布")
            fig_k, ax_k = plt.subplots(figsize=(6, 3))
            k_labels = list(kim_counts.keys())
            k_vals = [kim_counts[kk]/n_sims*100 for kk in k_labels]
            ax_k.bar(k_labels, k_vals, color='#3498db')
            for j, v in enumerate(k_vals):
                ax_k.text(j, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
            ax_k.set_ylabel('%')
            ax_k.set_title('決まり手分布')
            plt.tight_layout()
            st.pyplot(fig_k)
            plt.close()

        st.session_state['mc_done'] = True

    # ----------------------------------------------------------
    # 6-E. オッズ取得 & 期待値計算
    # ----------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
                padding: 20px; border-radius: 15px;'>
        <h2 style='color: #e94560; text-align: center;'>🎰 オッズ分析 & 期待値計算</h2>
        <p style='color: #eee; text-align: center;'>
            3連単オッズを取得 → 全券種の合成オッズ → シミュレーション確率 × オッズ = 期待値
        </p>
    </div>
    """, unsafe_allow_html=True)

    odds_method = st.radio(
        "オッズ取得方法",
        ["🌐 自動取得 (公式サイト)", "📋 テキスト貼り付け", "✏️ 手動入力"],
        horizontal=True, key="odds_method")

    if odds_method == "🌐 自動取得 (公式サイト)":
        st.info("💡 公式サイトからオッズを自動取得します（レース締切後のオッズ）")
        col_ov, col_od, col_or = st.columns(3)
        with col_ov:
            ov = st.selectbox("会場", venue_list,
                              index=venue_list.index(selected_venue), key="ov")
        with col_od:
            od = st.date_input("日付", value=race_date, key="od")
        with col_or:
            orn = st.number_input("レース", 1, 12, race_no, key="orn")

        if st.button("🔍 オッズ自動取得", key="btn_fetch", type="primary"):
            vc = VENUE_CODES[ov]
            ds = od.strftime("%Y%m%d")
            with st.spinner(f"📡 {ov} {od} {orn}R オッズ取得中..."):
                tri_odds = fetch_trifecta_odds(vc, ds, orn)
            if tri_odds and len(tri_odds) >= 60:
                st.success(f"✅ {len(tri_odds)}通りの3連単オッズを取得！")
                st.session_state['tri_odds'] = tri_odds
                # 検証表示
                with st.expander("📊 取得オッズ確認 (上位10件)", expanded=False):
                    sorted_odds = sorted(tri_odds.items(), key=lambda x: x[1])
                    check_rows = [{'買い目': k, 'オッズ': v} for k, v in sorted_odds[:10]]
                    st.dataframe(pd.DataFrame(check_rows), use_container_width=True)
            else:
                st.warning("⚠️ 取得できませんでした。テキスト貼り付けをお試しください。")

    elif odds_method == "📋 テキスト貼り付け":
        odds_txt = st.text_area(
            "3連単オッズテキスト", height=250,
            placeholder="公式サイトからコピーしたテキスト or '1-2-3 5.0' 形式...",
            key="odds_txt")
        if st.button("🔍 オッズ解析", key="btn_parse_odds"):
            if odds_txt.strip():
                tri_odds = parse_pasted_odds(odds_txt)
                if tri_odds and len(tri_odds) >= 60:
                    st.success(f"✅ {len(tri_odds)}通り解析完了！")
                    st.session_state['tri_odds'] = tri_odds
                else:
                    st.warning(f"⚠️ {len(tri_odds)}通りのみ。フォーマット確認してください。")

    elif odds_method == "✏️ 手動入力":
        manual_odds = st.text_area(
            "買い目とオッズ (1行1組)", height=200,
            placeholder="1-2-3 5.0\n1-3-2 8.5\n...", key="manual_odds")
        if st.button("📝 確定", key="btn_manual_odds"):
            tri_odds = {}
            for line in manual_odds.strip().split('\n'):
                m = re.match(r'(\d)\s*-\s*(\d)\s*-\s*(\d)\s+([\d,.]+)', line.strip())
                if m:
                    tri_odds[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"] = float(
                        m.group(4).replace(',', ''))
            if tri_odds:
                st.success(f"✅ {len(tri_odds)}通り登録！")
                st.session_state['tri_odds'] = tri_odds

    # --- 期待値計算 ---
    if 'tri_odds' in st.session_state and st.session_state['tri_odds']:
        st.markdown("---")
        st.markdown("### 💰 期待値計算")

        ev_sims = st.slider(
            "期待値シミュレーション回数",
            1000, 50000, 10000, 1000, key="ev_sims")

        if st.button("💰 期待値を計算する", key="btn_ev", type="primary"):
            tri_odds = st.session_state['tri_odds']

            with st.spinner("📐 合成オッズ計算中..."):
                synth = compute_synthetic_odds(tri_odds)

            st.write(f"🎲 モンテカルロ ({ev_sims}回)...")
            sim_probs = run_ev_simulation(
                agents, conditions, selected_venue, race_month, ev_sims)

            with st.spinner("💹 期待値算出中..."):
                ev_results = compute_expected_values(synth, sim_probs)

            st.success("✅ 計算完了！")

            ticket_names = {
                'trifecta': '3連単', 'trio': '3連複', 'exacta': '2連単',
                'quinella': '2連複', 'wide': '拡連複'
            }

            tabs = st.tabs([f"🎯 {n}" for n in ticket_names.values()])

            for (tt, tname), tab in zip(ticket_names.items(), tabs):
                with tab:
                    ev_data = ev_results[tt]
                    rows = []
                    for key, info in ev_data.items():
                        if info['odds'] > 0:
                            rows.append({
                                '買い目': key,
                                'オッズ': info['odds'],
                                '的中確率': f"{info['probability']*100:.2f}%",
                                '期待値': info['expected_value'],
                                '判定': '🟢 買い' if info['profitable'] else '🔴'
                            })
                    if rows:
                        df = pd.DataFrame(rows).sort_values('期待値', ascending=False)
                        prof_n = sum(1 for r in rows if r['判定'] == '🟢 買い')

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("総買い目", len(rows))
                        with c2:
                            st.metric("プラス期待値", f"{prof_n}件")
                        with c3:
                            st.metric("最大EV", f"{max(r['期待値'] for r in rows):.2f}")

                        st.markdown(f"#### 📈 {tname} 期待値ランキング Top20")
                        disp = df.head(20).reset_index(drop=True)
                        disp.index = disp.index + 1

                        def hl(row):
                            if row['期待値'] > 1.0:
                                return ['background-color: #d4edda'] * len(row)
                            elif row['期待値'] > 0.8:
                                return ['background-color: #fff3cd'] * len(row)
                            return [''] * len(row)

                        st.dataframe(
                            disp.style.apply(hl, axis=1),
                            use_container_width=True)

                        top15 = df.head(15)
                        fig_e, ax_e = plt.subplots(figsize=(10, 5))
                        c_ev = ['#27ae60' if v > 1.0 else '#f39c12' if v > 0.5 else '#e74c3c'
                                for v in top15['期待値']]
                        ax_e.barh(range(len(top15)), top15['期待値'].values, color=c_ev)
                        ax_e.set_yticks(range(len(top15)))
                        ax_e.set_yticklabels(top15['買い目'].values)
                        ax_e.axvline(x=1.0, color='red', linestyle='--',
                                     linewidth=2, label='EV=1.0')
                        ax_e.set_xlabel('期待値')
                        ax_e.set_title(f'{tname} 期待値 Top15')
                        ax_e.legend()
                        ax_e.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig_e)
                        plt.close()

            # おすすめ買い目
            st.markdown("---")
            st.markdown("### 🏆 おすすめ買い目 (期待値 > 1.0)")
            all_prof = []
            for tt, tname in ticket_names.items():
                for key, info in ev_results[tt].items():
                    if info['profitable'] and info['odds'] > 0:
                        all_prof.append({
                            '券種': tname, '買い目': key,
                            'オッズ': info['odds'],
                            '的中確率': f"{info['probability']*100:.2f}%",
                            '期待値': info['expected_value']
                        })

            if all_prof:
                pdf = pd.DataFrame(all_prof).sort_values('期待値', ascending=False)
                st.dataframe(pdf, use_container_width=True)

                st.markdown("#### 💡 買い目ガイド")
                for tname in ticket_names.values():
                    subset = [r for r in all_prof if r['券種'] == tname]
                    if subset:
                        top3_items = sorted(subset, key=lambda x: x['期待値'], reverse=True)[:3]
                        recs = " / ".join(
                            [f"**{r['買い目']}** (EV={r['期待値']:.2f})" for r in top3_items])
                        st.write(f"**{tname}**: {recs}")
            else:
                st.info("期待値 > 1.0 の買い目はありません。0.8以上も参考にしてください。")
                near = []
                for tt, tname in ticket_names.items():
                    for key, info in ev_results[tt].items():
                        if info['expected_value'] > 0.8 and info['odds'] > 0:
                            near.append({
                                '券種': tname, '買い目': key,
                                'オッズ': info['odds'],
                                '的中確率': f"{info['probability']*100:.2f}%",
                                '期待値': info['expected_value']
                            })
                if near:
                    st.markdown("#### 📊 準おすすめ (EV > 0.8)")
                    st.dataframe(
                        pd.DataFrame(near).sort_values('期待値', ascending=False).head(20),
                        use_container_width=True)

            st.caption(
                "⚠️ 期待値はシミュレーション推定値です。"
                "実際の結果を保証しません。投票は自己責任で。")

# ----------------------------------------------------------
# フッター
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    🚤 ボートレース AI シミュレーター v3.1 ｜ 場別プロファイル対応版<br>
    全24場のコース別成績・決まり手・水面特性を反映 ｜ オッズ自動取得修正済
</div>
""", unsafe_allow_html=True)
