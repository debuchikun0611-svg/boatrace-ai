# ============================================================
#  ボートレース AI シミュレーター v4.0  ─ app.py
#  完全エージェント版（30項目対応）
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import itertools
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from urllib.request import urlopen, Request

# ─────────────────────────────────────────────
# 1. 日本語フォント設定
# ─────────────────────────────────────────────
def setup_japanese_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/ipa-gothic/ipag.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            from matplotlib.font_manager import FontProperties
            fp = FontProperties(fname=p)
            matplotlib.rcParams['font.family'] = fp.get_name()
            break
    else:
        matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False

setup_japanese_font()

# ─────────────────────────────────────────────
# 2. 会場プロフィール（全24場）
# ─────────────────────────────────────────────
VENUE_PROFILES = {
    "桐生":   {"code":"01","water":"淡水","tide":False,
               "course_win_rate":[54.4,14.4,12.4,11.3,5.4,2.3],
               "course_top2":[73.0,30.0,27.0,25.0,18.0,10.0],
               "course_top3":[84.0,44.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.14,"捲り差し":0.12,"抜き":0.04,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"冬は赤城おろし（北風）"},
    "戸田":   {"code":"02","water":"淡水","tide":False,
               "course_win_rate":[44.0,17.2,13.5,13.8,7.5,4.2],
               "course_top2":[63.0,33.0,28.0,28.0,22.0,14.0],
               "course_top3":[76.0,47.0,42.0,42.0,36.0,26.0],
               "kimarite":{"逃げ":0.42,"差し":0.18,"捲り":0.16,"捲り差し":0.15,"抜き":0.07,"恵まれ":0.02},
               "wind_effect":0.4,"memo":"狭い水面、捲り有利"},
    "江戸川": {"code":"03","water":"汽水","tide":True,
               "course_win_rate":[49.5,15.2,12.8,12.5,6.8,3.5],
               "course_top2":[69.0,31.0,27.0,26.0,20.0,12.0],
               "course_top3":[81.0,45.0,41.0,40.0,34.0,24.0],
               "kimarite":{"逃げ":0.47,"差し":0.16,"捲り":0.15,"捲り差し":0.14,"抜き":0.06,"恵まれ":0.02},
               "wind_effect":0.7,"memo":"荒水面、風・潮影響大"},
    "平和島": {"code":"04","water":"汽水","tide":True,
               "course_win_rate":[49.8,16.0,13.2,12.0,5.8,3.5],
               "course_top2":[68.0,32.0,28.0,26.0,19.0,12.0],
               "course_top3":[80.0,46.0,42.0,40.0,33.0,23.0],
               "kimarite":{"逃げ":0.48,"差し":0.17,"捲り":0.15,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.02},
               "wind_effect":0.6,"memo":"ビル風の影響"},
    "多摩川": {"code":"05","water":"淡水","tide":False,
               "course_win_rate":[52.9,15.0,13.0,11.5,5.2,2.5],
               "course_top2":[72.0,31.0,27.0,25.0,18.0,10.0],
               "course_top3":[83.0,45.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.52,"差し":0.16,"捲り":0.14,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.3,"memo":"穏やかな水面"},
    "浜名湖": {"code":"06","water":"汽水","tide":True,
               "course_win_rate":[50.9,15.8,12.8,12.2,5.6,3.0],
               "course_top2":[70.0,32.0,27.0,26.0,19.0,11.0],
               "course_top3":[82.0,46.0,41.0,39.0,33.0,23.0],
               "kimarite":{"逃げ":0.50,"差し":0.16,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.02},
               "wind_effect":0.5,"memo":"西風が強い日あり"},
    "蒲郡":   {"code":"07","water":"汽水","tide":True,
               "course_win_rate":[54.4,14.5,12.5,11.0,5.3,2.5],
               "course_top2":[73.0,30.0,27.0,25.0,18.0,10.0],
               "course_top3":[84.0,44.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.4,"memo":"静水面、イン有利"},
    "常滑":   {"code":"08","water":"海水","tide":True,
               "course_win_rate":[57.8,14.0,11.5,10.0,4.8,2.2],
               "course_top2":[75.0,29.0,26.0,23.0,17.0,9.0],
               "course_top3":[86.0,43.0,40.0,37.0,31.0,21.0],
               "kimarite":{"逃げ":0.57,"差し":0.14,"捲り":0.12,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"向かい風でイン不利"},
    "津":     {"code":"09","water":"海水","tide":True,
               "course_win_rate":[56.5,14.2,12.0,10.5,4.8,2.2],
               "course_top2":[74.0,30.0,26.0,24.0,17.0,9.0],
               "course_top3":[85.0,44.0,40.0,37.0,31.0,21.0],
               "kimarite":{"逃げ":0.56,"差し":0.14,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"伊勢湾の潮影響"},
    "三国":   {"code":"10","water":"淡水","tide":False,
               "course_win_rate":[53.5,14.8,12.5,11.5,5.2,2.5],
               "course_top2":[72.0,31.0,27.0,25.0,18.0,10.0],
               "course_top3":[83.0,45.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.52,"差し":0.15,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.6,"memo":"冬の北風が強い"},
    "びわこ": {"code":"11","water":"淡水","tide":False,
               "course_win_rate":[52.0,15.5,13.0,11.5,5.5,2.8],
               "course_top2":[71.0,31.0,27.0,25.0,18.0,10.0],
               "course_top3":[83.0,45.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.51,"差し":0.16,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"比良おろし"},
    "住之江": {"code":"12","water":"淡水","tide":False,
               "course_win_rate":[55.0,14.5,12.0,11.0,5.0,2.5],
               "course_top2":[73.0,30.0,26.0,25.0,18.0,10.0],
               "course_top3":[85.0,44.0,40.0,38.0,31.0,21.0],
               "kimarite":{"逃げ":0.55,"差し":0.15,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.3,"memo":"ナイター、静水面"},
    "尼崎":   {"code":"13","water":"海水","tide":True,
               "course_win_rate":[58.5,13.5,11.5,10.0,4.5,2.2],
               "course_top2":[76.0,29.0,25.0,23.0,17.0,9.0],
               "course_top3":[87.0,43.0,39.0,36.0,30.0,20.0],
               "kimarite":{"逃げ":0.58,"差し":0.14,"捲り":0.12,"捲り差し":0.10,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.4,"memo":"センタープール"},
    "鳴門":   {"code":"14","water":"海水","tide":True,
               "course_win_rate":[53.0,15.0,12.5,11.5,5.5,2.8],
               "course_top2":[72.0,31.0,27.0,25.0,18.0,10.0],
               "course_top3":[83.0,45.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.52,"差し":0.15,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.6,"memo":"潮の干満差大"},
    "丸亀":   {"code":"15","water":"海水","tide":True,
               "course_win_rate":[55.5,14.5,12.0,10.5,5.0,2.5],
               "course_top2":[74.0,30.0,26.0,24.0,17.0,9.0],
               "course_top3":[85.0,44.0,40.0,37.0,31.0,21.0],
               "kimarite":{"逃げ":0.55,"差し":0.14,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"向い風で荒れやすい"},
    "児島":   {"code":"16","water":"海水","tide":True,
               "course_win_rate":[54.0,14.8,12.5,11.0,5.2,2.5],
               "course_top2":[73.0,30.0,27.0,25.0,18.0,10.0],
               "course_top3":[84.0,44.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.53,"差し":0.15,"捲り":0.14,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"瀬戸内海、潮流"},
    "宮島":   {"code":"17","water":"海水","tide":True,
               "course_win_rate":[54.5,14.5,12.5,11.0,5.0,2.5],
               "course_top2":[73.0,30.0,27.0,25.0,18.0,10.0],
               "course_top3":[84.0,44.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"潮の影響大"},
    "徳山":   {"code":"18","water":"海水","tide":True,
               "course_win_rate":[63.4,11.7,12.8,9.7,3.5,1.1],
               "course_top2":[80.4,30.5,20.0,17.5,12.2,5.2],
               "course_top3":[87.5,47.7,40.2,39.1,29.2,22.0],
               "kimarite":{"逃げ":0.63,"差し":0.12,"捲り":0.10,"捲り差し":0.09,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.6,
               "seasonal":{"春":[64.6,14.6,9.7,7.3,3.8,0.5],
                           "夏":[66.1,11.9,10.7,7.4,4.2,0.5],
                           "秋":[61.1,10.5,12.1,11.6,3.2,1.9],
                           "冬":[62.9,14.0,8.4,8.0,5.7,1.6]},
               "memo":"追い風安定、イン最強、BS広い"},
    "下関":   {"code":"19","water":"海水","tide":True,
               "course_win_rate":[56.0,14.5,12.0,10.5,4.8,2.2],
               "course_top2":[74.0,30.0,26.0,24.0,17.0,9.0],
               "course_top3":[85.0,44.0,40.0,37.0,31.0,21.0],
               "kimarite":{"逃げ":0.55,"差し":0.14,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"ナイター、海峡の風"},
    "若松":   {"code":"20","water":"海水","tide":True,
               "course_win_rate":[55.5,14.8,12.0,10.5,5.0,2.5],
               "course_top2":[74.0,30.0,26.0,24.0,17.0,9.0],
               "course_top3":[85.0,44.0,40.0,37.0,31.0,21.0],
               "kimarite":{"逃げ":0.55,"差し":0.15,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"潮の干満差あり"},
    "芦屋":   {"code":"21","water":"淡水","tide":False,
               "course_win_rate":[60.0,13.0,11.0,9.5,4.5,2.0],
               "course_top2":[78.0,28.0,25.0,22.0,16.0,8.0],
               "course_top3":[87.0,42.0,39.0,36.0,30.0,20.0],
               "kimarite":{"逃げ":0.60,"差し":0.13,"捲り":0.11,"捲り差し":0.10,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.3,"memo":"モーニング、イン有利"},
    "福岡":   {"code":"22","water":"汽水","tide":True,
               "course_win_rate":[52.0,15.5,13.0,11.5,5.5,2.8],
               "course_top2":[71.0,31.0,27.0,25.0,18.0,10.0],
               "course_top3":[83.0,45.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.51,"差し":0.16,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.6,"memo":"那珂川河口、うねり"},
    "唐津":   {"code":"23","water":"海水","tide":True,
               "course_win_rate":[56.0,14.5,12.0,10.5,5.0,2.2],
               "course_top2":[74.0,30.0,26.0,24.0,17.0,9.0],
               "course_top3":[85.0,44.0,40.0,37.0,31.0,21.0],
               "kimarite":{"逃げ":0.55,"差し":0.14,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"モーニング、追い風多い"},
    "大村":   {"code":"24","water":"海水","tide":True,
               "course_win_rate":[62.0,12.0,11.5,9.0,3.8,1.5],
               "course_top2":[79.0,28.0,25.0,22.0,16.0,8.0],
               "course_top3":[87.0,42.0,39.0,36.0,30.0,20.0],
               "kimarite":{"逃げ":0.62,"差し":0.12,"捲り":0.10,"捲り差し":0.10,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.4,"memo":"イン天国、ナイター"},
}

DEFAULT_VENUE_PROFILE = {
    "code":"00","water":"不明","tide":False,
    "course_win_rate":[55.9,14.5,12.5,11.0,4.8,2.2],
    "course_top2":[73.0,30.0,27.0,25.0,18.0,10.0],
    "course_top3":[84.0,44.0,41.0,38.0,32.0,22.0],
    "kimarite":{"逃げ":0.55,"差し":0.15,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},
    "wind_effect":0.5,"memo":"全国平均"
}

def get_venue_profile(venue_name: str) -> dict:
    return VENUE_PROFILES.get(venue_name, DEFAULT_VENUE_PROFILE)

def get_season(month: int) -> str:
    if month in [3,4,5]:   return "春"
    if month in [6,7,8]:   return "夏"
    if month in [9,10,11]: return "秋"
    return "冬"

# ─────────────────────────────────────────────
# 3. 完全版データクラス（30項目）
# ─────────────────────────────────────────────
@dataclass
class BoatAgent:
    """選手エージェント ─ 公式データから取得可能な全項目を保持"""
    # === 基本情報 ===
    lane: int                          # 枠番 (1-6)
    number: int = 0                    # 登録番号
    name: str = ""                     # 選手名
    rank: str = "B1"                   # 級別 (A1/A2/B1/B2)
    age: int = 30                      # 年齢
    weight: float = 52.0               # 体重(kg)
    branch: str = ""                   # 支部

    # === 成績（直近6ヶ月）===
    avg_st: float = 0.18               # 平均ST
    win_rate: float = 5.0              # 勝率
    top2_rate: float = 30.0            # 2連対率(%)
    top3_rate: float = 50.0            # 3連対率(%)

    # === 枠別成績 ===
    lane_win_rate: float = 10.0        # 枠別1着率(%)
    lane_top2_rate: float = 30.0       # 枠別2連対率(%)
    lane_top3_rate: float = 50.0       # 枠別3連対率(%)
    lane_avg_st: float = 0.18          # 枠別平均ST

    # === 能力・モーター ===
    ability: int = 50                  # 能力値（今期）
    motor_contribution: float = 0.0    # モーター貢献P（通算）
    motor_top2_rate: float = 30.0      # モーター2連対率(%)
    boat_top2_rate: float = 30.0       # ボート2連対率(%)

    # === 直前情報（展示） ===
    exhibition_time: float = 0.0       # 展示タイム(秒) 例: 6.90
    lap_time: float = 0.0             # 周回タイム(秒) 例: 37.35
    turn_time: float = 0.0            # 周り足タイム(秒) 例: 11.57
    straight_time: float = 0.0        # 直線タイム(秒)
    start_exhibition: float = 0.0     # スタート展示ST

    # === 直前情報（機材） ===
    tilt: float = -0.5                 # チルト角度
    adjusted_weight: float = 0.0       # 調整重量(kg)

    # === リスク要因 ===
    flying_count: int = 0              # フライング数（今期）
    accident_rate: float = 0.0         # 事故率

    # === 決まり手傾向（直近6ヶ月）===
    nige_count: int = 0                # 逃げ回数
    sashi_count: int = 0               # 差し回数
    makuri_count: int = 0              # 捲り回数
    makurisashi_count: int = 0         # 捲り差し回数

    def calculate_start_timing(self) -> float:
        """スタートタイミングを生成（展示STも考慮）"""
        base = self.avg_st
        # 展示STが取れていればそちらを重視
        if self.start_exhibition > 0:
            base = base * 0.4 + self.start_exhibition * 0.6
        variation = np.random.normal(0, 0.02)
        return max(0.01, base + variation)

    def get_power_score(self) -> float:
        """選手の総合パワースコア（0.0〜1.0）"""
        base = self.ability / 100.0
        motor = self.motor_contribution * 0.1
        rank_bonus = {"A1":0.08, "A2":0.04, "B1":0.0, "B2":-0.04}.get(self.rank, 0)
        return np.clip(base + motor + rank_bonus, 0.1, 1.0)

    def get_machine_score(self) -> float:
        """機力スコア（展示タイム・周回・周り足から算出）"""
        scores = []
        # 展示タイム（6.80が最速級、7.10が遅い）
        if self.exhibition_time > 0:
            et_score = np.clip((7.10 - self.exhibition_time) / 0.30, 0, 1)
            scores.append(et_score * 0.40)  # 重み40%
        # 周回タイム（36.0が最速級、39.0が遅い）
        if self.lap_time > 0:
            lt_score = np.clip((39.0 - self.lap_time) / 3.0, 0, 1)
            scores.append(lt_score * 0.30)  # 重み30%
        # 周り足（11.0が最速級、12.5が遅い）
        if self.turn_time > 0:
            tt_score = np.clip((12.5 - self.turn_time) / 1.5, 0, 1)
            scores.append(tt_score * 0.30)  # 重み30%

        if scores:
            return np.clip(sum(scores) / (0.40 + 0.30 + 0.30) * max(len(scores), 1), 0, 1)
        return 0.5  # データなし → 平均

    def get_weight_factor(self) -> float:
        """体重による補正（軽いほど有利、ただし荒水面は重い方が安定）"""
        # 基準: 52kg、1kgあたり約0.5%の影響
        diff = self.weight - 52.0
        return 1.0 - diff * 0.005

    def get_kimarite_tendency(self) -> dict:
        """決まり手傾向（正規化）"""
        total = (self.nige_count + self.sashi_count +
                 self.makuri_count + self.makurisashi_count)
        if total == 0:
            return {"逃げ":0.25, "差し":0.25, "捲り":0.25, "捲り差し":0.25}
        return {
            "逃げ": self.nige_count / total,
            "差し": self.sashi_count / total,
            "捲り": self.makuri_count / total,
            "捲り差し": self.makurisashi_count / total
        }


@dataclass
class RaceCondition:
    weather: str = "晴"
    temperature: float = 20.0
    wind_speed: float = 2.0
    wind_direction: str = "左横"
    water_temp: float = 20.0
    wave_height: float = 2.0
    tide: str = "満潮"
# ─────────────────────────────────────────────
# 4. 完全版シミュレーター
# ─────────────────────────────────────────────
class RaceSimulator:
    """
    全30項目を活用した完全版シミュレーター
    重み計算の内訳:
      - 会場コース勝率ベース     (20%)
      - 枠別1着率               (15%)
      - 選手勝率                (10%)
      - 連対率補正              (5%)
      - パワースコア(能力+級+モーター) (15%)
      - 機力スコア(展示+周回+周り足)  (15%)
      - ST品質                  (8%)
      - 体重補正                (3%)
      - チルト補正              (2%)
      - 風・潮補正              (4%)
      - フライングリスク         (1.5%)
      - 決まり手傾向補正         (1.5%)
    """

    def __init__(self, agents: List[BoatAgent], conditions: RaceCondition,
                 venue_name: str = "徳山", race_month: int = 2):
        self.agents = agents
        self.conditions = conditions
        self.venue_name = venue_name
        self.race_month = race_month
        self.profile = get_venue_profile(venue_name)

    def _compute_race_weights(self) -> List[float]:
        profile = self.profile
        season = get_season(self.race_month)

        if "seasonal" in profile and season in profile["seasonal"]:
            base_rates = profile["seasonal"][season]
        else:
            base_rates = profile["course_win_rate"]

        # 全艇の展示タイムから相対評価用の統計を取得
        ex_times = [a.exhibition_time for a in self.agents if a.exhibition_time > 0]
        ex_mean = np.mean(ex_times) if ex_times else 0
        ex_std = np.std(ex_times) if len(ex_times) > 1 else 0.05

        lap_times = [a.lap_time for a in self.agents if a.lap_time > 0]
        lap_mean = np.mean(lap_times) if lap_times else 0
        lap_std = np.std(lap_times) if len(lap_times) > 1 else 0.3

        turn_times = [a.turn_time for a in self.agents if a.turn_time > 0]
        turn_mean = np.mean(turn_times) if turn_times else 0
        turn_std = np.std(turn_times) if len(turn_times) > 1 else 0.15

        weights = []
        for agent in self.agents:
            idx = agent.lane - 1

            # ── 1. 会場コース勝率ベース (20%) ──
            venue_base = base_rates[idx] / 100.0 if idx < len(base_rates) else 0.05
            w = venue_base * 0.20

            # ── 2. 枠別1着率 (15%) ──
            lane_wr = agent.lane_win_rate / 100.0 if agent.lane_win_rate > 0 else venue_base
            w += lane_wr * 0.15

            # ── 3. 選手勝率 (10%) ──
            # 勝率は着順点÷出走数（最大10程度）。5.0を基準に正規化
            wr_factor = agent.win_rate / 8.0  # 8.0を上限目安
            w += np.clip(wr_factor, 0.01, 1.0) * 0.10

            # ── 4. 連対率補正 (5%) ──
            top2_factor = agent.top2_rate / 100.0
            top3_factor = agent.top3_rate / 100.0
            w += (top2_factor * 0.6 + top3_factor * 0.4) * 0.05

            # ── 5. パワースコア：能力値＋級別＋モーター貢献 (15%) ──
            power = agent.get_power_score()
            w += power * 0.15

            # ── 6. 機力スコア：展示タイム・周回・周り足 (15%) ──
            # 相対評価: 各タイムがレース内で上位なら加点
            machine = 0.5  # ベース（データなし時）
            machine_data_count = 0

            if agent.exhibition_time > 0 and ex_mean > 0 and ex_std > 0:
                # 速い=小さい値 → 差がマイナスなら良い
                ex_z = (ex_mean - agent.exhibition_time) / max(ex_std, 0.01)
                ex_relative = np.clip(0.5 + ex_z * 0.2, 0.0, 1.0)
                machine += ex_relative * 0.40
                machine_data_count += 0.40

            if agent.lap_time > 0 and lap_mean > 0 and lap_std > 0:
                lap_z = (lap_mean - agent.lap_time) / max(lap_std, 0.1)
                lap_relative = np.clip(0.5 + lap_z * 0.2, 0.0, 1.0)
                machine += lap_relative * 0.30
                machine_data_count += 0.30

            if agent.turn_time > 0 and turn_mean > 0 and turn_std > 0:
                turn_z = (turn_mean - agent.turn_time) / max(turn_std, 0.05)
                turn_relative = np.clip(0.5 + turn_z * 0.2, 0.0, 1.0)
                machine += turn_relative * 0.30
                machine_data_count += 0.30

            if machine_data_count > 0:
                machine_score = (machine - 0.5) / machine_data_count + 0.5
            else:
                machine_score = agent.get_machine_score()

            w += np.clip(machine_score, 0.1, 1.0) * 0.15

            # ── 7. ST品質 (8%) ──
            # 平均ST 0.12が最速級、0.25が遅い
            st_quality = np.clip(1.0 - (agent.avg_st - 0.12) * 4.0, 0.2, 1.0)
            w += st_quality * 0.08

            # ── 8. 体重補正 (3%) ──
            weight_factor = agent.get_weight_factor()
            # 荒れ水面では重い方が安定
            if self.conditions.wave_height >= 5:
                weight_factor = 1.0 + (agent.weight - 52.0) * 0.002
            w *= np.clip(weight_factor, 0.90, 1.10)

            # ── 9. チルト補正 (2%) ──
            # マイナスチルト → 出足型（イン有利）
            # プラスチルト → 伸び型（アウト有利）
            if agent.tilt <= -0.5 and agent.lane <= 2:
                w *= 1.02
            elif agent.tilt >= 0.5 and agent.lane >= 4:
                w *= 1.02
            elif agent.tilt >= 1.0 and agent.lane >= 5:
                w *= 1.04

            # ── 10. 風・潮補正 (4%) ──
            wind_spd = self.conditions.wind_speed
            wind_eff = profile.get("wind_effect", 0.5)
            if agent.lane <= 2:
                w *= (1.0 - wind_spd * 0.012 * wind_eff)
            elif agent.lane >= 5:
                w *= (1.0 + wind_spd * 0.006 * wind_eff)

            if profile.get("tide", False):
                if self.conditions.tide == "満潮" and agent.lane <= 2:
                    w *= 1.03
                elif self.conditions.tide == "干潮" and agent.lane >= 4:
                    w *= 1.02

            # ── 11. フライングリスク (1.5%) ──
            if agent.flying_count >= 1:
                # F持ちはスタートが慎重になる → 出遅れやすい
                w *= (1.0 - agent.flying_count * 0.05)

            # ── 12. 事故率補正 ──
            if agent.accident_rate > 0:
                w *= (1.0 - agent.accident_rate * 0.02)

            weights.append(max(w, 0.001))

        total = sum(weights)
        return [wt / total for wt in weights]

    def simulate_race(self) -> dict:
        # スタートタイミング
        st_times = {}
        for agent in self.agents:
            st_times[agent.lane] = agent.calculate_start_timing()

        # レース重み
        weights = self._compute_race_weights()
        adjusted = list(weights)

        # ── ST ボーナス（最速スタートに追加ボーナス）──
        min_st = min(st_times.values())
        for i, agent in enumerate(self.agents):
            st_diff = st_times[agent.lane] - min_st
            bonus = max(0, (0.05 - st_diff) * 2.5)
            adjusted[i] += bonus

        # ── 決まり手傾向ボーナス ──
        kimarite_probs = self.profile.get("kimarite", {})

        # 捲り発生判定
        if np.random.random() < kimarite_probs.get("捲り", 0.13):
            for i, agent in enumerate(self.agents):
                if agent.lane >= 3:
                    # 選手自身の捲り傾向も加味
                    tendency = agent.get_kimarite_tendency()
                    makuri_bonus = 1.0 + tendency.get("捲り", 0.25) * 0.3
                    adjusted[i] *= makuri_bonus
                    # 展示タイム1位なら捲り成功率UP
                    if agent.exhibition_time > 0:
                        ex_times_sorted = sorted(
                            [a.exhibition_time for a in self.agents if a.exhibition_time > 0]
                        )
                        if ex_times_sorted and agent.exhibition_time == ex_times_sorted[0]:
                            adjusted[i] *= 1.10

        # 差し発生判定
        if np.random.random() < kimarite_probs.get("差し", 0.15):
            for i, agent in enumerate(self.agents):
                if agent.lane == 2 or agent.lane == 3:
                    tendency = agent.get_kimarite_tendency()
                    sashi_bonus = 1.0 + tendency.get("差し", 0.25) * 0.3
                    adjusted[i] *= sashi_bonus

        # ── 展示タイム1位ボーナス（全体）──
        ex_times_valid = [(i, a.exhibition_time) for i, a in enumerate(self.agents)
                          if a.exhibition_time > 0]
        if ex_times_valid:
            best_idx = min(ex_times_valid, key=lambda x: x[1])[0]
            adjusted[best_idx] *= 1.06  # 展示タイム1位は1着率25%UP相当

            # 展示タイム最下位にペナルティ
            worst_idx = max(ex_times_valid, key=lambda x: x[1])[0]
            adjusted[worst_idx] *= 0.94

        # ── 周り足（ターン力）ボーナス ──
        turn_valid = [(i, a.turn_time) for i, a in enumerate(self.agents)
                      if a.turn_time > 0]
        if turn_valid:
            best_turn_idx = min(turn_valid, key=lambda x: x[1])[0]
            adjusted[best_turn_idx] *= 1.04

        # ── ランダム要素 ──
        for i in range(len(adjusted)):
            adjusted[i] *= np.random.uniform(0.85, 1.15)
            adjusted[i] = max(adjusted[i], 0.001)

        total = sum(adjusted)
        probs = [a / total for a in adjusted]

        # ── 着順決定 ──
        remaining = list(range(len(self.agents)))
        finish_order = []
        current_probs = list(probs)

        for _ in range(len(self.agents)):
            p = [current_probs[j] for j in remaining]
            p_sum = sum(p)
            if p_sum <= 0:
                finish_order.extend(remaining)
                break
            p_norm = [x / p_sum for x in p]
            chosen_idx = np.random.choice(len(remaining), p=p_norm)
            chosen = remaining[chosen_idx]
            finish_order.append(chosen)
            remaining.remove(chosen)

        result = {}
        for pos, agent_idx in enumerate(finish_order):
            result[pos + 1] = self.agents[agent_idx].lane

        kimarite = self._determine_kimarite(result[1], st_times)

        # ── ポジション履歴 ──
        positions = {agent.lane: [] for agent in self.agents}
        n_steps = 300
        current_pos = {agent.lane: float(agent.lane) for agent in self.agents}
        for step in range(n_steps):
            progress = step / n_steps
            for agent_idx, agent in enumerate(self.agents):
                target = float(finish_order.index(agent_idx) + 1)
                speed = 0.01 + progress * 0.04
                noise = np.random.normal(0, 0.1 * (1 - progress))
                current_pos[agent.lane] += (target - current_pos[agent.lane]) * speed + noise
                current_pos[agent.lane] = np.clip(current_pos[agent.lane], 0.5, 6.5)
                positions[agent.lane].append(current_pos[agent.lane])

        return {
            "finish_order": result,
            "st_times": st_times,
            "kimarite": kimarite,
            "positions": positions,
            "weights": weights
        }

    def _determine_kimarite(self, winner_lane: int, st_times: dict) -> str:
        winner_agent = None
        for a in self.agents:
            if a.lane == winner_lane:
                winner_agent = a
                break

        if winner_lane == 1:
            return "逃げ"

        # 選手の決まり手傾向を参照
        if winner_agent:
            tendency = winner_agent.get_kimarite_tendency()
        else:
            tendency = {"逃げ":0.25,"差し":0.25,"捲り":0.25,"捲り差し":0.25}

        if winner_lane == 2:
            p_sashi = 0.50 + tendency.get("差し", 0.25) * 0.3
            p_makuri = 1.0 - p_sashi
            return np.random.choice(["差し", "捲り"], p=[p_sashi, p_makuri])
        elif winner_lane == 3:
            return np.random.choice(
                ["捲り", "捲り差し", "差し"],
                p=[0.35 + tendency.get("捲り",0.25)*0.2,
                   0.35 + tendency.get("捲り差し",0.25)*0.2,
                   max(0.05, 0.30 - tendency.get("捲り",0.25)*0.2 - tendency.get("捲り差し",0.25)*0.2)]
            )
        elif winner_lane == 4:
            return np.random.choice(
                ["捲り", "捲り差し", "差し"],
                p=[0.40 + tendency.get("捲り",0.25)*0.15,
                   0.35 + tendency.get("捲り差し",0.25)*0.15,
                   max(0.05, 0.25 - tendency.get("捲り",0.25)*0.15 - tendency.get("捲り差し",0.25)*0.15)]
            )
        else:
            return np.random.choice(
                ["捲り", "捲り差し", "抜き"],
                p=[0.40, 0.40, 0.20]
            )


# ─────────────────────────────────────────────
# 5. 公式サイト コピペ用パーサー（完全版30項目対応）
# ─────────────────────────────────────────────
def parse_official_site_text(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
    lines = text.strip().split('\n')
    conditions = RaceCondition()

    def extract_values(line_text: str, count: int = 6):
        cleaned = line_text.replace('%', '').replace('℃', '').replace(',', '')
        parts = re.split(r'\t+|\s{2,}', cleaned.strip())
        vals = []
        for p in parts:
            p = p.strip()
            if p == '-' or p == '':
                vals.append(None)
            else:
                m = re.match(r'^([+\-]?\d+\.?\d*)$', p)
                if m:
                    vals.append(float(m.group(1)))
        return vals

    def safe_get(vals, idx, default, min_v=None, max_v=None):
        if vals and idx < len(vals) and vals[idx] is not None:
            v = vals[idx]
            if min_v is not None and v < min_v: return default
            if max_v is not None and v > max_v: return default
            return v
        return default

    # ── 登録番号 ──
    numbers = [0]*6
    for line in lines:
        nums = re.findall(r'\b(\d{4})\b', line)
        if len(nums) == 6:
            cands = [int(n) for n in nums]
            if all(1000 <= n <= 5999 for n in cands):
                numbers = cands
                break

    # ── 選手名 ──
    names = [f"選手{i+1}" for i in range(6)]
    exclude_words = {
        '号艇','基本情報','枠別情報','モータ情報','直前情報','今節成績',
        '選手情報','天候状況','決まり手','平均','直近','一般戦','コース変更',
        '狙いトク','日本財団','会長杯','争奪戦','初日','最終日','ナイター',
        '出走表','オッズ','レース','メニュー','お気入り','進入','展示',
        '変更','クリア','コメント','情報','検索','一覧','結果','出目',
        'ランク','天気','風向','風速','波高','水温','気温','部品交換',
        'プロペラ','チルト','調整重量','体重','周回','周り足','直線',
        '安定率','抜出率','出遅率','通算','今期','前期','未消化',
        '逃げ','差し','捲り','捲差','抜き','恵まれ','優勝','優出',
        '準優','フライング','能力値','事故率','事故点','勝率','連対率',
        '着率','決り手','徳山','朝トク','予選','狙い','締切','得点',
        '順位','進入順','画像','クリック','更新','最新'
    }
    for line in lines:
        cands = re.findall(r'[一-龥ぁ-んァ-ヶー]{2,}', line)
        filtered = [n for n in cands if n not in exclude_words and len(n) >= 2]
        if len(filtered) == 6:
            names = filtered
            break

    # ── 級別 ──
    ranks = ["B1"]*6
    for line in lines:
        rc = re.findall(r'\b([AB][12])\b', line)
        if len(rc) == 6:
            ranks = rc
            break

    # ── 年齢 ──
    ages = [30]*6
    for line in lines:
        if '歳' in line:
            ac = re.findall(r'(\d{2})歳', line)
            if len(ac) == 6:
                ages = [int(a) for a in ac]
                break

    # ── 体重 ──
    weights = [52.0]*6
    for line in lines:
        if '体重' in line or 'kg' in line:
            wc = re.findall(r'(\d{2}\.\d)', line)
            if len(wc) == 6:
                weights = [float(w) for w in wc]
                break

    # ── セクション別解析 ──
    avg_sts = [0.18]*6
    win_rates = [5.0]*6
    top2_rates = [30.0]*6
    top3_rates = [50.0]*6
    lane_win_rates = [10.0]*6
    lane_top2_rates = [30.0]*6
    lane_top3_rates = [50.0]*6
    lane_avg_sts = [0.18]*6
    abilities = [50]*6
    motors = [0.0]*6
    accident_rates = [0.0]*6
    flying_counts = [0]*6
    nige = [0]*6; sashi = [0]*6; makuri = [0]*6; makurisashi = [0]*6

    # 展示直前
    exhibition_times = [0.0]*6
    lap_times = [0.0]*6
    turn_times = [0.0]*6
    straight_times = [0.0]*6
    tilts = [-0.5]*6
    adj_weights = [0.0]*6
    start_exhibitions = [0.0]*6

        section = ""
    sub_section = ""
    found_winrate_6m = False
    found_top2_6m = False
    found_top3_6m = False
    found_lane_win_6m = False

    for line_idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue

        # ── セクション検出 ──
        if s == '基本情報':
            section = "basic"; sub_section = ""; continue
        elif s == '枠別情報':
            section = "frame"; sub_section = ""; continue
        elif 'モータ情報' in s or 'モーター情報' in s:
            section = "motor_info"; sub_section = ""; continue
        elif s == '直前情報':
            section = "beforeinfo"; sub_section = ""; continue
        elif '展示情報' in s:
            section = "exhibition"; sub_section = ""; continue
        elif '今節情報' in s:
            section = "thisnode"; sub_section = ""; continue

        # ── サブセクション検出（より厳密に）──
        if s == '平均ST' or (s.startswith('平均ST') and '総合' not in s and '枠' not in s):
            sub_section = "avg_st"; continue
        elif s.startswith('ST順位') and '総合' not in s:
            sub_section = "st_rank"; continue
        elif (s == '勝率' or s.startswith('勝率')) and '1着率' not in s and '展示' not in s and '総合' not in s:
            sub_section = "winrate"; found_winrate_6m = False; continue
        elif s.startswith('2連対率') and '総合' not in s:
            sub_section = "top2"; found_top2_6m = False; continue
        elif s.startswith('3連対率') and '総合' not in s:
            sub_section = "top3"; found_top3_6m = False; continue
        elif '1着率(総合)' in s or '1着率（総合）' in s:
            sub_section = "lane_win"; found_lane_win_6m = False; continue
        elif '2連対率(総合)' in s:
            sub_section = "lane_top2"; continue
        elif '3連対率(総合)' in s:
            sub_section = "lane_top3"; continue
        elif '平均ST(総合)' in s:
            sub_section = "lane_st"; continue
        elif s == '能力値' or s.startswith('能力値'):
            sub_section = "ability"; continue
        elif s == '決り手数' or '決まり手数' in s:
            sub_section = "kimarite_count"; continue
        elif '事故率' in s and '事故点' not in s and sub_section != "accident":
            sub_section = "accident"
        elif s == 'フライング' or s.startswith('フライング'):
            sub_section = "flying"; continue

        # ── 貢献P（独立検出）──
        if '貢献P' in s:
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    v = safe_get(vals, i, 0.0, -3.0, 3.0)
                    motors[i] = v
            continue

        # ── 直近6ヶ月データ取得 ──
        if '直近6ヶ月' in s or '直近6' in s:
            vals = extract_values(s)

            # 値が行内に足りない場合、次の行も結合して再試行
            if len(vals) < 6 and line_idx + 1 < len(lines):
                combined = s + " " + lines[line_idx + 1].strip()
                vals = extract_values(combined)

            if len(vals) >= 6:
                if sub_section == "avg_st":
                    for i in range(6):
                        avg_sts[i] = safe_get(vals, i, 0.18, 0.05, 0.35)
                elif sub_section == "winrate" and not found_winrate_6m:
                    for i in range(6):
                        win_rates[i] = safe_get(vals, i, 5.0, 0, 15)
                    found_winrate_6m = True
                elif sub_section == "top2" and not found_top2_6m:
                    for i in range(6):
                        top2_rates[i] = safe_get(vals, i, 30.0, 0, 100)
                    found_top2_6m = True
                elif sub_section == "top3" and not found_top3_6m:
                    for i in range(6):
                        top3_rates[i] = safe_get(vals, i, 50.0, 0, 100)
                    found_top3_6m = True
                elif sub_section == "lane_win" and not found_lane_win_6m:
                    for i in range(6):
                        lane_win_rates[i] = safe_get(vals, i, 10.0, 0, 100)
                    found_lane_win_6m = True
                elif sub_section == "lane_top2":
                    for i in range(6):
                        lane_top2_rates[i] = safe_get(vals, i, 30.0, 0, 100)
                elif sub_section == "lane_top3":
                    for i in range(6):
                        lane_top3_rates[i] = safe_get(vals, i, 50.0, 0, 100)
                elif sub_section == "lane_st":
                    for i in range(6):
                        lane_avg_sts[i] = safe_get(vals, i, 0.18, 0.05, 0.35)

        # ── 枠別1着率: パーセント+括弧の特殊フォーマット対応 ──
        # "38.5%\n(13)\t0.0%\n(14)..." のように値と(回数)が混在
        if sub_section == "lane_win" and not found_lane_win_6m:
            if '直近6ヶ月' in s or '直近6' in s:
                # パーセント値を直接抽出
                pct_vals = re.findall(r'([\d.]+)\s*%', s)
                if len(pct_vals) >= 6:
                    for i in range(6):
                        try:
                            lane_win_rates[i] = float(pct_vals[i])
                        except (ValueError, IndexError):
                            pass
                    found_lane_win_6m = True
                else:
                    # 次の数行も含めて探す
                    combined = s
                    for offset in range(1, 4):
                        if line_idx + offset < len(lines):
                            combined += " " + lines[line_idx + offset].strip()
                    pct_vals = re.findall(r'([\d.]+)\s*%', combined)
                    if len(pct_vals) >= 6:
                        for i in range(6):
                            try:
                                lane_win_rates[i] = float(pct_vals[i])
                            except (ValueError, IndexError):
                                pass
                        found_lane_win_6m = True

        # ── 能力値（今期）──
        if sub_section == "ability" and '今期' in s:
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    abilities[i] = int(safe_get(vals, i, 50, 1, 100))

        # ── 事故率 ──
        if sub_section == "accident" and '事故率' in s:
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    accident_rates[i] = safe_get(vals, i, 0.0, 0, 5)

        # ── フライング（今期）──
        if sub_section == "flying" and '今期' in s:
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    v = safe_get(vals, i, 0, 0, 10)
                    flying_counts[i] = int(v) if v is not None else 0

        # ── 決まり手数 ──
        if sub_section == "kimarite_count":
            if s.startswith('逃げ'):
                vals = extract_values(s)
                if len(vals) >= 6:
                    for i in range(6): nige[i] = int(safe_get(vals, i, 0, 0, 999))
            elif s.startswith('差し') and '捲' not in s:
                vals = extract_values(s)
                if len(vals) >= 6:
                    for i in range(6): sashi[i] = int(safe_get(vals, i, 0, 0, 999))
            elif s.startswith('捲差') or s.startswith('捲り差'):
                vals = extract_values(s)
                if len(vals) >= 6:
                    for i in range(6): makurisashi[i] = int(safe_get(vals, i, 0, 0, 999))
            elif s.startswith('捲り') or s.startswith('捲'):
                vals = extract_values(s)
                if len(vals) >= 6:
                    for i in range(6): makuri[i] = int(safe_get(vals, i, 0, 0, 999))

        # ── 展示タイム ──
        if (section in ["beforeinfo","exhibition"]) or '展示' in s:
            if s.startswith('展示') and '情報' not in s and '順位' not in s and 'タイム' not in s and '1位' not in s:
                vals = extract_values(s)
                if len(vals) >= 6:
                    for i in range(6):
                        v = safe_get(vals, i, 0.0, 6.0, 8.0)
                        if v > 0: exhibition_times[i] = v

        # ── 周回タイム ──
        if s.startswith('周回') and '展示' not in s and '前走' not in s and '平均' not in s:
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    v = safe_get(vals, i, 0.0, 30.0, 45.0)
                    if v > 0: lap_times[i] = v

        # ── 周り足 ──
        if s.startswith('周り足') and '前走' not in s and '平均' not in s:
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    v = safe_get(vals, i, 0.0, 10.0, 14.0)
                    if v > 0: turn_times[i] = v

        # ── チルト ──
        if s.startswith('チルト'):
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    tilts[i] = safe_get(vals, i, -0.5, -1.0, 3.5)

        # ── 調整重量 ──
        if s.startswith('調整重量'):
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    adj_weights[i] = safe_get(vals, i, 0.0, 0.0, 5.0)

        # ── 体重（直前情報内）──
        if section in ["beforeinfo","exhibition"] and s.startswith('体重'):
            vals = extract_values(s)
            if len(vals) >= 6:
                for i in range(6):
                    v = safe_get(vals, i, 52.0, 40.0, 70.0)
                    if v > 0: weights[i] = v

    # ── 天候 ──
    for line in lines:
        t_m = re.search(r'(\d+\.?\d*)\s*℃', line)
        if t_m and '水温' not in line[:max(0,line.find('℃')-5)]:
            conditions.temperature = float(t_m.group(1))
            break
    for line in lines:
        wt_m = re.search(r'水温\s*(\d+\.?\d*)', line)
        if wt_m:
            conditions.water_temp = float(wt_m.group(1))
            break
    for line in lines:
        if '風' in line:
            ws_m = re.search(r'(\d+\.?\d*)\s*m\b', line)
            if ws_m:
                conditions.wind_speed = float(ws_m.group(1))
                break
    for line in lines:
        wh_m = re.search(r'(\d+\.?\d*)\s*cm', line)
        if wh_m:
            conditions.wave_height = float(wh_m.group(1))
            break
    if '雨' in text:
        conditions.weather = "雨"
    elif '曇' in text:
        conditions.weather = "曇"

    # ── エージェント作成 ──
    agents = []
    for i in range(6):
        agents.append(BoatAgent(
            lane=i+1, number=numbers[i], name=names[i],
            rank=ranks[i], age=ages[i], weight=weights[i],
            avg_st=avg_sts[i], win_rate=win_rates[i],
            top2_rate=top2_rates[i], top3_rate=top3_rates[i],
            lane_win_rate=lane_win_rates[i],
            lane_top2_rate=lane_top2_rates[i],
            lane_top3_rate=lane_top3_rates[i],
            lane_avg_st=lane_avg_sts[i],
            ability=abilities[i], motor_contribution=motors[i],
            exhibition_time=exhibition_times[i],
            lap_time=lap_times[i], turn_time=turn_times[i],
            tilt=tilts[i], adjusted_weight=adj_weights[i],
            flying_count=flying_counts[i],
            accident_rate=accident_rates[i],
            nige_count=nige[i], sashi_count=sashi[i],
            makuri_count=makuri[i], makurisashi_count=makurisashi[i]
        ))

    return agents, conditions


# ─────────────────────────────────────────────
# 6. オッズ取得 & 合成オッズ
# ─────────────────────────────────────────────
def fetch_trifecta_odds(venue_code: str, date_str: str, race_no: int) -> dict:
    url = (f"https://www.boatrace.jp/owpc/pc/race/odds3t"
           f"?rno={race_no}&jcd={venue_code}&hd={date_str}")
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
                try: odds_vals.append(float(txt.replace(',','')))
                except ValueError: odds_vals.append(0.0)
            break
    if len(odds_vals) < 120:
        st.warning(f"⚠️ oddsPoint セルが {len(odds_vals)} 個しか見つかりません")
        return {}
    boats = [1,2,3,4,5,6]
    def get_col_order(first):
        others = sorted([b for b in boats if b != first])
        order = []
        for s in others:
            for t in sorted([b for b in others if b != s]):
                order.append((first, s, t))
        return order
    col_orders = [get_col_order(f) for f in boats]
    odds_dict = {}
    for r in range(20):
        for c in range(6):
            ci = r*6+c
            if ci < len(odds_vals):
                f,s,t = col_orders[c][r]
                odds_dict[f"{f}-{s}-{t}"] = odds_vals[ci]
    return odds_dict

def parse_pasted_odds(text: str) -> dict:
    odds_dict = {}
    p1 = re.findall(r'(\d)-(\d)-(\d)\s+([\d,.]+)', text)
    if p1:
        for m in p1:
            k = f"{m[0]}-{m[1]}-{m[2]}"
            try: odds_dict[k] = float(m[3].replace(',',''))
            except: pass
        if len(odds_dict) >= 10: return odds_dict
    nums = re.findall(r'[\d]+\.[\d]+', text)
    if len(nums) >= 120:
        boats = [1,2,3,4,5,6]
        def gco(first):
            others = sorted([b for b in boats if b != first])
            order = []
            for s in others:
                for t in sorted([b for b in others if b != s]):
                    order.append((first,s,t))
            return order
        co = [gco(f) for f in boats]
        for r in range(20):
            for c in range(6):
                ci = r*6+c
                if ci < len(nums):
                    f,s,t = co[c][r]
                    try: odds_dict[f"{f}-{s}-{t}"] = float(nums[ci])
                    except: pass
    return odds_dict

def compute_synthetic_odds(trifecta: dict) -> dict:
    boats = [1,2,3,4,5,6]
    result = {"trifecta":trifecta,"trio":{},"exacta":{},"quinella":{},"wide":{}}
    for combo in itertools.combinations(boats,3):
        inv = sum(1.0/trifecta[f"{p[0]}-{p[1]}-{p[2]}"]
                  for p in itertools.permutations(combo)
                  if f"{p[0]}-{p[1]}-{p[2]}" in trifecta and trifecta[f"{p[0]}-{p[1]}-{p[2]}"] > 0)
        result["trio"][f"{combo[0]}={combo[1]}={combo[2]}"] = round(1.0/inv,1) if inv>0 else 0
    for p in itertools.permutations(boats,2):
        inv = sum(1.0/trifecta[f"{p[0]}-{p[1]}-{t}"]
                  for t in boats if t!=p[0] and t!=p[1]
                  and f"{p[0]}-{p[1]}-{t}" in trifecta and trifecta[f"{p[0]}-{p[1]}-{t}"]>0)
        result["exacta"][f"{p[0]}-{p[1]}"] = round(1.0/inv,1) if inv>0 else 0
    for combo in itertools.combinations(boats,2):
        inv = sum(1.0/trifecta[f"{pm[0]}-{pm[1]}-{t}"]
                  for pm in itertools.permutations(combo)
                  for t in boats if t!=pm[0] and t!=pm[1]
                  and f"{pm[0]}-{pm[1]}-{t}" in trifecta and trifecta[f"{pm[0]}-{pm[1]}-{t}"]>0)
        result["quinella"][f"{combo[0]}={combo[1]}"] = round(1.0/inv,1) if inv>0 else 0
    for combo in itertools.combinations(boats,2):
        inv = sum(1.0/trifecta[f"{pm[0]}-{pm[1]}-{pm[2]}"]
                  for pm in itertools.permutations(boats,3)
                  if combo[0] in pm and combo[1] in pm
                  and f"{pm[0]}-{pm[1]}-{pm[2]}" in trifecta and trifecta[f"{pm[0]}-{pm[1]}-{pm[2]}"]>0)
        result["wide"][f"{combo[0]}={combo[1]}"] = round(1.0/inv,1) if inv>0 else 0
    return result

# ─────────────────────────────────────────────
# 7. モンテカルロ & 期待値
# ─────────────────────────────────────────────
def run_ev_simulation(agents, conditions, venue_name, month, n_sims=10000):
    boats = [1,2,3,4,5,6]
    counts = {'trifecta':{},'trio':{},'exacta':{},'quinella':{},'wide':{}}
    for p in itertools.permutations(boats,3): counts['trifecta'][f"{p[0]}-{p[1]}-{p[2]}"] = 0
    for c in itertools.combinations(boats,3): counts['trio'][f"{c[0]}={c[1]}={c[2]}"] = 0
    for p in itertools.permutations(boats,2): counts['exacta'][f"{p[0]}-{p[1]}"] = 0
    for c in itertools.combinations(boats,2):
        counts['quinella'][f"{c[0]}={c[1]}"] = 0
        counts['wide'][f"{c[0]}={c[1]}"] = 0
    sim = RaceSimulator(agents, conditions, venue_name, month)
    bar = st.progress(0)
    for i in range(n_sims):
        if i % max(1,n_sims//100) == 0: bar.progress(min(i/n_sims,1.0))
        result = sim.simulate_race()
        fo = result["finish_order"]
        f1,f2,f3 = fo[1],fo[2],fo[3]
        tk = f"{f1}-{f2}-{f3}"
        if tk in counts['trifecta']: counts['trifecta'][tk] += 1
        trk = "=".join(str(x) for x in sorted([f1,f2,f3]))
        if trk in counts['trio']: counts['trio'][trk] += 1
        ek = f"{f1}-{f2}"
        if ek in counts['exacta']: counts['exacta'][ek] += 1
        qk = "=".join(str(x) for x in sorted([f1,f2]))
        if qk in counts['quinella']: counts['quinella'][qk] += 1
        for combo in itertools.combinations(sorted([f1,f2,f3]),2):
            wk = f"{combo[0]}={combo[1]}"
            if wk in counts['wide']: counts['wide'][wk] += 1
    bar.progress(1.0)
    probs = {}
    for bt in counts:
        probs[bt] = {k: cnt/n_sims for k,cnt in counts[bt].items()}
    return probs

def compute_expected_values(probs, synthetic_odds):
    ev = {}
    for bt in probs:
        ev[bt] = {}
        om = synthetic_odds.get(bt,{})
        for k,prob in probs[bt].items():
            ov = om.get(k,0)
            ev_val = prob * ov
            if ev_val > 0:
                ev[bt][k] = {"prob":round(prob*100,2),"odds":ov,"ev":round(ev_val,3),
                             "flag":"◎" if ev_val>=1.2 else "○" if ev_val>=1.0 else "△" if ev_val>=0.8 else "×"}
    return ev
# =============================================================
#  8. Streamlit UI
# =============================================================
st.set_page_config(page_title="ボートレース AI v4.0", layout="wide")
st.title("🚤 ボートレース AI シミュレーター v4.0")
st.caption("完全エージェント（30項目）× 展示タイム × 会場特性 × モンテカルロ × 期待値")

# ── サイドバー ──
st.sidebar.header("⚙️ 設定")
venue_list = list(VENUE_PROFILES.keys())
venue_name = st.sidebar.selectbox("会場", venue_list, index=venue_list.index("徳山"))
venue_profile = get_venue_profile(venue_name)
venue_code = venue_profile["code"]

race_date = st.sidebar.date_input("日付", value=pd.Timestamp("2026-02-27"))
date_str = race_date.strftime("%Y%m%d")
race_month = race_date.month

race_no = st.sidebar.number_input("レース番号", 1, 12, 1)

st.sidebar.markdown("---")
st.sidebar.subheader(f"📍 {venue_name}の特徴")
st.sidebar.write(f"水面: {venue_profile['water']}　潮: {'あり' if venue_profile['tide'] else 'なし'}")
st.sidebar.write(f"風影響度: {venue_profile['wind_effect']}")
st.sidebar.write(f"メモ: {venue_profile.get('memo','')}")

fig_sb, ax_sb = plt.subplots(figsize=(4, 2.5))
courses_label = ["1C","2C","3C","4C","5C","6C"]
rates = venue_profile["course_win_rate"]
sb_colors = ['#e74c3c','#000000','#2ecc71','#3498db','#f1c40f','#9b59b6']
ax_sb.bar(courses_label, rates, color=sb_colors)
ax_sb.set_ylabel("1着率(%)")
ax_sb.set_title(f"{venue_name} コース別1着率")
for i, v in enumerate(rates):
    ax_sb.text(i, v+0.5, f"{v}%", ha='center', fontsize=7)
st.sidebar.pyplot(fig_sb)
plt.close(fig_sb)

boat_colors = {1:'#e74c3c',2:'#000000',3:'#2ecc71',4:'#3498db',5:'#f1c40f',6:'#9b59b6'}

# ── メイン: タブ構成 ──
tab_input, tab_sim, tab_mc, tab_odds, tab_ev = st.tabs(
    ["📝 データ入力","🏁 単発シミュレーション","📊 モンテカルロ",
     "💰 オッズ取得","📈 期待値計算"]
)

# ----------------------------
# タブ1: データ入力
# ----------------------------
with tab_input:
    st.subheader("選手データ入力")

    input_method = st.radio(
        "入力方法を選択",
        ["📋 公式サイトからコピペ（推奨）","📝 フォーム入力"],
        horizontal=True
    )

    if input_method == "📋 公式サイトからコピペ（推奨）":
        st.markdown("""
        **使い方**: ボートレース公式サイトや情報サイトの出走表ページを
        **まるごとコピー（Ctrl+A → Ctrl+C）** して下の欄に貼り付けてください。
        基本情報・成績・枠別・展示タイム・周回・周り足・チルト・体重・天候を自動抽出します。
        """)
        official_text = st.text_area(
            "公式サイトのデータを貼り付け",
            height=400,
            placeholder="出走表ページ全体をコピーして貼り付け...",
            key="official_paste"
        )

        if st.button("🔍 データ解析", type="primary", key="parse_official"):
            if official_text and len(official_text.strip()) > 50:
                agents, conditions = parse_official_site_text(official_text)
                st.session_state['agents'] = agents
                st.session_state['conditions'] = conditions
                all_default = all(a.name == f"選手{a.lane}" and a.win_rate == 5.0 for a in agents)
                if all_default:
                    st.warning("⚠️ 自動解析がうまくいきませんでした。フォーム入力をお試しください。")
                else:
                    st.success(f"✅ {len(agents)}艇のデータを解析しました！")
            else:
                st.warning("データが短すぎます。ページ全体をコピペしてください。")

    else:
        st.markdown("#### 基本情報")
        agents_list = []
        for i in range(1, 7):
            with st.expander(f"🚤 {i}号艇", expanded=(i <= 2)):
                c1, c2, c3, c4 = st.columns(4)
                number = c1.number_input("登録番号", 0, 9999, 0, key=f"num_{i}")
                name = c2.text_input("選手名", f"選手{i}", key=f"name_{i}")
                rank = c3.selectbox("級別", ["A1","A2","B1","B2"], index=2, key=f"rank_{i}")
                age = c4.number_input("年齢", 18, 70, 30, key=f"age_{i}")

                c5, c6, c7, c8 = st.columns(4)
                avg_st = c5.number_input("平均ST", 0.01, 0.35, 0.18, 0.01, key=f"st_{i}")
                win_rate = c6.number_input("勝率", 0.0, 15.0, 5.0, 0.01, key=f"wr_{i}")
                top2 = c7.number_input("2連対率(%)", 0.0, 100.0, 30.0, 0.1, key=f"t2_{i}")
                top3 = c8.number_input("3連対率(%)", 0.0, 100.0, 50.0, 0.1, key=f"t3_{i}")

                c9, c10, c11, c12 = st.columns(4)
                lane_wr = c9.number_input("枠別1着(%)", 0.0, 100.0, 10.0, 0.1, key=f"lw_{i}")
                ability = c10.number_input("能力値", 1, 100, 50, key=f"ab_{i}")
                motor = c11.number_input("モーター貢献P", -2.0, 2.0, 0.0, 0.01, key=f"mo_{i}")
                weight = c12.number_input("体重(kg)", 40.0, 70.0, 52.0, 0.1, key=f"wt_{i}")

                c13, c14, c15, c16 = st.columns(4)
                ex_time = c13.number_input("展示タイム", 6.0, 8.0, 0.0, 0.01, key=f"ex_{i}",
                                           help="0=データなし")
                lap_t = c14.number_input("周回タイム", 30.0, 45.0, 0.0, 0.01, key=f"lap_{i}",
                                         help="0=データなし")
                turn_t = c15.number_input("周り足", 10.0, 14.0, 0.0, 0.01, key=f"turn_{i}",
                                          help="0=データなし")
                tilt = c16.number_input("チルト", -1.0, 3.5, -0.5, 0.5, key=f"tilt_{i}")

                agents_list.append(BoatAgent(
                    lane=i, number=number, name=name, rank=rank, age=age,
                    weight=weight, avg_st=avg_st, win_rate=win_rate,
                    top2_rate=top2, top3_rate=top3, lane_win_rate=lane_wr,
                    ability=ability, motor_contribution=motor,
                    exhibition_time=ex_time, lap_time=lap_t,
                    turn_time=turn_t, tilt=tilt
                ))

        st.markdown("#### 天候")
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_weather = wc1.selectbox("天候", ["晴","曇","雨","雪"])
        w_temp = wc2.number_input("気温(℃)", -5.0, 45.0, 20.0, 0.5)
        w_wind = wc3.number_input("風速(m)", 0.0, 15.0, 2.0, 0.5)
        w_wave = wc4.number_input("波高(cm)", 0.0, 30.0, 2.0, 0.5)
        form_conditions = RaceCondition(weather=w_weather, temperature=w_temp,
                                        wind_speed=w_wind, wave_height=w_wave)

        if st.button("✅ データ確定", type="primary", key="confirm_form"):
            st.session_state['agents'] = agents_list
            st.session_state['conditions'] = form_conditions
            st.success(f"✅ {len(agents_list)}艇のデータを確定しました")

    # ─── 確定済みデータ表示 ───
    if 'agents' in st.session_state:
        st.markdown("---")
        st.markdown("#### ✅ 確定済み選手データ")

        # 基本テーブル
        agent_df = pd.DataFrame([
            {"艇": a.lane, "番号": a.number, "名前": a.name, "級": a.rank,
             "年齢": a.age, "体重": a.weight,
             "ST": a.avg_st, "勝率": a.win_rate,
             "2連対": a.top2_rate, "3連対": a.top3_rate,
             "枠1着": a.lane_win_rate, "能力": a.ability,
             "モーター": a.motor_contribution}
            for a in st.session_state['agents']
        ])
        st.dataframe(agent_df, use_container_width=True, hide_index=True)

        # 展示・機力テーブル
        with st.expander("🔧 展示・機力データ"):
            ex_df = pd.DataFrame([
                {"艇": a.lane, "名前": a.name,
                 "展示タイム": a.exhibition_time if a.exhibition_time > 0 else "-",
                 "周回": a.lap_time if a.lap_time > 0 else "-",
                 "周り足": a.turn_time if a.turn_time > 0 else "-",
                 "チルト": a.tilt,
                 "調整重量": a.adjusted_weight,
                 "機力スコア": f"{a.get_machine_score():.3f}",
                 "パワースコア": f"{a.get_power_score():.3f}"}
                for a in st.session_state['agents']
            ])
            st.dataframe(ex_df, use_container_width=True, hide_index=True)

        # 決まり手傾向
        with st.expander("🎯 決まり手傾向"):
            km_df = pd.DataFrame([
                {"艇": a.lane, "名前": a.name,
                 "逃げ": a.nige_count, "差し": a.sashi_count,
                 "捲り": a.makuri_count, "捲差": a.makurisashi_count,
                 "F数": a.flying_count, "事故率": a.accident_rate}
                for a in st.session_state['agents']
            ])
            st.dataframe(km_df, use_container_width=True, hide_index=True)

        # 天候
        cond = st.session_state['conditions']
        st.write(f"🌤 天候: {cond.weather} / 気温: {cond.temperature}℃ / "
                 f"風速: {cond.wind_speed}m / 波高: {cond.wave_height}cm / "
                 f"水温: {cond.water_temp}℃")

        # デバッグ
        with st.expander("🔧 デバッグ: 解析結果の詳細"):
            for a in st.session_state['agents']:
                flags = []
                if a.name == f"選手{a.lane}": flags.append("名前")
                if a.number == 0: flags.append("番号")
                if a.win_rate == 5.0: flags.append("勝率")
                if a.avg_st == 0.18: flags.append("ST")
                if a.lane_win_rate == 10.0: flags.append("枠1着")
                if a.ability == 50: flags.append("能力")
                if a.motor_contribution == 0.0: flags.append("モーター")
                if a.exhibition_time == 0.0: flags.append("展示タイム")
                if a.lap_time == 0.0: flags.append("周回")
                if a.turn_time == 0.0: flags.append("周り足")
                if flags:
                    st.write(f"⚠️ {a.lane}号艇 {a.name}: デフォルト値 → {', '.join(flags)}")
                else:
                    st.write(f"✅ {a.lane}号艇 {a.name}: 全項目取得OK")


# ----------------------------
# タブ2: 単発シミュレーション
# ----------------------------
with tab_sim:
    st.subheader("🏁 単発レースシミュレーション")
    if 'agents' not in st.session_state:
        st.info("先に「📝 データ入力」タブでデータを確定してください。")
    else:
        n_trials = st.slider("試行回数", 1, 10, 3)
        if st.button("▶️ シミュレーション実行", key="run_single"):
            agents = st.session_state['agents']
            conditions = st.session_state['conditions']
            name_map = {a.lane: a.name for a in agents}

            for trial in range(n_trials):
                st.markdown(f"---\n**第{trial+1}レース**")
                simulator = RaceSimulator(agents, conditions, venue_name, race_month)
                result = simulator.simulate_race()
                fo = result["finish_order"]
                st.write(f"決まり手: **{result['kimarite']}**")

                res_df = pd.DataFrame([
                    {"着順":pos,"艇番":boat,"選手名":name_map.get(boat,""),
                     "ST":f"{result['st_times'].get(boat,0):.3f}"}
                    for pos,boat in fo.items()
                ])
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                t1,t2,t3 = fo[1],fo[2],fo[3]
                trio_s = sorted([t1,t2,t3])
                st.write(f"3連単: **{t1}-{t2}-{t3}**　/　"
                         f"3連複: **{trio_s[0]}={trio_s[1]}={trio_s[2]}**　/　"
                         f"2連単: **{t1}-{t2}**")

                fig, ax = plt.subplots(figsize=(10, 4))
                for lane, ph in result["positions"].items():
                    ax.plot(ph, color=boat_colors.get(lane,'gray'),
                            label=f"{lane}号艇 {name_map.get(lane,'')}", linewidth=1.5)
                ax.set_xlabel("ステップ"); ax.set_ylabel("順位")
                ax.invert_yaxis()
                ax.set_title(f"レース展開（第{trial+1}試行）")
                ax.legend(loc='upper right', fontsize=7)
                ax.set_yticks([1,2,3,4,5,6])
                st.pyplot(fig); plt.close(fig)


# ----------------------------
# タブ3: モンテカルロ
# ----------------------------
with tab_mc:
    st.subheader("📊 モンテカルロシミュレーション")
    if 'agents' not in st.session_state:
        st.info("先に「📝 データ入力」タブでデータを確定してください。")
    else:
        n_mc = st.slider("シミュレーション回数", 1000, 50000, 10000, 1000, key="mc_slider")
        if st.button("▶️ モンテカルロ実行", type="primary", key="run_mc"):
            agents = st.session_state['agents']
            conditions = st.session_state['conditions']
            name_map = {a.lane: a.name for a in agents}

            win_c = {a.lane:0 for a in agents}
            top2_c = {a.lane:0 for a in agents}
            top3_c = {a.lane:0 for a in agents}
            km_c = {}

            sim = RaceSimulator(agents, conditions, venue_name, race_month)
            bar = st.progress(0)
            for i in range(n_mc):
                if i % max(1,n_mc//100) == 0: bar.progress(min(i/n_mc,1.0))
                r = sim.simulate_race()
                fo = r["finish_order"]
                win_c[fo[1]] += 1
                top2_c[fo[1]] += 1; top2_c[fo[2]] += 1
                top3_c[fo[1]] += 1; top3_c[fo[2]] += 1; top3_c[fo[3]] += 1
                km = r["kimarite"]
                km_c[km] = km_c.get(km,0) + 1
            bar.progress(1.0)

            st.markdown("#### 勝率・連対率・3連対率")
            mc_df = pd.DataFrame([
                {"艇番":l,"選手":name_map.get(l,""),
                 "1着率":f"{win_c[l]/n_mc*100:.1f}%",
                 "2連対率":f"{top2_c[l]/n_mc*100:.1f}%",
                 "3連対率":f"{top3_c[l]/n_mc*100:.1f}%"}
                for l in sorted(win_c.keys())
            ])
            st.dataframe(mc_df, use_container_width=True, hide_index=True)

            fig2, ax2 = plt.subplots(figsize=(8,4))
            lanes = sorted(win_c.keys())
            wp = [win_c[l]/n_mc*100 for l in lanes]
            bc = [boat_colors.get(l,'gray') for l in lanes]
            lb = [f"{l}号艇\n{name_map.get(l,'')}" for l in lanes]
            ax2.bar(lb, wp, color=bc)
            ax2.set_ylabel("1着率(%)")
            ax2.set_title(f"モンテカルロ {n_mc:,}回 - 1着率")
            for i,v in enumerate(wp):
                ax2.text(i, v+0.3, f"{v:.1f}%", ha='center', fontsize=9)
            st.pyplot(fig2); plt.close(fig2)

            st.markdown("#### 決まり手分布")
            km_df = pd.DataFrame([
                {"決まり手":k,"回数":v,"割合":f"{v/n_mc*100:.1f}%"}
                for k,v in sorted(km_c.items(), key=lambda x:-x[1])
            ])
            st.dataframe(km_df, use_container_width=True, hide_index=True)

            # 重み内訳を表示
            with st.expander("📊 重み計算内訳（1回分のサンプル）"):
                sample_sim = RaceSimulator(agents, conditions, venue_name, race_month)
                sample_w = sample_sim._compute_race_weights()
                w_df = pd.DataFrame([
                    {"艇番": a.lane, "名前": a.name,
                     "最終重み": f"{sample_w[i]*100:.2f}%",
                     "パワー": f"{a.get_power_score():.3f}",
                     "機力": f"{a.get_machine_score():.3f}",
                     "体重補正": f"{a.get_weight_factor():.3f}"}
                    for i, a in enumerate(agents)
                ])
                st.dataframe(w_df, use_container_width=True, hide_index=True)

            st.session_state['mc_done'] = True


# ----------------------------
# タブ4: オッズ取得
# ----------------------------
with tab_odds:
    st.subheader("💰 オッズ取得")
    odds_method = st.radio("取得方法",
        ["🌐 自動取得（公式サイト）","📋 テキスト貼り付け","✏️ 手動入力"],
        horizontal=True, key="odds_method")

    if odds_method == "🌐 自動取得（公式サイト）":
        st.info(f"会場: {venue_name}（{venue_code}） / 日付: {date_str} / レース: {race_no}R")
        if st.button("🔄 オッズ自動取得", type="primary", key="fetch_odds"):
            with st.spinner("取得中..."):
                odds = fetch_trifecta_odds(venue_code, date_str, race_no)
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りのオッズを取得")
            else:
                st.error("取得失敗")

    elif odds_method == "📋 テキスト貼り付け":
        odds_text = st.text_area("オッズデータ", height=200, key="odds_text")
        if st.button("📥 解析", key="parse_odds"):
            odds = parse_pasted_odds(odds_text)
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りを解析")
            else:
                st.error("解析失敗")
    else:
        mot = st.text_area("オッズ入力（1-2-3 6.2）", height=200, key="manual_odds")
        if st.button("📥 登録", key="register_odds"):
            odds = {}
            for line in mot.strip().split('\n'):
                m = re.match(r'(\d)-(\d)-(\d)\s+([\d.]+)', line.strip())
                if m: odds[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"] = float(m.group(4))
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りを登録")

    if 'trifecta_odds' in st.session_state:
        odds = st.session_state['trifecta_odds']
        st.markdown("---")
        st.markdown("#### 取得済み3連単オッズ（低配当順 Top 20）")
        so = sorted(odds.items(), key=lambda x: x[1])
        st.dataframe(pd.DataFrame([{"買い目":k,"オッズ":v} for k,v in so[:20]]),
                     use_container_width=True, hide_index=True)
        st.write(f"合計: {len(odds)}通り / 最低: {so[0][1]} / 最高: {so[-1][1]}")

        synthetic = compute_synthetic_odds(odds)
        st.session_state['synthetic_odds'] = synthetic
        with st.expander("📊 合成オッズ"):
            for bt,lb in [("exacta","2連単"),("quinella","2連複"),("trio","3連複"),("wide","拡連複")]:
                st.markdown(f"**{lb}**")
                s = sorted(synthetic[bt].items(), key=lambda x:x[1])
                st.dataframe(pd.DataFrame([{"買い目":k,"合成オッズ":v} for k,v in s[:15]]),
                             use_container_width=True, hide_index=True)


# ----------------------------
# タブ5: 期待値計算
# ----------------------------
with tab_ev:
    st.subheader("📈 期待値 (EV) 計算")

    if 'agents' not in st.session_state:
        st.info("先に「📝 データ入力」タブでデータを確定してください。")
    elif 'trifecta_odds' not in st.session_state:
        st.info("先に「💰 オッズ取得」タブでオッズを取得してください。")
    else:
        ev_sims = st.slider("シミュレーション回数", 1000, 50000, 10000, 1000, key="ev_sims")
        if st.button("🚀 期待値計算", type="primary", key="run_ev"):
            agents = st.session_state['agents']
            conditions = st.session_state['conditions']
            synthetic = st.session_state['synthetic_odds']
            st.write("⏳ シミュレーション中...")
            probs = run_ev_simulation(agents, conditions, venue_name, race_month, ev_sims)
            ev_results = compute_expected_values(probs, synthetic)
            st.session_state['ev_results'] = ev_results
            st.success("✅ 完了！")

        if 'ev_results' in st.session_state:
            ev_results = st.session_state['ev_results']
            ev_tabs = st.tabs(["3連単","3連複","2連単","2連複","拡連複"])
            bet_types = ["trifecta","trio","exacta","quinella","wide"]
            bet_labels = ["3連単","3連複","2連単","2連複","拡連複"]

            for ev_tab, bt, bl in zip(ev_tabs, bet_types, bet_labels):
                with ev_tab:
                    data = ev_results.get(bt,{})
                    if not data: st.write("データなし"); continue
                    se = sorted(data.items(), key=lambda x:-x[1]['ev'])
                    tn = se[:20]
                    st.dataframe(pd.DataFrame([
                        {"買い目":k,"確率(%)":v['prob'],"オッズ":v['odds'],
                         "期待値":v['ev'],"判定":v['flag']}
                        for k,v in tn
                    ]), use_container_width=True, hide_index=True)

                    if tn:
                        t15 = tn[:15]
                        fig_ev, ax_ev = plt.subplots(figsize=(10,5))
                        keys = [x[0] for x in t15]
                        vals = [x[1]['ev'] for x in t15]
                        bc2 = ['#2ecc71' if v>=1.0 else '#f39c12' if v>=0.8 else '#e74c3c' for v in vals]
                        ax_ev.barh(keys[::-1], vals[::-1], color=bc2[::-1])
                        ax_ev.axvline(x=1.0, color='red', linestyle='--', label='EV=1.0')
                        ax_ev.set_xlabel("期待値"); ax_ev.set_title(f"{bl} EV Top15")
                        ax_ev.legend()
                        st.pyplot(fig_ev); plt.close(fig_ev)

            # おすすめ
            st.markdown("---")
            st.markdown("### 🎯 おすすめ買い目サマリー")
            all_good = []
            for bt,bl in zip(bet_types, bet_labels):
                for k,v in ev_results.get(bt,{}).items():
                    if v['ev'] >= 1.0:
                        all_good.append({"券種":bl,"買い目":k,"確率(%)":v['prob'],
                                         "オッズ":v['odds'],"期待値":v['ev'],"判定":v['flag']})
            if all_good:
                all_good.sort(key=lambda x:-x['期待値'])
                st.markdown(f"**EV ≥ 1.0: {len(all_good)}件**")
                st.dataframe(pd.DataFrame(all_good), use_container_width=True, hide_index=True)
            else:
                st.warning("EV ≥ 1.0 の買い目は見つかりませんでした。")

            all_ok = []
            for bt,bl in zip(bet_types, bet_labels):
                for k,v in ev_results.get(bt,{}).items():
                    if 0.8 <= v['ev'] < 1.0:
                        all_ok.append({"券種":bl,"買い目":k,"確率(%)":v['prob'],
                                       "オッズ":v['odds'],"期待値":v['ev'],"判定":v['flag']})
            if all_ok:
                with st.expander(f"📋 EV 0.8〜1.0 ({len(all_ok)}件)"):
                    all_ok.sort(key=lambda x:-x['期待値'])
                    st.dataframe(pd.DataFrame(all_ok), use_container_width=True, hide_index=True)


# ── フッター ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em;'>"
    "🚤 ボートレース AI シミュレーター v4.0 ─ 完全エージェント(30項目) × 展示タイム × 会場特性 × モンテカルロ × 期待値"
    "</div>",
    unsafe_allow_html=True
)
