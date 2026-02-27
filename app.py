# ============================================================
#  ボートレース AI シミュレーター v4.0  ─  app.py  (Part 1/3)
#  完全版: 30項目エージェント × 会場特性 × モンテカルロ × 期待値
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
               "wind_effect":0.5,"memo":"向かい風でイン不利になる場合"},
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
               "wind_effect":0.5,"memo":"比良おろし（冬の北西風）"},
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
               "wind_effect":0.4,"memo":"センタープール、静水面"},
    "鳴門":   {"code":"14","water":"海水","tide":True,
               "course_win_rate":[53.0,15.0,12.5,11.5,5.5,2.8],
               "course_top2":[72.0,31.0,27.0,25.0,18.0,10.0],
               "course_top3":[83.0,45.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.52,"差し":0.15,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.6,"memo":"潮の干満差が大きい"},
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
               "wind_effect":0.5,"memo":"瀬戸内海、潮流あり"},
    "宮島":   {"code":"17","water":"海水","tide":True,
               "course_win_rate":[54.5,14.5,12.5,11.0,5.0,2.5],
               "course_top2":[73.0,30.0,27.0,25.0,18.0,10.0],
               "course_top3":[84.0,44.0,41.0,38.0,32.0,22.0],
               "kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.5,"memo":"潮の影響が大きい"},
    "徳山":   {"code":"18","water":"海水","tide":True,
               "course_win_rate":[63.4,11.7,12.8,9.7,3.5,1.1],
               "course_top2":[80.4,30.5,20.0,17.5,12.2,5.2],
               "course_top3":[87.5,47.7,40.2,39.1,29.2,22.0],
               "kimarite":{"逃げ":0.63,"差し":0.12,"捲り":0.10,"捲り差し":0.09,"抜き":0.05,"恵まれ":0.01},
               "wind_effect":0.6,
               "seasonal":{
                   "春":[64.6,14.6,9.7,7.3,3.8,0.5],
                   "夏":[66.1,11.9,10.7,7.4,4.2,0.5],
                   "秋":[61.1,10.5,12.1,11.6,3.2,1.9],
                   "冬":[62.9,14.0,8.4,8.0,5.7,1.6]
               },
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
               "wind_effect":0.6,"memo":"那珂川河口、うねり注意"},
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

def get_venue_profile(name: str) -> dict:
    return VENUE_PROFILES.get(name, DEFAULT_VENUE_PROFILE)

def get_season(month: int) -> str:
    if month in [3,4,5]:   return "春"
    if month in [6,7,8]:   return "夏"
    if month in [9,10,11]: return "秋"
    return "冬"

# ─────────────────────────────────────────────
# 3. データクラス（30項目完全エージェント）
# ─────────────────────────────────────────────
@dataclass
class BoatAgent:
    """1艇の全情報を保持するエージェント"""
    # ── 基本情報 ──
    lane: int = 1                    # 枠番 1-6
    number: int = 0                  # 登録番号
    name: str = ""                   # 選手名
    rank: str = "B1"                 # 級別 A1/A2/B1/B2
    age: int = 30                    # 年齢
    branch: str = ""                 # 支部
    weight: float = 52.0             # 体重(kg)
    flying_count: int = 0            # F数
    late_count: int = 0              # L数

    # ── 成績データ ──
    avg_st: float = 0.18             # 平均ST
    win_rate: float = 5.00           # 全国勝率
    top2_rate: float = 30.0          # 2連対率(%)
    top3_rate: float = 50.0          # 3連対率(%)
    lane_win_rate: float = 10.0      # 枠別1着率(%)
    local_win_rate: float = 5.00     # 当地勝率
    local_top2_rate: float = 30.0    # 当地2連対率(%)
    local_top3_rate: float = 50.0    # 当地3連対率(%)
    accident_rate: float = 0.0       # 事故率

    # ── 能力値 ──
    ability: int = 50                # 能力値 (1-100)
    motor_no: int = 0                # モーター番号
    motor_contribution: float = 0.0  # モーター貢献度P
    motor_top2_rate: float = 30.0    # モーター2連対率(%)
    boat_no: int = 0                 # ボート番号
    boat_top2_rate: float = 30.0     # ボート2連対率(%)

    # ── 直前情報（展示） ──
    exhibition_time: float = 6.80    # 展示タイム(秒)
    lap_time: float = 0.0            # 周回展示タイム(秒)
    turn_time: float = 0.0           # 周り足タイム(秒)
    straight_time: float = 0.0       # 直線タイム(秒)
    exhibition_rank: int = 3         # 展示順位 (1-6)
    tilt: float = -0.5               # チルト角度

    # ── 決まり手傾向 ──
    nige_count: int = 0              # 逃げ回数
    sashi_count: int = 0             # 差し回数
    makuri_count: int = 0            # 捲り回数
    makuri_sashi_count: int = 0      # 捲り差し回数
    nuki_count: int = 0              # 抜き回数

    def calculate_start_timing(self) -> float:
        """実レースのST生成（平均STにランダムばらつき加算）"""
        base = self.avg_st
        variation = np.random.normal(0, 0.015)
        # フライング持ちは慎重になる
        if self.flying_count > 0:
            base += 0.01 * self.flying_count
        return max(0.01, base + variation)

    def get_power_score(self) -> float:
        """選手能力スコア (0-1)"""
        base = self.ability / 100.0
        rank_bonus = {"A1": 0.08, "A2": 0.04, "B1": 0.0, "B2": -0.04}.get(self.rank, 0)
        return np.clip(base + rank_bonus, 0.1, 1.0)

    def get_machine_score(self) -> float:
        """機力スコア (0-1)：モーター貢献度 + モーター2連率 + 展示タイム"""
        motor_p = np.clip(self.motor_contribution * 0.8, -0.15, 0.15)
        motor_rate_score = (self.motor_top2_rate - 30.0) / 100.0
        # 展示タイム: 6.50秒が最速クラス、6.90秒が平均
        exhibition_score = (6.90 - self.exhibition_time) / 0.80
        exhibition_score = np.clip(exhibition_score, -0.2, 0.2)
        # チルト補正: マイナスチルトは出足型、プラスは伸び型
        tilt_score = 0.0
        if self.tilt < -0.5:
            tilt_score = 0.02  # 出足重視（1マーク有利）
        elif self.tilt > 0.5:
            tilt_score = -0.01  # 伸び重視（直線有利だがターン不利）
        base = 0.5 + motor_p + motor_rate_score + exhibition_score + tilt_score
        return np.clip(base, 0.1, 1.0)

    def get_turn_score(self) -> float:
        """旋回力スコア: 周り足 + 展示順位 + 体重"""
        turn_s = 0.0
        if self.turn_time > 0:
            turn_s = (5.0 - self.turn_time) / 5.0
        rank_s = (4 - self.exhibition_rank) * 0.03
        # 軽量が有利（52kgが標準）
        weight_s = (52.0 - self.weight) * 0.003
        return np.clip(0.5 + turn_s + rank_s + weight_s, 0.1, 1.0)

    def get_kimarite_tendency(self) -> Dict[str, float]:
        """決まり手傾向の正規化"""
        total = max(1, self.nige_count + self.sashi_count + self.makuri_count
                    + self.makuri_sashi_count + self.nuki_count)
        return {
            "逃げ": self.nige_count / total,
            "差し": self.sashi_count / total,
            "捲り": self.makuri_count / total,
            "捲り差し": self.makuri_sashi_count / total,
            "抜き": self.nuki_count / total,
        }

@dataclass
class RaceCondition:
    weather: str = "晴"
    temperature: float = 20.0
    wind_speed: float = 2.0
    wind_direction: str = ""
    water_temp: float = 20.0
    wave_height: float = 2.0
    tide: str = ""

# ─────────────────────────────────────────────
# 4. レースシミュレーター
# ─────────────────────────────────────────────
class RaceSimulator:
    """30項目を使った重み付きモンテカルロシミュレーター"""

    def __init__(self, agents: List[BoatAgent], conditions: RaceCondition,
                 venue_name: str = "徳山", month: int = 2):
        self.agents = sorted(agents, key=lambda a: a.lane)
        self.conditions = conditions
        self.venue = get_venue_profile(venue_name)
        self.season = get_season(month)

    def _compute_race_weights(self) -> Dict[int, float]:
        """30項目から各艇の勝ち重みを算出"""
        vp = self.venue
        # 会場のコース別勝率（季節補正あり）
        if "seasonal" in vp and self.season in vp["seasonal"]:
            base_rates = vp["seasonal"][self.season]
        else:
            base_rates = vp["course_win_rate"]

        weights = {}
        for agent in self.agents:
            idx = agent.lane - 1
            w = 0.0

            # ①会場コース別勝率(20%): 会場のデータが最重要
            venue_base = base_rates[idx] / 100.0
            w += venue_base * 2.0

            # ②選手の枠別1着率(12%): 個人のコース適性
            lane_factor = agent.lane_win_rate / 100.0
            w += lane_factor * 1.2

            # ③全国勝率(10%)
            wr_factor = agent.win_rate / 10.0
            w += wr_factor * 1.0

            # ④2連対率/3連対率(5%)
            t2_factor = agent.top2_rate / 200.0
            t3_factor = agent.top3_rate / 300.0
            w += (t2_factor + t3_factor) * 0.5

            # ⑤当地勝率(5%): その会場での実績
            local_factor = agent.local_win_rate / 10.0
            w += local_factor * 0.5

            # ⑥パワースコア(10%): 級別 + 能力値
            power = agent.get_power_score()
            w += power * 1.0

            # ⑦機力スコア(15%): モーター貢献P + モーター2連率 + 展示タイム + チルト
            machine = agent.get_machine_score()
            w += machine * 1.5

            # ⑧旋回力スコア(8%): 周り足 + 展示順位 + 体重
            turn = agent.get_turn_score()
            w += turn * 0.8

            # ⑨STクオリティ(8%)
            st_quality = max(0, (0.20 - agent.avg_st) / 0.10)
            w += st_quality * 0.8

            # ⑩体重補正(2%): 軽い方が有利
            weight_adj = (54.0 - agent.weight) / 20.0
            w += weight_adj * 0.2

            # ⑪風の影響(3%)
            wind_effect = self.conditions.wind_speed * vp["wind_effect"] / 20.0
            if agent.lane == 1:
                w -= wind_effect * 0.3  # インは風で不利
            elif agent.lane >= 4:
                w += wind_effect * 0.1  # アウトは風で有利

            # ⑫波高の影響(1%)
            if self.conditions.wave_height > 5:
                if agent.weight > 54:
                    w += 0.02  # 重い方が安定
                else:
                    w -= 0.01

            # ⑬事故率(1%): 高いとマイナス
            w -= agent.accident_rate * 0.01

            # ⑭フライング持ち(1%): 慎重になるためST遅れやすい
            if agent.flying_count > 0:
                w -= 0.02 * agent.flying_count

            weights[agent.lane] = max(w, 0.01)

        # 正規化
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

        return weights

    def simulate_race(self) -> dict:
        """1レース分のシミュレーション"""
        weights = self._compute_race_weights()

        # STタイム生成
        st_times = {}
        for agent in self.agents:
            st_times[agent.lane] = agent.calculate_start_timing()

        # ST最速ボーナス
        min_st = min(st_times.values())
        adjusted = dict(weights)
        for lane, st in st_times.items():
            diff = st - min_st
            if diff < 0.02:
                adjusted[lane] *= 1.15  # ST最速グループ
            elif diff < 0.05:
                adjusted[lane] *= 1.05
            elif diff > 0.10:
                adjusted[lane] *= 0.85

        # 展示タイムボーナス: 全艇中最速ならさらに加算
        ex_times = {a.lane: a.exhibition_time for a in self.agents if a.exhibition_time > 0}
        if ex_times:
            fastest_ex = min(ex_times.values())
            for lane, et in ex_times.items():
                if et <= fastest_ex + 0.02:
                    adjusted[lane] *= 1.08
                elif et >= fastest_ex + 0.15:
                    adjusted[lane] *= 0.95

        # ランダムノイズ（波乱要素）
        noise_level = 0.05
        if self.conditions.wave_height > 5 or self.conditions.wind_speed > 5:
            noise_level = 0.10
        if self.conditions.weather in ["雨", "雪"]:
            noise_level += 0.03

        for lane in adjusted:
            noise = np.random.normal(0, noise_level * adjusted[lane])
            adjusted[lane] = max(adjusted[lane] + noise, 0.001)

        # 正規化
        total = sum(adjusted.values())
        for k in adjusted:
            adjusted[k] /= total

        # 着順決定（重み付きランダムドロー）
        lanes = list(adjusted.keys())
        probs = [adjusted[l] for l in lanes]
        finish_order_list = []
        remaining_lanes = list(lanes)
        remaining_probs = list(probs)
        for _ in range(6):
            total_p = sum(remaining_probs)
            norm_probs = [p / total_p for p in remaining_probs]
            chosen_idx = np.random.choice(len(remaining_lanes), p=norm_probs)
            finish_order_list.append(remaining_lanes[chosen_idx])
            remaining_lanes.pop(chosen_idx)
            remaining_probs.pop(chosen_idx)

        finish_order = {pos + 1: lane for pos, lane in enumerate(finish_order_list)}

        # ポジション履歴（グラフ用）
        n_steps = 300
        positions = {a.lane: [] for a in self.agents}
        for step in range(n_steps):
            progress = step / n_steps
            for a in self.agents:
                final_pos = list(finish_order.values()).index(a.lane) + 1
                if progress < 0.1:
                    pos = a.lane + np.random.normal(0, 0.3)
                elif progress < 0.3:
                    pos = a.lane * (1 - progress) + final_pos * progress + np.random.normal(0, 0.5)
                else:
                    pos = final_pos + np.random.normal(0, max(0.1, 0.5 * (1 - progress)))
                positions[a.lane].append(np.clip(pos, 0.8, 6.2))

        kimarite = self._determine_kimarite(finish_order, st_times)

        return {
            "finish_order": finish_order,
            "st_times": st_times,
            "kimarite": kimarite,
            "positions": positions,
            "weights": weights,
        }

    def _determine_kimarite(self, finish_order: dict, st_times: dict) -> str:
        """決まり手を判定"""
        winner = finish_order[1]
        agent_map = {a.lane: a for a in self.agents}
        winner_agent = agent_map[winner]

        if winner == 1:
            return "逃げ"

        winner_st = st_times.get(winner, 0.18)
        inner_st = st_times.get(1, 0.18)

        tendency = winner_agent.get_kimarite_tendency()
        # 決まり手の候補リスト（1着が1コース以外）
        options = []
        if winner == 2:
            options = [("差し", 0.5), ("捲り", 0.2), ("抜き", 0.2), ("恵まれ", 0.1)]
        elif winner == 3:
            options = [("捲り", 0.35), ("捲り差し", 0.3), ("差し", 0.15), ("抜き", 0.15), ("恵まれ", 0.05)]
        elif winner == 4:
            options = [("捲り", 0.35), ("差し", 0.25), ("捲り差し", 0.2), ("抜き", 0.15), ("恵まれ", 0.05)]
        elif winner == 5:
            options = [("捲り", 0.4), ("捲り差し", 0.25), ("抜き", 0.2), ("差し", 0.1), ("恵まれ", 0.05)]
        else:
            options = [("捲り", 0.4), ("捲り差し", 0.2), ("抜き", 0.2), ("差し", 0.1), ("恵まれ", 0.1)]

        # 個人傾向で補正
        adjusted_options = []
        for km, base_p in options:
            personal = tendency.get(km, 0.0)
            p = base_p * 0.7 + personal * 0.3
            adjusted_options.append((km, p))

        total_p = sum(p for _, p in adjusted_options)
        names = [n for n, _ in adjusted_options]
        probs = [p / total_p for _, p in adjusted_options]
        return np.random.choice(names, p=probs)
# ============================================================
#  ボートレース AI シミュレーター v4.0  ─  app.py  (Part 2/3)
#  パーサー × オッズ取得 × 合成オッズ × モンテカルロEV
# ============================================================

# ─────────────────────────────────────────────
# 5. パーサー（3方式対応）
# ─────────────────────────────────────────────

def _extract_floats(text: str) -> List[float]:
    """文字列から浮動小数点数をすべて抽出"""
    return [float(x) for x in re.findall(r'-?\d+\.\d+|\d+', text)]

def _extract_six_floats(lines: List[str], start_idx: int) -> Tuple[List[float], int]:
    """start_idx 以降の行から6個の数値を収集（複数行にまたがる場合も対応）"""
    vals = []
    idx = start_idx
    while idx < len(lines) and len(vals) < 6:
        nums = re.findall(r'[\d]+\.[\d]+|[\d]+', lines[idx])
        for n in nums:
            try:
                vals.append(float(n))
            except ValueError:
                pass
            if len(vals) >= 6:
                break
        idx += 1
    return vals[:6], idx


def parse_manual_format(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
    """
    1行1艇の手動入力フォーマット:
    1号艇: 4251 川崎誠志 B1 43歳 山口 平均ST0.21 勝率4.61 2連対26.6 3連対42.2 ...
    """
    agents = []
    conditions = RaceCondition()

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # ── 天候行 ──
        if ('天候' in line or '℃' in line) and '号艇' not in line:
            m = re.search(r'(\d+\.?\d*)\s*℃', line)
            if m:
                conditions.temperature = float(m.group(1))
            m = re.search(r'水温\s*(\d+\.?\d*)', line)
            if m:
                conditions.water_temp = float(m.group(1))
            m = re.search(r'風速?\s*(\d+\.?\d*)', line)
            if m:
                conditions.wind_speed = float(m.group(1))
            m = re.search(r'波高?\s*(\d+\.?\d*)', line)
            if m:
                conditions.wave_height = float(m.group(1))
            if '雨' in line:
                conditions.weather = "雨"
            elif '曇' in line:
                conditions.weather = "曇"
            elif '雪' in line:
                conditions.weather = "雪"
            continue

        # ── 選手行 ──
        lane_m = re.search(r'(\d)\s*号艇', line)
        if not lane_m:
            continue
        lane = int(lane_m.group(1))

        a = BoatAgent(lane=lane)

        # 登録番号
        m = re.search(r'[:：\s]\s*(\d{3,5})\b', line)
        if m:
            a.number = int(m.group(1))

        # 選手名（番号の後ろの漢字/かな）
        after = line[m.end():] if m else line[lane_m.end():]
        m = re.search(r'\s*([一-龥ぁ-んァ-ヶー]{2,})', after)
        if m:
            a.name = m.group(1)

        # 級別
        for r in ["A1", "A2", "B1", "B2"]:
            if r in line:
                a.rank = r
                break

        # 年齢
        m = re.search(r'(\d{2,3})\s*歳', line)
        if m:
            a.age = int(m.group(1))

        # 支部
        for pref in ["北海道","青森","岩手","宮城","秋田","山形","福島",
                      "茨城","栃木","群馬","埼玉","千葉","東京","神奈川",
                      "新潟","富山","石川","福井","山梨","長野","岐阜",
                      "静岡","愛知","三重","滋賀","京都","大阪","兵庫",
                      "奈良","和歌山","鳥取","島根","岡山","広島","山口",
                      "徳島","香川","愛媛","高知","福岡","佐賀","長崎",
                      "熊本","大分","宮崎","鹿児島","沖縄"]:
            if pref in line:
                a.branch = pref
                break

        # 体重
        m = re.search(r'体重\s*(\d+\.?\d*)', line)
        if m:
            a.weight = float(m.group(1))

        # F数/L数
        m = re.search(r'F\s*(\d+)', line)
        if m:
            a.flying_count = int(m.group(1))
        m = re.search(r'L\s*(\d+)', line)
        if m:
            a.late_count = int(m.group(1))

        # 平均ST
        m = re.search(r'(?:平均)?ST\s*(0\.\d+)', line)
        if m:
            a.avg_st = float(m.group(1))

        # 勝率
        m = re.search(r'勝率\s*([\d.]+)', line)
        if m:
            a.win_rate = float(m.group(1))

        # 2連対率
        m = re.search(r'2連対?\s*([\d.]+)', line)
        if m:
            a.top2_rate = float(m.group(1))

        # 3連対率
        m = re.search(r'3連対?\s*([\d.]+)', line)
        if m:
            a.top3_rate = float(m.group(1))

        # 枠別1着率
        m = re.search(r'枠別?[1１]?着?\s*([\d.]+)', line)
        if m:
            a.lane_win_rate = float(m.group(1))

        # 当地勝率
        m = re.search(r'当地勝率?\s*([\d.]+)', line)
        if m:
            a.local_win_rate = float(m.group(1))

        # 能力
        m = re.search(r'能力\s*(\d+)', line)
        if m:
            a.ability = int(m.group(1))

        # モーター貢献P
        m = re.search(r'モーター?\s*([+\-]?\s*[\d.]+)', line)
        if m:
            a.motor_contribution = float(m.group(1).replace(' ', ''))

        # モーター2連率
        m = re.search(r'モ2連\s*([\d.]+)', line)
        if m:
            a.motor_top2_rate = float(m.group(1))

        # 展示タイム
        m = re.search(r'展示T?\s*([\d.]+)', line)
        if m:
            a.exhibition_time = float(m.group(1))

        # 展示順位
        m = re.search(r'展示順?\s*(\d)', line)
        if m:
            a.exhibition_rank = int(m.group(1))

        # 周回タイム
        m = re.search(r'周回T?\s*([\d.]+)', line)
        if m:
            a.lap_time = float(m.group(1))

        # 周り足
        m = re.search(r'周り足\s*([\d.]+)', line)
        if m:
            a.turn_time = float(m.group(1))

        # チルト
        m = re.search(r'チルト\s*([+\-]?\s*[\d.]+)', line)
        if m:
            a.tilt = float(m.group(1).replace(' ', ''))

        # 事故率
        m = re.search(r'事故率?\s*([\d.]+)', line)
        if m:
            a.accident_rate = float(m.group(1))

        # 決まり手
        m = re.search(r'逃げ\s*(\d+)', line)
        if m:
            a.nige_count = int(m.group(1))
        m = re.search(r'差し\s*(\d+)', line)
        if m:
            a.sashi_count = int(m.group(1))
        m = re.search(r'捲り\s*(\d+)', line)
        if m:
            a.makuri_count = int(m.group(1))
        m = re.search(r'捲り差し?\s*(\d+)', line)
        if m:
            a.makuri_sashi_count = int(m.group(1))
        m = re.search(r'抜き\s*(\d+)', line)
        if m:
            a.nuki_count = int(m.group(1))

        agents.append(a)

    if not agents:
        for i in range(1, 7):
            agents.append(BoatAgent(lane=i, name=f"選手{i}"))

    return agents, conditions


def parse_official_site_text(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
    """
    公式サイトからコピーした列方向テキストをパース。
    数値が縦に6個並ぶパターンを検出し、セクションに応じて振り分ける。
    """
    agents = [BoatAgent(lane=i+1) for i in range(6)]
    conditions = RaceCondition()
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines]

    # ─── 天候抽出（先にスキャン）───
    for line in lines:
        if '℃' in line and '号艇' not in line and '勝率' not in line:
            m = re.search(r'気温\s*(\d+\.?\d*)', line)
            if not m:
                m = re.search(r'(\d+\.?\d*)\s*℃', line)
            if m:
                conditions.temperature = float(m.group(1))
            m = re.search(r'水温\s*(\d+\.?\d*)', line)
            if m:
                conditions.water_temp = float(m.group(1))
        m = re.search(r'風速?\s*(\d+\.?\d*)\s*m', line)
        if m:
            conditions.wind_speed = float(m.group(1))
        m = re.search(r'波高?\s*(\d+\.?\d*)\s*cm', line)
        if m:
            conditions.wave_height = float(m.group(1))
        if '雨' in line and '号艇' not in line:
            conditions.weather = "雨"
        elif '曇' in line and '号艇' not in line:
            conditions.weather = "曇"

    # ─── セクション検出キーワード ───
    # 公式サイトの出走表は以下のセクションを持つ:
    #   登録番号/名前/級, 平均ST, 全国勝率/2連/3連, 当地勝率/2連/3連,
    #   モーター番号/2連率, ボート番号/2連率, 展示タイム, 体重, チルト, etc.
    #
    # 戦略: 全行を走査し、キーワードで「次に来る6個の数値列」の意味を決定する

    def find_six_values_after(keyword: str) -> Optional[List[float]]:
        """キーワード行を見つけ、その後の行から6個の数値を収集"""
        for i, line in enumerate(lines):
            if keyword in line:
                vals = []
                j = i + 1
                while j < len(lines) and len(vals) < 6:
                    nums = re.findall(r'[\-]?\d+\.\d+|\d+', lines[j])
                    for n in nums:
                        try:
                            v = float(n)
                            vals.append(v)
                        except ValueError:
                            pass
                        if len(vals) >= 6:
                            break
                    j += 1
                if len(vals) >= 6:
                    return vals[:6]
        return None

    def find_six_ints_after(keyword: str) -> Optional[List[int]]:
        """キーワード行の後から6個の整数を収集"""
        for i, line in enumerate(lines):
            if keyword in line:
                vals = []
                j = i + 1
                while j < len(lines) and len(vals) < 6:
                    nums = re.findall(r'\d+', lines[j])
                    for n in nums:
                        vals.append(int(n))
                        if len(vals) >= 6:
                            break
                    j += 1
                if len(vals) >= 6:
                    return vals[:6]
        return None

    # ── 登録番号 ──
    reg_nums = []
    for line in lines:
        m = re.findall(r'\b([3-5]\d{3})\b', line)
        if len(m) >= 6:
            reg_nums = [int(x) for x in m[:6]]
            break
    if not reg_nums:
        # 連続する行から探す
        buf = []
        for line in lines:
            m = re.search(r'\b([3-5]\d{3})\b', line)
            if m:
                buf.append(int(m.group(1)))
                if len(buf) >= 6:
                    reg_nums = buf[:6]
                    break
            elif buf:
                buf = []  # 途切れたらリセット
    if reg_nums:
        for i, num in enumerate(reg_nums):
            agents[i].number = num

    # ── 選手名（漢字2-4文字）──
    name_candidates = []
    for line in lines:
        names_in_line = re.findall(r'([一-龥]{2,4})', line)
        # 会場名や列名ではない名前を収集
        filtered = [n for n in names_in_line
                    if n not in ["競艇","徳山","勝率","連対","決まり","選手","当地","全国",
                                 "平均","枠別","能力","事故","貢献","モーター","ボート",
                                 "展示","周回","直線","天候","気温","水温","風速","波高",
                                 "日本","財団","会長","争奪","初日","最終","一般"]]
        if len(filtered) >= 6:
            name_candidates = filtered[:6]
            break
    if not name_candidates:
        buf = []
        for line in lines:
            m = re.findall(r'([一-龥ぁ-んァ-ヶー]{2,})', line)
            for n in m:
                if len(n) >= 2 and n not in ["競艇","徳山","勝率","連対"]:
                    buf.append(n)
                    if len(buf) >= 6:
                        break
            if len(buf) >= 6:
                name_candidates = buf[:6]
                break
    if name_candidates:
        for i, nm in enumerate(name_candidates):
            agents[i].name = nm

    # ── 級別 ──
    for line in lines:
        ranks = re.findall(r'\b(A1|A2|B1|B2)\b', line)
        if len(ranks) >= 6:
            for i, r in enumerate(ranks[:6]):
                agents[i].rank = r
            break

    # ── 年齢 ──
    for line in lines:
        ages = re.findall(r'(\d{2})歳', line)
        if len(ages) >= 6:
            for i, a in enumerate(ages[:6]):
                agents[i].age = int(a)
            break

    # ── 体重 ──
    vals = find_six_values_after("体重")
    if vals:
        for i, v in enumerate(vals):
            if 40 <= v <= 70:
                agents[i].weight = v

    # ── 平均ST ──
    vals = find_six_values_after("平均ST")
    if not vals:
        vals = find_six_values_after("平均ＳＴ")
    if vals:
        for i, v in enumerate(vals):
            if 0.05 <= v <= 0.30:
                agents[i].avg_st = v

    # ── 全国勝率 ──
    vals = find_six_values_after("全国勝率")
    if not vals:
        vals = find_six_values_after("勝率")
    if vals:
        for i, v in enumerate(vals):
            if 1.0 <= v <= 10.0:
                agents[i].win_rate = v

    # ── 全国2連対率 ──
    vals = find_six_values_after("2連対率")
    if not vals:
        vals = find_six_values_after("2連率")
    if vals:
        for i, v in enumerate(vals):
            if 0.0 <= v <= 100.0:
                agents[i].top2_rate = v

    # ── 全国3連対率 ──
    vals = find_six_values_after("3連対率")
    if not vals:
        vals = find_six_values_after("3連率")
    if vals:
        for i, v in enumerate(vals):
            if 0.0 <= v <= 100.0:
                agents[i].top3_rate = v

    # ── 当地勝率 ──
    vals = find_six_values_after("当地勝率")
    if vals:
        for i, v in enumerate(vals):
            if 0.0 <= v <= 10.0:
                agents[i].local_win_rate = v

    # ── 枠別1着率 ──
    # 公式サイトでは「1着率(総合)」や「枠別」の行の後に % 付きの値がある
    for i, line in enumerate(lines):
        if '1着率' in line or '枠別' in line:
            vals = []
            j = i + 1
            while j < min(len(lines), i + 15) and len(vals) < 6:
                pcts = re.findall(r'([\d]+\.[\d]+)\s*%?', lines[j])
                for p in pcts:
                    vals.append(float(p))
                    if len(vals) >= 6:
                        break
                j += 1
            if len(vals) >= 6:
                for k, v in enumerate(vals[:6]):
                    if 0.0 <= v <= 100.0:
                        agents[k].lane_win_rate = v
                break

    # ── 能力値 ──
    vals = find_six_ints_after("能力")
    if vals:
        for i, v in enumerate(vals):
            if 1 <= v <= 100:
                agents[i].ability = v

    # ── モーター貢献P ──
    for i, line in enumerate(lines):
        if '貢献' in line or '貢献P' in line:
            vals = []
            j = i
            while j < min(len(lines), i + 5) and len(vals) < 6:
                nums = re.findall(r'([+\-]?\d+\.\d+)', lines[j])
                for n in nums:
                    vals.append(float(n))
                    if len(vals) >= 6:
                        break
                j += 1
            if len(vals) >= 6:
                for k, v in enumerate(vals[:6]):
                    agents[k].motor_contribution = v
                break

    # ── モーター番号・2連率 ──
    vals = find_six_values_after("モーター")
    if vals:
        for i, v in enumerate(vals):
            if v > 10:
                agents[i].motor_no = int(v)
    # モーター2連率
    for i, line in enumerate(lines):
        if 'モーター' in line and '2連' in line:
            vs = find_six_values_after("モーター")
            break
    vals = find_six_values_after("モ2連")
    if vals:
        for i, v in enumerate(vals):
            if 0.0 <= v <= 100.0:
                agents[i].motor_top2_rate = v

    # ── 展示タイム ──
    vals = find_six_values_after("展示タイム")
    if not vals:
        vals = find_six_values_after("展示T")
    if vals:
        for i, v in enumerate(vals):
            if 6.0 <= v <= 7.5:
                agents[i].exhibition_time = v

    # ── 展示順位 ──
    vals = find_six_ints_after("展示順")
    if vals:
        for i, v in enumerate(vals):
            if 1 <= v <= 6:
                agents[i].exhibition_rank = v

    # ── 周回タイム ──
    vals = find_six_values_after("周回")
    if vals:
        for i, v in enumerate(vals):
            if v > 0:
                agents[i].lap_time = v

    # ── 周り足 ──
    vals = find_six_values_after("周り足")
    if not vals:
        vals = find_six_values_after("周足")
    if vals:
        for i, v in enumerate(vals):
            if v > 0:
                agents[i].turn_time = v

    # ── チルト ──
    for i, line in enumerate(lines):
        if 'チルト' in line:
            vals = []
            j = i
            while j < min(len(lines), i + 5) and len(vals) < 6:
                nums = re.findall(r'([+\-]?\d+\.?\d*)', lines[j])
                for n in nums:
                    vals.append(float(n))
                    if len(vals) >= 6:
                        break
                j += 1
            if len(vals) >= 6:
                for k, v in enumerate(vals[:6]):
                    if -3.0 <= v <= 3.0:
                        agents[k].tilt = v
                break

    # ── 事故率 ──
    vals = find_six_values_after("事故率")
    if vals:
        for i, v in enumerate(vals):
            agents[i].accident_rate = v

    # ── F数 / L数 ──
    for line in lines:
        if 'F' in line and 'L' in line:
            fl = re.findall(r'F(\d)\s*L(\d)', line)
            if len(fl) >= 6:
                for i, (f, l) in enumerate(fl[:6]):
                    agents[i].flying_count = int(f)
                    agents[i].late_count = int(l)
                break

    # ── 決まり手回数 ──
    for km_name, attr_name in [("逃げ","nige_count"),("差し","sashi_count"),
                                ("捲り差し","makuri_sashi_count"),
                                ("捲り","makuri_count"),("抜き","nuki_count")]:
        vals = find_six_ints_after(km_name)
        if vals:
            for i, v in enumerate(vals):
                setattr(agents[i], attr_name, v)

    return agents, conditions


def parse_any_text(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
    """テキスト形式を自動判定してパース"""
    if '号艇' in text and ('平均ST' in text or '勝率' in text):
        # 1行1艇の手動形式
        return parse_manual_format(text)
    else:
        # 公式サイトコピー形式
        return parse_official_site_text(text)


# ─────────────────────────────────────────────
# 6. オッズ取得・合成・期待値計算
# ─────────────────────────────────────────────

def fetch_trifecta_odds(venue_code: str, date_str: str, race_no: int) -> Dict[str, float]:
    """公式サイトから3連単オッズを取得"""
    url = (f"https://www.boatrace.jp/owpc/pc/race/odds3t"
           f"?rno={race_no}&jcd={venue_code}&hd={date_str}")
    try:
        from urllib.request import urlopen, Request
        from bs4 import BeautifulSoup
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=15).read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        cells = soup.select("td.oddsPoint")
        if len(cells) < 120:
            return {}
        odds = {}
        # 公式サイトの3連単テーブル: 6列 × 20行
        # 列順: 1着=1,2,3,4,5,6 の各グループ
        # 各列内で 2着-3着の組み合わせが行順に並ぶ
        perms = []
        for first in range(1, 7):
            others = [x for x in range(1, 7) if x != first]
            for second in others:
                for third in [x for x in others if x != second]:
                    perms.append((first, second, third))
        for i, (f, s, t) in enumerate(perms):
            if i < len(cells):
                txt = cells[i].get_text(strip=True).replace(",", "")
                try:
                    odds[f"{f}-{s}-{t}"] = float(txt)
                except ValueError:
                    pass
        return odds
    except Exception as e:
        st.warning(f"オッズ取得エラー: {e}")
        return {}


def parse_pasted_odds(text: str) -> Dict[str, float]:
    """貼り付けテキストからオッズを解析"""
    odds = {}

    # パターン1: "1-2-3 6.2" 形式
    pattern1 = re.findall(r'(\d)-(\d)-(\d)\s+([\d.]+)', text)
    if pattern1:
        for f, s, t, o in pattern1:
            key = f"{f}-{s}-{t}"
            try:
                odds[key] = float(o)
            except ValueError:
                pass
        if len(odds) >= 10:
            return odds

    # パターン2: 数値だけの列（120個以上）
    all_nums = re.findall(r'[\d]+\.[\d]+', text)
    if len(all_nums) >= 120:
        perms = []
        for first in range(1, 7):
            others = [x for x in range(1, 7) if x != first]
            for second in others:
                for third in [x for x in others if x != second]:
                    perms.append((first, second, third))
        for i, (f, s, t) in enumerate(perms):
            if i < len(all_nums):
                try:
                    odds[f"{f}-{s}-{t}"] = float(all_nums[i])
                except ValueError:
                    pass
        if len(odds) >= 60:
            return odds

    # パターン3: 部分的な "f-s-t odds" 混在テキスト
    for line in text.strip().split('\n'):
        m = re.findall(r'(\d)[‐\-](\d)[‐\-](\d)[\s,]+([\d.]+)', line)
        for f, s, t, o in m:
            try:
                odds[f"{f}-{s}-{t}"] = float(o)
            except ValueError:
                pass

    return odds


def compute_synthetic_odds(trifecta: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """3連単オッズから合成オッズ（3連複・2連単・2連複・拡連複）を計算"""
    trio = {}    # 3連複
    exacta = {}  # 2連単
    quinella = {} # 2連複
    wide = {}    # 拡連複

    # 逆数の合計で合成
    inv_trio = {}
    inv_exacta = {}
    inv_quinella = {}
    inv_wide = {}

    for key, odds_val in trifecta.items():
        if odds_val <= 0:
            continue
        parts = key.split("-")
        if len(parts) != 3:
            continue
        f, s, t = int(parts[0]), int(parts[1]), int(parts[2])
        inv = 1.0 / odds_val

        # 3連複: 同じ3艇の全順列を合算
        trio_key = "=".join(str(x) for x in sorted([f, s, t]))
        inv_trio[trio_key] = inv_trio.get(trio_key, 0) + inv

        # 2連単: 1着-2着が同じものを合算
        ex_key = f"{f}-{s}"
        inv_exacta[ex_key] = inv_exacta.get(ex_key, 0) + inv

        # 2連複: 1着-2着の順序なし
        q_key = "-".join(str(x) for x in sorted([f, s]))
        inv_quinella[q_key] = inv_quinella.get(q_key, 0) + inv

        # 拡連複: 3着以内の2艇ペア
        for pair in itertools.combinations(sorted([f, s, t]), 2):
            w_key = "-".join(str(x) for x in pair)
            inv_wide[w_key] = inv_wide.get(w_key, 0) + inv

    for key, inv_sum in inv_trio.items():
        if inv_sum > 0:
            trio[key] = round(1.0 / inv_sum, 1)
    for key, inv_sum in inv_exacta.items():
        if inv_sum > 0:
            exacta[key] = round(1.0 / inv_sum, 1)
    for key, inv_sum in inv_quinella.items():
        if inv_sum > 0:
            quinella[key] = round(1.0 / inv_sum, 1)
    for key, inv_sum in inv_wide.items():
        if inv_sum > 0:
            wide[key] = round(1.0 / inv_sum, 1)

    return {"trio": trio, "exacta": exacta, "quinella": quinella, "wide": wide}


def run_ev_simulation(agents: List[BoatAgent], conditions: RaceCondition,
                      venue_name: str, month: int,
                      n_sims: int = 10000) -> Dict[str, Dict[str, float]]:
    """モンテカルロで各賭式の出現確率を計算"""
    sim = RaceSimulator(agents, conditions, venue_name, month)

    trifecta_count = {}
    trio_count = {}
    exacta_count = {}
    quinella_count = {}
    wide_count = {}

    bar = st.progress(0)
    update_interval = max(1, n_sims // 100)

    for i in range(n_sims):
        if i % update_interval == 0:
            bar.progress(min(i / n_sims, 1.0))

        result = sim.simulate_race()
        fo = result["finish_order"]
        f1, f2, f3 = fo[1], fo[2], fo[3]

        # 3連単
        tri_key = f"{f1}-{f2}-{f3}"
        trifecta_count[tri_key] = trifecta_count.get(tri_key, 0) + 1

        # 3連複
        trio_key = "=".join(str(x) for x in sorted([f1, f2, f3]))
        trio_count[trio_key] = trio_count.get(trio_key, 0) + 1

        # 2連単
        ex_key = f"{f1}-{f2}"
        exacta_count[ex_key] = exacta_count.get(ex_key, 0) + 1

        # 2連複
        q_key = "-".join(str(x) for x in sorted([f1, f2]))
        quinella_count[q_key] = quinella_count.get(q_key, 0) + 1

        # 拡連複
        for pair in itertools.combinations(sorted([f1, f2, f3]), 2):
            w_key = "-".join(str(x) for x in pair)
            wide_count[w_key] = wide_count.get(w_key, 0) + 1

    bar.progress(1.0)

    def to_prob(counts):
        return {k: round(v / n_sims, 6) for k, v in counts.items()}

    return {
        "trifecta": to_prob(trifecta_count),
        "trio": to_prob(trio_count),
        "exacta": to_prob(exacta_count),
        "quinella": to_prob(quinella_count),
        "wide": to_prob(wide_count),
    }


def compute_expected_values(probs: Dict[str, Dict[str, float]],
                            synthetic_odds: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
    """確率 × オッズ で期待値を計算"""
    results = {}
    bet_map = {
        "trifecta": "trifecta",  # 3連単のオッズは元データをそのまま使う（後で渡す）
        "trio": "trio",
        "exacta": "exacta",
        "quinella": "quinella",
        "wide": "wide",
    }

    for bet_type in ["trifecta", "trio", "exacta", "quinella", "wide"]:
        prob_dict = probs.get(bet_type, {})
        if bet_type == "trifecta":
            odds_dict = synthetic_odds.get("_trifecta_raw", {})
        else:
            odds_dict = synthetic_odds.get(bet_type, {})

        ev_data = {}
        for key, prob in prob_dict.items():
            if key in odds_dict and odds_dict[key] > 0:
                ev = round(prob * odds_dict[key], 4)
                if ev >= 1.2:
                    flag = "◎"
                elif ev >= 1.0:
                    flag = "○"
                elif ev >= 0.8:
                    flag = "△"
                else:
                    flag = "×"
                ev_data[key] = {
                    "prob": round(prob * 100, 2),
                    "odds": odds_dict[key],
                    "ev": round(ev, 3),
                    "flag": flag,
                }
        results[bet_type] = ev_data

    return results
# ============================================================
#  ボートレース AI シミュレーター v4.0  ─  app.py  (Part 3/3)
#  Streamlit UI
# ============================================================

st.set_page_config(page_title="ボートレース AI v4.0", layout="wide")
st.title("🚤 ボートレース AI シミュレーター v4.0")
st.caption("30項目完全エージェント × 会場別特性 × モンテカルロ × 期待値計算")

# ── 艇番カラー ──
BOAT_COLORS = {1:'#e74c3c', 2:'#000000', 3:'#2ecc71',
               4:'#3498db', 5:'#f1c40f', 6:'#9b59b6'}

# ──────────────────────────────────────
# サイドバー
# ──────────────────────────────────────
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
courses = ["1C","2C","3C","4C","5C","6C"]
rates = venue_profile["course_win_rate"]
ax_sb.bar(courses, rates, color=[BOAT_COLORS[i] for i in range(1,7)])
ax_sb.set_ylabel("1着率(%)")
ax_sb.set_title(f"{venue_name} コース別1着率")
for i, v in enumerate(rates):
    ax_sb.text(i, v + 0.5, f"{v}%", ha='center', fontsize=7)
st.sidebar.pyplot(fig_sb)
plt.close(fig_sb)

# ──────────────────────────────────────
# メインタブ
# ──────────────────────────────────────
tab_input, tab_sim, tab_mc, tab_odds, tab_ev = st.tabs(
    ["📝 データ入力", "🏁 単発シミュレーション", "📊 モンテカルロ",
     "💰 オッズ取得", "📈 期待値計算"]
)

# ============================================================
# タブ1: データ入力
# ============================================================
with tab_input:
    st.subheader("選手データ入力")
    input_method = st.radio(
        "入力方法", ["テキスト貼り付け（自動判定）", "フォーム入力"], horizontal=True
    )

    if input_method == "テキスト貼り付け（自動判定）":
        sample = """1号艇: 4251 川崎誠志 B1 43歳 山口 体重52.0 F0 L0 平均ST0.21 勝率4.61 2連対26.6 3連対42.2 当地勝率4.42 枠別1着38.5 能力46 モーター+0.15 モ2連30.5 展示T6.78 展示順2 周り足4.8 チルト-0.5 事故率0.00 逃げ5 差し2 捲り0 捲り差し0 抜き1
2号艇: 3660 長谷川親王 B1 56歳 香川 体重54.0 F0 L0 平均ST0.20 勝率3.18 2連対6.4 3連対20.5 当地勝率3.00 枠別1着0.0 能力43 モーター+0.02 モ2連28.0 展示T6.85 展示順5 周り足5.1 チルト-0.5 事故率0.00 逃げ0 差し0 捲り0 捲り差し0 抜き0
3号艇: 3554 仲口博崇 A1 53歳 愛知 体重52.5 F0 L0 平均ST0.18 勝率6.66 2連対54.2 3連対71.6 当地勝率6.50 枠別1着7.7 能力53 モーター-0.32 モ2連25.0 展示T6.72 展示順1 周り足4.5 チルト-0.5 事故率0.00 逃げ0 差し3 捲り5 捲り差し2 抜き2
4号艇: 4729 佐藤謙史朗 B2 37歳 福岡 体重51.5 F0 L0 平均ST0.20 勝率4.00 2連対17.3 3連対25.0 当地勝率3.80 枠別1着12.5 能力46 モーター+0.11 モ2連32.0 展示T6.82 展示順4 周り足4.9 チルト-0.5 事故率0.00 逃げ0 差し1 捲り0 捲り差し1 抜き0
5号艇: 4973 栗原直也 A2 33歳 埼玉 体重52.0 F0 L0 平均ST0.17 勝率5.15 2連対30.2 3連対50.9 当地勝率5.00 枠別1着5.9 能力50 モーター-0.39 モ2連22.0 展示T6.75 展示順3 周り足4.6 チルト-0.5 事故率0.00 逃げ0 差し1 捲り2 捲り差し3 抜き1
6号艇: 3614 谷勝幸 B1 53歳 広島 体重53.5 F0 L0 平均ST0.16 勝率4.37 2連対28.9 3連対39.2 当地勝率4.20 枠別1着0.0 能力46 モーター-0.29 モ2連24.0 展示T6.88 展示順6 周り足5.2 チルト-0.5 事故率0.00 逃げ0 差し0 捲り1 捲り差し0 抜き1
天候: 気温12℃ 雨 風速1m 水温12℃ 波高1cm"""
        text_data = st.text_area("出走表データを貼り付け", value=sample, height=320)
        st.info("💡 公式サイトからのコピペでも、上記のような1行形式でも自動判定します。")

    else:
        # ─── フォーム入力 ───
        text_data = None
        st.markdown("#### 各艇の基本情報")
        form_agents = []
        for i in range(1, 7):
            with st.expander(f"🚤 {i}号艇", expanded=(i <= 2)):
                c1, c2, c3, c4 = st.columns(4)
                number = c1.number_input("登録番号", 0, 9999, 0, key=f"num_{i}")
                name = c2.text_input("名前", f"選手{i}", key=f"name_{i}")
                rank = c3.selectbox("級", ["A1","A2","B1","B2"], index=2, key=f"rank_{i}")
                age = c4.number_input("年齢", 18, 70, 30, key=f"age_{i}")

                c5, c6, c7, c8 = st.columns(4)
                weight = c5.number_input("体重(kg)", 40.0, 70.0, 52.0, 0.1, key=f"wt_{i}")
                avg_st = c6.number_input("平均ST", 0.01, 0.30, 0.18, 0.01, key=f"st_{i}")
                win_rate = c7.number_input("勝率", 0.0, 12.0, 5.0, 0.01, key=f"wr_{i}")
                local_wr = c8.number_input("当地勝率", 0.0, 12.0, 5.0, 0.01, key=f"lwr_{i}")

                c9, c10, c11, c12 = st.columns(4)
                top2 = c9.number_input("2連対率(%)", 0.0, 100.0, 30.0, 0.1, key=f"t2_{i}")
                top3 = c10.number_input("3連対率(%)", 0.0, 100.0, 50.0, 0.1, key=f"t3_{i}")
                lane_wr = c11.number_input("枠別1着率(%)", 0.0, 100.0, 10.0, 0.1, key=f"lw_{i}")
                ability = c12.number_input("能力", 1, 100, 50, key=f"ab_{i}")

                c13, c14, c15, c16 = st.columns(4)
                motor_p = c13.number_input("モーター貢献P", -1.0, 1.0, 0.0, 0.01, key=f"mo_{i}")
                motor_t2 = c14.number_input("モーター2連率(%)", 0.0, 100.0, 30.0, 0.1, key=f"mt2_{i}")
                ex_time = c15.number_input("展示タイム(秒)", 6.0, 7.5, 6.80, 0.01, key=f"ext_{i}")
                ex_rank = c16.number_input("展示順位", 1, 6, 3, key=f"exr_{i}")

                c17, c18, c19, c20 = st.columns(4)
                turn_t = c17.number_input("周り足", 0.0, 10.0, 0.0, 0.1, key=f"turn_{i}")
                tilt = c18.number_input("チルト", -3.0, 3.0, -0.5, 0.5, key=f"tilt_{i}")
                acc_rate = c19.number_input("事故率", 0.0, 1.0, 0.0, 0.01, key=f"acc_{i}")
                fly_cnt = c20.number_input("F数", 0, 5, 0, key=f"fly_{i}")

                c21, c22, c23, c24, c25 = st.columns(5)
                nige = c21.number_input("逃げ", 0, 50, 0, key=f"nige_{i}")
                sashi = c22.number_input("差し", 0, 50, 0, key=f"sashi_{i}")
                makuri = c23.number_input("捲り", 0, 50, 0, key=f"makuri_{i}")
                makuri_s = c24.number_input("捲差", 0, 50, 0, key=f"makuris_{i}")
                nuki = c25.number_input("抜き", 0, 50, 0, key=f"nuki_{i}")

                form_agents.append(BoatAgent(
                    lane=i, number=number, name=name, rank=rank, age=age,
                    weight=weight, avg_st=avg_st, win_rate=win_rate,
                    local_win_rate=local_wr, top2_rate=top2, top3_rate=top3,
                    lane_win_rate=lane_wr, ability=ability,
                    motor_contribution=motor_p, motor_top2_rate=motor_t2,
                    exhibition_time=ex_time, exhibition_rank=ex_rank,
                    turn_time=turn_t, tilt=tilt, accident_rate=acc_rate,
                    flying_count=fly_cnt,
                    nige_count=nige, sashi_count=sashi, makuri_count=makuri,
                    makuri_sashi_count=makuri_s, nuki_count=nuki,
                ))

        st.markdown("#### 天候条件")
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        w_weather = wc1.selectbox("天候", ["晴","曇","雨","雪"])
        w_temp = wc2.number_input("気温(℃)", -5.0, 45.0, 12.0, 0.5)
        w_wind = wc3.number_input("風速(m)", 0.0, 15.0, 1.0, 0.5)
        w_wtemp = wc4.number_input("水温(℃)", 0.0, 35.0, 12.0, 0.5)
        w_wave = wc5.number_input("波高(cm)", 0.0, 30.0, 1.0, 0.5)
        form_conditions = RaceCondition(
            weather=w_weather, temperature=w_temp,
            wind_speed=w_wind, water_temp=w_wtemp, wave_height=w_wave
        )

    # ─── 確定ボタン ───
    if st.button("✅ データ確定", type="primary", key="btn_confirm"):
        if input_method == "テキスト貼り付け（自動判定）":
            agents, conditions = parse_any_text(text_data)
        else:
            agents = form_agents
            conditions = form_conditions
        st.session_state['agents'] = agents
        st.session_state['conditions'] = conditions
        st.success(f"✅ {len(agents)}艇のデータを確定しました")

    # ─── 確定済みデータ表示 ───
    if 'agents' in st.session_state:
        st.markdown("---")
        st.markdown("#### ✅ 確定済み選手データ")

        # 基本情報テーブル
        st.markdown("**基本情報・成績**")
        df_basic = pd.DataFrame([{
            "艇": a.lane, "番号": a.number, "名前": a.name, "級": a.rank,
            "年齢": a.age, "体重": a.weight, "F": a.flying_count, "L": a.late_count,
            "平均ST": a.avg_st, "勝率": a.win_rate, "2連対": a.top2_rate,
            "3連対": a.top3_rate, "当地勝率": a.local_win_rate,
            "枠1着%": a.lane_win_rate, "能力": a.ability, "事故率": a.accident_rate,
        } for a in st.session_state['agents']])
        st.dataframe(df_basic, use_container_width=True, hide_index=True)

        # 機力・展示テーブル
        st.markdown("**機力・展示・直前情報**")
        df_machine = pd.DataFrame([{
            "艇": a.lane, "モーターP": a.motor_contribution,
            "モ2連率": a.motor_top2_rate, "展示T": a.exhibition_time,
            "展示順": a.exhibition_rank, "周り足": a.turn_time,
            "チルト": a.tilt,
            "機力スコア": round(a.get_machine_score(), 3),
            "旋回スコア": round(a.get_turn_score(), 3),
            "パワースコア": round(a.get_power_score(), 3),
        } for a in st.session_state['agents']])
        st.dataframe(df_machine, use_container_width=True, hide_index=True)

        # 決まり手テーブル
        st.markdown("**決まり手傾向**")
        df_km = pd.DataFrame([{
            "艇": a.lane, "名前": a.name,
            "逃げ": a.nige_count, "差し": a.sashi_count,
            "捲り": a.makuri_count, "捲差": a.makuri_sashi_count,
            "抜き": a.nuki_count,
        } for a in st.session_state['agents']])
        st.dataframe(df_km, use_container_width=True, hide_index=True)

        # 天候
        cond = st.session_state['conditions']
        st.markdown(f"**天候:** {cond.weather} / 気温 {cond.temperature}℃ / "
                    f"風速 {cond.wind_speed}m / 水温 {cond.water_temp}℃ / "
                    f"波高 {cond.wave_height}cm")

        # デフォルト値チェック
        with st.expander("🔍 デバッグ: デフォルト値の項目チェック"):
            defaults = {
                "枠1着%": (10.0, "lane_win_rate"),
                "モーターP": (0.0, "motor_contribution"),
                "展示T": (6.80, "exhibition_time"),
                "周り足": (0.0, "turn_time"),
                "当地勝率": (5.0, "local_win_rate"),
                "モ2連率": (30.0, "motor_top2_rate"),
            }
            warns = []
            for a in st.session_state['agents']:
                for label, (default_val, attr) in defaults.items():
                    val = getattr(a, attr)
                    if val == default_val:
                        warns.append(f"⚠️ {a.lane}号艇 {a.name}: {label} = {val}（デフォルト）")
            if warns:
                for w in warns:
                    st.write(w)
            else:
                st.write("✅ すべての項目が正常に取得されています。")


# ============================================================
# タブ2: 単発シミュレーション
# ============================================================
with tab_sim:
    st.subheader("🏁 単発レースシミュレーション")
    if 'agents' not in st.session_state:
        st.info("先に「📝 データ入力」タブでデータを確定してください。")
    else:
        n_trials = st.slider("試行回数", 1, 10, 3, key="sim_trials")
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

                res_df = pd.DataFrame([{
                    "着順": pos, "艇番": boat,
                    "選手名": name_map.get(boat, ""),
                    "ST": f"{result['st_times'].get(boat, 0):.3f}"
                } for pos, boat in fo.items()])
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                t1, t2, t3 = fo[1], fo[2], fo[3]
                trio_s = sorted([t1, t2, t3])
                st.write(
                    f"3連単: **{t1}-{t2}-{t3}**　/　"
                    f"3連複: **{trio_s[0]}={trio_s[1]}={trio_s[2]}**　/　"
                    f"2連単: **{t1}-{t2}**"
                )

                # レース展開グラフ
                fig, ax = plt.subplots(figsize=(10, 4))
                for lane, pos_hist in result["positions"].items():
                    ax.plot(pos_hist, color=BOAT_COLORS.get(lane, 'gray'),
                            label=f"{lane}号艇 {name_map.get(lane,'')}", linewidth=1.5)
                ax.set_xlabel("ステップ")
                ax.set_ylabel("順位")
                ax.invert_yaxis()
                ax.set_title(f"レース展開（第{trial+1}試行）")
                ax.legend(loc='upper right', fontsize=7)
                ax.set_yticks([1,2,3,4,5,6])
                st.pyplot(fig)
                plt.close(fig)

                # 重みの内訳表示
                with st.expander(f"📊 第{trial+1}レース 重み詳細"):
                    w_df = pd.DataFrame([{
                        "艇番": lane,
                        "選手": name_map.get(lane, ""),
                        "重み": f"{w:.4f}",
                        "確率(%)": f"{w*100:.1f}%"
                    } for lane, w in sorted(result["weights"].items())])
                    st.dataframe(w_df, use_container_width=True, hide_index=True)


# ============================================================
# タブ3: モンテカルロ
# ============================================================
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

            win_counts = {a.lane: 0 for a in agents}
            top2_counts = {a.lane: 0 for a in agents}
            top3_counts = {a.lane: 0 for a in agents}
            kimarite_counts = {}
            trifecta_counts = {}

            sim = RaceSimulator(agents, conditions, venue_name, race_month)
            bar = st.progress(0)
            update_interval = max(1, n_mc // 100)

            for i in range(n_mc):
                if i % update_interval == 0:
                    bar.progress(min(i / n_mc, 1.0))
                result = sim.simulate_race()
                fo = result["finish_order"]

                win_counts[fo[1]] += 1
                top2_counts[fo[1]] += 1
                top2_counts[fo[2]] += 1
                top3_counts[fo[1]] += 1
                top3_counts[fo[2]] += 1
                top3_counts[fo[3]] += 1

                km = result["kimarite"]
                kimarite_counts[km] = kimarite_counts.get(km, 0) + 1

                tri_key = f"{fo[1]}-{fo[2]}-{fo[3]}"
                trifecta_counts[tri_key] = trifecta_counts.get(tri_key, 0) + 1

            bar.progress(1.0)

            # 勝率テーブル
            st.markdown("#### 勝率・連対率・3連対率")
            mc_df = pd.DataFrame([{
                "艇番": lane,
                "選手": name_map.get(lane, ""),
                "1着率": f"{win_counts[lane]/n_mc*100:.1f}%",
                "2連対率": f"{top2_counts[lane]/n_mc*100:.1f}%",
                "3連対率": f"{top3_counts[lane]/n_mc*100:.1f}%",
            } for lane in sorted(win_counts.keys())])
            st.dataframe(mc_df, use_container_width=True, hide_index=True)

            # 勝率棒グラフ
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            lanes = sorted(win_counts.keys())
            win_pcts = [win_counts[l]/n_mc*100 for l in lanes]
            labels = [f"{l}号艇\n{name_map.get(l,'')}" for l in lanes]
            ax2.bar(labels, win_pcts, color=[BOAT_COLORS.get(l,'gray') for l in lanes])
            ax2.set_ylabel("1着率(%)")
            ax2.set_title(f"モンテカルロ {n_mc:,}回 - 1着率")
            for idx, v in enumerate(win_pcts):
                ax2.text(idx, v + 0.3, f"{v:.1f}%", ha='center', fontsize=9)
            st.pyplot(fig2)
            plt.close(fig2)

            # 決まり手分布
            st.markdown("#### 決まり手分布")
            km_df = pd.DataFrame([{
                "決まり手": k, "回数": v, "割合": f"{v/n_mc*100:.1f}%"
            } for k, v in sorted(kimarite_counts.items(), key=lambda x: -x[1])])
            st.dataframe(km_df, use_container_width=True, hide_index=True)

            # 3連単上位
            st.markdown("#### 3連単 出現頻度 Top20")
            sorted_tri = sorted(trifecta_counts.items(), key=lambda x: -x[1])[:20]
            tri_df = pd.DataFrame([{
                "買い目": k, "回数": v, "確率": f"{v/n_mc*100:.2f}%"
            } for k, v in sorted_tri])
            st.dataframe(tri_df, use_container_width=True, hide_index=True)

            st.session_state['mc_done'] = True


# ============================================================
# タブ4: オッズ取得
# ============================================================
with tab_odds:
    st.subheader("💰 オッズ取得")
    odds_method = st.radio(
        "取得方法",
        ["🌐 自動取得（公式サイト）", "📋 テキスト貼り付け", "✏️ 手動入力"],
        horizontal=True, key="odds_method"
    )

    if odds_method == "🌐 自動取得（公式サイト）":
        st.info(f"会場: {venue_name}（{venue_code}） / 日付: {date_str} / {race_no}R")
        if st.button("🔄 オッズ自動取得", type="primary", key="fetch_odds"):
            with st.spinner("公式サイトからオッズを取得中..."):
                odds = fetch_trifecta_odds(venue_code, date_str, race_no)
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りのオッズを取得しました")
            else:
                st.error("取得失敗。テキスト貼り付けをお試しください。")

    elif odds_method == "📋 テキスト貼り付け":
        odds_text = st.text_area(
            "オッズデータを貼り付け", height=200, key="odds_text",
            placeholder="1-2-3 6.2\n1-3-2 8.5\nまたは公式サイトの表をそのままコピペ"
        )
        if st.button("📥 解析", key="parse_odds"):
            odds = parse_pasted_odds(odds_text)
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りのオッズを解析しました")
            else:
                st.error("解析失敗。形式を確認してください。")

    else:
        manual_text = st.text_area(
            "オッズを入力（1行1組: 1-2-3 6.2）", height=200, key="manual_odds",
            placeholder="1-2-3 6.2\n1-3-2 8.5\n..."
        )
        if st.button("📥 登録", key="register_odds"):
            odds = {}
            for line in manual_text.strip().split('\n'):
                m = re.match(r'(\d)-(\d)-(\d)\s+([\d.]+)', line.strip())
                if m:
                    key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    odds[key] = float(m.group(4))
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りを登録しました")

    # オッズ表示
    if 'trifecta_odds' in st.session_state:
        odds = st.session_state['trifecta_odds']
        st.markdown("---")
        st.markdown("#### 取得済み3連単オッズ（低配当順 Top 20）")
        sorted_odds = sorted(odds.items(), key=lambda x: x[1])
        top_df = pd.DataFrame([{"買い目": k, "オッズ": v} for k, v in sorted_odds[:20]])
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.write(f"合計: {len(odds)}通り / 最低: {sorted_odds[0][1]} / 最高: {sorted_odds[-1][1]}")

        # 合成オッズ
        synthetic = compute_synthetic_odds(odds)
        synthetic["_trifecta_raw"] = odds  # EV計算用に3連単も保持
        st.session_state['synthetic_odds'] = synthetic

        with st.expander("📊 合成オッズ（2連単・2連複・3連複・拡連複）"):
            for bet_type, label in [("exacta","2連単"),("quinella","2連複"),
                                     ("trio","3連複"),("wide","拡連複")]:
                st.markdown(f"**{label}**")
                s = sorted(synthetic[bet_type].items(), key=lambda x: x[1])
                sdf = pd.DataFrame([{"買い目": k, "合成オッズ": v} for k, v in s[:15]])
                st.dataframe(sdf, use_container_width=True, hide_index=True)


# ============================================================
# タブ5: 期待値計算
# ============================================================
with tab_ev:
    st.subheader("📈 期待値 (EV) 計算")
    if 'agents' not in st.session_state:
        st.info("先に「📝 データ入力」タブでデータを確定してください。")
    elif 'trifecta_odds' not in st.session_state:
        st.info("先に「💰 オッズ取得」タブでオッズを取得してください。")
    else:
        ev_sims = st.slider("EV計算用シミュレーション回数", 1000, 50000, 10000, 1000, key="ev_sims")

        if st.button("🚀 期待値計算 実行", type="primary", key="run_ev"):
            agents = st.session_state['agents']
            conditions = st.session_state['conditions']
            synthetic = st.session_state['synthetic_odds']

            st.write("⏳ モンテカルロシミュレーション実行中...")
            probs = run_ev_simulation(agents, conditions, venue_name, race_month, ev_sims)
            ev_results = compute_expected_values(probs, synthetic)
            st.session_state['ev_results'] = ev_results
            st.success("✅ 期待値計算完了！")

        if 'ev_results' in st.session_state:
            ev_results = st.session_state['ev_results']
            bet_types = ["trifecta", "trio", "exacta", "quinella", "wide"]
            bet_labels = ["3連単", "3連複", "2連単", "2連複", "拡連複"]
            ev_tabs = st.tabs(bet_labels)

            for ev_tab, bt, bl in zip(ev_tabs, bet_types, bet_labels):
                with ev_tab:
                    data = ev_results.get(bt, {})
                    if not data:
                        st.write("データなし（オッズまたは出現データが不足）")
                        continue

                    sorted_ev = sorted(data.items(), key=lambda x: -x[1]['ev'])
                    top_n = sorted_ev[:20]

                    ev_df = pd.DataFrame([{
                        "買い目": k,
                        "確率(%)": v['prob'],
                        "オッズ": v['odds'],
                        "期待値": v['ev'],
                        "判定": v['flag']
                    } for k, v in top_n])
                    st.dataframe(ev_df, use_container_width=True, hide_index=True)

                    # EV棒グラフ
                    if len(top_n) > 0:
                        top15 = top_n[:15]
                        fig_ev, ax_ev = plt.subplots(figsize=(10, 5))
                        keys = [x[0] for x in top15]
                        vals = [x[1]['ev'] for x in top15]
                        bar_cols = ['#2ecc71' if v >= 1.0 else '#f39c12' if v >= 0.8
                                    else '#e74c3c' for v in vals]
                        ax_ev.barh(keys[::-1], vals[::-1], color=bar_cols[::-1])
                        ax_ev.axvline(x=1.0, color='red', linestyle='--', label='EV=1.0')
                        ax_ev.set_xlabel("期待値 (EV)")
                        ax_ev.set_title(f"{bl} 期待値ランキング Top15")
                        ax_ev.legend()
                        st.pyplot(fig_ev)
                        plt.close(fig_ev)

            # サマリー
            st.markdown("---")
            st.markdown("### 🎯 おすすめ買い目サマリー")
            all_good = []
            for bt, bl in zip(bet_types, bet_labels):
                data = ev_results.get(bt, {})
                for k, v in data.items():
                    if v['ev'] >= 1.0:
                        all_good.append({
                            "券種": bl, "買い目": k,
                            "確率(%)": v['prob'], "オッズ": v['odds'],
                            "期待値": v['ev'], "判定": v['flag']
                        })

            if all_good:
                all_good.sort(key=lambda x: -x['期待値'])
                st.markdown(f"**EV ≥ 1.0 の買い目: {len(all_good)}件**")
                good_df = pd.DataFrame(all_good)
                st.dataframe(good_df, use_container_width=True, hide_index=True)
            else:
                st.warning("EV ≥ 1.0 の買い目は見つかりませんでした。")

            # 準推奨
            all_ok = []
            for bt, bl in zip(bet_types, bet_labels):
                data = ev_results.get(bt, {})
                for k, v in data.items():
                    if 0.8 <= v['ev'] < 1.0:
                        all_ok.append({
                            "券種": bl, "買い目": k,
                            "確率(%)": v['prob'], "オッズ": v['odds'],
                            "期待値": v['ev'], "判定": v['flag']
                        })
            if all_ok:
                with st.expander(f"📋 EV 0.8〜1.0 の買い目 ({len(all_ok)}件)"):
                    all_ok.sort(key=lambda x: -x['期待値'])
                    ok_df = pd.DataFrame(all_ok)
                    st.dataframe(ok_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────
# フッター
# ──────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em;'>"
    "🚤 ボートレース AI シミュレーター v4.0<br>"
    "30項目完全エージェント × 会場別特性(全24場) × 季節補正 × モンテカルロ × 合成オッズ × 期待値計算"
    "</div>",
    unsafe_allow_html=True
)
