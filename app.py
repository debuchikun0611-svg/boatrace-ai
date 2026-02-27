# ============================================================
#  ボートレース AI シミュレーター v3.3  ─ app.py
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

def get_venue_profile(venue_name: str) -> dict:
    return VENUE_PROFILES.get(venue_name, DEFAULT_VENUE_PROFILE)

def get_season(month: int) -> str:
    if month in [3,4,5]:   return "春"
    if month in [6,7,8]:   return "夏"
    if month in [9,10,11]: return "秋"
    return "冬"

# ─────────────────────────────────────────────
# 3. データクラス
# ─────────────────────────────────────────────
@dataclass
class BoatAgent:
    lane: int
    number: int = 0
    name: str = ""
    rank: str = "B1"
    age: int = 30
    avg_st: float = 0.18
    win_rate: float = 5.0
    top2_rate: float = 30.0
    top3_rate: float = 50.0
    lane_win_rate: float = 10.0
    ability: int = 50
    motor_contribution: float = 0.0
    accident_rate: float = 0.0

    def calculate_start_timing(self) -> float:
        base = self.avg_st
        variation = np.random.normal(0, 0.02)
        return max(0.01, base + variation)

    def get_power_score(self) -> float:
        base = self.ability / 100.0
        motor = self.motor_contribution * 0.1
        rank_bonus = {"A1":0.08,"A2":0.04,"B1":0.0,"B2":-0.04}.get(self.rank, 0)
        return np.clip(base + motor + rank_bonus, 0.1, 1.0)

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
# 4. シミュレーター
# ─────────────────────────────────────────────
class RaceSimulator:
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

        weights = []
        for agent in self.agents:
            idx = agent.lane - 1
            venue_base = base_rates[idx] / 100.0 if idx < len(base_rates) else 0.05
            player_lane = agent.lane_win_rate / 100.0 if agent.lane_win_rate > 0 else venue_base
            w = venue_base * 0.6 + player_lane * 0.4

            power = agent.get_power_score()
            w *= (0.7 + power * 0.6)

            st_quality = max(0.5, 1.0 - (agent.avg_st - 0.12) * 3.0)
            w *= st_quality

            w *= (1.0 + agent.motor_contribution * 0.15)

            wind_spd = self.conditions.wind_speed
            wind_eff = profile.get("wind_effect", 0.5)
            if agent.lane <= 2:
                w *= (1.0 - wind_spd * 0.01 * wind_eff)
            elif agent.lane >= 5:
                w *= (1.0 + wind_spd * 0.005 * wind_eff)

            if profile.get("tide", False):
                if self.conditions.tide == "満潮" and agent.lane <= 2:
                    w *= 1.03
                elif self.conditions.tide == "干潮" and agent.lane >= 4:
                    w *= 1.02

            weights.append(max(w, 0.001))

        total = sum(weights)
        return [w / total for w in weights]

    def simulate_race(self) -> dict:
        st_times = {}
        for agent in self.agents:
            st_times[agent.lane] = agent.calculate_start_timing()

        weights = self._compute_race_weights()
        adjusted = list(weights)

        min_st = min(st_times.values())
        for i, agent in enumerate(self.agents):
            st_diff = st_times[agent.lane] - min_st
            bonus = max(0, (0.05 - st_diff) * 2)
            adjusted[i] += bonus

        kimarite_probs = self.profile.get("kimarite", {})
        if np.random.random() < kimarite_probs.get("捲り", 0.13):
            for i, agent in enumerate(self.agents):
                if agent.lane >= 3:
                    adjusted[i] *= 1.15

        for i in range(len(adjusted)):
            adjusted[i] *= np.random.uniform(0.85, 1.15)
            adjusted[i] = max(adjusted[i], 0.001)

        total = sum(adjusted)
        probs = [a / total for a in adjusted]

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
        if winner_lane == 1:
            return "逃げ"
        elif winner_lane == 2:
            if st_times.get(2, 0.2) < st_times.get(1, 0.2):
                return np.random.choice(["差し", "捲り"], p=[0.65, 0.35])
            return "差し"
        elif winner_lane == 3:
            return np.random.choice(["捲り", "捲り差し", "差し"], p=[0.45, 0.40, 0.15])
        elif winner_lane == 4:
            return np.random.choice(["捲り", "捲り差し", "差し"], p=[0.45, 0.35, 0.20])
        else:
            return np.random.choice(["捲り", "捲り差し", "抜き"], p=[0.40, 0.40, 0.20])


# ─────────────────────────────────────────────
# 5. オッズ取得 & 合成オッズ
# ─────────────────────────────────────────────
def fetch_trifecta_odds(venue_code: str, date_str: str, race_no: int) -> dict:
    """
    boatrace.jp の3連単オッズを取得。
    テーブル構造: 6列（1着=1~6号艇）× 20行（2着-3着の組合せ）= 120セル。
    """
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
                try:
                    odds_vals.append(float(txt.replace(',', '')))
                except ValueError:
                    odds_vals.append(0.0)
            break

    if len(odds_vals) < 120:
        st.warning(f"⚠️ oddsPoint セルが {len(odds_vals)} 個しか見つかりません")
        return {}

    boats = [1, 2, 3, 4, 5, 6]

    def get_column_order(first: int) -> list:
        others = sorted([b for b in boats if b != first])
        order = []
        for second in others:
            thirds = sorted([b for b in others if b != second])
            for third in thirds:
                order.append((first, second, third))
        return order

    column_orders = [get_column_order(f) for f in boats]

    odds_dict = {}
    for row_idx in range(20):
        for col_idx in range(6):
            cell_idx = row_idx * 6 + col_idx
            if cell_idx < len(odds_vals):
                f, s, t = column_orders[col_idx][row_idx]
                odds_dict[f"{f}-{s}-{t}"] = odds_vals[cell_idx]

    return odds_dict


def parse_pasted_odds(text: str) -> dict:
    """貼り付けテキストからオッズを解析"""
    odds_dict = {}

    # パターン1: "1-2-3  6.2" 形式
    pattern1 = re.findall(r'(\d)-(\d)-(\d)\s+([\d,.]+)', text)
    if pattern1:
        for m in pattern1:
            key = f"{m[0]}-{m[1]}-{m[2]}"
            try:
                odds_dict[key] = float(m[3].replace(',', ''))
            except ValueError:
                pass
        if len(odds_dict) >= 10:
            return odds_dict

    # パターン2: 数値の羅列（120個以上）
    nums = re.findall(r'[\d]+\.[\d]+', text)
    if len(nums) >= 120:
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
                if cell_idx < len(nums):
                    f, s, t = column_orders[col_idx][row_idx]
                    try:
                        odds_dict[f"{f}-{s}-{t}"] = float(nums[cell_idx])
                    except ValueError:
                        pass
        return odds_dict

    # パターン3: スペース区切り
    all_nums = []
    for line in text.strip().split('\n'):
        found = re.findall(r'[\d]+\.?\d*', line.strip())
        for n in found:
            try:
                v = float(n)
                if v > 1.0:
                    all_nums.append(v)
            except ValueError:
                pass

    if len(all_nums) >= 120:
        boats = [1, 2, 3, 4, 5, 6]

        def get_column_order2(first):
            others = sorted([b for b in boats if b != first])
            order = []
            for second in others:
                thirds = sorted([b for b in others if b != second])
                for third in thirds:
                    order.append((first, second, third))
            return order

        column_orders = [get_column_order2(f) for f in boats]
        for row_idx in range(20):
            for col_idx in range(6):
                cell_idx = row_idx * 6 + col_idx
                if cell_idx < len(all_nums):
                    f, s, t = column_orders[col_idx][row_idx]
                    odds_dict[f"{f}-{s}-{t}"] = all_nums[cell_idx]

    return odds_dict


def compute_synthetic_odds(trifecta: dict) -> dict:
    """3連単オッズから他券種の合成オッズを算出"""
    boats = [1, 2, 3, 4, 5, 6]
    result = {"trifecta": trifecta, "trio": {}, "exacta": {},
              "quinella": {}, "wide": {}}

    # 3連複
    for combo in itertools.combinations(boats, 3):
        inv_sum = 0
        key = f"{combo[0]}={combo[1]}={combo[2]}"
        for perm in itertools.permutations(combo):
            pk = f"{perm[0]}-{perm[1]}-{perm[2]}"
            if pk in trifecta and trifecta[pk] > 0:
                inv_sum += 1.0 / trifecta[pk]
        result["trio"][key] = round(1.0 / inv_sum, 1) if inv_sum > 0 else 0

    # 2連単
    for p in itertools.permutations(boats, 2):
        inv_sum = 0
        key = f"{p[0]}-{p[1]}"
        for third in boats:
            if third != p[0] and third != p[1]:
                pk = f"{p[0]}-{p[1]}-{third}"
                if pk in trifecta and trifecta[pk] > 0:
                    inv_sum += 1.0 / trifecta[pk]
        result["exacta"][key] = round(1.0 / inv_sum, 1) if inv_sum > 0 else 0

    # 2連複
    for combo in itertools.combinations(boats, 2):
        inv_sum = 0
        key = f"{combo[0]}={combo[1]}"
        for perm in itertools.permutations(combo):
            for third in boats:
                if third != perm[0] and third != perm[1]:
                    pk = f"{perm[0]}-{perm[1]}-{third}"
                    if pk in trifecta and trifecta[pk] > 0:
                        inv_sum += 1.0 / trifecta[pk]
        result["quinella"][key] = round(1.0 / inv_sum, 1) if inv_sum > 0 else 0

    # 拡連複
    for combo in itertools.combinations(boats, 2):
        inv_sum = 0
        key = f"{combo[0]}={combo[1]}"
        for perm in itertools.permutations(boats, 3):
            if combo[0] in perm and combo[1] in perm:
                pk = f"{perm[0]}-{perm[1]}-{perm[2]}"
                if pk in trifecta and trifecta[pk] > 0:
                    inv_sum += 1.0 / trifecta[pk]
        result["wide"][key] = round(1.0 / inv_sum, 1) if inv_sum > 0 else 0

    return result


# ─────────────────────────────────────────────
# 6. モンテカルロ期待値シミュレーション
# ─────────────────────────────────────────────
def run_ev_simulation(agents, conditions, venue_name, month, n_sims=10000):
    boats = [1, 2, 3, 4, 5, 6]
    counts = {
        'trifecta': {}, 'trio': {}, 'exacta': {},
        'quinella': {}, 'wide': {}
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

    for i in range(n_sims):
        if i % max(1, n_sims // 100) == 0:
            bar.progress(min(i / n_sims, 1.0))

        result = sim.simulate_race()
        fo = result["finish_order"]
        first, second, third = fo[1], fo[2], fo[3]

        # 3連単
        tk = f"{first}-{second}-{third}"
        if tk in counts['trifecta']:
            counts['trifecta'][tk] += 1

        # 3連複
        trio_key = "=".join(str(x) for x in sorted([first, second, third]))
        if trio_key in counts['trio']:
            counts['trio'][trio_key] += 1

        # 2連単
        ek = f"{first}-{second}"
        if ek in counts['exacta']:
            counts['exacta'][ek] += 1

        # 2連複
        qk = "=".join(str(x) for x in sorted([first, second]))
        if qk in counts['quinella']:
            counts['quinella'][qk] += 1

        # 拡連複
        top3 = sorted([first, second, third])
        for combo in itertools.combinations(top3, 2):
            wk = f"{combo[0]}={combo[1]}"
            if wk in counts['wide']:
                counts['wide'][wk] += 1

    bar.progress(1.0)

    probs = {}
    for bet_type in counts:
        probs[bet_type] = {}
        for key, cnt in counts[bet_type].items():
            probs[bet_type][key] = cnt / n_sims

    return probs


def compute_expected_values(probs: dict, synthetic_odds: dict) -> dict:
    """EV = 確率 × オッズ"""
    ev = {}
    for bet_type in probs:
        ev[bet_type] = {}
        odds_map = synthetic_odds.get(bet_type, {})
        for key, prob in probs[bet_type].items():
            odds_val = odds_map.get(key, 0)
            ev_val = prob * odds_val
            if ev_val > 0:
                ev[bet_type][key] = {
                    "prob": round(prob * 100, 2),
                    "odds": odds_val,
                    "ev": round(ev_val, 3),
                    "flag": "◎" if ev_val >= 1.2 else
                            "○" if ev_val >= 1.0 else
                            "△" if ev_val >= 0.8 else "×"
                }
    return ev
# =============================================================
#  7. Streamlit UI
# =============================================================
st.set_page_config(page_title="ボートレース AI v3.3", layout="wide")
st.title("🚤 ボートレース AI シミュレーター v3.3")
st.caption("会場別特性 × モンテカルロ × オッズ自動取得 × 期待値計算")

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
courses = ["1C", "2C", "3C", "4C", "5C", "6C"]
rates = venue_profile["course_win_rate"]
sb_colors = ['#e74c3c', '#000000', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6']
ax_sb.bar(courses, rates, color=sb_colors)
ax_sb.set_ylabel("1着率(%)")
ax_sb.set_title(f"{venue_name} コース別1着率")
for i, v in enumerate(rates):
    ax_sb.text(i, v + 0.5, f"{v}%", ha='center', fontsize=7)
st.sidebar.pyplot(fig_sb)
plt.close(fig_sb)

# ── 艇番カラー辞書（グラフ用） ──
boat_colors = {1:'#e74c3c', 2:'#000000', 3:'#2ecc71',
               4:'#3498db', 5:'#f1c40f', 6:'#9b59b6'}

# ── メイン: タブ構成 ──
tab_input, tab_sim, tab_mc, tab_odds, tab_ev = st.tabs(
    ["📝 データ入力", "🏁 単発シミュレーション", "📊 モンテカルロ",
     "💰 オッズ取得", "📈 期待値計算"]
)

# ----------------------------
# タブ1: データ入力
# ----------------------------
with tab_input:
    st.subheader("選手データ入力")
    input_method = st.radio("入力方法", ["テキスト貼り付け", "フォーム入力"], horizontal=True)

    if input_method == "テキスト貼り付け":
        sample = """1号艇: 4251 川崎誠志 B1 平均ST0.21 勝率4.61 2連対26.6 3連対42.2 枠別1着38.5 能力46 モーター+0.15
2号艇: 3660 長谷川親王 B1 平均ST0.20 勝率3.18 2連対6.4 3連対20.5 枠別1着0.0 能力43 モーター+0.02
3号艇: 3554 仲口博崇 A1 平均ST0.18 勝率6.66 2連対54.2 3連対71.6 枠別1着7.7 能力53 モーター-0.32
4号艇: 4729 佐藤謙史朗 B2 平均ST0.20 勝率4.00 2連対17.3 3連対25.0 枠別1着12.5 能力46 モーター+0.11
5号艇: 4973 栗原直也 A2 平均ST0.17 勝率5.15 2連対30.2 3連対50.9 枠別1着5.9 能力50 モーター-0.39
6号艇: 3614 谷勝幸 B1 平均ST0.16 勝率4.37 2連対28.9 3連対39.2 枠別1着0.0 能力46 モーター-0.29
天候: 12℃ 雨 風速1m 波高1cm"""
        text_data = st.text_area("出走表データを貼り付け", value=sample, height=280)
    else:
        text_data = None

    # ─── パーサー（v3.3 修正版）───
    def parse_input_text(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
        agents = []
        conditions = RaceCondition()

        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # ── 天候行 ──
            if ('天候' in line or '℃' in line) and '号艇' not in line:
                t_m = re.search(r'(\d+\.?\d*)\s*℃', line)
                if t_m:
                    conditions.temperature = float(t_m.group(1))
                w_m = re.search(r'風速\s*(\d+\.?\d*)', line)
                if w_m:
                    conditions.wind_speed = float(w_m.group(1))
                wv_m = re.search(r'波高?\s*(\d+\.?\d*)', line)
                if wv_m:
                    conditions.wave_height = float(wv_m.group(1))
                if '雨' in line:
                    conditions.weather = "雨"
                elif '曇' in line:
                    conditions.weather = "曇"
                continue

            # ── 選手行: "X号艇" を探す ──
            lane_m = re.search(r'(\d)\s*号艇', line)
            if not lane_m:
                continue
            lane = int(lane_m.group(1))

            # 登録番号（3〜5桁）
            num_m = re.search(r'[:：\s]\s*(\d{3,5})\b', line)
            number = int(num_m.group(1)) if num_m else 0

            # 選手名（番号の後の漢字2文字以上）
            name = f"選手{lane}"
            if num_m:
                after_num = line[num_m.end():]
            else:
                after_num = line[lane_m.end():]
            name_m = re.search(r'\s*([一-龥ぁ-んァ-ヶー]{2,})', after_num)
            if name_m:
                name = name_m.group(1)

            # 級別
            rank = "B1"
            for r in ["A1", "A2", "B1", "B2"]:
                if r in line:
                    rank = r
                    break

            # 平均ST（"平均ST0.21" "ST 0.21" 両対応）
            avg_st = 0.18
            st_m = re.search(r'(?:平均)?ST\s*(0\.\d+)', line)
            if st_m:
                avg_st = float(st_m.group(1))

            # 勝率
            win_rate = 5.0
            wr_m = re.search(r'勝率\s*([\d.]+)', line)
            if wr_m:
                win_rate = float(wr_m.group(1))

            # 2連対率
            top2 = 30.0
            t2_m = re.search(r'2連対?\s*([\d.]+)', line)
            if t2_m:
                top2 = float(t2_m.group(1))

            # 3連対率
            top3 = 50.0
            t3_m = re.search(r'3連対?\s*([\d.]+)', line)
            if t3_m:
                top3 = float(t3_m.group(1))

            # 枠別1着率
            lane_wr = 10.0
            lw_m = re.search(r'枠別[1１]?着?\s*([\d.]+)', line)
            if lw_m:
                lane_wr = float(lw_m.group(1))

            # 能力値
            ability = 50
            ab_m = re.search(r'能力\s*(\d+)', line)
            if ab_m:
                ability = int(ab_m.group(1))

            # モーター貢献（"+0.15" "-0.32" "+ 0.15" 対応）
            motor = 0.0
            mo_m = re.search(r'モーター?\s*([+\-]?\s*[\d.]+)', line)
            if mo_m:
                motor = float(mo_m.group(1).replace(' ', ''))

            agents.append(BoatAgent(
                lane=lane, number=number, name=name, rank=rank,
                age=30, avg_st=avg_st, win_rate=win_rate,
                top2_rate=top2, top3_rate=top3,
                lane_win_rate=lane_wr, ability=ability,
                motor_contribution=motor
            ))

        if not agents:
            for i in range(1, 7):
                agents.append(BoatAgent(lane=i, name=f"選手{i}"))

        return agents, conditions

    # ─── フォーム入力 ───
    if input_method == "フォーム入力":
        st.markdown("#### 各艇の情報")
        agents_list = []
        cols_header = st.columns([1,1,2,1,1,1,1,1,1,1,1])
        headers = ["艇","番号","名前","級","ST","勝率","2連対","3連対","枠1着","能力","モーター"]
        for c, h in zip(cols_header, headers):
            c.write(f"**{h}**")

        for i in range(1, 7):
            cols = st.columns([1,1,2,1,1,1,1,1,1,1,1])
            cols[0].write(f"**{i}**")
            number = cols[1].number_input("番号", 0, 9999, 0, key=f"num_{i}", label_visibility="collapsed")
            name = cols[2].text_input("名前", f"選手{i}", key=f"name_{i}", label_visibility="collapsed")
            rank = cols[3].selectbox("級", ["A1","A2","B1","B2"], index=2, key=f"rank_{i}", label_visibility="collapsed")
            avg_st = cols[4].number_input("ST", 0.01, 0.30, 0.18, 0.01, key=f"st_{i}", label_visibility="collapsed")
            win_rate = cols[5].number_input("勝率", 0.0, 15.0, 5.0, 0.01, key=f"wr_{i}", label_visibility="collapsed")
            top2 = cols[6].number_input("2連", 0.0, 100.0, 30.0, 0.1, key=f"t2_{i}", label_visibility="collapsed")
            top3 = cols[7].number_input("3連", 0.0, 100.0, 50.0, 0.1, key=f"t3_{i}", label_visibility="collapsed")
            lane_wr = cols[8].number_input("枠1着", 0.0, 100.0, 10.0, 0.1, key=f"lw_{i}", label_visibility="collapsed")
            ability = cols[9].number_input("能力", 1, 100, 50, key=f"ab_{i}", label_visibility="collapsed")
            motor = cols[10].number_input("モーター", -1.0, 1.0, 0.0, 0.01, key=f"mo_{i}", label_visibility="collapsed")
            agents_list.append(BoatAgent(
                lane=i, number=number, name=name, rank=rank,
                avg_st=avg_st, win_rate=win_rate,
                top2_rate=top2, top3_rate=top3,
                lane_win_rate=lane_wr, ability=ability,
                motor_contribution=motor
            ))

        st.markdown("#### 天候")
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_weather = wc1.selectbox("天候", ["晴","曇","雨","雪"])
        w_temp = wc2.number_input("気温(℃)", -5.0, 45.0, 20.0, 0.5)
        w_wind = wc3.number_input("風速(m)", 0.0, 15.0, 2.0, 0.5)
        w_wave = wc4.number_input("波高(cm)", 0.0, 30.0, 2.0, 0.5)
        form_conditions = RaceCondition(
            weather=w_weather, temperature=w_temp,
            wind_speed=w_wind, wave_height=w_wave
        )

    # ─── 確定ボタン ───
    if st.button("✅ データ確定", type="primary"):
        if input_method == "テキスト貼り付け":
            agents, conditions = parse_input_text(text_data)
        else:
            agents = agents_list
            conditions = form_conditions

        st.session_state['agents'] = agents
        st.session_state['conditions'] = conditions
        st.success(f"✅ {len(agents)}艇のデータを確定しました")

    # ─── 確定済みデータ表示 ───
    if 'agents' in st.session_state:
        st.markdown("#### 確定済み選手データ")
        agent_df = pd.DataFrame([
            {"艇": a.lane, "番号": a.number, "名前": a.name, "級": a.rank,
             "ST": a.avg_st, "勝率": a.win_rate, "2連対": a.top2_rate,
             "3連対": a.top3_rate, "枠1着": a.lane_win_rate,
             "能力": a.ability, "モーター": a.motor_contribution}
            for a in st.session_state['agents']
        ])
        st.dataframe(agent_df, use_container_width=True, hide_index=True)

        st.markdown("#### 天候条件")
        cond = st.session_state['conditions']
        st.write(f"天候: {cond.weather} / 気温: {cond.temperature}℃ / "
                 f"風速: {cond.wind_speed}m / 波高: {cond.wave_height}cm")


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
                    {"着順": pos, "艇番": boat,
                     "選手名": name_map.get(boat, ""),
                     "ST": f"{result['st_times'].get(boat, 0):.3f}"}
                    for pos, boat in fo.items()
                ])
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                t1, t2, t3 = fo[1], fo[2], fo[3]
                trio_sorted = sorted([t1, t2, t3])
                st.write(f"3連単: **{t1}-{t2}-{t3}**　/　"
                         f"3連複: **{trio_sorted[0]}={trio_sorted[1]}={trio_sorted[2]}**　/　"
                         f"2連単: **{t1}-{t2}**")

                fig, ax = plt.subplots(figsize=(10, 4))
                for lane, pos_hist in result["positions"].items():
                    ax.plot(pos_hist, color=boat_colors.get(lane, 'gray'),
                            label=f"{lane}号艇 {name_map.get(lane,'')}", linewidth=1.5)
                ax.set_xlabel("ステップ")
                ax.set_ylabel("順位")
                ax.invert_yaxis()
                ax.set_title(f"レース展開（第{trial+1}試行）")
                ax.legend(loc='upper right', fontsize=7)
                ax.set_yticks([1,2,3,4,5,6])
                st.pyplot(fig)
                plt.close(fig)


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

            win_counts = {a.lane: 0 for a in agents}
            top2_counts = {a.lane: 0 for a in agents}
            top3_counts = {a.lane: 0 for a in agents}
            kimarite_counts = {}

            sim = RaceSimulator(agents, conditions, venue_name, race_month)
            bar = st.progress(0)

            for i in range(n_mc):
                if i % max(1, n_mc // 100) == 0:
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

            bar.progress(1.0)

            st.markdown("#### 勝率・連対率・3連対率")
            mc_df = pd.DataFrame([
                {"艇番": lane,
                 "選手": name_map.get(lane, ""),
                 "1着率": f"{win_counts[lane]/n_mc*100:.1f}%",
                 "2連対率": f"{top2_counts[lane]/n_mc*100:.1f}%",
                 "3連対率": f"{top3_counts[lane]/n_mc*100:.1f}%"}
                for lane in sorted(win_counts.keys())
            ])
            st.dataframe(mc_df, use_container_width=True, hide_index=True)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            lanes = sorted(win_counts.keys())
            win_pcts = [win_counts[l]/n_mc*100 for l in lanes]
            bar_colors_mc = [boat_colors.get(l, 'gray') for l in lanes]
            labels = [f"{l}号艇\n{name_map.get(l,'')}" for l in lanes]
            ax2.bar(labels, win_pcts, color=bar_colors_mc)
            ax2.set_ylabel("1着率(%)")
            ax2.set_title(f"モンテカルロ {n_mc:,}回 - 1着率")
            for i, v in enumerate(win_pcts):
                ax2.text(i, v + 0.3, f"{v:.1f}%", ha='center', fontsize=9)
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown("#### 決まり手分布")
            km_df = pd.DataFrame([
                {"決まり手": k, "回数": v, "割合": f"{v/n_mc*100:.1f}%"}
                for k, v in sorted(kimarite_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(km_df, use_container_width=True, hide_index=True)

            st.session_state['mc_done'] = True
            st.session_state['n_mc'] = n_mc


# ----------------------------
# タブ4: オッズ取得
# ----------------------------
with tab_odds:
    st.subheader("💰 オッズ取得")
    odds_method = st.radio(
        "取得方法",
        ["🌐 自動取得（公式サイト）", "📋 テキスト貼り付け", "✏️ 手動入力"],
        horizontal=True, key="odds_method"
    )

    if odds_method == "🌐 自動取得（公式サイト）":
        st.info(f"会場: {venue_name}（{venue_code}） / 日付: {date_str} / レース: {race_no}R")
        if st.button("🔄 オッズ自動取得", type="primary", key="fetch_odds"):
            with st.spinner("公式サイトからオッズを取得中..."):
                odds = fetch_trifecta_odds(venue_code, date_str, race_no)
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りのオッズを取得しました")
            else:
                st.error("取得失敗。テキスト貼り付けをお試しください。")

    elif odds_method == "📋 テキスト貼り付け":
        odds_text = st.text_area("オッズデータを貼り付け", height=200, key="odds_text",
                                 placeholder="公式サイトの3連単オッズ表をコピーして貼り付け")
        if st.button("📥 解析", key="parse_odds"):
            odds = parse_pasted_odds(odds_text)
            if odds:
                st.session_state['trifecta_odds'] = odds
                st.success(f"✅ {len(odds)} 通りのオッズを解析しました")
            else:
                st.error("解析失敗。形式を確認してください。")

    else:
        manual_text = st.text_area("オッズを入力（1行1組: 1-2-3 6.2）", height=200,
                                   key="manual_odds",
                                   placeholder="1-2-3 6.2\n1-3-2 8.5\n...")
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

    # --- オッズ表示 ---
    if 'trifecta_odds' in st.session_state:
        odds = st.session_state['trifecta_odds']
        st.markdown("---")
        st.markdown("#### 取得済み3連単オッズ（低配当順 Top 20）")
        sorted_odds = sorted(odds.items(), key=lambda x: x[1])
        top_df = pd.DataFrame([
            {"買い目": k, "オッズ": v} for k, v in sorted_odds[:20]
        ])
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.write(f"合計: {len(odds)} 通り / 最低: {sorted_odds[0][1]} / 最高: {sorted_odds[-1][1]}")

        synthetic = compute_synthetic_odds(odds)
        st.session_state['synthetic_odds'] = synthetic

        with st.expander("📊 合成オッズ（2連単・2連複・3連複・拡連複）"):
            for bet_type, label in [("exacta","2連単"),("quinella","2連複"),
                                     ("trio","3連複"),("wide","拡連複")]:
                st.markdown(f"**{label}**")
                s = sorted(synthetic[bet_type].items(), key=lambda x: x[1])
                sdf = pd.DataFrame([{"買い目":k,"合成オッズ":v} for k,v in s[:15]])
                st.dataframe(sdf, use_container_width=True, hide_index=True)


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
        ev_sims = st.slider("EV計算用シミュレーション回数", 1000, 50000, 10000, 1000, key="ev_sims")

        if st.button("🚀 期待値計算 実行", type="primary", key="run_ev"):
            agents = st.session_state['agents']
            conditions = st.session_state['conditions']
            synthetic = st.session_state['synthetic_odds']

            st.write("⏳ モンテカルロでシミュレーション中...")
            probs = run_ev_simulation(agents, conditions, venue_name, race_month, ev_sims)
            ev_results = compute_expected_values(probs, synthetic)
            st.session_state['ev_results'] = ev_results
            st.success("✅ 期待値計算完了！")

        if 'ev_results' in st.session_state:
            ev_results = st.session_state['ev_results']

            ev_tabs = st.tabs(["3連単","3連複","2連単","2連複","拡連複"])
            bet_types = ["trifecta","trio","exacta","quinella","wide"]
            bet_labels = ["3連単","3連複","2連単","2連複","拡連複"]

            for ev_tab, bt, bl in zip(ev_tabs, bet_types, bet_labels):
                with ev_tab:
                    data = ev_results.get(bt, {})
                    if not data:
                        st.write("データなし")
                        continue

                    sorted_ev = sorted(data.items(), key=lambda x: -x[1]['ev'])
                    top_n = sorted_ev[:20]

                    ev_df = pd.DataFrame([
                        {"買い目": k,
                         "確率(%)": v['prob'],
                         "オッズ": v['odds'],
                         "期待値": v['ev'],
                         "判定": v['flag']}
                        for k, v in top_n
                    ])
                    st.dataframe(ev_df, use_container_width=True, hide_index=True)

                    if len(top_n) > 0:
                        top15 = top_n[:15]
                        fig_ev, ax_ev = plt.subplots(figsize=(10, 5))
                        keys = [x[0] for x in top15]
                        vals = [x[1]['ev'] for x in top15]
                        bar_cols = ['#2ecc71' if v >= 1.0 else '#f39c12' if v >= 0.8 else '#e74c3c'
                                    for v in vals]
                        ax_ev.barh(keys[::-1], vals[::-1], color=bar_cols[::-1])
                        ax_ev.axvline(x=1.0, color='red', linestyle='--', label='EV=1.0')
                        ax_ev.set_xlabel("期待値 (EV)")
                        ax_ev.set_title(f"{bl} 期待値ランキング Top15")
                        ax_ev.legend()
                        st.pyplot(fig_ev)
                        plt.close(fig_ev)

            # おすすめ買い目
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


# ── フッター ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em;'>"
    "🚤 ボートレース AI シミュレーター v3.3 ─ 会場別特性 × モンテカルロ × オッズ自動取得 × 期待値計算"
    "</div>",
    unsafe_allow_html=True
)
