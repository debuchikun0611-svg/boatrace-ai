# ============================================================
#  ボートレース AI シミュレーター v5.1  ─  app.py  (Part 1/3)
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
# 1. 日本語フォント
# ─────────────────────────────────────────────
def setup_japanese_font():
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
              "/usr/share/fonts/ipa-gothic/ipag.ttf"]:
        if os.path.exists(p):
            from matplotlib.font_manager import FontProperties
            matplotlib.rcParams['font.family'] = FontProperties(fname=p).get_name()
            break
    else:
        matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False

setup_japanese_font()

# ─────────────────────────────────────────────
# 2. 会場プロフィール（全24場）
# ─────────────────────────────────────────────
VENUE_PROFILES = {
    "桐生":{"code":"01","water":"淡水","tide":False,"course_win_rate":[54.4,14.4,12.4,11.3,5.4,2.3],"course_top2":[73,30,27,25,18,10],"course_top3":[84,44,41,38,32,22],"kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.14,"捲り差し":0.12,"抜き":0.04,"恵まれ":0.01},"wind_effect":0.5,"memo":"冬は赤城おろし"},
    "戸田":{"code":"02","water":"淡水","tide":False,"course_win_rate":[44.0,17.2,13.5,13.8,7.5,4.2],"course_top2":[63,33,28,28,22,14],"course_top3":[76,47,42,42,36,26],"kimarite":{"逃げ":0.42,"差し":0.18,"捲り":0.16,"捲り差し":0.15,"抜き":0.07,"恵まれ":0.02},"wind_effect":0.4,"memo":"狭い水面、捲り有利"},
    "江戸川":{"code":"03","water":"汽水","tide":True,"course_win_rate":[49.5,15.2,12.8,12.5,6.8,3.5],"course_top2":[69,31,27,26,20,12],"course_top3":[81,45,41,40,34,24],"kimarite":{"逃げ":0.47,"差し":0.16,"捲り":0.15,"捲り差し":0.14,"抜き":0.06,"恵まれ":0.02},"wind_effect":0.7,"memo":"荒水面"},
    "平和島":{"code":"04","water":"汽水","tide":True,"course_win_rate":[49.8,16.0,13.2,12.0,5.8,3.5],"course_top2":[68,32,28,26,19,12],"course_top3":[80,46,42,40,33,23],"kimarite":{"逃げ":0.48,"差し":0.17,"捲り":0.15,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.02},"wind_effect":0.6,"memo":"ビル風"},
    "多摩川":{"code":"05","water":"淡水","tide":False,"course_win_rate":[52.9,15.0,13.0,11.5,5.2,2.5],"course_top2":[72,31,27,25,18,10],"course_top3":[83,45,41,38,32,22],"kimarite":{"逃げ":0.52,"差し":0.16,"捲り":0.14,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.3,"memo":"穏やか"},
    "浜名湖":{"code":"06","water":"汽水","tide":True,"course_win_rate":[50.9,15.8,12.8,12.2,5.6,3.0],"course_top2":[70,32,27,26,19,11],"course_top3":[82,46,41,39,33,23],"kimarite":{"逃げ":0.50,"差し":0.16,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.02},"wind_effect":0.5,"memo":"西風"},
    "蒲郡":{"code":"07","water":"汽水","tide":True,"course_win_rate":[54.4,14.5,12.5,11.0,5.3,2.5],"course_top2":[73,30,27,25,18,10],"course_top3":[84,44,41,38,32,22],"kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.4,"memo":"静水面、イン有利"},
    "常滑":{"code":"08","water":"海水","tide":True,"course_win_rate":[57.8,14.0,11.5,10.0,4.8,2.2],"course_top2":[75,29,26,23,17,9],"course_top3":[86,43,40,37,31,21],"kimarite":{"逃げ":0.57,"差し":0.14,"捲り":0.12,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"向かい風注意"},
    "津":{"code":"09","water":"海水","tide":True,"course_win_rate":[56.5,14.2,12.0,10.5,4.8,2.2],"course_top2":[74,30,26,24,17,9],"course_top3":[85,44,40,37,31,21],"kimarite":{"逃げ":0.56,"差し":0.14,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"伊勢湾の潮"},
    "三国":{"code":"10","water":"淡水","tide":False,"course_win_rate":[53.5,14.8,12.5,11.5,5.2,2.5],"course_top2":[72,31,27,25,18,10],"course_top3":[83,45,41,38,32,22],"kimarite":{"逃げ":0.52,"差し":0.15,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.6,"memo":"冬の北風"},
    "びわこ":{"code":"11","water":"淡水","tide":False,"course_win_rate":[52.0,15.5,13.0,11.5,5.5,2.8],"course_top2":[71,31,27,25,18,10],"course_top3":[83,45,41,38,32,22],"kimarite":{"逃げ":0.51,"差し":0.16,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"比良おろし"},
    "住之江":{"code":"12","water":"淡水","tide":False,"course_win_rate":[55.0,14.5,12.0,11.0,5.0,2.5],"course_top2":[73,30,26,25,18,10],"course_top3":[85,44,40,38,31,21],"kimarite":{"逃げ":0.55,"差し":0.15,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.3,"memo":"ナイター、静水面"},
    "尼崎":{"code":"13","water":"海水","tide":True,"course_win_rate":[58.5,13.5,11.5,10.0,4.5,2.2],"course_top2":[76,29,25,23,17,9],"course_top3":[87,43,39,36,30,20],"kimarite":{"逃げ":0.58,"差し":0.14,"捲り":0.12,"捲り差し":0.10,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.4,"memo":"センタープール"},
    "鳴門":{"code":"14","water":"海水","tide":True,"course_win_rate":[53.0,15.0,12.5,11.5,5.5,2.8],"course_top2":[72,31,27,25,18,10],"course_top3":[83,45,41,38,32,22],"kimarite":{"逃げ":0.52,"差し":0.15,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.6,"memo":"潮の干満差大"},
    "丸亀":{"code":"15","water":"海水","tide":True,"course_win_rate":[55.5,14.5,12.0,10.5,5.0,2.5],"course_top2":[74,30,26,24,17,9],"course_top3":[85,44,40,37,31,21],"kimarite":{"逃げ":0.55,"差し":0.14,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"向い風で荒れ"},
    "児島":{"code":"16","water":"海水","tide":True,"course_win_rate":[54.0,14.8,12.5,11.0,5.2,2.5],"course_top2":[73,30,27,25,18,10],"course_top3":[84,44,41,38,32,22],"kimarite":{"逃げ":0.53,"差し":0.15,"捲り":0.14,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"瀬戸内海"},
    "宮島":{"code":"17","water":"海水","tide":True,"course_win_rate":[54.5,14.5,12.5,11.0,5.0,2.5],"course_top2":[73,30,27,25,18,10],"course_top3":[84,44,41,38,32,22],"kimarite":{"逃げ":0.54,"差し":0.15,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"潮の影響大"},
    "徳山":{"code":"18","water":"海水","tide":True,"course_win_rate":[63.4,11.7,12.8,9.7,3.5,1.1],"course_top2":[80.4,30.5,20,17.5,12.2,5.2],"course_top3":[87.5,47.7,40.2,39.1,29.2,22],"kimarite":{"逃げ":0.63,"差し":0.12,"捲り":0.10,"捲り差し":0.09,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.6,"seasonal":{"春":[64.6,14.6,9.7,7.3,3.8,0.5],"夏":[66.1,11.9,10.7,7.4,4.2,0.5],"秋":[61.1,10.5,12.1,11.6,3.2,1.9],"冬":[62.9,14.0,8.4,8.0,5.7,1.6]},"memo":"イン最強、BS広い"},
    "下関":{"code":"19","water":"海水","tide":True,"course_win_rate":[56.0,14.5,12.0,10.5,4.8,2.2],"course_top2":[74,30,26,24,17,9],"course_top3":[85,44,40,37,31,21],"kimarite":{"逃げ":0.55,"差し":0.14,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"ナイター"},
    "若松":{"code":"20","water":"海水","tide":True,"course_win_rate":[55.5,14.8,12.0,10.5,5.0,2.5],"course_top2":[74,30,26,24,17,9],"course_top3":[85,44,40,37,31,21],"kimarite":{"逃げ":0.55,"差し":0.15,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"潮の干満差"},
    "芦屋":{"code":"21","water":"淡水","tide":False,"course_win_rate":[60.0,13.0,11.0,9.5,4.5,2.0],"course_top2":[78,28,25,22,16,8],"course_top3":[87,42,39,36,30,20],"kimarite":{"逃げ":0.60,"差し":0.13,"捲り":0.11,"捲り差し":0.10,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.3,"memo":"モーニング、イン有利"},
    "福岡":{"code":"22","water":"汽水","tide":True,"course_win_rate":[52.0,15.5,13.0,11.5,5.5,2.8],"course_top2":[71,31,27,25,18,10],"course_top3":[83,45,41,38,32,22],"kimarite":{"逃げ":0.51,"差し":0.16,"捲り":0.14,"捲り差し":0.13,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.6,"memo":"那珂川河口"},
    "唐津":{"code":"23","water":"海水","tide":True,"course_win_rate":[56.0,14.5,12.0,10.5,5.0,2.2],"course_top2":[74,30,26,24,17,9],"course_top3":[85,44,40,37,31,21],"kimarite":{"逃げ":0.55,"差し":0.14,"捲り":0.13,"捲り差し":0.12,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"モーニング"},
    "大村":{"code":"24","water":"海水","tide":True,"course_win_rate":[62.0,12.0,11.5,9.0,3.8,1.5],"course_top2":[79,28,25,22,16,8],"course_top3":[87,42,39,36,30,20],"kimarite":{"逃げ":0.62,"差し":0.12,"捲り":0.10,"捲り差し":0.10,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.4,"memo":"イン天国、ナイター"},
}
DEFAULT_VENUE_PROFILE = {"code":"00","water":"不明","tide":False,"course_win_rate":[55.9,14.5,12.5,11,4.8,2.2],"course_top2":[73,30,27,25,18,10],"course_top3":[84,44,41,38,32,22],"kimarite":{"逃げ":0.55,"差し":0.15,"捲り":0.13,"捲り差し":0.11,"抜き":0.05,"恵まれ":0.01},"wind_effect":0.5,"memo":"全国平均"}

def get_venue_profile(name: str) -> dict:
    return VENUE_PROFILES.get(name, DEFAULT_VENUE_PROFILE)
def get_season(month: int) -> str:
    if month in [3,4,5]: return "春"
    if month in [6,7,8]: return "夏"
    if month in [9,10,11]: return "秋"
    return "冬"

# ─────────────────────────────────────────────
# 3. データクラス
# ─────────────────────────────────────────────
@dataclass
class BoatAgent:
    lane: int = 1; number: int = 0; name: str = ""; rank: str = "B1"
    age: int = 30; branch: str = ""; weight: float = 52.0
    flying_count: int = 0; late_count: int = 0
    avg_st: float = 0.18; win_rate: float = 5.0; top2_rate: float = 30.0
    top3_rate: float = 50.0; lane_win_rate: float = 10.0
    local_win_rate: float = 5.0; local_top2_rate: float = 30.0
    local_top3_rate: float = 50.0; accident_rate: float = 0.0
    ability: int = 50; motor_no: int = 0; motor_contribution: float = 0.0
    motor_top2_rate: float = 30.0; boat_no: int = 0; boat_top2_rate: float = 30.0
    exhibition_time: float = 6.80; lap_time: float = 0.0; turn_time: float = 0.0
    straight_time: float = 0.0; exhibition_rank: int = 3; tilt: float = -0.5
    nige_count: int = 0; sashi_count: int = 0; makuri_count: int = 0
    makuri_sashi_count: int = 0; nuki_count: int = 0

    def calculate_start_timing(self) -> float:
        base = self.avg_st + (0.01 * self.flying_count)
        return max(0.01, base + np.random.normal(0, 0.015))

    def get_power_score(self) -> float:
        base = self.ability / 100.0
        rb = {"A1":0.08,"A2":0.04,"B1":0.0,"B2":-0.04}.get(self.rank, 0)
        return float(np.clip(base + rb, 0.1, 1.0))

    def get_machine_score(self) -> float:
        mp = np.clip(self.motor_contribution * 0.8, -0.15, 0.15)
        mr = (self.motor_top2_rate - 30.0) / 100.0
        ex = np.clip((6.90 - self.exhibition_time) / 0.80, -0.2, 0.2)
        ti = 0.02 if self.tilt < -0.5 else (-0.01 if self.tilt > 0.5 else 0.0)
        return float(np.clip(0.5 + mp + mr + ex + ti, 0.1, 1.0))

    def get_turn_score(self) -> float:
        ts = (5.5 - self.turn_time) / 3.0 if self.turn_time > 0 else 0.0
        rs = (4 - self.exhibition_rank) * 0.03
        ws = (52.0 - self.weight) * 0.003
        return float(np.clip(0.5 + ts + rs + ws, 0.1, 1.0))

    def get_kimarite_tendency(self) -> Dict[str, float]:
        t = max(1, self.nige_count+self.sashi_count+self.makuri_count+self.makuri_sashi_count+self.nuki_count)
        return {"逃げ":self.nige_count/t,"差し":self.sashi_count/t,"捲り":self.makuri_count/t,"捲り差し":self.makuri_sashi_count/t,"抜き":self.nuki_count/t}

@dataclass
class RaceCondition:
    weather: str = "晴"; temperature: float = 20.0; wind_speed: float = 2.0
    wind_direction: str = ""; water_temp: float = 20.0; wave_height: float = 2.0; tide: str = ""

# ─────────────────────────────────────────────
# 4. シミュレーター
# ─────────────────────────────────────────────
class RaceSimulator:
    def __init__(self, agents, conditions, venue_name="徳山", month=2):
        self.agents = sorted(agents, key=lambda a: a.lane)
        self.conditions = conditions
        self.venue = get_venue_profile(venue_name)
        self.season = get_season(month)

    def _compute_race_weights(self):
        vp = self.venue
        br = vp.get("seasonal",{}).get(self.season, vp["course_win_rate"])
        weights = {}
        for a in self.agents:
            idx = a.lane - 1
            w = br[idx]/100*2.0 + a.lane_win_rate/100*1.2 + a.win_rate/10*1.0
            w += (a.top2_rate/200 + a.top3_rate/300)*0.5 + a.local_win_rate/10*0.5
            w += a.get_power_score()*1.0 + a.get_machine_score()*1.5 + a.get_turn_score()*0.8
            w += max(0,(0.20-a.avg_st)/0.10)*0.8 + (54.0-a.weight)/20*0.2
            we = self.conditions.wind_speed*vp["wind_effect"]/20
            if a.lane==1: w -= we*0.3
            elif a.lane>=4: w += we*0.1
            if self.conditions.wave_height>5:
                w += 0.02 if a.weight>54 else -0.01
            w -= a.accident_rate*0.01 + 0.02*a.flying_count
            weights[a.lane] = max(w, 0.01)
        t = sum(weights.values())
        return {k: v/t for k,v in weights.items()}

    def simulate_race(self):
        weights = self._compute_race_weights()
        st_times = {a.lane: a.calculate_start_timing() for a in self.agents}
        min_st = min(st_times.values())
        adj = dict(weights)
        for ln, sv in st_times.items():
            d = sv - min_st
            if d < 0.02: adj[ln] *= 1.15
            elif d < 0.05: adj[ln] *= 1.05
            elif d > 0.10: adj[ln] *= 0.85
        ext = {a.lane: a.exhibition_time for a in self.agents if a.exhibition_time > 0}
        if ext:
            fe = min(ext.values())
            for ln, et in ext.items():
                if et <= fe+0.02: adj[ln] *= 1.08
                elif et >= fe+0.15: adj[ln] *= 0.95
        nl = 0.05
        if self.conditions.wave_height>5 or self.conditions.wind_speed>5: nl=0.10
        if self.conditions.weather in ["雨","雪"]: nl+=0.03
        for ln in adj:
            adj[ln] = max(adj[ln]+np.random.normal(0, nl*adj[ln]), 0.001)
        t = sum(adj.values())
        adj = {k: v/t for k,v in adj.items()}
        rem_l = list(adj.keys()); rem_p = [adj[l] for l in rem_l]; fol = []
        for _ in range(6):
            tp = sum(rem_p); np_ = [p/tp for p in rem_p]
            ci = np.random.choice(len(rem_l), p=np_)
            fol.append(rem_l[ci]); rem_l.pop(ci); rem_p.pop(ci)
        fo = {i+1: ln for i, ln in enumerate(fol)}
        pos = {a.lane: [] for a in self.agents}
        for step in range(300):
            pr = step/300
            for a in self.agents:
                fp = list(fo.values()).index(a.lane)+1
                if pr<0.1: p=a.lane+np.random.normal(0,0.3)
                elif pr<0.3: p=a.lane*(1-pr)+fp*pr+np.random.normal(0,0.5)
                else: p=fp+np.random.normal(0,max(0.1,0.5*(1-pr)))
                pos[a.lane].append(float(np.clip(p,0.8,6.2)))
        km = self._determine_kimarite(fo, st_times)
        return {"finish_order":fo,"st_times":st_times,"kimarite":km,"positions":pos,"weights":weights}

    def _determine_kimarite(self, fo, st_times):
        w = fo[1]; am = {a.lane:a for a in self.agents}; wa = am[w]
        if w==1: return "逃げ"
        td = wa.get_kimarite_tendency()
        opts = {2:[("差し",0.5),("捲り",0.2),("抜き",0.2),("恵まれ",0.1)],
                3:[("捲り",0.35),("捲り差し",0.3),("差し",0.15),("抜き",0.15),("恵まれ",0.05)],
                4:[("捲り",0.35),("差し",0.25),("捲り差し",0.2),("抜き",0.15),("恵まれ",0.05)],
                5:[("捲り",0.4),("捲り差し",0.25),("抜き",0.2),("差し",0.1),("恵まれ",0.05)],
                6:[("捲り",0.4),("捲り差し",0.2),("抜き",0.2),("差し",0.1),("恵まれ",0.1)]}
        ao = [(n, bp*0.7+td.get(n,0)*0.3) for n,bp in opts.get(w, opts[6])]
        tp = sum(p for _,p in ao)
        return np.random.choice([n for n,_ in ao], p=[p/tp for _,p in ao])
# ============================================================
#  ボートレース AI シミュレーター v5.1  ─  app.py  (Part 2/3)
#  パーサー + オッズ取得 + 合成オッズ + EV計算
# ============================================================

# ─────────────────────────────────────────────
# 5. パーサー用ユーティリティ
# ─────────────────────────────────────────────
def _tab_split_numbers(line: str) -> List[float]:
    parts = re.split(r'\t|  +', line)
    vals = []
    for p in parts:
        p = p.strip().replace('%','').replace(',','')
        if re.match(r'^\(\d+\)$', p):
            continue
        m = re.match(r'^-?(\d+\.?\d*)$', p)
        if m:
            vals.append(float(m.group(0)))
    return vals

def _find_row_values(lines, section_kw, row_kw, n=6, vrange=None):
    in_sec = False
    for i, line in enumerate(lines):
        if section_kw in line:
            in_sec = True
        if not in_sec:
            continue
        if row_kw not in line:
            continue
        block = line
        for j in range(1, 5):
            if i+j < len(lines):
                block += "\t" + lines[i+j]
        vals = _tab_split_numbers(block)
        if vrange:
            vals = [v for v in vals if vrange[0] <= v <= vrange[1]]
        if len(vals) >= n:
            return vals[:n]
    return None

def _find_pct_row(lines, section_kw, row_kw, n=6):
    in_sec = False
    for i, line in enumerate(lines):
        if section_kw in line:
            in_sec = True
        if not in_sec:
            continue
        if row_kw not in line:
            continue
        block = line
        for j in range(1, 8):
            if i+j < len(lines):
                nl = lines[i+j].strip()
                if any(k in nl for k in ['直近3ヶ月','直近1ヶ月','一般戦','SG/G1','当地','ナイター','全国','今期','波5cm']):
                    break
                block += "\t" + nl
        pcts = re.findall(r'(\d+\.?\d*)\s*%', block)
        if len(pcts) >= n:
            return [float(p) for p in pcts[:n]]
        vals = _tab_split_numbers(block)
        if len(vals) >= n:
            return vals[:n]
    return None

# ─────────────────────────────────────────────
# 6. 公式サイトコピペパーサー
# ─────────────────────────────────────────────
SKIP_NAMES = {"競艇","徳山","勝率","連対","決まり","選手","当地","全国","平均",
    "枠別","能力","事故","貢献","展示","周回","直線","天候","気温","水温","風速",
    "波高","日本","財団","会長","争奪","初日","最終","一般","基本","情報","直前",
    "結果","予選","締切","登録","変更","進入","更新","検索","解除","モータ","ボート",
    "号艇","決り","恵まれ","安定","抜出","出遅","通算","今節","得点","順位","前走",
    "部品","交換","プロペラ","コメント","風向き","オッズ","早見","出走","投票",
    "レース","メニュー","選手情報","最新情報","コース","変更クリア"}

def parse_official_site_text(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
    agents = [BoatAgent(lane=i+1) for i in range(6)]
    conditions = RaceCondition()
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    # ── 登録番号（4桁 3000-5999）──
    for line in lines:
        nums = re.findall(r'\b(\d{4})\b', line)
        bn = [int(n) for n in nums if 3000 <= int(n) <= 5999]
        if len(bn) >= 6:
            for i, n in enumerate(bn[:6]):
                agents[i].number = n
            break

    # ── 選手名 ──
    name_found = False
    # 方法A: タブ区切りで漢字名が6個並ぶ行
    for line in lines:
        parts = re.split(r'\t', line)
        np_ = []
        for p in parts:
            p = p.strip().replace('\u3000','').replace(' ','')
            if re.match(r'^[一-龥ぁ-んァ-ヶー]{2,}$', p) and p not in SKIP_NAMES:
                np_.append(p)
        if len(np_) >= 6:
            for i, nm in enumerate(np_[:6]):
                agents[i].name = nm
            name_found = True
            break

    # 方法B: 1行中に漢字名が6個（全角スペース区切りも含む）
    if not name_found:
        for line in lines:
            cands = re.findall(r'([一-龥ぁ-んァ-ヶー]{1,2}[\s\u3000]?[一-龥ぁ-んァ-ヶー]{1,3})', line)
            cleaned = []
            for c in cands:
                cc = c.replace(' ','').replace('\u3000','')
                if len(cc) >= 2 and cc not in SKIP_NAMES:
                    cleaned.append(cc)
            if len(cleaned) >= 6:
                for i, nm in enumerate(cleaned[:6]):
                    agents[i].name = nm
                name_found = True
                break

    # 方法C: 登録番号行の直後
    if not name_found:
        for i, line in enumerate(lines):
            nums = re.findall(r'\b(\d{4})\b', line)
            bn = [int(n) for n in nums if 3000 <= int(n) <= 5999]
            if len(bn) >= 6 and i+1 < len(lines):
                cands = re.findall(r'([一-龥ぁ-んァ-ヶー]{2,4})', lines[i+1])
                fl = [n for n in cands if n not in SKIP_NAMES]
                if len(fl) >= 6:
                    for j, nm in enumerate(fl[:6]):
                        agents[j].name = nm
                    name_found = True
                break

    # 方法D: 全行から蓄積
    if not name_found:
        buf = []
        for line in lines:
            parts = re.split(r'\t|  +', line)
            for p in parts:
                p = p.strip().replace('\u3000','').replace(' ','')
                if re.match(r'^[一-龥ぁ-んァ-ヶー]{2,4}$', p) and p not in SKIP_NAMES:
                    buf.append(p)
                    if len(buf) >= 6:
                        break
            if len(buf) >= 6:
                for j, nm in enumerate(buf[:6]):
                    agents[j].name = nm
                break

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
    if agents[0].age == 30:
        for i, line in enumerate(lines):
            if '年齢' in line:
                block = line
                for j in range(1,4):
                    if i+j < len(lines): block += "\t" + lines[i+j]
                ages = re.findall(r'(\d{2})歳', block)
                if len(ages) >= 6:
                    for k, a in enumerate(ages[:6]):
                        agents[k].age = int(a)
                break

    # ── 支部 ──
    prefs = ["北海道","青森","岩手","宮城","秋田","山形","福島","茨城","栃木",
             "群馬","埼玉","千葉","東京","神奈川","新潟","富山","石川","福井",
             "山梨","長野","岐阜","静岡","愛知","三重","滋賀","京都","大阪",
             "兵庫","奈良","和歌山","鳥取","島根","岡山","広島","山口","徳島",
             "香川","愛媛","高知","福岡","佐賀","長崎","熊本","大分","宮崎",
             "鹿児島","沖縄"]
    buf = []
    for line in lines:
        for p in prefs:
            if p in line and p not in buf:
                buf.append(p)
        if len(buf) >= 6:
            for i, p in enumerate(buf[:6]):
                agents[i].branch = p
            break

    # ── 平均ST ──
    v = _find_row_values(lines, "平均ST", "直近6ヶ月", 6, (0.05, 0.30))
    if v:
        for i, x in enumerate(v): agents[i].avg_st = x

    # ── 勝率 ──
    v = _find_row_values(lines, "勝率", "直近6ヶ月", 6, (1.0, 12.0))
    if v:
        for i, x in enumerate(v): agents[i].win_rate = x

    # ── 2連対率 ──
    v = _find_pct_row(lines, "2連対率", "直近6ヶ月")
    if v:
        for i, x in enumerate(v): agents[i].top2_rate = x

    # ── 3連対率 ──
    v = _find_pct_row(lines, "3連対率", "直近6ヶ月")
    if v:
        for i, x in enumerate(v): agents[i].top3_rate = x

    # ── 当地勝率 ──
    v = _find_row_values(lines, "勝率", "当地", 6, (0.0, 12.0))
    if v:
        for i, x in enumerate(v): agents[i].local_win_rate = x

    # ── 枠別1着率 ──
    v = _find_pct_row(lines, "1着率", "直近6ヶ月")
    if v:
        for i, x in enumerate(v): agents[i].lane_win_rate = x

    # ── 能力値 ──
    v = _find_row_values(lines, "能力値", "今期", 6, (1, 100))
    if v:
        for i, x in enumerate(v): agents[i].ability = int(x)

    # ── 事故率 ──
    for i, line in enumerate(lines):
        if line.strip().startswith('事故率') and '事故点' not in line:
            vals = _tab_split_numbers(line)
            fv = [x for x in vals if 0.0 <= x <= 5.0]
            if len(fv) >= 6:
                for k, x in enumerate(fv[:6]): agents[k].accident_rate = x
                break
            if i+1 < len(lines):
                vals = _tab_split_numbers(line + "\t" + lines[i+1])
                fv = [x for x in vals if 0.0 <= x <= 5.0]
                if len(fv) >= 6:
                    for k, x in enumerate(fv[:6]): agents[k].accident_rate = x
                    break

    # ── フライング ──
    v = _find_row_values(lines, "フライング", "今期", 6, (0, 10))
    if v:
        for i, x in enumerate(v): agents[i].flying_count = int(x)

    # ── モーター貢献P ──
    for i, line in enumerate(lines):
        if '貢献P' in line or '貢献Ｐ' in line:
            vals = []
            for j in range(0, 4):
                if i+j < len(lines):
                    nums = re.findall(r'([+\-]?\d+\.\d+)', lines[i+j])
                    vals.extend([float(n) for n in nums])
                    if len(vals) >= 6: break
            if len(vals) >= 6:
                for k, x in enumerate(vals[:6]):
                    agents[k].motor_contribution = x
            break

    # ── 展示タイム ──
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('展示') and '展示順' not in s and '展示情報' not in s and '展示タイム1位' not in s and '平均展示' not in s and '前走展示' not in s:
            block = line
            if i+1 < len(lines): block += "\t" + lines[i+1]
            vals = _tab_split_numbers(block)
            ev = [x for x in vals if 6.0 <= x <= 7.5]
            if len(ev) >= 6:
                for k, x in enumerate(ev[:6]): agents[k].exhibition_time = x
                break

    # ── 体重 ──
    for i, line in enumerate(lines):
        if line.strip().startswith('体重'):
            block = line
            if i+1 < len(lines): block += "\t" + lines[i+1]
            vals = _tab_split_numbers(block)
            wv = [x for x in vals if 40.0 <= x <= 70.0]
            if len(wv) >= 6:
                for k, x in enumerate(wv[:6]): agents[k].weight = x
                break

    # ── チルト ──
    for i, line in enumerate(lines):
        if line.strip().startswith('チルト'):
            nums = re.findall(r'([+\-]?\d+\.?\d*)', line)
            fv = [float(n) for n in nums if -3.0 <= float(n) <= 3.0]
            if len(fv) >= 6:
                for k, x in enumerate(fv[:6]): agents[k].tilt = x
                break

    # ── 周回タイム ──
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('周回') and '周回展示' not in s and '前走周回' not in s and '平均周回' not in s:
            vals = _tab_split_numbers(line)
            lv = [x for x in vals if 30.0 <= x <= 45.0]
            if len(lv) >= 6:
                for k, x in enumerate(lv[:6]): agents[k].lap_time = x
                break

    # ── 周り足 ──
    for i, line in enumerate(lines):
        s = line.strip()
        if (s.startswith('周り足') or s.startswith('周足')) and '前走周足' not in s and '平均周足' not in s:
            vals = _tab_split_numbers(line)
            tv = [x for x in vals if 3.0 <= x <= 8.0]
            if len(tv) >= 6:
                for k, x in enumerate(tv[:6]): agents[k].turn_time = x
                break

    # ── 直線タイム ──
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('直線') and '前走直線' not in s and '平均直線' not in s:
            vals = _tab_split_numbers(line)
            sv = [x for x in vals if 6.0 <= x <= 9.0]
            if len(sv) >= 6:
                for k, x in enumerate(sv[:6]): agents[k].straight_time = x
                break

    # ── 決まり手数 ──
    for i, line in enumerate(lines):
        if '決り手数' in line or '決まり手数' in line:
            for j in range(i+1, min(i+20, len(lines))):
                row = lines[j].strip()
                def _get_ints(r):
                    v = _tab_split_numbers(r)
                    return [int(x) for x in v if x == int(x) and 0 <= x <= 200]
                if row.startswith('逃げ'):
                    iv = _get_ints(row)
                    if len(iv)>=6:
                        for k,x in enumerate(iv[:6]): agents[k].nige_count=x
                elif row.startswith('差し'):
                    iv = _get_ints(row)
                    if len(iv)>=6:
                        for k,x in enumerate(iv[:6]): agents[k].sashi_count=x
                elif row.startswith('捲差') or row.startswith('捲り差'):
                    iv = _get_ints(row)
                    if len(iv)>=6:
                        for k,x in enumerate(iv[:6]): agents[k].makuri_sashi_count=x
                elif row.startswith('捲り'):
                    iv = _get_ints(row)
                    if len(iv)>=6:
                        for k,x in enumerate(iv[:6]): agents[k].makuri_count=x
                elif row.startswith('抜き'):
                    iv = _get_ints(row)
                    if len(iv)>=6:
                        for k,x in enumerate(iv[:6]): agents[k].nuki_count=x
                if any(kw in row for kw in ['優勝','優出','フライング','能力','事故']):
                    break
            break

    # ── 天候 ──
    for line in lines:
        if '℃' in line and ('気温' in line or '天気' in line or '水温' in line):
            m = re.search(r'気温\s*(\d+\.?\d*)\s*℃', line)
            if not m: m = re.search(r'(\d+\.?\d*)\s*℃', line)
            if m: conditions.temperature = float(m.group(1))
            m = re.search(r'水温\s*(\d+\.?\d*)', line)
            if m: conditions.water_temp = float(m.group(1))
    for line in lines:
        m = re.search(r'(\d+\.?\d*)\s*m\b', line)
        if m:
            v = float(m.group(1))
            if 0 < v <= 20: conditions.wind_speed = v; break
    for line in lines:
        m = re.search(r'(\d+\.?\d*)\s*cm', line)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 50: conditions.wave_height = v; break
    for line in lines:
        if '雨' in line and '号艇' not in line and '勝率' not in line:
            conditions.weather = "雨"; break
        elif '曇' in line and '号艇' not in line:
            conditions.weather = "曇"; break
        elif '雪' in line and '号艇' not in line:
            conditions.weather = "雪"; break

    return agents, conditions

# ─────────────────────────────────────────────
# 7. 1行1艇パーサー
# ─────────────────────────────────────────────
def parse_manual_format(text: str) -> Tuple[List[BoatAgent], RaceCondition]:
    agents = []; conditions = RaceCondition()
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if ('天候' in line or '℃' in line) and '号艇' not in line:
            m = re.search(r'(\d+\.?\d*)\s*℃', line)
            if m: conditions.temperature = float(m.group(1))
            m = re.search(r'水温\s*(\d+\.?\d*)', line)
            if m: conditions.water_temp = float(m.group(1))
            m = re.search(r'風速?\s*(\d+\.?\d*)', line)
            if m: conditions.wind_speed = float(m.group(1))
            m = re.search(r'波高?\s*(\d+\.?\d*)', line)
            if m: conditions.wave_height = float(m.group(1))
            if '雨' in line: conditions.weather="雨"
            elif '曇' in line: conditions.weather="曇"
            continue
        lm = re.search(r'(\d)\s*号艇', line)
        if not lm: continue
        a = BoatAgent(lane=int(lm.group(1)))
        m = re.search(r'[:：\s]\s*(\d{3,5})\b', line)
        if m: a.number = int(m.group(1))
        after = line[m.end():] if m else line[lm.end():]
        m = re.search(r'\s*([一-龥ぁ-んァ-ヶー]{2,})', after)
        if m: a.name = m.group(1)
        for r in ["A1","A2","B1","B2"]:
            if r in line: a.rank = r; break
        m = re.search(r'(\d{2,3})\s*歳', line)
        if m: a.age = int(m.group(1))
        m = re.search(r'体重\s*(\d+\.?\d*)', line)
        if m: a.weight = float(m.group(1))
        m = re.search(r'F\s*(\d+)', line)
        if m: a.flying_count = int(m.group(1))
        m = re.search(r'(?:平均)?ST\s*(0\.\d+)', line)
        if m: a.avg_st = float(m.group(1))
        m = re.search(r'勝率\s*([\d.]+)', line)
        if m: a.win_rate = float(m.group(1))
        m = re.search(r'2連対?\s*([\d.]+)', line)
        if m: a.top2_rate = float(m.group(1))
        m = re.search(r'3連対?\s*([\d.]+)', line)
        if m: a.top3_rate = float(m.group(1))
        m = re.search(r'枠別?[1１]?着?\s*([\d.]+)', line)
        if m: a.lane_win_rate = float(m.group(1))
        m = re.search(r'当地勝率?\s*([\d.]+)', line)
        if m: a.local_win_rate = float(m.group(1))
        m = re.search(r'能力\s*(\d+)', line)
        if m: a.ability = int(m.group(1))
        m = re.search(r'モーター?\s*([+\-]?\s*[\d.]+)', line)
        if m: a.motor_contribution = float(m.group(1).replace(' ',''))
        m = re.search(r'モ2連\s*([\d.]+)', line)
        if m: a.motor_top2_rate = float(m.group(1))
        m = re.search(r'展示T?\s*([\d.]+)', line)
        if m:
            v = float(m.group(1))
            if 6.0 <= v <= 7.5: a.exhibition_time = v
        m = re.search(r'展示順?\s*(\d)', line)
        if m: a.exhibition_rank = int(m.group(1))
        m = re.search(r'周り足\s*([\d.]+)', line)
        if m: a.turn_time = float(m.group(1))
        m = re.search(r'チルト\s*([+\-]?\s*[\d.]+)', line)
        if m: a.tilt = float(m.group(1).replace(' ',''))
        m = re.search(r'事故率?\s*([\d.]+)', line)
        if m: a.accident_rate = float(m.group(1))
        m = re.search(r'逃げ\s*(\d+)', line)
        if m: a.nige_count = int(m.group(1))
        m = re.search(r'差し\s*(\d+)', line)
        if m: a.sashi_count = int(m.group(1))
        m = re.search(r'(?<!捲り)捲り\s*(\d+)', line)
        if m: a.makuri_count = int(m.group(1))
        m = re.search(r'捲り差し?\s*(\d+)', line)
        if m: a.makuri_sashi_count = int(m.group(1))
        m = re.search(r'抜き\s*(\d+)', line)
        if m: a.nuki_count = int(m.group(1))
        agents.append(a)
    if not agents:
        agents = [BoatAgent(lane=i, name=f"選手{i}") for i in range(1,7)]
    return agents, conditions

def parse_any_text(text):
    if '号艇:' in text or '号艇：' in text:
        return parse_manual_format(text)
    return parse_official_site_text(text)

# ─────────────────────────────────────────────
# 8. オッズ取得（公式サイト自動スクレイピング）
# ─────────────────────────────────────────────
def fetch_trifecta_odds(venue_code: str, date_str: str, race_no: int) -> Dict[str, float]:
    """
    公式サイト3連単オッズ: 20行×6列テーブル
    行=2着-3着の組み合わせ(20通り), 列=1着(6艇)
    → 転置して1着ごとに20通りを展開
    """
    url = (f"https://www.boatrace.jp/owpc/pc/race/odds3t"
           f"?rno={race_no}&jcd={venue_code}&hd={date_str}")
    try:
        from urllib.request import urlopen, Request
        from bs4 import BeautifulSoup

        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=15).read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # テーブル検索（複数の方法でフォールバック）
        table = None
        # 方法1: is-w495 クラス
        tables = soup.select("table.is-w495")
        if tables:
            table = tables[0]
        # 方法2: oddsPoint を含むテーブル
        if table is None:
            for t in soup.select("table"):
                if t.select("td.oddsPoint"):
                    table = t
                    break
        # 方法3: tbody 内の oddsPoint
        if table is None:
            all_cells = soup.select("td.oddsPoint")
            if len(all_cells) >= 120:
                # テーブルなしでセル直接取得
                vals = []
                for c in all_cells[:120]:
                    txt = c.get_text(strip=True).replace(",","")
                    try: vals.append(float(txt))
                    except: vals.append(0.0)
                # 120個を20行×6列として転置
                matrix = np.array(vals).reshape(20, 6)
                flat = matrix.T.reshape(-1).tolist()
                odds = {}; idx = 0
                for first in range(1,7):
                    others = [x for x in range(1,7) if x!=first]
                    for second in others:
                        for third in [x for x in others if x!=second]:
                            if idx < len(flat) and flat[idx] > 0:
                                odds[f"{first}-{second}-{third}"] = flat[idx]
                            idx += 1
                return odds

        if table is None:
            st.warning("オッズテーブルが見つかりません")
            return {}

        # テーブルからセル取得
        tbody = table.select_one("tbody") or table
        rows = tbody.select("tr")

        odds_matrix = []
        for row in rows[:20]:
            cells = row.select("td.oddsPoint")
            if len(cells) >= 6:
                rv = []
                for c in cells[:6]:
                    txt = c.get_text(strip=True).replace(",","")
                    try: rv.append(float(txt))
                    except: rv.append(0.0)
                odds_matrix.append(rv)
            else:
                # oddsPoint がない場合は td から数値を探す
                all_tds = row.select("td")
                rv = []
                for td in all_tds:
                    txt = td.get_text(strip=True).replace(",","")
                    try:
                        v = float(txt)
                        if v > 0: rv.append(v)
                    except: pass
                if len(rv) >= 6:
                    odds_matrix.append(rv[:6])

        if len(odds_matrix) < 20:
            st.warning(f"オッズ行数不足: {len(odds_matrix)}行")
            return {}

        # 20行×6列 → 転置 → 6×20 → フラット
        matrix = np.array(odds_matrix[:20])
        flat = matrix.T.reshape(-1).tolist()

        odds = {}; idx = 0
        for first in range(1,7):
            others = [x for x in range(1,7) if x!=first]
            for second in others:
                for third in [x for x in others if x!=second]:
                    if idx < len(flat) and flat[idx] > 0:
                        odds[f"{first}-{second}-{third}"] = flat[idx]
                    idx += 1
        return odds

    except Exception as e:
        st.warning(f"オッズ取得エラー: {e}")
        return {}

# ─────────────────────────────────────────────
# 9. テキスト貼り付けオッズ解析
# ─────────────────────────────────────────────
def parse_pasted_odds(text: str) -> Dict[str, float]:
    odds = {}

    # パターン1: "1-2-3 6.2" 形式
    p1 = re.findall(r'(\d)[‐\-](\d)[‐\-](\d)\s+([\d,.]+)', text)
    if p1:
        for f,s,t,o in p1:
            try: odds[f"{f}-{s}-{t}"] = float(o.replace(",",""))
            except: pass
        if len(odds) >= 10: return odds

    # パターン2: 数値120個以上（テーブルコピー）
    all_nums = []
    for n in re.findall(r'[\d,]+\.[\d]+|\d+\.\d+', text):
        try: all_nums.append(float(n.replace(",","")))
        except: pass
    if len(all_nums) >= 120:
        matrix = np.array(all_nums[:120]).reshape(20, 6)
        flat = matrix.T.reshape(-1).tolist()
        idx = 0
        for first in range(1,7):
            others = [x for x in range(1,7) if x!=first]
            for second in others:
                for third in [x for x in others if x!=second]:
                    if idx < len(flat) and flat[idx] > 0:
                        odds[f"{first}-{second}-{third}"] = flat[idx]
                    idx += 1
        if len(odds) >= 60: return odds

    # パターン3: 行ごとに解析
    for line in text.strip().split('\n'):
        ms = re.findall(r'(\d)[‐\-](\d)[‐\-](\d)[\s,]+([\d,.]+)', line)
        for f,s,t,o in ms:
            try: odds[f"{f}-{s}-{t}"] = float(o.replace(",",""))
            except: pass
    return odds

# ─────────────────────────────────────────────
# 10. 合成オッズ
# ─────────────────────────────────────────────
def compute_synthetic_odds(trifecta):
    inv = {"trio":{},"exacta":{},"quinella":{},"wide":{}}
    for key, ov in trifecta.items():
        if ov <= 0: continue
        pts = key.split("-")
        if len(pts)!=3: continue
        f,s,t = int(pts[0]),int(pts[1]),int(pts[2])
        iv = 1.0/ov
        tk = "=".join(str(x) for x in sorted([f,s,t]))
        inv["trio"][tk] = inv["trio"].get(tk,0)+iv
        ek = f"{f}-{s}"
        inv["exacta"][ek] = inv["exacta"].get(ek,0)+iv
        qk = "-".join(str(x) for x in sorted([f,s]))
        inv["quinella"][qk] = inv["quinella"].get(qk,0)+iv
        for pair in itertools.combinations(sorted([f,s,t]),2):
            wk = "-".join(str(x) for x in pair)
            inv["wide"][wk] = inv["wide"].get(wk,0)+iv
    result = {}
    for bt in ["trio","exacta","quinella","wide"]:
        result[bt] = {k: round(1.0/v,1) for k,v in inv[bt].items() if v > 0}
    return result

# ─────────────────────────────────────────────
# 11. モンテカルロEV
# ─────────────────────────────────────────────
def run_ev_simulation(agents, conditions, venue_name, month, n_sims=10000):
    sim = RaceSimulator(agents, conditions, venue_name, month)
    counts = {bt:{} for bt in ["trifecta","trio","exacta","quinella","wide"]}
    bar = st.progress(0); upd = max(1, n_sims//100)
    for i in range(n_sims):
        if i % upd == 0: bar.progress(min(i/n_sims, 1.0))
        r = sim.simulate_race(); fo = r["finish_order"]
        f1,f2,f3 = fo[1],fo[2],fo[3]
        tk = f"{f1}-{f2}-{f3}"
        counts["trifecta"][tk] = counts["trifecta"].get(tk,0)+1
        trk = "=".join(str(x) for x in sorted([f1,f2,f3]))
        counts["trio"][trk] = counts["trio"].get(trk,0)+1
        ek = f"{f1}-{f2}"
        counts["exacta"][ek] = counts["exacta"].get(ek,0)+1
        qk = "-".join(str(x) for x in sorted([f1,f2]))
        counts["quinella"][qk] = counts["quinella"].get(qk,0)+1
        for pair in itertools.combinations(sorted([f1,f2,f3]),2):
            wk = "-".join(str(x) for x in pair)
            counts["wide"][wk] = counts["wide"].get(wk,0)+1
    bar.progress(1.0)
    return {bt:{k:round(v/n_sims,6) for k,v in c.items()} for bt,c in counts.items()}

def compute_expected_values(probs, synthetic_odds):
    results = {}
    for bt in ["trifecta","trio","exacta","quinella","wide"]:
        pd_ = probs.get(bt,{})
        od = synthetic_odds.get("_trifecta_raw",{}) if bt=="trifecta" else synthetic_odds.get(bt,{})
        ev_data = {}
        for k,p in pd_.items():
            if k in od and od[k]>0:
                ev = round(p*od[k],4)
                flag = "◎" if ev>=1.2 else "○" if ev>=1.0 else "△" if ev>=0.8 else "×"
                ev_data[k] = {"prob":round(p*100,2),"odds":od[k],"ev":round(ev,3),"flag":flag}
        results[bt] = ev_data
    return results
# ── ページ設定 ──────────────────────────────────────
st.set_page_config(page_title="ボートレース AI v5.1", layout="wide")
setup_japanese_font()
st.title("🚤 ボートレース AI シミュレーター v5.1")
st.caption("30項目エージェント × 会場別特性(全24場) × 季節補正 × モンテカルロ × 合成オッズ × 期待値計算")

BOAT_COLORS = {1:"#e74c3c", 2:"#000000", 3:"#2ecc71",
               4:"#3498db", 5:"#f1c40f", 6:"#9b59b6"}

# ── サイドバー ──────────────────────────────────────
st.sidebar.header("⚙️ レース設定")
venue_name = st.sidebar.selectbox("会場", list(VENUE_PROFILES.keys()),
                                  index=list(VENUE_PROFILES.keys()).index("徳山"))
venue_profile = get_venue_profile(venue_name)
race_date = st.sidebar.date_input("日付", value=pd.Timestamp("2026-02-27"))
race_no = st.sidebar.number_input("レース番号", 1, 12, 3)

st.sidebar.subheader(f"📍 {venue_name}の特徴")
vp = venue_profile
st.sidebar.write(f"水面: {vp.get('water','—')}　潮: {vp.get('tide','—')}")
st.sidebar.write(f"風影響: {vp.get('wind_effect','—')}　メモ: {vp.get('memo','—')}")
fig_sb, ax_sb = plt.subplots(figsize=(4, 2.2))
bars = ax_sb.bar(range(1, 7), vp["course_win_rate"],
                 color=[BOAT_COLORS[i] for i in range(1, 7)])
ax_sb.set_xlabel("コース")
ax_sb.set_ylabel("1着率 (%)")
ax_sb.set_title(f"{venue_name} コース別1着率")
ax_sb.set_xticks(range(1, 7))
for b in bars:
    ax_sb.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
               f"{b.get_height():.1f}", ha="center", fontsize=7)
plt.tight_layout()
st.sidebar.pyplot(fig_sb)
plt.close(fig_sb)

# ── メインタブ ──────────────────────────────────────
tab_input, tab_sim, tab_mc, tab_odds, tab_ev = st.tabs(
    ["📝 データ入力", "🏁 単発シミュレーション",
     "📊 モンテカルロ", "💰 オッズ取得", "📈 期待値計算"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 1 ─ データ入力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_input:
    st.subheader("📝 出走データ入力")
    input_method = st.radio("入力方式", ["テキスト貼り付け", "フォーム入力"], horizontal=True)

    if input_method == "テキスト貼り付け":
        sample_text = """1号艇: 4753 森照夫 B1 46 52 0.15 4.92 34.5 44.8 4.50 18.0 48 1.13 30.0 6.78 3 6.70 -0.5 0.34 12 8 6 4 2
2号艇: 5090 生方靖亜 B1 28 52 0.16 5.41 33.0 53.0 5.00 15.0 49 -0.16 28.0 6.84 4 6.72 0.0 0.15 5 10 8 6 3
3号艇: 4226 村田浩司 B1 50 54 0.16 4.77 29.9 42.3 4.30 12.0 48 0.78 26.0 6.80 2 6.68 0.0 0.00 8 6 10 5 2
4号艇: 5121 定松勇樹 A1 26 51 0.14 7.61 62.3 74.7 6.80 22.0 64 0.26 38.0 6.79 1 6.65 -0.5 0.17 15 12 8 6 3
5号艇: 4486 野村誠 B1 42 54 0.19 5.09 37.9 48.3 4.60 14.0 47 -0.22 24.0 6.85 5 6.74 0.0 0.69 6 8 12 5 2
6号艇: 4262 馬場貴也 A1 40 52 0.14 7.26 48.5 62.5 6.50 20.0 70 0.54 36.0 6.83 6 6.71 -0.5 0.00 10 14 10 8 4
天候: 気温13.0℃ 雨 風速3m 水温12.0℃ 波高3cm"""
        raw_text = st.text_area("出走データを貼り付け", value=sample_text, height=300,
                                help="公式サイトからのコピー or 1行1艇フォーマット")
    else:  # フォーム入力
        raw_text = None
        form_agents = []
        for i in range(1, 7):
            with st.expander(f"🚤 {i}号艇", expanded=(i <= 2)):
                cols = st.columns(5)
                reg   = cols[0].number_input(f"登録番号#{i}", value=0, key=f"reg_{i}")
                name  = cols[1].text_input(f"名前#{i}", value=f"選手{i}", key=f"name_{i}")
                rank  = cols[2].selectbox(f"級#{i}", ["A1","A2","B1","B2"], index=2, key=f"rank_{i}")
                age   = cols[3].number_input(f"年齢#{i}", value=30, key=f"age_{i}")
                wt    = cols[4].number_input(f"体重#{i}", value=52.0, step=0.5, key=f"wt_{i}")

                cols2 = st.columns(5)
                avg_st   = cols2[0].number_input(f"平均ST#{i}", value=0.15, format="%.2f", key=f"st_{i}")
                win_r    = cols2[1].number_input(f"勝率#{i}", value=5.00, format="%.2f", key=f"wr_{i}")
                top2     = cols2[2].number_input(f"2連対率#{i}", value=30.0, format="%.1f", key=f"t2_{i}")
                top3     = cols2[3].number_input(f"3連対率#{i}", value=45.0, format="%.1f", key=f"t3_{i}")
                local_wr = cols2[4].number_input(f"当地勝率#{i}", value=4.50, format="%.2f", key=f"lw_{i}")

                cols3 = st.columns(5)
                lane1 = cols3[0].number_input(f"枠別1着率#{i}", value=10.0, format="%.1f", key=f"l1_{i}")
                abil  = cols3[1].number_input(f"能力#{i}", value=50, key=f"ab_{i}")
                mp    = cols3[2].number_input(f"モーターP#{i}", value=0.0, format="%.2f", key=f"mp_{i}")
                m2r   = cols3[3].number_input(f"モ2連率#{i}", value=30.0, format="%.1f", key=f"m2_{i}")
                ext   = cols3[4].number_input(f"展示T#{i}", value=6.80, format="%.2f", key=f"ex_{i}")

                cols4 = st.columns(5)
                exr   = cols4[0].number_input(f"展示順#{i}", value=i, key=f"exr_{i}")
                turn  = cols4[1].number_input(f"周り足#{i}", value=6.70, format="%.2f", key=f"tu_{i}")
                tilt  = cols4[2].number_input(f"チルト#{i}", value=0.0, format="%.1f", key=f"ti_{i}")
                acc   = cols4[3].number_input(f"事故率#{i}", value=0.0, format="%.2f", key=f"ac_{i}")
                f_cnt = cols4[4].number_input(f"F数#{i}", value=0, key=f"fc_{i}")

                cols5 = st.columns(5)
                k_nige  = cols5[0].number_input(f"逃げ#{i}", value=0, key=f"kn_{i}")
                k_sashi = cols5[1].number_input(f"差し#{i}", value=0, key=f"ks_{i}")
                k_makuri = cols5[2].number_input(f"捲り#{i}", value=0, key=f"km_{i}")
                k_makusa = cols5[3].number_input(f"捲差#{i}", value=0, key=f"kms_{i}")
                k_nuki  = cols5[4].number_input(f"抜き#{i}", value=0, key=f"knk_{i}")

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
                    kimarite_nuki=k_nuki
                ))
        # 天候
        st.markdown("---")
        st.subheader("🌤️ 天候条件")
        wc = st.columns(5)
        w_weather = wc[0].selectbox("天候", ["晴","曇","雨","雪","霧"], index=2)
        w_temp    = wc[1].number_input("気温(℃)", value=13.0, format="%.1f")
        w_wind    = wc[2].number_input("風速(m/s)", value=3.0, format="%.1f")
        w_wtemp   = wc[3].number_input("水温(℃)", value=12.0, format="%.1f")
        w_wave    = wc[4].number_input("波高(cm)", value=3, min_value=0)

    # ── データ確定ボタン ──
    if st.button("✅ データ確定", type="primary", use_container_width=True):
        if input_method == "テキスト貼り付け":
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
        agents = st.session_state["agents"]
        conditions = st.session_state["conditions"]
        st.markdown("---")
        st.subheader("📋 確定データ")

        # 基本情報
        df_basic = pd.DataFrame([{
            "枠": a.lane, "登番": a.number, "名前": a.name, "級": a.rank,
            "年齢": a.age, "体重": a.weight, "平均ST": a.avg_st,
            "勝率": a.national_win_rate, "2連対": a.national_top2_rate,
            "3連対": a.national_top3_rate, "当地勝率": a.local_win_rate,
            "枠別1着": a.lane_win_rate, "能力": a.ability
        } for a in agents])
        st.dataframe(df_basic, use_container_width=True, hide_index=True)

        # 機力・展示
        df_machine = pd.DataFrame([{
            "枠": a.lane, "名前": a.name, "モーターP": a.motor_contribution,
            "モ2連率": a.motor_top2_rate, "展示T": a.exhibition_time,
            "展示順": a.exhibition_rank, "周り足": a.turn_time,
            "チルト": a.tilt, "事故率": a.accident_rate, "F数": a.flying_count
        } for a in agents])
        st.dataframe(df_machine, use_container_width=True, hide_index=True)

        # 決まり手
        df_kima = pd.DataFrame([{
            "枠": a.lane, "名前": a.name, "逃げ": a.kimarite_nige,
            "差し": a.kimarite_sashi, "捲り": a.kimarite_makuri,
            "捲差": a.kimarite_makuzashi, "抜き": a.kimarite_nuki
        } for a in agents])
        st.dataframe(df_kima, use_container_width=True, hide_index=True)

        # 天候
        st.info(f"🌤️ {conditions.weather}　気温{conditions.temperature}℃　"
                f"風速{conditions.wind_speed}m/s　水温{conditions.water_temperature}℃　"
                f"波高{conditions.wave_height}cm")

        # デバッグ
        with st.expander("🔍 デバッグ: デフォルト値チェック"):
            defaults = {"avg_st": 0.15, "national_win_rate": 5.0,
                        "ability": 50, "motor_contribution": 0.0,
                        "exhibition_time": 6.80, "turn_time": 6.70}
            warnings = []
            for a in agents:
                for field, dval in defaults.items():
                    val = getattr(a, field)
                    if abs(val - dval) < 0.001:
                        warnings.append(f"⚠️ {a.lane}号艇 {a.name}: {field} = {val}（デフォルト値の可能性）")
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("✅ すべてのフィールドがデフォルト以外の値です")

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
                finish_cols = st.columns(6)
                for pos, lane in enumerate(order):
                    a = agents[lane - 1]
                    finish_cols[pos].markdown(
                        f"<div style='text-align:center; padding:8px; "
                        f"background:{BOAT_COLORS[lane]}; color:white; "
                        f"border-radius:8px; margin:2px;'>"
                        f"<b>{pos+1}着</b><br>{lane}号艇<br>{a.name}</div>",
                        unsafe_allow_html=True)

                st.write(f"**決まり手:** {kimarite}")

                # ST一覧
                st_df = pd.DataFrame([{
                    "枠": i+1, "名前": agents[i].name,
                    "ST": f"{st_times[i]:.2f}"
                } for i in range(6)])
                st.dataframe(st_df, use_container_width=True, hide_index=True)

                # 3連単・3連複・2連単
                tri = f"{order[0]}-{order[1]}-{order[2]}"
                trio_sorted = sorted(order[:3])
                trio_str = f"{trio_sorted[0]}-{trio_sorted[1]}-{trio_sorted[2]}"
                exacta = f"{order[0]}-{order[1]}"
                st.write(f"**3連単:** {tri}　**3連複:** {trio_str}　**2連単:** {exacta}")

                # レース展開グラフ
                positions = result["positions"]
                fig_race, ax_race = plt.subplots(figsize=(10, 4))
                for lane in range(1, 7):
                    y = positions[lane]
                    x = list(range(len(y)))
                    ax_race.plot(x, y, color=BOAT_COLORS[lane], linewidth=2,
                                 label=f"{lane}号艇 {agents[lane-1].name}")
                ax_race.set_xlabel("ポイント")
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
                        "確率": f"{weights[i]/sum(weights)*100:.1f}%"
                    } for i in range(6)])
                    st.dataframe(w_df, use_container_width=True, hide_index=True)

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
            for i in range(n_mc):
                if i % 100 == 0:
                    progress.progress(i / n_mc)
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
            st.session_state["mc_trifecta_counts"] = trifecta_counts
            st.session_state["mc_n"] = n_mc

            # 勝率テーブル
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
            fig_mc, ax_mc = plt.subplots(figsize=(8, 4))
            bars = ax_mc.bar(range(1, 7), win_counts/n_mc*100,
                             color=[BOAT_COLORS[i] for i in range(1, 7)])
            ax_mc.set_xlabel("枠番")
            ax_mc.set_ylabel("1着率 (%)")
            ax_mc.set_title(f"モンテカルロ 1着率（{n_mc:,}回）")
            ax_mc.set_xticks(range(1, 7))
            for b in bars:
                ax_mc.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                           f"{b.get_height():.1f}%", ha="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_mc)
            plt.close(fig_mc)

            # 決まり手
            st.markdown("### 🥊 決まり手分布")
            kima_df = pd.DataFrame([
                {"決まり手": k, "回数": v, "割合": f"{v/n_mc*100:.1f}%"}
                for k, v in sorted(kima_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(kima_df, use_container_width=True, hide_index=True)

            # 3連単 Top20
            st.markdown("### 🎯 3連単 出現頻度 Top20")
            sorted_tri = sorted(trifecta_counts.items(), key=lambda x: -x[1])[:20]
            tri_df = pd.DataFrame([
                {"3連単": k, "回数": v, "確率": f"{v/n_mc*100:.2f}%"}
                for k, v in sorted_tri
            ])
            st.dataframe(tri_df, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 4 ─ オッズ取得
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_odds:
    st.subheader("💰 オッズ取得")
    odds_method = st.radio("取得方法",
                           ["自動取得（公式サイト）", "テキスト貼り付け", "手動入力"],
                           horizontal=True, key="odds_method")

    venue_code = f"{list(VENUE_PROFILES.keys()).index(venue_name)+1:02d}"
    # 徳山=18 なので正確に取得
    venue_code_map = {
        "桐生":"01","戸田":"02","江戸川":"03","平和島":"04","多摩川":"05",
        "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
        "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
        "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
        "芦屋":"21","福岡":"22","唐津":"23","大村":"24"
    }
    venue_code = venue_code_map.get(venue_name, "18")
    date_str = race_date.strftime("%Y%m%d") if race_date else "20260227"

    if odds_method == "自動取得（公式サイト）":
        st.info(f"🌐 {venue_name} {date_str} {race_no}R の3連単オッズを取得します")
        if st.button("🔄 オッズ取得", key="fetch_odds"):
            with st.spinner("取得中…"):
                odds = fetch_trifecta_odds(venue_code, date_str, race_no)
            if odds:
                st.success(f"✅ {len(odds)}件のオッズを取得しました")
                st.session_state["trifecta_odds"] = odds
            else:
                st.error("❌ オッズ取得に失敗しました。テキスト貼り付けをお試しください。")

    elif odds_method == "テキスト貼り付け":
        odds_text = st.text_area("オッズテキストを貼り付け", height=300,
                                  help="公式サイトからコピー or 「1-2-3 27.7」形式",
                                  key="odds_paste")
        if st.button("📋 オッズ解析", key="parse_odds"):
            odds = parse_pasted_odds(odds_text)
            if odds:
                st.success(f"✅ {len(odds)}件のオッズを取得")
                st.session_state["trifecta_odds"] = odds
            else:
                st.error("❌ 解析失敗。形式を確認してください。")

    else:  # 手動入力
        odds_manual = st.text_area("オッズを入力（1行1組: 1-2-3 27.7）", height=200,
                                    key="odds_manual")
        if st.button("✏️ オッズ登録", key="register_odds"):
            odds = {}
            for line in odds_manual.strip().split("\n"):
                m = re.match(r'(\d)-(\d)-(\d)\s+([\d,.]+)', line.strip())
                if m:
                    key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    odds[key] = float(m.group(4).replace(",", ""))
            if odds:
                st.success(f"✅ {len(odds)}件登録")
                st.session_state["trifecta_odds"] = odds
            else:
                st.error("❌ 有効な行がありません")

    # オッズ表示
    if "trifecta_odds" in st.session_state:
        odds = st.session_state["trifecta_odds"]
        st.markdown("---")
        st.markdown("### 📊 3連単オッズ")
        sorted_odds = sorted(odds.items(), key=lambda x: x[1])
        odds_df = pd.DataFrame([
            {"3連単": k, "オッズ": v} for k, v in sorted_odds[:30]
        ])
        st.dataframe(odds_df, use_container_width=True, hide_index=True)
        st.write(f"**合計:** {len(odds)}件　**最低:** {sorted_odds[0][1]:.1f}　"
                 f"**最高:** {sorted_odds[-1][1]:.1f}")

        # 合成オッズ
        st.markdown("### 🔧 合成オッズ")
        synthetic = compute_synthetic_odds(odds)
        st.session_state["synthetic_odds"] = synthetic
        for bet_type, bet_odds in synthetic.items():
            label_map = {"trio": "3連複", "exacta": "2連単",
                         "quinella": "2連複", "wide": "ワイド"}
            with st.expander(f"📎 {label_map.get(bet_type, bet_type)}（{len(bet_odds)}件）"):
                sorted_bo = sorted(bet_odds.items(), key=lambda x: x[1])[:20]
                bo_df = pd.DataFrame([{"組合せ": k, "オッズ": f"{v:.1f}"} for k, v in sorted_bo])
                st.dataframe(bo_df, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  タブ 5 ─ 期待値計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ev:
    st.subheader("📈 期待値計算")
    if "agents" not in st.session_state:
        st.warning("⬅️ 先に「データ入力」でデータを確定してください")
    elif "trifecta_odds" not in st.session_state:
        st.warning("⬅️ 先に「オッズ取得」でオッズを取得してください")
    else:
        n_ev = st.slider("シミュレーション回数", 1000, 50000, 10000, step=1000, key="ev_n")
        if st.button("📈 期待値計算 実行", type="primary", key="run_ev"):
            agents = st.session_state["agents"]
            conditions = st.session_state["conditions"]
            odds = st.session_state["trifecta_odds"]
            synthetic = st.session_state.get("synthetic_odds", {})
            month = race_date.month if race_date else 2

            all_odds = {"trifecta": odds}
            all_odds.update(synthetic)

            with st.spinner("モンテカルロ実行中…"):
                mc_results = run_ev_simulation(agents, conditions,
                                               venue_profile, month, n_ev)
            ev_results = compute_expected_values(mc_results, all_odds, n_ev)

            # 券種別タブ
            label_map = {"trifecta": "3連単", "trio": "3連複",
                         "exacta": "2連単", "quinella": "2連複", "wide": "ワイド"}
            ev_tabs = st.tabs([label_map.get(k, k) for k in ev_results.keys()])

            all_recommended = []

            for ev_tab, (bet_type, ev_data) in zip(ev_tabs, ev_results.items()):
                with ev_tab:
                    if not ev_data:
                        st.info("該当データなし")
                        continue

                    sorted_ev = sorted(ev_data, key=lambda x: -x["ev"])[:20]
                    ev_df = pd.DataFrame(sorted_ev)
                    display_cols = ["combination", "probability", "odds", "ev", "flag"]
                    rename = {"combination": "組合せ", "probability": "確率",
                              "odds": "オッズ", "ev": "期待値", "flag": "評価"}
                    ev_df_disp = ev_df[display_cols].rename(columns=rename)
                    st.dataframe(ev_df_disp, use_container_width=True, hide_index=True)

                    # EVバーチャート
                    top15 = sorted_ev[:15]
                    fig_ev, ax_ev = plt.subplots(figsize=(10, 5))
                    colors = ["#2ecc71" if d["ev"] >= 1.0 else
                              "#f39c12" if d["ev"] >= 0.8 else "#e74c3c"
                              for d in top15]
                    ax_ev.barh([d["combination"] for d in top15][::-1],
                               [d["ev"] for d in top15][::-1],
                               color=colors[::-1])
                    ax_ev.axvline(x=1.0, color="red", linestyle="--", alpha=0.7)
                    ax_ev.set_xlabel("期待値")
                    ax_ev.set_title(f"{label_map.get(bet_type, bet_type)} 期待値 Top15")
                    plt.tight_layout()
                    st.pyplot(fig_ev)
                    plt.close(fig_ev)

                    recommended = [d for d in ev_data if d["ev"] >= 1.0]
                    for d in recommended:
                        d["bet_type"] = label_map.get(bet_type, bet_type)
                    all_recommended.extend(recommended)

            # おすすめ買い目
            st.markdown("---")
            st.markdown("### 🎯 おすすめ買い目（EV ≥ 1.0）")
            if all_recommended:
                rec_sorted = sorted(all_recommended, key=lambda x: -x["ev"])
                rec_df = pd.DataFrame(rec_sorted)
                display_cols = ["bet_type", "combination", "probability", "odds", "ev"]
                rename = {"bet_type": "券種", "combination": "組合せ",
                          "probability": "確率", "odds": "オッズ", "ev": "期待値"}
                rec_disp = rec_df[display_cols].rename(columns=rename)
                st.dataframe(rec_disp, use_container_width=True, hide_index=True)
                st.info(f"🎯 推奨買い目: {len(all_recommended)}件")
            else:
                st.info("EV ≥ 1.0 の買い目はありません")

            # 準推奨
            with st.expander("📋 準推奨（EV 0.8〜1.0）"):
                semi = []
                for bet_type, ev_data in ev_results.items():
                    for d in ev_data:
                        if 0.8 <= d["ev"] < 1.0:
                            d["bet_type"] = label_map.get(bet_type, bet_type)
                            semi.append(d)
                if semi:
                    semi_sorted = sorted(semi, key=lambda x: -x["ev"])[:30]
                    semi_df = pd.DataFrame(semi_sorted)
                    display_cols = ["bet_type", "combination", "probability", "odds", "ev"]
                    rename = {"bet_type": "券種", "combination": "組合せ",
                              "probability": "確率", "odds": "オッズ", "ev": "期待値"}
                    semi_disp = semi_df[display_cols].rename(columns=rename)
                    st.dataframe(semi_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("該当なし")

# ── フッター ────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em;'>"
    "🚤 ボートレース AI シミュレーター v5.1<br>"
    "30項目完全エージェント × 会場別特性(全24場) × 季節補正 × モンテカルロ × 合成オッズ × 期待値計算"
    "</div>", unsafe_allow_html=True)
