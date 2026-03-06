# ============================================================
# app.py を LambdaRank v2 対応に書き換え → 保存
# ============================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os

BASE = "/content/drive/MyDrive/boatrace"

new_app = r'''import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import re
import os
import time
import datetime
import lightgbm as lgb
from itertools import permutations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ============================================================
# 設定
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# v6 ペアワイズ（ペアワイズ勝率マトリクス表示用に残す）
MODEL_V6_PATH = os.path.join(BASE_DIR, "pairwise_model_v6.txt")
BOAT_FEATURES_V6_PATH = os.path.join(BASE_DIR, "boat_features_v6.json")
COLUMN_MAPPING_V6_PATH = os.path.join(BASE_DIR, "column_mapping_v6.json")

# LambdaRank v2（三連単予測のメインエンジン）
MODEL_LR2_PATH = os.path.join(BASE_DIR, "lambdarank_model_v2.txt")
LR2_FEATURES_PATH = os.path.join(BASE_DIR, "lambdarank_features_v2.json")

PLACE_STATS_PATH = os.path.join(BASE_DIR, "place_stats_v4.json")
TEMPERATURE_PATH = os.path.join(BASE_DIR, "temperature_v6.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

PLACE_MAP = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"
}


# ============================================================
# モデル読み込み
# ============================================================
@st.cache_resource
def load_model():
    # v6 ペアワイズ（マトリクス表示用）
    model_v6 = lgb.Booster(model_file=MODEL_V6_PATH)
    with open(BOAT_FEATURES_V6_PATH, "r") as f:
        v6_boat_features = json.load(f)
    with open(COLUMN_MAPPING_V6_PATH, "r") as f:
        v6_feature_names = json.load(f)

    # LambdaRank v2（三連単メイン）
    model_lr2 = lgb.Booster(model_file=MODEL_LR2_PATH)
    with open(LR2_FEATURES_PATH, "r") as f:
        lr2_feature_names = json.load(f)

    try:
        with open(PLACE_STATS_PATH, "r") as f:
            place_stats_raw = json.load(f)
    except Exception:
        place_stats_raw = {}
    try:
        with open(TEMPERATURE_PATH, "r") as f:
            temperature = json.load(f).get("temperature", 5.0)
    except Exception:
        temperature = 5.0

    place_stats = {}
    if "place" in place_stats_raw and "win_rate" in place_stats_raw:
        pid_to_name = place_stats_raw.get("place", {})
        win_rates = place_stats_raw.get("win_rate", {})
        for pid, pname in pid_to_name.items():
            place_stats[pname] = {
                "win_rate": win_rates.get(pid, 50.0) / 100.0,
            }

    return (model_v6, v6_boat_features, v6_feature_names,
            model_lr2, lr2_feature_names,
            place_stats, temperature)


# ============================================================
# スクレイピング（変更なし）
# ============================================================
def get_br_split(cell):
    for br in cell.find_all("br"):
        br.replace_with("|")
    return [t.strip() for t in cell.get_text().split("|") if t.strip()]


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def scrape_racelist(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception:
        return None

    tables = soup.find_all("table")
    if len(tables) < 2:
        return None

    rows = tables[1].find_all("tr")
    boats = []

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 8:
            continue
        cell0_class = cells[0].get("class", [])
        if not any("is-boatColor" in c for c in cell0_class):
            continue

        waku_text = cells[0].get_text(strip=True)
        waku_map = {"１": 1, "２": 2, "３": 3, "４": 4, "５": 5, "６": 6,
                    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
        waku = waku_map.get(waku_text, 0)
        if waku == 0:
            continue

        boat = {"waku": waku}

        racer_text = cells[2].get_text(separator="|", strip=True)
        racer_parts = [p.strip() for p in racer_text.split("|") if p.strip()]

        if racer_parts:
            boat["toban"] = safe_float(re.sub(r"\D", "", racer_parts[0]), 0)

        for rp in racer_parts:
            for g in ["A1", "A2", "B1", "B2"]:
                if g in rp:
                    boat["grade"] = g
                    break

        for rp in racer_parts:
            cleaned = rp.replace("\u3000", "").replace(" ", "").strip()
            if re.match(r"^[\u3040-\u9fff]+$", cleaned) and len(cleaned) >= 2:
                boat["name"] = rp.replace("\u3000", " ").strip()
                break

        for rp in racer_parts:
            age_m = re.search(r"(\d+)歳", rp)
            if age_m:
                boat["age"] = int(age_m.group(1))
            wt_m = re.search(r"([\d.]+)kg", rp)
            if wt_m:
                boat["weight"] = float(wt_m.group(1))

        fl_parts = get_br_split(cells[3])
        for p in fl_parts:
            try:
                val = float(p)
                if 0 <= val <= 1.0:
                    boat["avg_st"] = val
            except Exception:
                pass

        nat = get_br_split(cells[4])
        if len(nat) >= 1: boat["national_win_rate"] = safe_float(nat[0])
        if len(nat) >= 2: boat["national_2rate"] = safe_float(nat[1])
        if len(nat) >= 3: boat["national_3rate"] = safe_float(nat[2])

        loc = get_br_split(cells[5])
        if len(loc) >= 1: boat["local_win_rate"] = safe_float(loc[0])
        if len(loc) >= 2: boat["local_2rate"] = safe_float(loc[1])
        if len(loc) >= 3: boat["local_3rate"] = safe_float(loc[2])

        mot = get_br_split(cells[6])
        if len(mot) >= 1: boat["motor_no"] = safe_float(mot[0])
        if len(mot) >= 2: boat["motor_2rate"] = safe_float(mot[1])
        if len(mot) >= 3: boat["motor_3rate"] = safe_float(mot[2])

        bot = get_br_split(cells[7])
        if len(bot) >= 1: boat["boat_no"] = safe_float(bot[0])
        if len(bot) >= 2: boat["boat_2rate"] = safe_float(bot[1])
        if len(bot) >= 3: boat["boat_3rate"] = safe_float(bot[2])

        boat.setdefault("grade", "B1")
        boat.setdefault("name", f"枠{waku}")
        boat.setdefault("age", 35)
        boat.setdefault("weight", 52.0)
        boat.setdefault("national_win_rate", 4.0)
        boat.setdefault("national_2rate", 20.0)
        boat.setdefault("national_3rate", 30.0)
        boat.setdefault("local_win_rate", 4.0)
        boat.setdefault("local_2rate", 20.0)
        boat.setdefault("local_3rate", 30.0)
        boat.setdefault("motor_2rate", 30.0)
        boat.setdefault("motor_3rate", 40.0)
        boat.setdefault("boat_2rate", 30.0)
        boat.setdefault("boat_3rate", 40.0)
        boat.setdefault("avg_st", 0.15)

        boats.append(boat)

    return boats if len(boats) == 6 else None


def scrape_beforeinfo(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception:
        return None

    tables = soup.find_all("table")
    if len(tables) < 2:
        return None

    checks = {
        "exhibition_time": [],
        "entry_course": False,
        "exhibition_st": [],
        "wind": False,
    }

    info = {}
    for row in tables[1].find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 8:
            continue
        cell0_class = cells[0].get("class", [])
        if not any("is-boatColor" in c for c in cell0_class):
            continue
        waku_text = cells[0].get_text(strip=True)
        try:
            waku = int(waku_text)
        except ValueError:
            continue
        if waku < 1 or waku > 6:
            continue
        ex_time = safe_float(cells[4].get_text(strip=True), 0)
        if 6.0 <= ex_time <= 7.5:
            info[waku] = {"exhibition_time": ex_time}
            checks["exhibition_time"].append(waku)
        else:
            info[waku] = {"exhibition_time": 6.80}

    boat_images = soup.find_all("div", class_="table1_boatImage1")
    entry_courses = {}
    exhibition_sts = {}

    for course_pos, div in enumerate(boat_images, 1):
        num_span = div.find("span", class_="table1_boatImage1Number")
        waku = None
        if num_span:
            try:
                waku = int(num_span.get_text(strip=True))
                entry_courses[waku] = course_pos
            except ValueError:
                continue

        time_span = div.find("span", class_="table1_boatImage1Time")
        if time_span and waku:
            st_text = time_span.get_text(strip=True)
            if st_text:
                is_flying = "F" in st_text
                num_part = st_text.replace("F", "").strip()
                try:
                    if num_part.startswith("."):
                        st_val = float("0" + num_part)
                    else:
                        st_val = float(num_part)
                    if is_flying:
                        st_val = -st_val
                    exhibition_sts[waku] = st_val
                    checks["exhibition_st"].append(waku)
                except ValueError:
                    pass

    if len(entry_courses) == 6:
        checks["entry_course"] = True

    wind_info = {"wind_speed": 3, "wave": 3, "wind_dir_num": 0, "weather_num": 1,
                 "weather_name": "不明", "wind_dir_name": "不明"}

    weather_body = soup.find("div", class_="weather1Body")
    if weather_body:
        weather_text = weather_body.get_text()
        ws_m = re.search(r"風速\s*(\d+)\s*m", weather_text)
        if ws_m:
            wind_info["wind_speed"] = int(ws_m.group(1))
            checks["wind"] = True
        wv_m = re.search(r"波高\s*(\d+)\s*cm", weather_text)
        if wv_m:
            wind_info["wave"] = int(wv_m.group(1))
        weather_map = {"晴": 1, "曇り": 2, "曇": 2, "雨": 3, "雪": 4, "霧": 5}
        for wn, wv in weather_map.items():
            if wn in weather_text:
                wind_info["weather_num"] = wv
                wind_info["weather_name"] = wn
                break

    if not checks["wind"]:
        all_spans = soup.find_all("span")
        for span in all_spans:
            txt = span.get_text(strip=True)
            m = re.match(r"^(\d+)m$", txt)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 20:
                    wind_info["wind_speed"] = val
                    checks["wind"] = True
                    break
        for span in all_spans:
            txt = span.get_text(strip=True)
            m = re.match(r"^(\d+)cm$", txt)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 30:
                    wind_info["wave"] = val
                    break

    weather_span = soup.find("span", class_=re.compile(r"is-weather"))
    if weather_span:
        w_cls = " ".join(weather_span.get("class", []))
        if "is-weather1" in w_cls:
            wind_info["weather_num"] = 1; wind_info["weather_name"] = "晴"
        elif "is-weather2" in w_cls:
            wind_info["weather_num"] = 2; wind_info["weather_name"] = "曇"
        elif "is-weather3" in w_cls:
            wind_info["weather_num"] = 3; wind_info["weather_name"] = "雨"

    wind_dirs = {
        "北北東": 1, "北北西": 15, "北東": 2, "北西": 14, "北": 0,
        "東北東": 3, "東南東": 5, "南南東": 7, "南南西": 9,
        "西南西": 11, "西北西": 13, "南東": 6, "南西": 10,
        "東": 4, "南": 8, "西": 12,
    }
    wind_div = soup.find("div", class_=re.compile(r"is-wind"))
    if wind_div:
        w_cls = " ".join(wind_div.get("class", []))
        m = re.search(r"is-wind(\d+)", w_cls)
        if m:
            wind_info["wind_dir_num"] = int(m.group(1))

    body_text = soup.get_text()
    for wd_name, wd_num in sorted(wind_dirs.items(), key=lambda x: -len(x[0])):
        if wd_name in body_text:
            wind_info["wind_dir_name"] = wd_name
            if wind_info["wind_dir_num"] == 0:
                wind_info["wind_dir_num"] = wd_num
            checks["wind"] = True
            break

    for waku in range(1, 7):
        if waku not in info:
            info[waku] = {"exhibition_time": 6.80}
        if waku in entry_courses:
            info[waku]["entry_course"] = entry_courses[waku]
        else:
            info[waku]["entry_course"] = waku
        if waku in exhibition_sts:
            info[waku]["exhibition_st"] = exhibition_sts[waku]
        else:
            info[waku]["exhibition_st"] = 0.15

    info["_wind"] = wind_info
    info["_checks"] = checks
    return info


def check_venue_open(jcd, hd):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd={jcd}&hd={hd}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            if len(soup.find_all("table")) >= 2:
                return True
    except Exception:
        pass
    return False


# ============================================================
# 予測エンジン（LambdaRank v2 + v6ペアワイズ併用）
# ============================================================
def predict_race(jcd, hd, rno, model_v6, v6_boat_features, v6_feature_names,
                 model_lr2, lr2_feature_names,
                 place_stats, temperature):
    racelist = scrape_racelist(jcd, hd, rno)
    if racelist is None:
        return None

    beforeinfo = scrape_beforeinfo(jcd, hd, rno)
    if beforeinfo is None:
        beforeinfo = {
            w: {"exhibition_time": 6.80, "entry_course": w, "exhibition_st": 0.15}
            for w in range(1, 7)
        }
        beforeinfo["_wind"] = {"wind_speed": 3, "wave": 3,
                                "wind_dir_num": 0, "weather_num": 1,
                                "weather_name": "不明", "wind_dir_name": "不明"}
        beforeinfo["_checks"] = {"exhibition_time": [], "entry_course": False,
                                  "exhibition_st": [], "wind": False}

    place_name = PLACE_MAP.get(jcd, "")
    wind = beforeinfo.get("_wind", {"wind_speed": 3, "wave": 3,
                                     "wind_dir_num": 0, "weather_num": 1})
    checks = beforeinfo.get("_checks", {})

    ws = wind.get("wind_speed", 3)
    wd = wind.get("wind_dir_num", 0)
    is_strong = 1 if ws >= 5 else 0
    if wd in [8, 9, 10]:
        wind_effect = ws
    elif wd in [0, 1, 15]:
        wind_effect = -ws
    else:
        wind_effect = 0

    place_1_winrate = place_stats.get(place_name, {}).get("win_rate", 0.50)

    # データ取得チェック
    data_checks = []
    names_ok = sum(1 for b in racelist if not b.get("name", "").startswith("枠"))
    data_checks.append(("選手名", f"{names_ok}/6",
                        "OK" if names_ok == 6 else "一部欠損"))
    rates_ok = sum(1 for b in racelist if b.get("national_win_rate", 4.0) != 4.0)
    data_checks.append(("全国勝率", f"{rates_ok}/6",
                        "OK" if rates_ok >= 5 else "一部欠損"))
    ages_ok = sum(1 for b in racelist if b.get("age", 35) != 35)
    data_checks.append(("年齢/体重", f"{ages_ok}/6",
                        "OK" if ages_ok >= 5 else "一部欠損"))
    et_ok = len(checks.get("exhibition_time", []))
    data_checks.append(("展示タイム", f"{et_ok}/6",
                        "OK" if et_ok == 6 else "一部欠損"))
    ec_ok = checks.get("entry_course", False)
    data_checks.append(("進入コース", "6/6" if ec_ok else "0/6",
                        "OK" if ec_ok else "未取得（枠順で代用）"))
    es_ok = len(checks.get("exhibition_st", []))
    data_checks.append(("展示ST", f"{es_ok}/6",
                        "OK" if es_ok >= 4 else "一部欠損"))
    wd_ok = checks.get("wind", False)
    data_checks.append(("風/天候", "あり" if wd_ok else "デフォルト",
                        "OK" if wd_ok else "未取得"))
    data_checks.append(("場の統計", "あり" if place_name in place_stats else "なし",
                        "OK" if place_name in place_stats else "デフォルト"))

    # --- 各艇の特徴量構築 ---
    waku_base_rates = {
        1: place_1_winrate,
        2: (1 - place_1_winrate) * 0.22,
        3: (1 - place_1_winrate) * 0.20,
        4: (1 - place_1_winrate) * 0.22,
        5: (1 - place_1_winrate) * 0.19,
        6: (1 - place_1_winrate) * 0.17,
    }

    boat_data = []
    for b in racelist:
        w = b["waku"]
        bi = beforeinfo.get(w, {"exhibition_time": 6.80, "entry_course": w,
                                 "exhibition_st": 0.15})
        grade_num = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}.get(b.get("grade", "B1"), 2)
        machine_score = (b.get("motor_2rate", 30) + b.get("boat_2rate", 30)) / 2
        pw_win = waku_base_rates.get(w, 0.15)

        features = {
            "waku": w,
            "grade_num": grade_num,
            "age": b.get("age", 35),
            "weight": b.get("weight", 52.0),
            "national_win_rate": b.get("national_win_rate", 4.0),
            "national_2rate": b.get("national_2rate", 20.0),
            "local_win_rate": b.get("local_win_rate", 4.0),
            "local_2rate": b.get("local_2rate", 20.0),
            "motor_2rate": b.get("motor_2rate", 30.0),
            "boat_2rate": b.get("boat_2rate", 30.0),
            "machine_score": machine_score,
            "exhibition_time": bi.get("exhibition_time", 6.80),
            "entry_course": bi.get("entry_course", w),
            "avg_st": b.get("avg_st", 0.15),
            "weather_num": wind.get("weather_num", 1),
            "wind_dir_num": wd,
            "wind_speed": ws,
            "wave": wind.get("wave", 3),
            "wind_effect": wind_effect,
            "is_strong_wind": is_strong,
            "place_w1_winrate": place_1_winrate,
            "place_upset_rate": 1.0 - place_1_winrate,
            "place_waku_win_rate": pw_win,
            "place_waku_top3_rate": min(pw_win * 2.5, 0.80),
            "place_in_win_rate": pw_win * 1.1 if w <= 2 else pw_win * 0.9,
            "waku_win_hist": {1:0.55,2:0.14,3:0.12,4:0.10,5:0.06,6:0.03}.get(w, 0.1),
            # v6互換
            "racer_avg_st_20": b.get("avg_st", 0.15),
            "racer_avg_st_10": b.get("avg_st", 0.15),
            "is_waku1": 1 if w == 1 else 0,
            "is_waku2": 1 if w == 2 else 0,
            "is_waku3": 1 if w == 3 else 0,
            "place_waku_advantage": place_1_winrate - 0.50 if w == 1 else (0.50 - place_1_winrate) / 5,
        }
        features["_name"] = b.get("name", f"枠{w}")
        features["_grade"] = b.get("grade", "B1")
        features["_exhibition_st"] = bi.get("exhibition_st", 0.15)
        boat_data.append(features)

    if len(boat_data) != 6:
        return None

    # --- レース内相対特徴量（LR2用） ---
    nwr_vals = [bd["national_win_rate"] for bd in boat_data]
    m2r_vals = [bd["motor_2rate"] for bd in boat_data]
    et_vals = [bd["exhibition_time"] for bd in boat_data]
    gn_vals = [bd["grade_num"] for bd in boat_data]
    ms_vals = [bd["machine_score"] for bd in boat_data]

    def z_score(vals):
        m = np.mean(vals); s = np.std(vals) + 1e-8
        return [(v - m) / s for v in vals]

    def ranks_desc(vals):
        indexed = sorted(enumerate(vals), key=lambda x: -x[1])
        ranks = [0] * len(vals)
        for rank, (idx, _) in enumerate(indexed, 1):
            ranks[idx] = rank
        return ranks

    nwr_z = z_score(nwr_vals)
    m2r_z = z_score(m2r_vals)
    et_z = z_score(et_vals)
    gn_z = z_score(gn_vals)
    nwr_rank = ranks_desc(nwr_vals)
    m2r_rank = ranks_desc(m2r_vals)
    et_rank = ranks_desc(et_vals)
    gn_rank = ranks_desc(gn_vals)

    # 1号艇データ
    w1_data = boat_data[0]  # waku==1 は常にindex 0
    et_mean = np.mean(et_vals)
    et_best = np.min(et_vals)

    for i, bd in enumerate(boat_data):
        bd["national_win_rate_z"] = nwr_z[i]
        bd["motor_2rate_z"] = m2r_z[i]
        bd["exhibition_time_z"] = et_z[i]
        bd["grade_num_z"] = gn_z[i]
        bd["national_win_rate_race_rank"] = nwr_rank[i]
        bd["motor_2rate_race_rank"] = m2r_rank[i]
        bd["exhibition_time_race_rank"] = et_rank[i]
        bd["grade_num_race_rank"] = gn_rank[i]
        bd["diff_w1_national_win_rate"] = bd["national_win_rate"] - w1_data["national_win_rate"]
        bd["diff_w1_motor_2rate"] = bd["motor_2rate"] - w1_data["motor_2rate"]
        bd["diff_w1_exhibition_time"] = bd["exhibition_time"] - w1_data["exhibition_time"]
        bd["diff_w1_grade_num"] = bd["grade_num"] - w1_data["grade_num"]
        bd["diff_w1_machine_score"] = bd["machine_score"] - w1_data["machine_score"]
        bd["et_diff_mean"] = bd["exhibition_time"] - et_mean
        bd["et_diff_best"] = bd["exhibition_time"] - et_best
        bd["race_national_win_rate_mean"] = np.mean(nwr_vals)
        bd["race_national_win_rate_std"] = np.std(nwr_vals)
        bd["race_motor_2rate_mean"] = np.mean(m2r_vals)
        bd["race_motor_2rate_std"] = np.std(m2r_vals)
        bd["race_grade_num_mean"] = np.mean(gn_vals)
        bd["race_grade_num_std"] = np.std(gn_vals)

    # ==========================================
    # v6 ペアワイズ予測（マトリクス表示 + LR2特徴量用）
    # ==========================================
    pair_features_list = []
    pair_ij = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            pair_feat = {}
            for fn in v6_feature_names:
                if fn.startswith("i_"):
                    base = fn[2:]
                    pair_feat[fn] = boat_data[i].get(base, 0)
                elif fn.startswith("j_"):
                    base = fn[2:]
                    pair_feat[fn] = boat_data[j].get(base, 0)
                elif fn.endswith("_diff"):
                    base = fn.replace("_diff", "")
                    pair_feat[fn] = boat_data[i].get(base, 0) - boat_data[j].get(base, 0)
                elif fn.endswith("_ratio"):
                    base = fn.replace("_ratio", "")
                    jv = boat_data[j].get(base, 1)
                    if jv == 0: jv = 0.001
                    pair_feat[fn] = boat_data[i].get(base, 0) / jv
                else:
                    pair_feat[fn] = boat_data[i].get(fn, 0)
            pair_features_list.append(pair_feat)
            pair_ij.append((i, j))

    X_v6 = np.zeros((len(pair_features_list), len(v6_feature_names)))
    for k, pf in enumerate(pair_features_list):
        for fi, fn in enumerate(v6_feature_names):
            X_v6[k, fi] = pf.get(fn, 0)

    v6_preds = model_v6.predict(X_v6)

    # ペアワイズ勝率マトリクス
    pairwise_prob = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i == j:
                pairwise_prob[i][j] = 0.5
    for k, (i, j) in enumerate(pair_ij):
        pairwise_prob[i][j] = 1.0 / (1.0 + np.exp(-v6_preds[k]))

    # v6 スコア集計（LR2の特徴量として使用）
    v6_scores = np.zeros(6)
    for k, (i, j) in enumerate(pair_ij):
        v6_scores[i] += 1.0 / (1.0 + np.exp(-v6_preds[k]))
    v6_top1_idx = np.argmax(v6_scores)

    # v6の温度スケーリング勝率（表示用）
    scaled = v6_scores / temperature
    exp_s = np.exp(scaled - np.max(scaled))
    v6_win_probs = exp_s / exp_s.sum()

    # ==========================================
    # LambdaRank v2 三連単予測
    # ==========================================
    # LR2用の艇特徴量リスト（バックテストと同じ順序）
    lr_boat_feats = [
        "waku","grade_num","age","weight",
        "national_win_rate","national_2rate","local_win_rate","local_2rate",
        "motor_2rate","boat_2rate","machine_score",
        "exhibition_time","entry_course","avg_st",
        "weather_num","wind_dir_num","wind_speed","wave","wind_effect","is_strong_wind",
        "place_w1_winrate","place_upset_rate","place_waku_win_rate",
        "place_waku_top3_rate","place_in_win_rate","waku_win_hist",
        "national_win_rate_z","motor_2rate_z","exhibition_time_z","grade_num_z",
        "national_win_rate_race_rank","motor_2rate_race_rank",
        "exhibition_time_race_rank","grade_num_race_rank",
        "diff_w1_national_win_rate","diff_w1_motor_2rate",
        "diff_w1_exhibition_time","diff_w1_grade_num","diff_w1_machine_score",
        "et_diff_mean","et_diff_best",
        "race_national_win_rate_mean","race_national_win_rate_std",
        "race_motor_2rate_mean","race_motor_2rate_std",
        "race_grade_num_mean","race_grade_num_std",
    ]
    diff_feats = ["national_win_rate","motor_2rate","exhibition_time",
                  "grade_num","machine_score","entry_course"]
    trio_feats = ["national_win_rate","motor_2rate","grade_num","machine_score"]

    n_lr2_feat = len(lr2_feature_names)
    wakus = [bd["waku"] for bd in boat_data]
    all_perms = list(permutations(range(6), 3))  # 120通り

    X_lr2 = np.zeros((120, n_lr2_feat), dtype=np.float32)

    for pi, perm in enumerate(all_perms):
        i1, i2, i3 = perm
        b1, b2, b3 = boat_data[i1], boat_data[i2], boat_data[i3]
        idx = 0

        # 1st, 2nd, 3rd 個別特徴量
        for b in [b1, b2, b3]:
            for f in lr_boat_feats:
                X_lr2[pi, idx] = b.get(f, 0)
                idx += 1

        # ペア差分
        for f in diff_feats:
            X_lr2[pi, idx] = b1.get(f, 0) - b2.get(f, 0); idx += 1
            X_lr2[pi, idx] = b1.get(f, 0) - b3.get(f, 0); idx += 1
            X_lr2[pi, idx] = b2.get(f, 0) - b3.get(f, 0); idx += 1

        # trio 統計
        for f in trio_feats:
            vs = [b1.get(f, 0), b2.get(f, 0), b3.get(f, 0)]
            X_lr2[pi, idx] = np.mean(vs); idx += 1
            X_lr2[pi, idx] = np.std(vs); idx += 1
            X_lr2[pi, idx] = np.min(vs); idx += 1
            X_lr2[pi, idx] = np.max(vs); idx += 1

        # 枠番特徴
        w1, w2, w3 = wakus[i1], wakus[i2], wakus[i3]
        X_lr2[pi, idx] = w1 + w2 + w3; idx += 1
        X_lr2[pi, idx] = w1 * w2 * w3; idx += 1
        X_lr2[pi, idx] = 1 if w1 == 1 else 0; idx += 1

        # v6 スコア
        X_lr2[pi, idx] = v6_scores[i1]; idx += 1
        X_lr2[pi, idx] = v6_scores[i2]; idx += 1
        X_lr2[pi, idx] = v6_scores[i3]; idx += 1
        X_lr2[pi, idx] = v6_scores[i1] - v6_scores[i2]; idx += 1
        X_lr2[pi, idx] = 1 if i1 == v6_top1_idx else 0; idx += 1

    lr2_scores = model_lr2.predict(X_lr2)

    # スコア → 確率（softmax）
    lr2_max = np.max(lr2_scores)
    lr2_exp = np.exp(lr2_scores - lr2_max)
    lr2_probs = lr2_exp / lr2_exp.sum()

    combos = []
    for pi, perm in enumerate(all_perms):
        i1, i2, i3 = perm
        combo_str = f"{wakus[i1]}-{wakus[i2]}-{wakus[i3]}"
        combos.append({"combo": combo_str, "prob": float(lr2_probs[pi]),
                        "score": float(lr2_scores[pi])})

    combos.sort(key=lambda x: -x["prob"])
    conf_score = combos[0]["prob"] - combos[1]["prob"] if len(combos) >= 2 else 0

    # 各艇の1着確率（LR2から集計）
    win_probs_lr2 = np.zeros(6)
    for pi, perm in enumerate(all_perms):
        win_probs_lr2[perm[0]] += lr2_probs[pi]

    return {
        "jcd": jcd, "place": place_name, "rno": rno,
        "boat_data": boat_data,
        "scores": v6_scores,  # v6スコア（表示用）
        "win_probs": win_probs_lr2,  # LR2ベースの1着確率
        "v6_win_probs": v6_win_probs,  # v6の1着確率（参考）
        "ranked": np.argsort(-win_probs_lr2),
        "combos": combos[:20],
        "conf_score": conf_score,
        "top1_prob": combos[0]["prob"],
        "top1_combo": combos[0]["combo"],
        "wind": wind, "data_checks": data_checks,
        "pairwise_prob": pairwise_prob,
        "model_name": "LambdaRank v2",
    }


# ============================================================
# 展示ST表示用フォーマット関数
# ============================================================
def format_st(val):
    if val is None or val == 0.15:
        return "-"
    if val < 0:
        return f"F{abs(val):.2f}"
    else:
        return f".{val:.2f}"[1:]


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="ボートレース AI 予測", layout="wide")
st.title("🚤 ボートレース AI 予測 v7")
st.caption("LambdaRank v2 三連単直接最適化 + ペアワイズv6 マトリクス表示")

(model_v6, v6_boat_features, v6_feature_names,
 model_lr2, lr2_feature_names,
 place_stats, temperature) = load_model()
st.sidebar.success(
    f"モデル読込完了\\n"
    f"- LambdaRank v2 ({len(lr2_feature_names)}特徴量)\\n"
    f"- ペアワイズv6 ({len(v6_boat_features)}基礎特徴量, T={temperature})"
)

st.sidebar.header("設定")
today = datetime.date.today()
selected_date = st.sidebar.date_input("日付", value=today)
hd = selected_date.strftime("%Y%m%d")

mode = st.sidebar.radio("モード選択", ["全場一括予測", "個別レース予測"])

# ============================================================
# 全場一括予測
# ============================================================
if mode == "全場一括予測":
    st.header("📊 全場全レース 一括予測")

    if st.button("一括予測を開始", type="primary"):
        with st.spinner("開催場を検出中..."):
            venues = []
            for code, name in PLACE_MAP.items():
                if check_venue_open(code, hd):
                    venues.append({"jcd": code, "name": name})
                time.sleep(0.3)

        if not venues:
            st.error("開催場が見つかりません。日付を確認してください。")
        else:
            venue_names = ", ".join(venue["name"] for venue in venues)
            st.info(f"開催場: {venue_names} ({len(venues)}場)")

            all_results = []
            total = len(venues) * 12
            bar = st.progress(0)
            status = st.empty()

            for vi, venue in enumerate(venues):
                for race_no in range(1, 13):
                    n = vi * 12 + race_no
                    status.text(f"{venue['name']} {race_no}R ({n}/{total})")
                    result = predict_race(
                        venue["jcd"], hd, race_no,
                        model_v6, v6_boat_features, v6_feature_names,
                        model_lr2, lr2_feature_names,
                        place_stats, temperature,
                    )
                    if result:
                        all_results.append(result)
                    time.sleep(0.5)
                    bar.progress(n / total)

            bar.progress(1.0)
            status.text(f"完了: {len(all_results)} レース予測")

            if all_results:
                all_results.sort(key=lambda x: -x["conf_score"])

                st.subheader("🏆 今日の勝負レース TOP20（確信度順）")
                top_rows = []
                for rank, res in enumerate(all_results[:20], 1):
                    top_rows.append({
                        "順位": rank,
                        "場": res["place"],
                        "R": f"{res['rno']}R",
                        "予測1位": res["top1_combo"],
                        "確率": f"{res['top1_prob']*100:.2f}%",
                        "確信度": f"{res['conf_score']*100:.3f}%",
                    })
                st.dataframe(
                    pd.DataFrame(top_rows),
                    use_container_width=True, hide_index=True,
                )

                st.subheader("📋 全レース一覧")
                venue_name_list = list(
                    dict.fromkeys(venue["name"] for venue in venues)
                )
                for vname in venue_name_list:
                    venue_results = [
                        r for r in all_results if r["place"] == vname
                    ]
                    if not venue_results:
                        continue
                    with st.expander(f"🏟️ {vname} ({len(venue_results)}R)"):
                        for res in sorted(venue_results, key=lambda x: x["rno"]):
                            col_left, col_right = st.columns([3, 1])
                            with col_left:
                                top3_str = "　".join(
                                    f"{c['combo']}({c['prob']*100:.2f}%)"
                                    for c in res["combos"][:3]
                                )
                                st.text(f"{res['rno']}R: {top3_str}")
                            with col_right:
                                st.text(
                                    f"確信度 {res['conf_score']*100:.3f}%"
                                )

                st.session_state["all_results"] = all_results

# ============================================================
# 個別レース予測
# ============================================================
elif mode == "個別レース予測":
    st.header("🎯 個別レース予測")

    col1, col2 = st.columns(2)
    with col1:
        place_opts = {name: code for code, name in PLACE_MAP.items()}
        sel_place = st.selectbox("場", list(place_opts.keys()))
        jcd = place_opts[sel_place]
    with col2:
        rno = st.selectbox(
            "レース", range(1, 13), format_func=lambda x: f"{x}R"
        )

    if st.button("予測する", type="primary"):
        with st.spinner(f"{sel_place} {rno}R 予測中..."):
            result = predict_race(
                jcd, hd, rno,
                model_v6, v6_boat_features, v6_feature_names,
                model_lr2, lr2_feature_names,
                place_stats, temperature,
            )

        if result is None:
            st.error(
                "データ取得失敗。日付・場・レース番号を確認してください。"
            )
        else:
            st.success(f"{result['place']} {result['rno']}R 予測完了 ({result.get('model_name', 'LR2')})")

            # --- データ取得チェック ---
            dc = result.get("data_checks", [])
            if dc:
                with st.expander("📋 データ取得チェック", expanded=False):
                    check_rows = []
                    for item_name, count, status in dc:
                        if status == "OK":
                            icon = "✅"
                        elif "欠損" in status or "未取得" in status:
                            icon = "⚠️"
                        else:
                            icon = "ℹ️"
                        check_rows.append({
                            "": icon,
                            "項目": item_name,
                            "取得数": count,
                            "状態": status,
                        })
                    st.dataframe(
                        pd.DataFrame(check_rows),
                        use_container_width=True, hide_index=True,
                    )

            # 天候情報
            wind = result.get("wind", {})
            if wind:
                weather_name = wind.get("weather_name", "不明")
                if weather_name == "不明":
                    weather_names = {0: "-", 1: "晴", 2: "曇", 3: "雨", 4: "雪"}
                    weather_name = weather_names.get(wind.get("weather_num", 0), "-")
                wind_dir_name = wind.get("wind_dir_name", "")
                st.info(
                    f"天候: {weather_name}　"
                    f"風向: {wind_dir_name}　"
                    f"風速: {wind.get('wind_speed', '-')}m　"
                    f"波高: {wind.get('wave', '-')}cm"
                )

            # 選手情報
            st.subheader("🚤 選手情報 & AIスコア")
            boat_rows = []
            for i, bd in enumerate(result["boat_data"]):
                boat_rows.append({
                    "枠": bd["waku"],
                    "選手": bd["_name"],
                    "級": bd["_grade"],
                    "全国勝率": bd.get("national_win_rate", 0),
                    "当地勝率": bd.get("local_win_rate", 0),
                    "モーター2連": bd.get("motor_2rate", 0),
                    "展示T": bd.get("exhibition_time", 0),
                    "進入C": bd.get("entry_course", bd["waku"]),
                    "展示ST": format_st(bd.get("_exhibition_st", 0.15)),
                    "1着確率": f"{result['win_probs'][i]*100:.1f}%",
                })
            st.dataframe(
                pd.DataFrame(boat_rows),
                use_container_width=True, hide_index=True,
            )

            # スタート展示
            st.subheader("🏁 スタート展示")
            st_rows = []
            course_data = []
            for bd in result["boat_data"]:
                course_data.append({
                    "course": bd.get("entry_course", bd["waku"]),
                    "waku": bd["waku"],
                    "name": bd["_name"],
                    "st": bd.get("_exhibition_st", 0.15),
                })
            course_data.sort(key=lambda x: x["course"])

            for cd in course_data:
                st_val = cd["st"]
                if st_val < 0:
                    st_display = f"F{abs(st_val):.2f}"
                    st_status = "⚠️ フライング"
                elif st_val == 0.15:
                    st_display = "-"
                    st_status = "未取得"
                elif st_val <= 0.05:
                    st_display = f".{st_val:.2f}"[1:]
                    st_status = "🟢 好スタート"
                elif st_val <= 0.10:
                    st_display = f".{st_val:.2f}"[1:]
                    st_status = "🟡 普通"
                else:
                    st_display = f".{st_val:.2f}"[1:]
                    st_status = "🔴 遅い"

                st_rows.append({
                    "コース": cd["course"],
                    "枠": f"{cd['waku']}号艇",
                    "選手": cd["name"],
                    "ST": st_display,
                    "評価": st_status,
                })

            st.dataframe(
                pd.DataFrame(st_rows),
                use_container_width=True, hide_index=True,
            )

            # 3連単予測
            st.subheader("🎯 3連単予測 TOP10")
            combo_rows = []
            for rank, combo in enumerate(result["combos"][:10], 1):
                expected = (
                    f"{1/combo['prob']:.0f}倍" if combo["prob"] > 0 else "-"
                )
                combo_rows.append({
                    "順位": rank,
                    "3連単": combo["combo"],
                    "確率": f"{combo['prob']*100:.2f}%",
                    "スコア": f"{combo['score']:.3f}",
                    "期待倍率": expected,
                })
            st.dataframe(
                pd.DataFrame(combo_rows),
                use_container_width=True, hide_index=True,
            )

            st.metric(
                label="確信度スコア",
                value=f"{result['conf_score']*100:.3f}%",
                help="Top1確率 − Top2確率。大きいほど予測に自信あり。",
            )

            # ペアワイズ勝率マトリクス
            st.subheader("⚔️ ペアワイズ勝率マトリクス（v6参考）")
            st.caption("ペアワイズv6による各艇同士の勝率。三連単予測はLambdaRank v2が担当。")

            pw = result["pairwise_prob"]
            pw_boats = result["boat_data"]

            col_labels = [f"{b['waku']}号艇" for b in pw_boats]
            row_labels = [f"{b['waku']}号艇 {b['_name']}" for b in pw_boats]

            pw_display = pd.DataFrame(
                [[f"{pw[i][j]*100:.1f}%" if i != j else "―" for j in range(6)]
                 for i in range(6)],
                columns=col_labels,
                index=row_labels,
            )
            st.dataframe(pw_display, use_container_width=True)

            # ヒートマップ
            fig, ax = plt.subplots(figsize=(8, 6))
            pw_heatmap = pw.copy()
            for i in range(6):
                pw_heatmap[i][i] = np.nan

            im = ax.imshow(pw_heatmap, cmap="RdYlGn", vmin=0.3, vmax=0.7)
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels([f"{b['waku']}" for b in pw_boats], fontsize=12)
            ax.set_yticklabels([f"{b['waku']}" for b in pw_boats], fontsize=12)
            ax.set_xlabel("Opponent (j)", fontsize=12)
            ax.set_ylabel("Boat (i)", fontsize=12)
            ax.set_title("P(i beats j) - Pairwise Win Probability", fontsize=14)

            for i in range(6):
                for j in range(6):
                    if i == j:
                        ax.text(j, i, "―", ha="center", va="center",
                                fontsize=11, color="gray")
                    else:
                        val = pw[i][j] * 100
                        color = "white" if val > 65 or val < 35 else "black"
                        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                                fontsize=10, fontweight="bold", color=color)

            plt.colorbar(im, ax=ax, label="Win Probability")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ペアワイズ平均勝率ランキング
            st.subheader("🏅 ペアワイズ平均勝率ランキング (v6参考)")
            avg_pw = []
            for i in range(6):
                wins = [pw[i][j] for j in range(6) if i != j]
                avg = np.mean(wins)
                avg_pw.append({
                    "順位": 0,
                    "枠": pw_boats[i]["waku"],
                    "選手名": pw_boats[i]["_name"],
                    "平均勝率": f"{avg*100:.1f}%",
                    "avg_raw": avg,
                    "対戦詳細": " / ".join([
                        f"vs{pw_boats[j]['waku']}号艇:{pw[i][j]*100:.0f}%"
                        for j in range(6) if i != j
                    ]),
                })
            avg_pw.sort(key=lambda x: x["avg_raw"], reverse=True)
            for rank, item in enumerate(avg_pw, 1):
                item["順位"] = rank

            avg_df = pd.DataFrame(avg_pw)[["順位", "枠", "選手名", "平均勝率", "対戦詳細"]]
            st.dataframe(avg_df, use_container_width=True, hide_index=True)

# ============================================================
# フッター
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**モデル情報**
- **メイン: LambdaRank v2** (三連単直接最適化)
- サブ: ペアワイズv6 (マトリクス表示用)
- バックテスト: Top1 10.05%, ROI 106.2%
- 確信度5%以上: 的中率19%, ROI 127%
"""
)
st.sidebar.caption("※ 予測は参考情報です。投票は自己責任でお願いします。")
'''

# 保存
# まず旧版をバックアップ
backup_path = os.path.join(BASE, "app_v6_backup.py")
app_path = os.path.join(BASE, "app.py")

if os.path.exists(app_path):
    import shutil
    shutil.copy2(app_path, backup_path)
    print(f"✅ バックアップ: {backup_path}")

with open(app_path, "w") as f:
    f.write(new_app)

print(f"✅ 新 app.py 保存: {app_path}")
print(f"   行数: {len(new_app.splitlines())}")

# 必要ファイル確認
required = [
    "pairwise_model_v6.txt",
    "boat_features_v6.json",
    "column_mapping_v6.json",
    "lambdarank_model_v2.txt",
    "lambdarank_features_v2.json",
    "place_stats_v4.json",
    "temperature_v6.json",
]
print("\n📋 必要ファイル確認:")
for fname in required:
    path = os.path.join(BASE, fname)
    exists = os.path.exists(path)
    size = os.path.getsize(path) / 1024 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.0f} KB)")

print("\n✅ 完了！")
print("app.pyをデプロイ先にアップロードしてください。")
print("lambdarank_model_v2.txt と lambdarank_features_v2.json も同じディレクトリに配置してください。")
