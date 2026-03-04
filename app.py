import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import json
import re
import os
import time
import datetime
import lightgbm as lgb
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib

# ============================================================
# 設定
# ============================================================
BASE_DIR = "/content/drive/MyDrive/boatrace_model"
MODEL_PATH = os.path.join(BASE_DIR, "pairwise_model_v6.txt")
FEATURES_PATH = os.path.join(BASE_DIR, "boat_features_v6.json")
COLUMNS_PATH = os.path.join(BASE_DIR, "column_mapping_v6.json")
PLACE_STATS_PATH = os.path.join(BASE_DIR, "place_stats_v4.json")
TEMP_PATH = os.path.join(BASE_DIR, "temperature_v6.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

PLACE_MAP = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

PLACE_ID_MAP = {
    "桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5,
    "浜名湖": 6, "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10,
    "びわこ": 11, "住之江": 12, "尼崎": 13, "鳴門": 14, "丸亀": 15,
    "児島": 16, "宮島": 17, "徳山": 18, "下関": 19, "若松": 20,
    "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24,
}

GRADE_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
WEATHER_MAP = {"晴": 1, "曇り": 2, "曇": 2, "雨": 3, "雪": 4, "霧": 5}
WIND_DIR_MAP = {
    "北": 0, "北北東": 1, "北東": 2, "東北東": 3,
    "東": 4, "東南東": 5, "南東": 6, "南南東": 7,
    "南": 8, "南南西": 9, "南西": 10, "西南西": 11,
    "西": 12, "西北西": 13, "北西": 14, "北北西": 15,
}


# ============================================================
# モデルロード
# ============================================================
@st.cache_resource
def load_model():
    model = lgb.Booster(model_file=MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        boat_features = json.load(f)

    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(PLACE_STATS_PATH, "r", encoding="utf-8") as f:
        raw_ps = json.load(f)

    # place_stats の変換
    place_stats = {}
    if "place" in raw_ps and "win_rate" in raw_ps:
        place_dict = raw_ps["place"]
        wr_dict = raw_ps["win_rate"]
        for key, place_name in place_dict.items():
            if key in wr_dict:
                wr = wr_dict[key]
                if wr > 1:
                    wr = wr / 100.0
                place_stats[place_name] = {"win_rate": wr}
    else:
        place_stats = raw_ps

    temperature = 5.0
    if os.path.exists(TEMP_PATH):
        with open(TEMP_PATH, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
            temperature = temp_data.get("temperature", temp_data.get("T", 5.0))

    return model, boat_features, feature_names, place_stats, temperature


# ============================================================
# スクレイピング: レース一覧
# ============================================================
def scrape_racelist(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"レース一覧取得エラー: {e}")
        return [default_boat(i) for i in range(1, 7)]

    soup = BeautifulSoup(resp.content, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 2:
        return [default_boat(i) for i in range(1, 7)]

    table = tables[1]
    boats = []

    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 8:
            continue

        # 枠番セルに is-boatColor クラスがあるか
        cell0_classes = cells[0].get("class", [])
        if not any("is-boatColor" in c for c in cell0_classes):
            continue

        waku = len(boats) + 1

        # Cell[2]: 選手情報（登番、級別、名前）
        cell2_text = cells[2].get_text(separator="|", strip=True)
        cell2_parts = [p.strip() for p in cell2_text.split("|") if p.strip()]

        toban = ""
        grade = "B1"
        name = f"枠{waku}"

        for part in cell2_parts:
            if re.match(r"^\d{4}$", part):
                toban = part
            elif part in GRADE_MAP:
                grade = part
            elif re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", part) and len(part) >= 2:
                cleaned = re.sub(r"\s+", "", part)
                if len(cleaned) >= 2 and cleaned not in ["北海道", "青森", "岩手", "宮城", "秋田",
                    "山形", "福島", "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
                    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知",
                    "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山", "鳥取", "島根",
                    "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀",
                    "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"]:
                    name = cleaned

        # Cell[3]: 体重・年齢など
        cell3_text = cells[3].get_text(separator="|", strip=True)
        cell3_parts = [p.strip() for p in cell3_text.split("|") if p.strip()]
        age = 30
        weight = 52.0
        for part in cell3_parts:
            m_age = re.search(r"(\d+)\s*歳", part)
            if m_age:
                age = int(m_age.group(1))
            m_weight = re.search(r"([\d.]+)\s*kg", part)
            if m_weight:
                weight = float(m_weight.group(1))

        # Cell[4]: 全国勝率
        cell4_parts = get_br_split(cells[4])
        national_win_rate = safe_float(cell4_parts[0]) if len(cell4_parts) > 0 else 5.0
        national_2rate = safe_float(cell4_parts[1]) if len(cell4_parts) > 1 else 30.0

        # Cell[5]: 当地勝率
        cell5_parts = get_br_split(cells[5])
        local_win_rate = safe_float(cell5_parts[0]) if len(cell5_parts) > 0 else 0.0
        local_2rate = safe_float(cell5_parts[1]) if len(cell5_parts) > 1 else 0.0

        # Cell[6]: モーター
        cell6_parts = get_br_split(cells[6])
        motor_no = safe_float(cell6_parts[0]) if len(cell6_parts) > 0 else 0
        motor_2rate = safe_float(cell6_parts[1]) if len(cell6_parts) > 1 else 35.0

        # Cell[7]: ボート
        cell7_parts = get_br_split(cells[7])
        boat_no = safe_float(cell7_parts[0]) if len(cell7_parts) > 0 else 0
        boat_2rate = safe_float(cell7_parts[1]) if len(cell7_parts) > 1 else 35.0

        # ST
        st_text = cells[3].get_text() if len(cells) > 3 else ""
        avg_st = 0.15
        m_st = re.search(r"(\d+\.\d+)", cell3_text)
        if m_st:
            val = float(m_st.group(1))
            if 0.01 <= val <= 0.30:
                avg_st = val

        boats.append({
            "waku": waku,
            "name": name,
            "toban": toban,
            "grade": grade,
            "grade_num": GRADE_MAP.get(grade, 2),
            "age": age,
            "weight": weight,
            "national_win_rate": national_win_rate,
            "national_2rate": national_2rate,
            "local_win_rate": local_win_rate,
            "local_2rate": local_2rate,
            "motor_no": motor_no,
            "motor_2rate": motor_2rate,
            "boat_no": boat_no,
            "boat_2rate": boat_2rate,
            "machine_score": (motor_2rate + boat_2rate) / 2,
            "racer_avg_st_20": avg_st,
            "racer_avg_st_10": avg_st,
            "is_waku1": 1 if waku == 1 else 0,
            "is_waku2": 1 if waku == 2 else 0,
            "is_waku3": 1 if waku == 3 else 0,
        })

    while len(boats) < 6:
        boats.append(default_boat(len(boats) + 1))

    return boats[:6]


def default_boat(waku):
    return {
        "waku": waku, "name": f"枠{waku}", "toban": "", "grade": "B1",
        "grade_num": 2, "age": 30, "weight": 52.0,
        "national_win_rate": 5.0, "national_2rate": 30.0,
        "local_win_rate": 0.0, "local_2rate": 0.0,
        "motor_no": 0, "motor_2rate": 35.0, "boat_no": 0, "boat_2rate": 35.0,
        "machine_score": 35.0, "racer_avg_st_20": 0.15, "racer_avg_st_10": 0.15,
        "is_waku1": 1 if waku == 1 else 0,
        "is_waku2": 1 if waku == 2 else 0,
        "is_waku3": 1 if waku == 3 else 0,
    }


def get_br_split(cell):
    cell_copy = BeautifulSoup(str(cell), "html.parser")
    for br in cell_copy.find_all("br"):
        br.replace_with("|")
    return [t.strip() for t in cell_copy.get_text().split("|") if t.strip()]


def safe_float(s, default=0.0):
    try:
        val = re.sub(r"[^\d.\-]", "", str(s))
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


# ============================================================
# スクレイピング: 直前情報
# ============================================================
def scrape_beforeinfo(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
    result = {
        "exhibition_times": [6.80] * 6,
        "entry_courses": list(range(1, 7)),
        "exhibition_sts": [0.15] * 6,
        "wind_speed": 3,
        "wave": 3,
        "wind_dir": "北",
        "wind_dir_num": 0,
        "weather": "晴",
        "weather_num": 1,
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception:
        return result

    soup = BeautifulSoup(resp.content, "html.parser")

    # --- 展示タイム取得 ---
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        boat_idx = 0
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 8:
                cell0_classes = cells[0].get("class", [])
                if any("is-boatColor" in c for c in cell0_classes):
                    # 展示タイムはセルの中から数値を探す
                    for ci in range(3, min(len(cells), 10)):
                        txt = cells[ci].get_text(strip=True)
                        m = re.match(r"^(\d+\.\d{2})$", txt)
                        if m:
                            val = float(m.group(1))
                            if 6.0 <= val <= 8.0:
                                if boat_idx < 6:
                                    result["exhibition_times"][boat_idx] = val
                                break
                    boat_idx += 1

    # --- 進入コース取得 ---
    # table1_boatImage1 内の div から取得
    course_divs = soup.find_all("div", class_=re.compile(r"table1_boatImage1$"))
    if course_divs:
        entry = [0] * 6
        for course_idx, div in enumerate(course_divs):
            if course_idx >= 6:
                break
            # 枠番を取得
            num_span = div.find("span", class_=re.compile(r"boatImage1Number"))
            if num_span:
                txt = num_span.get_text(strip=True)
                m = re.search(r"(\d)", txt)
                if m:
                    waku = int(m.group(1))
                    if 1 <= waku <= 6:
                        entry[course_idx] = waku
            # 展示ST
            st_span = div.find("span", class_=re.compile(r"boatImage1Time"))
            if st_span:
                st_txt = st_span.get_text(strip=True)
                m_st = re.search(r"[F.]?(\d*\.?\d+)", st_txt)
                if m_st:
                    try:
                        st_val = float(m_st.group(1))
                        if st_val < 1.0 and waku and 1 <= waku <= 6:
                            result["exhibition_sts"][waku - 1] = st_val
                    except ValueError:
                        pass
        if all(e > 0 for e in entry):
            # entry[course] = waku → waku の entry_course を設定
            for course_idx, waku in enumerate(entry):
                result["entry_courses"][waku - 1] = course_idx + 1
    else:
        # フォールバック: テーブルから「進入」行を探す
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                txt = row.get_text(strip=True)
                if "進入" in txt or "コース" in txt:
                    cells = row.find_all(["td", "th"])
                    nums = []
                    for c in cells:
                        m = re.search(r"(\d)", c.get_text(strip=True))
                        if m:
                            nums.append(int(m.group(1)))
                    if len(nums) == 6 and all(1 <= n <= 6 for n in nums):
                        for course_idx, waku in enumerate(nums):
                            result["entry_courses"][waku - 1] = course_idx + 1
                        break

    # --- 風速・波・風向・天候 ---
    page_text = soup.get_text()

    # 風速
    m_wind = re.search(r"風速\s*(\d+)\s*m", page_text)
    if m_wind:
        result["wind_speed"] = int(m_wind.group(1))

    # 波
    m_wave = re.search(r"波高\s*(\d+)\s*cm", page_text)
    if m_wave:
        result["wave"] = int(m_wave.group(1))

    # 風向
    wind_spans = soup.find_all("span", class_=re.compile(r"wind"))
    for span in wind_spans:
        cls_list = span.get("class", [])
        for cls in cls_list:
            if "is-wind" in cls:
                m_dir = re.search(r"is-wind(\d+)", cls)
                if m_dir:
                    result["wind_dir_num"] = int(m_dir.group(1))

    for direction, num in WIND_DIR_MAP.items():
        if direction in page_text:
            result["wind_dir"] = direction
            if result["wind_dir_num"] == 0:
                result["wind_dir_num"] = num
            break

    # 天候
    for weather_name, weather_num in WEATHER_MAP.items():
        if weather_name in page_text:
            result["weather"] = weather_name
            result["weather_num"] = weather_num
            break

    return result


# ============================================================
# 開催場取得
# ============================================================
def get_today_venues(hd):
    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={hd}"
    venues = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
        links = soup.find_all("a", href=re.compile(r"raceindex\?jcd="))
        seen = set()
        for link in links:
            m = re.search(r"jcd=(\d{2})", link["href"])
            if m:
                jcd = m.group(1)
                if jcd not in seen:
                    seen.add(jcd)
                    name = PLACE_MAP.get(jcd, jcd)
                    venues.append({"jcd": jcd, "name": name})
    except Exception as e:
        st.warning(f"開催場取得エラー: {e}")
    return venues


# ============================================================
# 予測メイン
# ============================================================
def predict_race(model, boat_features, feature_names, place_stats, temperature,
                 jcd, hd, rno, place_name):

    boats = scrape_racelist(jcd, hd, rno)
    before = scrape_beforeinfo(jcd, hd, rno)

    # --- 場統計 ---
    ps = place_stats.get(place_name, {})
    place_win_rate = ps.get("win_rate", 0.5)
    place_id = PLACE_ID_MAP.get(place_name, 0)

    # --- 枠別統計（概算） ---
    waku_win_rates = {1: 0.58, 2: 0.15, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.04}
    waku_top3_rates = {1: 0.80, 2: 0.55, 3: 0.50, 4: 0.45, 5: 0.35, 6: 0.30}
    upset_rate = 1.0 - place_win_rate if place_win_rate > 0.5 else 0.5

    # --- 各艇にデータ統合 ---
    boat_data = []
    checks = {
        "選手名": True,
        "全国勝率": True,
        "年齢/体重": True,
        "展示タイム": True,
        "進入コース": True,
        "展示ST": True,
        "風/天候": True,
        "場統計": place_name in place_stats,
    }

    for i, boat in enumerate(boats):
        waku = boat["waku"]
        ex_time = before["exhibition_times"][i]
        entry_course = before["entry_courses"][i]
        ex_st = before["exhibition_sts"][i]
        wind_speed = before["wind_speed"]
        wave = before["wave"]
        wind_dir_num = before["wind_dir_num"]
        weather_num = before["weather_num"]

        # 風の影響スコア
        wind_effect = wind_speed * (1 if wind_dir_num <= 4 or wind_dir_num >= 12 else -1)
        is_strong_wind = 1 if wind_speed >= 5 else 0

        features = {
            "waku": waku,
            "grade_num": boat["grade_num"],
            "age": boat["age"],
            "weight": boat["weight"],
            "national_win_rate": boat["national_win_rate"],
            "national_2rate": boat["national_2rate"],
            "local_win_rate": boat["local_win_rate"],
            "local_2rate": boat["local_2rate"],
            "motor_2rate": boat["motor_2rate"],
            "boat_2rate": boat["boat_2rate"],
            "machine_score": boat["machine_score"],
            "racer_avg_st_20": boat["racer_avg_st_20"],
            "racer_avg_st_10": boat["racer_avg_st_10"],
            "is_waku1": boat["is_waku1"],
            "is_waku2": boat["is_waku2"],
            "is_waku3": boat["is_waku3"],
            "exhibition_time": ex_time,
            "entry_course": entry_course,
            "wind_speed": wind_speed,
            "wave": wave,
            "wind_dir_num": wind_dir_num,
            "wind_effect": wind_effect,
            "is_strong_wind": is_strong_wind,
            "weather_num": weather_num,
            "place_in_win_rate": place_win_rate,
            "place_waku_win_rate": waku_win_rates.get(waku, 0.1),
            "place_waku_top3_rate": waku_top3_rates.get(waku, 0.4),
            "place_waku_advantage": waku_win_rates.get(waku, 0.1) - 1 / 6,
            "place_upset_rate": upset_rate,
        }

        boat_data.append(features)

        # チェック
        if boat["name"].startswith("枠"):
            checks["選手名"] = False
        if boat["national_win_rate"] == 5.0:
            checks["全国勝率"] = False
        if boat["age"] == 30 and boat["weight"] == 52.0:
            checks["年齢/体重"] = False
        if ex_time == 6.80:
            checks["展示タイム"] = False

    # --- ペアワイズ特徴量生成 ---
    pair_features_list = []
    pair_ij = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            pair_feat = {}
            for fn in feature_names:
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
                    if jv == 0:
                        jv = 0.001
                    pair_feat[fn] = boat_data[i].get(base, 0) / jv
                else:
                    pair_feat[fn] = boat_data[i].get(fn, 0)
            pair_features_list.append(pair_feat)
            pair_ij.append((i, j))

    # --- 特徴量行列 ---
    X_pred = np.zeros((len(pair_features_list), len(feature_names)))
    for k, pf in enumerate(pair_features_list):
        for fi, fn in enumerate(feature_names):
            X_pred[k, fi] = pf.get(fn, 0)

    # --- 予測 ---
    raw_predictions = model.predict(X_pred)

    # --- ペアワイズ勝率行列（シグモイド変換） ---
    pairwise_matrix_raw = np.zeros((6, 6))
    for idx, (i, j) in enumerate(pair_ij):
        pairwise_matrix_raw[i][j] = raw_predictions[idx]

    pairwise_prob = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i == j:
                pairwise_prob[i][j] = 0.5
            else:
                score_ij = pairwise_matrix_raw[i][j]
                pairwise_prob[i][j] = 1.0 / (1.0 + np.exp(-score_ij))

    # --- スコア集計 ---
    scores = np.zeros(6)
    for idx, (i, j) in enumerate(pair_ij):
        scores[i] += raw_predictions[idx]

    # --- 温度スケーリング + softmax → 勝率 ---
    scaled = scores / temperature
    exp_s = np.exp(scaled - np.max(scaled))
    probs = exp_s / exp_s.sum()

    # --- 3連単確率（Bradley-Terry） ---
    top_combos = []
    try:
        for perm in permutations(range(6), 3):
            i, j, k = perm
            p1 = probs[i] / (probs[i] + probs[j] + probs[k] + 1e-10)
            p2 = probs[j] / (probs[j] + probs[k] + 1e-10)
            combo_prob = p1 * p2
            top_combos.append({
                "combo": f"{boats[i]['waku']}-{boats[j]['waku']}-{boats[k]['waku']}",
                "prob": combo_prob,
                "names": f"{boats[i]['name']}-{boats[j]['name']}-{boats[k]['name']}",
            })
        top_combos.sort(key=lambda x: x["prob"], reverse=True)
        top_combos = top_combos[:20]
    except Exception:
        pass

    # --- 確信度 ---
    sorted_probs = sorted(probs, reverse=True)
    confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0
    top1_prob = sorted_probs[0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(6)

    # --- 艇情報リスト ---
    boat_info_list = []
    for i, boat in enumerate(boats):
        boat_info_list.append({
            **boat,
            "exhibition_time": before["exhibition_times"][i],
            "entry_course": before["entry_courses"][i],
            "exhibition_st": before["exhibition_sts"][i],
        })

    return {
        "place": place_name,
        "race_no": rno,
        "boats": boat_info_list,
        "scores": scores,
        "probs": probs,
        "top_combos": top_combos,
        "confidence": confidence,
        "top1_prob": top1_prob,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "checks": checks,
        "pairwise_prob": pairwise_prob,
        "wind_speed": before["wind_speed"],
        "wave": before["wave"],
        "wind_dir": before.get("wind_dir", "不明"),
        "weather": before.get("weather", "不明"),
    }


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(page_title="ボートレース AI 予測 v6", layout="wide")
    st.title("ボートレース AI 予測 v6")
    st.caption("LightGBM ペアワイズランキングモデル + 温度スケーリング校正")

    # --- サイドバー ---
    st.sidebar.header("モデル情報")
    st.sidebar.markdown("""
    - **モデル**: LightGBM v6 (pairwise lambdarank)
    - **特徴量**: 29 × 116 (pairwise)
    - **温度スケーリング**: 校正済み
    - **バックテスト**: Top1 10.0%, ROI 105.7%
    """)

    # --- モデルロード ---
    try:
        model, boat_features, feature_names, place_stats, temperature = load_model()
        st.sidebar.success(f"モデルロード完了 (T={temperature})")
    except Exception as e:
        st.error(f"モデルロードエラー: {e}")
        st.info(f"モデルパス: {MODEL_PATH}")
        return

    st.sidebar.markdown(f"**特徴量数**: {len(boat_features)} (boat) / {len(feature_names)} (pair)")
    st.sidebar.markdown(f"**場統計**: {len(place_stats)} 場")

    # --- 日付選択 ---
    today = datetime.date.today()
    hd_date = st.sidebar.date_input("日付", value=today)
    hd = hd_date.strftime("%Y%m%d")

    # --- モード選択 ---
    mode = st.sidebar.radio("モード", ["個別レース予測", "全場一括予測"])

    # ==========================
    # 個別レース予測
    # ==========================
    if mode == "個別レース予測":
        st.header("個別レース予測")

        c1, c2 = st.columns(2)
        with c1:
            place_opts = {name: code for code, name in PLACE_MAP.items()}
            sel_place = st.selectbox("場", list(place_opts.keys()))
            jcd = place_opts[sel_place]
        with c2:
            rno = st.selectbox("レース", range(1, 13), format_func=lambda x: f"{x}R")

        if st.button("予測実行", type="primary"):
            with st.spinner("データ取得・予測中..."):
                result = predict_race(
                    model, boat_features, feature_names, place_stats, temperature,
                    jcd, hd, rno, sel_place
                )

            # --- データ取得チェック ---
            st.subheader("データ取得チェック")
            check_cols = st.columns(4)
            for idx, (item, ok) in enumerate(result["checks"].items()):
                with check_cols[idx % 4]:
                    if ok:
                        st.success(f"{item}: OK")
                    else:
                        st.warning(f"{item}: 一部欠損")

            # --- 天候情報 ---
            st.subheader("コンディション")
            w_cols = st.columns(4)
            w_cols[0].metric("天候", result["weather"])
            w_cols[1].metric("風速", f"{result['wind_speed']}m")
            w_cols[2].metric("風向", result["wind_dir"])
            w_cols[3].metric("波高", f"{result['wave']}cm")

            # --- 選手情報・予測結果テーブル ---
            st.subheader("予測結果")
            rows = []
            for i, boat in enumerate(result["boats"]):
                rows.append({
                    "枠": boat["waku"],
                    "選手名": boat["name"],
                    "級別": boat.get("grade", ""),
                    "全国勝率": boat["national_win_rate"],
                    "当地勝率": boat["local_win_rate"],
                    "モーター2率": boat["motor_2rate"],
                    "ボート2率": boat["boat_2rate"],
                    "展示タイム": boat["exhibition_time"],
                    "進入C": boat["entry_course"],
                    "展示ST": boat["exhibition_st"],
                    "AIスコア": round(result["scores"][i], 4),
                    "1着確率": f"{result['probs'][i]*100:.1f}%",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # --- 確信度 ---
            st.subheader("確信度")
            conf_cols = st.columns(3)
            conf_cols[0].metric("Top1確率", f"{result['top1_prob']*100:.1f}%")
            conf_cols[1].metric("Top1-Top2差", f"{result['confidence']*100:.1f}%")
            entropy_ratio = result["entropy"] / result["max_entropy"]
            conf_cols[2].metric("エントロピー比", f"{entropy_ratio:.2f}",
                                help="0に近いほど確信度が高い（1.0 = 完全ランダム）")

            # --- ペアワイズ勝率マトリクス ---
            st.subheader("ペアワイズ勝率マトリクス（各艇同士の勝率）")
            st.caption("行の艇が列の艇に勝つ確率。例: 行「1号艇」× 列「4号艇」＝ 1号艇が4号艇に勝つ確率")

            pw = result["pairwise_prob"]
            pw_boats = result["boats"]
            col_labels = [f"{b['waku']}号艇" for b in pw_boats]
            row_labels = [f"{b['waku']}号艇 {b['name']}" for b in pw_boats]

            pw_df = pd.DataFrame(pw, columns=col_labels, index=row_labels)
            pw_display = pw_df.copy()
            for col in pw_display.columns:
                pw_display[col] = pw_display[col].apply(lambda x: f"{x*100:.1f}%")
            for i in range(6):
                pw_display.iloc[i, i] = "―"
            st.dataframe(pw_display, use_container_width=True)

            # --- ヒートマップ ---
            fig, ax = plt.subplots(figsize=(8, 6))
            matplotlib.rcParams["font.family"] = "DejaVu Sans"

            pw_heatmap = pw.copy()
            for i in range(6):
                pw_heatmap[i][i] = np.nan

            im = ax.imshow(pw_heatmap, cmap="RdYlGn", vmin=0.3, vmax=0.7)
            waku_labels_short = [f"{b['waku']}" for b in pw_boats]
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels(waku_labels_short, fontsize=12)
            ax.set_yticklabels(waku_labels_short, fontsize=12)
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

            # --- ペアワイズ平均勝率ランキング ---
            st.subheader("ペアワイズ平均勝率ランキング")
            st.caption("全対戦相手に対する平均勝率（高いほど総合的に強い）")

            avg_pw = []
            for i in range(6):
                wins = [pw[i][j] for j in range(6) if i != j]
                avg = np.mean(wins)
                avg_pw.append({
                    "順位": 0,
                    "枠": pw_boats[i]["waku"],
                    "選手名": pw_boats[i]["name"],
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

            # --- 3連単予測 ---
            st.subheader("3連単 AI 予測 Top10")
            if result["top_combos"]:
                combo_rows = []
                for ci, combo in enumerate(result["top_combos"][:10], 1):
                    combo_rows.append({
                        "順位": ci,
                        "組合せ": combo["combo"],
                        "選手": combo["names"],
                        "確率": f"{combo['prob']*100:.2f}%",
                    })
                combo_df = pd.DataFrame(combo_rows)
                st.dataframe(combo_df, use_container_width=True, hide_index=True)

            # --- スコア・確率チャート ---
            st.subheader("AIスコア & 1着確率")
            chart_cols = st.columns(2)

            with chart_cols[0]:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                waku_nums = [b["waku"] for b in pw_boats]
                colors = ["#FFFFFF", "#000000", "#FF0000", "#0000FF", "#FFFF00", "#00FF00"]
                edge_colors = ["#333"] * 6
                bar_colors = [colors[w - 1] for w in waku_nums]
                ax1.bar(waku_nums, result["scores"],
                        color=bar_colors, edgecolor=edge_colors, linewidth=1.5)
                ax1.set_xlabel("Waku")
                ax1.set_ylabel("AI Score")
                ax1.set_title("AI Score by Boat")
                ax1.set_xticks(waku_nums)
                st.pyplot(fig1)
                plt.close()

            with chart_cols[1]:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.bar(waku_nums, result["probs"] * 100,
                        color=bar_colors, edgecolor=edge_colors, linewidth=1.5)
                ax2.set_xlabel("Waku")
                ax2.set_ylabel("Win Probability (%)")
                ax2.set_title("Win Probability by Boat")
                ax2.set_xticks(waku_nums)
                ax2.axhline(y=100 / 6, color="red", linestyle="--", alpha=0.5, label="Random (16.7%)")
                ax2.legend()
                st.pyplot(fig2)
                plt.close()

    # ==========================
    # 全場一括予測
    # ==========================
    elif mode == "全場一括予測":
        st.header("全場一括予測")

        if st.button("全場予測開始", type="primary"):
            venues = get_today_venues(hd)
            if not venues:
                st.warning("本日の開催場が見つかりません。")
                return

            st.info(f"開催場: {', '.join(v['name'] for v in venues)} ({len(venues)}場)")

            all_results = []
            progress = st.progress(0)
            total = len(venues) * 12

            for vi, venue in enumerate(venues):
                for rno in range(1, 13):
                    try:
                        result = predict_race(
                            model, boat_features, feature_names, place_stats, temperature,
                            venue["jcd"], hd, rno, venue["name"]
                        )
                        all_results.append(result)
                    except Exception as e:
                        st.warning(f"{venue['name']} {rno}R エラー: {e}")
                    progress.progress((vi * 12 + rno) / total)
                    time.sleep(0.3)

            progress.empty()

            if not all_results:
                st.warning("予測結果がありません。")
                return

            # --- 高確信度レース Top20 ---
            st.subheader("高確信度レース Top20")
            all_results.sort(key=lambda x: x["top1_prob"], reverse=True)

            top_rows = []
            for ri, r in enumerate(all_results[:20], 1):
                best_idx = np.argmax(r["probs"])
                top_rows.append({
                    "順位": ri,
                    "場": r["place"],
                    "R": f"{r['race_no']}R",
                    "本命": f"{r['boats'][best_idx]['waku']}号艇 {r['boats'][best_idx]['name']}",
                    "1着確率": f"{r['top1_prob']*100:.1f}%",
                    "確信度差": f"{r['confidence']*100:.1f}%",
                    "3連単1位": r["top_combos"][0]["combo"] if r["top_combos"] else "-",
                })
            top_df = pd.DataFrame(top_rows)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            # --- 場別全レース結果 ---
            st.subheader("全レース一覧")
            for venue_name in dict.fromkeys(v["name"] for v in venues):
                venue_results = [r for r in all_results if r["place"] == venue_name]
                if not venue_results:
                    continue
                venue_results.sort(key=lambda x: x["race_no"])

                with st.expander(f"{venue_name} ({len(venue_results)}R)", expanded=False):
                    for r in venue_results:
                        best_idx = np.argmax(r["probs"])
                        rows = []
                        for i, boat in enumerate(r["boats"]):
                            rows.append({
                                "枠": boat["waku"],
                                "選手名": boat["name"],
                                "全国勝率": boat["national_win_rate"],
                                "展示タイム": boat["exhibition_time"],
                                "AIスコア": round(r["scores"][i], 4),
                                "1着確率": f"{r['probs'][i]*100:.1f}%",
                            })
                        st.markdown(f"**{r['race_no']}R** | 本命: {r['boats'][best_idx]['waku']}号艇 "
                                    f"{r['boats'][best_idx]['name']} ({r['probs'][best_idx]*100:.1f}%) "
                                    f"| 3連単: {r['top_combos'][0]['combo'] if r['top_combos'] else '-'}")
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
