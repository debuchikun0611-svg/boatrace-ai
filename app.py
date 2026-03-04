import streamlit as st
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

# ============================================================
# 設定
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pairwise_model_v6.txt")
BOAT_FEATURES_PATH = os.path.join(BASE_DIR, "boat_features_v6.json")
COLUMN_MAPPING_PATH = os.path.join(BASE_DIR, "column_mapping_v6.json")
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

# jcd → place_stats内部ID のマッピング
# place_stats_v4.json の "place" dict から逆引き
JCD_TO_STATS_ID = {}

# ============================================================
# モデル読み込み
# ============================================================
@st.cache_resource
def load_model():
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(BOAT_FEATURES_PATH, "r") as f:
        boat_features = json.load(f)
    with open(COLUMN_MAPPING_PATH, "r") as f:
        feature_names = json.load(f)
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

    # place_stats を使いやすい形に変換
    # 元: {"place": {"8": "大村", ...}, "win_rate": {"8": 62.5, ...}, ...}
    # → {"大村": {"win_rate": 62.5, ...}, ...}
    place_stats = {}
    if "place" in place_stats_raw and "win_rate" in place_stats_raw:
        place_id_to_name = place_stats_raw.get("place", {})
        win_rates = place_stats_raw.get("win_rate", {})
        totals = place_stats_raw.get("total", {})
        wins = place_stats_raw.get("win", {})
        for pid, pname in place_id_to_name.items():
            place_stats[pname] = {
                "win_rate": win_rates.get(pid, 50.0) / 100.0,
                "total": totals.get(pid, 1000),
                "wins": wins.get(pid, 500),
            }

    return model, boat_features, feature_names, place_stats, temperature


# ============================================================
# スクレイピング
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

        # Cell[2]: 選手情報
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

        # Cell[3]: F/L/平均ST
        fl_parts = get_br_split(cells[3])
        for p in fl_parts:
            try:
                val = float(p)
                if 0 <= val <= 1.0:
                    boat["avg_st"] = val
            except Exception:
                pass

        # Cell[4]: 全国
        nat = get_br_split(cells[4])
        if len(nat) >= 1: boat["national_win_rate"] = safe_float(nat[0])
        if len(nat) >= 2: boat["national_2rate"] = safe_float(nat[1])
        if len(nat) >= 3: boat["national_3rate"] = safe_float(nat[2])

        # Cell[5]: 当地
        loc = get_br_split(cells[5])
        if len(loc) >= 1: boat["local_win_rate"] = safe_float(loc[0])
        if len(loc) >= 2: boat["local_2rate"] = safe_float(loc[1])
        if len(loc) >= 3: boat["local_3rate"] = safe_float(loc[2])

        # Cell[6]: モーター
        mot = get_br_split(cells[6])
        if len(mot) >= 1: boat["motor_no"] = safe_float(mot[0])
        if len(mot) >= 2: boat["motor_2rate"] = safe_float(mot[1])
        if len(mot) >= 3: boat["motor_3rate"] = safe_float(mot[2])

        # Cell[7]: ボート
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

    # --- Table[1] パース ---
    # 構造: 10セル行=選手行(枠,写真,名前,体重,展示T,チルト,プロペラ,部品交換...)
    #        2セル行=進入行("進入", コース番号)
    #        3セル行=ST行(調整重量, "ST", ST値)
    info = {}
    current_waku = None
    entry_order = []  # 進入順序を記録

    for row in tables[1].find_all("tr"):
        cells = row.find_all(["td", "th"])
        num_cells = len(cells)

        if num_cells >= 8:
            # 選手行: Cell[0]=枠, Cell[4]=展示タイム
            cell0_class = cells[0].get("class", [])
            if any("is-boatColor" in c for c in cell0_class):
                waku_text = cells[0].get_text(strip=True)
                try:
                    current_waku = int(waku_text)
                except ValueError:
                    continue
                ex_time = safe_float(cells[4].get_text(strip=True), 6.80)
                info[current_waku] = {"exhibition_time": ex_time}

        elif num_cells == 2:
            first = cells[0].get_text(strip=True)
            second = cells[1].get_text(strip=True)
            if first == "進入" and current_waku is not None:
                # 進入コース番号
                try:
                    course_pos = int(second)
                    info[current_waku]["entry_course"] = course_pos
                    entry_order.append((current_waku, course_pos))
                except (ValueError, KeyError):
                    pass

        elif num_cells == 3:
            # ST行: Cell[1]="ST", Cell[2]=ST値
            if cells[1].get_text(strip=True) == "ST" and current_waku is not None:
                st_text = cells[2].get_text(strip=True)
                if st_text:
                    st_val = safe_float(st_text.lstrip("."), 0.15)
                    if st_val < 1:
                        pass  # STは展示STとして参考情報のみ

    # --- Table[2]: スタート展示 ---
    # 各行1セル: "\n\n1\n\nF.04\n\n" → コース1, フライング, ST 0.04
    exhibition_st = {}
    if len(tables) >= 3:
        course_idx = 0
        for row in tables[2].find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) == 1:
                text = cells[0].get_text(strip=True).replace("\n", "")
                match = re.match(r"(\d)(F?)\.(\d+)", text)
                if match:
                    course_idx += 1
                    is_f = match.group(2) == "F"
                    st_val = float(f"0.{match.group(3)}")
                    if is_f:
                        st_val = -st_val
                    exhibition_st[course_idx] = st_val

    # 風・天候
    wind_info = {"wind_speed": 3, "wave": 3, "wind_dir_num": 0, "weather_num": 1}
    body_text = soup.get_text()
    ws_m = re.search(r"(\d+)m", body_text)
    if ws_m:
        wind_info["wind_speed"] = int(ws_m.group(1))
    wv_m = re.search(r"(\d+)cm", body_text)
    if wv_m:
        wind_info["wave"] = int(wv_m.group(1))
    wind_dirs = {
        "北北東": 1, "北北西": 15, "北東": 2, "北西": 14, "北": 0,
        "東北東": 3, "東南東": 5, "南南東": 7, "南南西": 9,
        "西南西": 11, "西北西": 13, "南東": 6, "南西": 10,
        "東": 4, "南": 8, "西": 12,
    }
    for wd_name, wd_num in sorted(wind_dirs.items(), key=lambda x: -len(x[0])):
        if wd_name in body_text:
            wind_info["wind_dir_num"] = wd_num
            break
    weather_map = {"晴": 1, "曇": 2, "雨": 3, "雪": 4}
    for wn, wv in weather_map.items():
        if wn in body_text:
            wind_info["weather_num"] = wv
            break

    # デフォルト補完
    for waku in range(1, 7):
        if waku not in info:
            info[waku] = {"exhibition_time": 6.80}
        info[waku].setdefault("entry_course", waku)
        cpos = info[waku]["entry_course"]
        info[waku]["exhibition_st"] = exhibition_st.get(cpos, 0.15)

    info["_wind"] = wind_info
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
# 予測エンジン
# ============================================================
def predict_race(jcd, hd, rno, model, boat_features, feature_names,
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
                                "wind_dir_num": 0, "weather_num": 1}

    place_name = PLACE_MAP.get(jcd, "")
    wind = beforeinfo.get("_wind", {"wind_speed": 3, "wave": 3,
                                     "wind_dir_num": 0, "weather_num": 1})

    ws = wind.get("wind_speed", 3)
    wd = wind.get("wind_dir_num", 0)
    is_strong = 1 if ws >= 5 else 0
    if wd in [8, 9, 10]:
        wind_effect = ws
    elif wd in [0, 1, 15]:
        wind_effect = -ws
    else:
        wind_effect = 0

    # 場の1号艇勝率（place_statsから取得）
    place_1_winrate = place_stats.get(place_name, {}).get("win_rate", 0.50)

    boat_data = []
    for b in racelist:
        w = b["waku"]
        bi = beforeinfo.get(
            w, {"exhibition_time": 6.80, "entry_course": w, "exhibition_st": 0.15}
        )
        grade_num = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}.get(
            b.get("grade", "B1"), 2
        )
        machine_score = (b.get("motor_2rate", 30) + b.get("boat_2rate", 30)) / 2

        # 場×枠の統計（簡易推定）
        # 1号艇勝率をベースに枠ごとの勝率を推定
        waku_base_rates = {
            1: place_1_winrate,
            2: (1 - place_1_winrate) * 0.22,
            3: (1 - place_1_winrate) * 0.20,
            4: (1 - place_1_winrate) * 0.22,
            5: (1 - place_1_winrate) * 0.19,
            6: (1 - place_1_winrate) * 0.17,
        }
        pw_win = waku_base_rates.get(w, 0.15)
        pw_top3 = min(pw_win * 2.5, 0.80)
        pw_in_win = pw_win * 1.1 if w <= 2 else pw_win * 0.9
        pw_upset = 1.0 - place_1_winrate if w >= 4 else place_1_winrate * 0.3
        pw_advantage = place_1_winrate - 0.50 if w == 1 else (0.50 - place_1_winrate) / 5

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
            "racer_avg_st_20": b.get("avg_st", 0.15),
            "racer_avg_st_10": b.get("avg_st", 0.15),
            "is_waku1": 1 if w == 1 else 0,
            "is_waku2": 1 if w == 2 else 0,
            "is_waku3": 1 if w == 3 else 0,
            "exhibition_time": bi.get("exhibition_time", 6.80),
            "entry_course": bi.get("entry_course", w),
            "wind_speed": ws,
            "wave": wind.get("wave", 3),
            "wind_dir_num": wd,
            "wind_effect": wind_effect,
            "is_strong_wind": is_strong,
            "weather_num": wind.get("weather_num", 1),
            "place_waku_win_rate": pw_win,
            "place_waku_top3_rate": pw_top3,
            "place_in_win_rate": pw_in_win,
            "place_upset_rate": pw_upset,
            "place_waku_advantage": pw_advantage,
        }
        features["_name"] = b.get("name", f"枠{w}")
        features["_grade"] = b.get("grade", "B1")
        boat_data.append(features)

    if len(boat_data) != 6:
        return None

        # ペアワイズ特徴量（修正版）
    # boat_features = ["waku", "grade_num", ...] （29個の生特徴量名）
    # feature_names = ["i_waku", ..., "j_waku", ..., "waku_diff", ..., "waku_ratio", ...] （116個）
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

    X_pred = np.zeros((len(pair_features_list), len(feature_names)))
    for k, pf in enumerate(pair_features_list):
        for fi, fn in enumerate(feature_names):
            X_pred[k, fi] = pf.get(fn, 0)


    X_pred = np.zeros((len(pair_features_list), len(feature_names)))
    for k, pf in enumerate(pair_features_list):
        for fi, fn in enumerate(feature_names):
            X_pred[k, fi] = pf.get(fn, 0)

    preds = model.predict(X_pred)
    scores = np.zeros(6)
    for k, (i, j) in enumerate(pair_ij):
        scores[i] += preds[k]

    ranked = np.argsort(-scores)
    wakus = [bd["waku"] for bd in boat_data]

    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    win_probs = exp_scores / exp_scores.sum()

    combos = []
    for perm in permutations(range(6), 3):
        i1, i2, i3 = perm
        remaining1 = list(range(6))
        p1 = win_probs[i1] / sum(win_probs[r] for r in remaining1)
        remaining2 = [r for r in remaining1 if r != i1]
        p2 = win_probs[i2] / sum(win_probs[r] for r in remaining2)
        remaining3 = [r for r in remaining2 if r != i2]
        p3 = win_probs[i3] / sum(win_probs[r] for r in remaining3)
        combo_prob = p1 * p2 * p3
        combo_str = f"{wakus[i1]}-{wakus[i2]}-{wakus[i3]}"
        combos.append({"combo": combo_str, "prob": combo_prob})

    combos.sort(key=lambda x: -x["prob"])
    conf_score = combos[0]["prob"] - combos[1]["prob"] if len(combos) >= 2 else 0

    return {
        "jcd": jcd, "place": place_name, "rno": rno,
        "boat_data": boat_data, "scores": scores, "win_probs": win_probs,
        "ranked": ranked, "combos": combos[:20], "conf_score": conf_score,
        "top1_prob": combos[0]["prob"], "top1_combo": combos[0]["combo"],
        "wind": wind,
    }


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="ボートレース AI 予測", layout="wide")
st.title("🚤 ボートレース AI 予測 v6")
st.caption("LightGBM ペアワイズランキングモデル + Temperature Scaling 校正済み")

model, boat_features, feature_names, place_stats, temperature = load_model()
st.sidebar.success(f"モデル読込完了 (特徴量{len(boat_features)}, T={temperature})")

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
                        model, boat_features, feature_names,
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
                        "確率": f"{res['top1_prob']*100:.1f}%",
                        "確信度": f"{res['conf_score']*100:.2f}%",
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
                        for res in sorted(
                            venue_results, key=lambda x: x["rno"]
                        ):
                            col_left, col_right = st.columns([3, 1])
                            with col_left:
                                top3_str = "　".join(
                                    f"{c['combo']}({c['prob']*100:.1f}%)"
                                    for c in res["combos"][:3]
                                )
                                st.text(f"{res['rno']}R: {top3_str}")
                            with col_right:
                                st.text(
                                    f"確信度 {res['conf_score']*100:.2f}%"
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
                model, boat_features, feature_names,
                place_stats, temperature,
            )

        if result is None:
            st.error(
                "データ取得失敗。日付・場・レース番号を確認してください。"
            )
        else:
            st.success(f"{result['place']} {result['rno']}R 予測完了")

            wind = result.get("wind", {})
            if wind:
                weather_names = {0: "-", 1: "晴", 2: "曇", 3: "雨", 4: "雪"}
                st.info(
                    f"天候: {weather_names.get(wind.get('weather_num', 0), '-')}　"
                    f"風速: {wind.get('wind_speed', '-')}m　"
                    f"波高: {wind.get('wave', '-')}cm"
                )

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
                    "AIスコア": f"{result['scores'][i]:.2f}",
                    "1着確率": f"{result['win_probs'][i]*100:.1f}%",
                })
            st.dataframe(
                pd.DataFrame(boat_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("🎯 3連単予測 TOP10")
            combo_rows = []
            for rank, combo in enumerate(result["combos"][:10], 1):
                expected = (
                    f"{1/combo['prob']:.0f}倍"
                    if combo["prob"] > 0
                    else "-"
                )
                combo_rows.append({
                    "順位": rank,
                    "3連単": combo["combo"],
                    "確率": f"{combo['prob']*100:.2f}%",
                    "期待倍率": expected,
                })
            st.dataframe(
                pd.DataFrame(combo_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.metric(
                label="確信度スコア",
                value=f"{result['conf_score']*100:.2f}%",
                help="Top1確率 − Top2確率。大きいほど予測に自信あり。",
            )

# ============================================================
# フッター
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**モデル情報**
- LightGBM v6 (pairwise lambdarank)
- 29特徴量 × ペアワイズ116特徴量
- 展示T/進入C/勝率/モーター/風/場統計
- 温度スケーリング校正済み (T=5.0)
- バックテスト: Top1 10.0%, ROI 105.7%
"""
)
st.sidebar.caption("※ 予測は参考情報です。投票は自己責任でお願いします。")
