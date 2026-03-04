# app.py - ボートレース AI 予測 v6 完全版
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

# ============================================================
# モデル読み込み（キャッシュ）
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
            place_stats = json.load(f)
    except Exception:
        place_stats = {}
    try:
        with open(TEMPERATURE_PATH, "r") as f:
            temperature = json.load(f).get("temperature", 5.0)
    except Exception:
        temperature = 5.0
    return model, boat_features, feature_names, place_stats, temperature

# ============================================================
# スクレイピング関数
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

    table = tables[1]
    rows = table.find_all("tr")
    boats = []
    idx = 0
    while idx < len(rows):
        cells = rows[idx].find_all(["td", "th"])
        if len(cells) < 3:
            idx += 1
            continue
        waku_text = cells[0].get_text(strip=True)
        if waku_text not in ["1", "2", "3", "4", "5", "6"]:
            idx += 1
            continue

        waku = int(waku_text)
        boat = {"waku": waku}

        parts = get_br_split(cells[2])
        if len(parts) >= 2:
            boat["toban"] = safe_float(parts[0], 0)
            boat["name"] = parts[1]
        if len(parts) >= 3:
            for g in ["A1", "A2", "B1", "B2"]:
                if g in parts[-1]:
                    boat["grade"] = g
                    break

        if len(cells) > 3:
            fl_parts = get_br_split(cells[3])
            for p in fl_parts:
                try:
                    val = float(p)
                    if 0 <= val <= 1.0:
                        boat["avg_st"] = val
                except Exception:
                    pass

        if len(cells) > 4:
            rp = get_br_split(cells[4])
            if len(rp) >= 1:
                boat["national_win_rate"] = safe_float(rp[0])
            if len(rp) >= 2:
                boat["national_2rate"] = safe_float(rp[1])
            if len(rp) >= 3:
                boat["national_3rate"] = safe_float(rp[2])

        if len(cells) > 5:
            rp = get_br_split(cells[5])
            if len(rp) >= 1:
                boat["local_win_rate"] = safe_float(rp[0])
            if len(rp) >= 2:
                boat["local_2rate"] = safe_float(rp[1])
            if len(rp) >= 3:
                boat["local_3rate"] = safe_float(rp[2])

        if len(cells) > 6:
            rp = get_br_split(cells[6])
            if len(rp) >= 2:
                boat["motor_no"] = safe_float(rp[0])
                boat["motor_2rate"] = safe_float(rp[1])
            if len(rp) >= 3:
                boat["motor_3rate"] = safe_float(rp[2])

        if len(cells) > 7:
            rp = get_br_split(cells[7])
            if len(rp) >= 2:
                boat["boat_no"] = safe_float(rp[0])
                boat["boat_2rate"] = safe_float(rp[1])
            if len(rp) >= 3:
                boat["boat_3rate"] = safe_float(rp[2])

        boat.setdefault("grade", "B1")
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
        boat.setdefault("name", f"枠{waku}")

        boats.append(boat)
        idx += 1

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

    info = {}
    table1 = tables[1]
    rows1 = table1.find_all("tr")
    for row in rows1:
        cells = row.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        waku_text = cells[0].get_text(strip=True)
        if waku_text in ["1", "2", "3", "4", "5", "6"]:
            waku = int(waku_text)
            ex_time = 6.80
            for cell in cells:
                for p in get_br_split(cell):
                    try:
                        val = float(p)
                        if 6.0 <= val <= 7.5:
                            ex_time = val
                            break
                    except Exception:
                        pass
            info[waku] = {"exhibition_time": ex_time}

    course_map = {}
    if len(tables) >= 3:
        table2 = tables[2]
        for cell in table2.find_all(["td", "th"]):
            text = cell.get_text(strip=True)
            match = re.match(r"(\d)(F?)\.(\d+)", text)
            if match:
                cpos = int(match.group(1))
                is_f = match.group(2) == "F"
                st_val = float(f"0.{match.group(3)}")
                if is_f:
                    st_val = -st_val
                course_map[cpos] = st_val

    entry_courses = {}
    for i in range(1, 7):
        elem = soup.find(class_=f"table1_boatImage1Number{i}")
        if elem:
            try:
                entry_courses[int(elem.get_text(strip=True))] = i
            except Exception:
                pass

    for waku in range(1, 7):
        if waku not in info:
            info[waku] = {"exhibition_time": 6.80}
        if waku in entry_courses:
            info[waku]["entry_course"] = entry_courses[waku]
        else:
            info[waku]["entry_course"] = waku
        cpos = info[waku]["entry_course"]
        info[waku]["exhibition_st"] = course_map.get(cpos, 0.15)

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

    place_name = PLACE_MAP.get(jcd, "")

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

        pw_key = f"{place_name}_{w}"
        pw_win = (
            place_stats.get(pw_key, {}).get("win_rate", 0.15)
            if place_stats else 0.15
        )
        pw_top3 = (
            place_stats.get(pw_key, {}).get("top3_rate", 0.45)
            if place_stats else 0.45
        )
        pw_upset = (
            place_stats.get(pw_key, {}).get("upset_rate", 0.10)
            if place_stats else 0.10
        )

        features = {
            "waku": w,
            "grade_num": grade_num,
            "national_win_rate": b.get("national_win_rate", 4.0),
            "national_2rate": b.get("national_2rate", 20.0),
            "national_3rate": b.get("national_3rate", 30.0),
            "local_win_rate": b.get("local_win_rate", 4.0),
            "local_2rate": b.get("local_2rate", 20.0),
            "local_3rate": b.get("local_3rate", 30.0),
            "motor_2rate": b.get("motor_2rate", 30.0),
            "motor_3rate": b.get("motor_3rate", 40.0),
            "boat_2rate": b.get("boat_2rate", 30.0),
            "boat_3rate": b.get("boat_3rate", 40.0),
            "machine_score": machine_score,
            "exhibition_time": bi.get("exhibition_time", 6.80),
            "entry_course": bi.get("entry_course", w),
            "racer_avg_st_20": b.get("avg_st", 0.15),
            "racer_avg_st_10": b.get("avg_st", 0.15),
            "is_waku1": 1 if w == 1 else 0,
            "is_inner": 1 if w <= 3 else 0,
            "place_waku_win_rate": pw_win,
            "place_waku_top3_rate": pw_top3,
            "place_upset_rate": pw_upset,
        }
        features["_name"] = b.get("name", f"枠{w}")
        features["_grade"] = b.get("grade", "B1")
        boat_data.append(features)

    if len(boat_data) != 6:
        return None

    # ペアワイズ特徴量
    pair_features_list = []
    pair_ij = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            pair_feat = {}
            for bf in boat_features:
                if bf.startswith("i_"):
                    pair_feat[bf] = boat_data[i].get(bf[2:], 0)
                elif bf.startswith("j_"):
                    pair_feat[bf] = boat_data[j].get(bf[2:], 0)
                elif bf.endswith("_diff"):
                    base = bf.replace("_diff", "")
                    pair_feat[bf] = (
                        boat_data[i].get(base, 0) - boat_data[j].get(base, 0)
                    )
                elif bf.endswith("_ratio"):
                    base = bf.replace("_ratio", "")
                    jv = boat_data[j].get(base, 1)
                    if jv == 0:
                        jv = 0.001
                    pair_feat[bf] = boat_data[i].get(base, 0) / jv
            pair_features_list.append(pair_feat)
            pair_ij.append((i, j))

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

    # 温度スケーリング
    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    win_probs = exp_scores / exp_scores.sum()

    # 3連単確率（Bradley-Terry近似）
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
    conf_score = (
        combos[0]["prob"] - combos[1]["prob"] if len(combos) >= 2 else 0
    )

    return {
        "jcd": jcd,
        "place": place_name,
        "rno": rno,
        "boat_data": boat_data,
        "scores": scores,
        "win_probs": win_probs,
        "ranked": ranked,
        "combos": combos[:20],
        "conf_score": conf_score,
        "top1_prob": combos[0]["prob"],
        "top1_combo": combos[0]["combo"],
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
                    status.text(
                        f"{venue['name']} {race_no}R ({n}/{total})"
                    )
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

                # --- 勝負レース TOP20 ---
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
                    use_container_width=True,
                    hide_index=True,
                )

                # --- 場別一覧 ---
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
                    with st.expander(
                        f"🏟️ {vname} ({len(venue_results)}R)"
                    ):
                        for res in sorted(venue_results, key=lambda x: x["rno"]):
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

            # --- 選手情報 ---
            st.subheader("🚤 選手情報 & AIスコア")
            boat_rows = []
            for i, bd in enumerate(result["boat_data"]):
                boat_rows.append({
                    "枠": bd["waku"],
                    "選手": bd["_name"],
                    "級": bd["_grade"],
                    "全国勝率": bd.get("national_win_rate", 0),
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

            # --- 3連単予測 ---
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
                    "期待倍率": expected,
                })
            st.dataframe(
                pd.DataFrame(combo_rows),
                use_container_width=True,
                hide_index=True,
            )

            # --- 確信度 ---
            st.metric(
                label="確信度スコア",
                value=f"{result['conf_score']*100:.2f}%",
                help="Top1確率 − Top2確率。大きいほど予測に自信あり。",
            )

# ============================================================
# サイドバー フッター
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**モデル情報**
- LightGBM v6 (pairwise lambdarank)
- 展示タイム / 進入コース / 勝率 / モーター等
- 実ST除外 / 選手平均ST使用
- 温度スケーリング校正済み (T=5.0)
- バックテスト: Top1 10.0%, ROI 105.7%
"""
)
st.sidebar.caption("※ 予測は参考情報です。投票は自己責任でお願いします。")
