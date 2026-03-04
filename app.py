import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import time
import datetime
import lightgbm as lgb
from itertools import permutations
import pandas as pd

# ============================================================
# 設定（Streamlit Cloud用 - リポジトリ直下のファイルを参照）
# ============================================================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "pairwise_model_v6.txt")
BOAT_FEATURES_PATH = os.path.join(BASE_DIR, "boat_features_v6.json")
COLUMN_MAPPING_PATH = os.path.join(BASE_DIR, "column_mapping_v6.json")
PLACE_STATS_PATH = os.path.join(BASE_DIR, "place_stats_v4.json")
TEMPERATURE_PATH = os.path.join(BASE_DIR, "temperature_v6.json")


# ============================================================
# モデルとデータの読み込み（キャッシュ）
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
    except:
        place_stats = {}
    try:
        with open(TEMPERATURE_PATH, "r") as f:
            temp_data = json.load(f)
            temperature = temp_data.get("temperature", 5.0)
    except:
        temperature = 5.0
    return model, boat_features, feature_names, place_stats, temperature

# ============================================================
# スクレイピング関数
# ============================================================
def get_br_split(cell):
    """セル内の <br/> を区切り文字にしてテキストを分離"""
    for br in cell.find_all("br"):
        br.replace_with("|")
    return [t.strip() for t in cell.get_text().split("|") if t.strip()]

def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

def scrape_racelist(jcd, hd, rno):
    """出走表をスクレイピング"""
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        return None

    tables = soup.find_all("table")
    if len(tables) < 2:
        return None

    table = tables[1]
    rows = table.find_all("tr")

    boats = []
    i = 0
    while i < len(rows):
        cells = rows[i].find_all(["td", "th"])
        if len(cells) < 3:
            i += 1
            continue

        # 枠番を探す
        waku_text = cells[0].get_text(strip=True)
        if waku_text not in ["1", "2", "3", "4", "5", "6"]:
            i += 1
            continue

        waku = int(waku_text)
        boat = {"waku": waku}

        # 登番・選手名・級別
        parts = get_br_split(cells[2])
        if len(parts) >= 2:
            boat["toban"] = safe_float(parts[0], 0)
            boat["name"] = parts[1] if len(parts) > 1 else ""
        if len(parts) >= 3:
            grade_str = parts[-1].strip()
            for g in ["A1", "A2", "B1", "B2"]:
                if g in grade_str:
                    boat["grade"] = g
                    break

        # F/L/平均ST
        if len(cells) > 3:
            fl_parts = get_br_split(cells[3])
            for p in fl_parts:
                try:
                    v = float(p)
                    if 0 <= v <= 1.0:
                        boat["avg_st"] = v
                except:
                    pass

        # 全国勝率
        if len(cells) > 4:
            rate_parts = get_br_split(cells[4])
            if len(rate_parts) >= 1:
                boat["national_win_rate"] = safe_float(rate_parts[0])
            if len(rate_parts) >= 2:
                boat["national_2rate"] = safe_float(rate_parts[1])
            if len(rate_parts) >= 3:
                boat["national_3rate"] = safe_float(rate_parts[2])

        # 当地勝率
        if len(cells) > 5:
            rate_parts = get_br_split(cells[5])
            if len(rate_parts) >= 1:
                boat["local_win_rate"] = safe_float(rate_parts[0])
            if len(rate_parts) >= 2:
                boat["local_2rate"] = safe_float(rate_parts[1])
            if len(rate_parts) >= 3:
                boat["local_3rate"] = safe_float(rate_parts[2])

        # モーター
        if len(cells) > 6:
            motor_parts = get_br_split(cells[6])
            if len(motor_parts) >= 2:
                boat["motor_no"] = safe_float(motor_parts[0])
                boat["motor_2rate"] = safe_float(motor_parts[1])
            if len(motor_parts) >= 3:
                boat["motor_3rate"] = safe_float(motor_parts[2])

        # ボート
        if len(cells) > 7:
            boat_parts = get_br_split(cells[7])
            if len(boat_parts) >= 2:
                boat["boat_no"] = safe_float(boat_parts[0])
                boat["boat_2rate"] = safe_float(boat_parts[1])
            if len(boat_parts) >= 3:
                boat["boat_3rate"] = safe_float(boat_parts[2])

        # デフォルト値
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

        boats.append(boat)
        i += 1

    return boats if len(boats) == 6 else None


def scrape_beforeinfo(jcd, hd, rno):
    """直前情報をスクレイピング"""
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        return None

    tables = soup.find_all("table")
    if len(tables) < 2:
        return None

    # Table[1]: 展示タイム等（枠順）
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
            parts = get_br_split(cells[-1]) if len(cells) > 3 else []
            ex_time = 6.80
            for p in parts:
                try:
                    v = float(p)
                    if 6.0 <= v <= 7.5:
                        ex_time = v
                        break
                except:
                    pass
            info[waku] = {"exhibition_time": ex_time}

    # Table[2]: スタート展示（コース順 → 進入コース & 展示ST）
    if len(tables) >= 3:
        table2 = tables[2]
        rows2 = table2.find_all("tr")
        course_map = {}  # course_pos → {waku, st}
        for row in rows2:
            cells = row.find_all(["td", "th"])
            for cell in cells:
                text = cell.get_text(strip=True)
                # パターン: "1.19" → コース1, ST 0.19
                # パターン: "4F.01" → コース4, フライング ST -0.01
                match = re.match(r"(\d+)(F?)\.(\d+)", text) if 'text' in dir() else None
                import re
                match = re.match(r"(\d+)(F?)\.(\d+)", text)
                if match:
                    course_pos = int(match.group(1))
                    is_flying = match.group(2) == "F"
                    st_val = float(f"0.{match.group(3)}")
                    if is_flying:
                        st_val = -st_val
                    course_map[course_pos] = st_val

    # 進入コースをHTMLから抽出
    entry_courses = {}
    for i in range(1, 7):
        img_elem = soup.find(class_=f"table1_boatImage1Number{i}")
        if img_elem:
            img_text = img_elem.get_text(strip=True)
            try:
                entry_courses[int(img_text)] = i  # waku → course position
            except:
                pass

    # 情報を統合
    for waku in range(1, 7):
        if waku not in info:
            info[waku] = {"exhibition_time": 6.80}
        if waku in entry_courses:
            info[waku]["entry_course"] = entry_courses[waku]
        else:
            info[waku]["entry_course"] = waku  # デフォルト: 枠順=進入コース
        course_pos = info[waku]["entry_course"]
        if course_pos in course_map:
            info[waku]["exhibition_st"] = course_map[course_pos]
        else:
            info[waku]["exhibition_st"] = 0.15

    return info


def scrape_today_venues(hd):
    """本日の開催場一覧を取得"""
    venues = []
    for jcd, name in PLACE_MAP.items():
        url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd={jcd}&hd={hd}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                tables = soup.find_all("table")
                if len(tables) >= 2:
                    venues.append({"jcd": jcd, "name": name})
            time.sleep(0.3)
        except:
            pass
    return venues


# ============================================================
# 予測エンジン
# ============================================================
def predict_race(jcd, hd, rno, model, boat_features, feature_names, place_stats, temperature):
    """1レースの予測を実行"""
    # スクレイピング
    racelist = scrape_racelist(jcd, hd, rno)
    if racelist is None:
        return None

    beforeinfo = scrape_beforeinfo(jcd, hd, rno)
    if beforeinfo is None:
        # 直前情報がなくても出走表だけで予測可能（精度は落ちる）
        beforeinfo = {w: {"exhibition_time": 6.80, "entry_course": w, "exhibition_st": 0.15} for w in range(1, 7)}

    place_name = PLACE_MAP.get(jcd, "")

    # 各艇の特徴量を構築
    boat_data = []
    for b in racelist:
        w = b["waku"]
        bi = beforeinfo.get(w, {"exhibition_time": 6.80, "entry_course": w, "exhibition_st": 0.15})

        grade_num = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}.get(b.get("grade", "B1"), 2)
        machine_score = (b.get("motor_2rate", 30) + b.get("boat_2rate", 30)) / 2

        # 場の統計
        pw_key = f"{place_name}_{w}"
        pw_win = place_stats.get(pw_key, {}).get("win_rate", 0.15) if place_stats else 0.15
        pw_top3 = place_stats.get(pw_key, {}).get("top3_rate", 0.45) if place_stats else 0.45
        pw_upset = place_stats.get(pw_key, {}).get("upset_rate", 0.10) if place_stats else 0.10

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

        # 名前を保存（表示用）
        features["_name"] = b.get("name", f"枠{w}")
        features["_grade"] = b.get("grade", "B1")

        boat_data.append(features)

    if len(boat_data) != 6:
        return None

    # ペアワイズ特徴量を生成
    diff_suffix = "_diff"
    ratio_suffix = "_ratio"
    pair_features_list = []
    pair_ij = []

    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            pair_feat = {}
            for bf in boat_features:
                if bf.startswith("i_"):
                    col = bf[2:]
                    pair_feat[bf] = boat_data[i].get(col, 0)
                elif bf.startswith("j_"):
                    col = bf[2:]
                    pair_feat[bf] = boat_data[j].get(col, 0)
                else:
                    # diff or ratio
                    base = bf.replace("_diff", "").replace("_ratio", "")
                    if bf.endswith("_diff"):
                        pair_feat[bf] = boat_data[i].get(base, 0) - boat_data[j].get(base, 0)
                    elif bf.endswith("_ratio"):
                        jv = boat_data[j].get(base, 1)
                        if jv == 0:
                            jv = 0.001
                        pair_feat[bf] = boat_data[i].get(base, 0) / jv
            pair_features_list.append(pair_feat)
            pair_ij.append((i, j))

    # 特徴量行列を構築
    X_pred = np.zeros((len(pair_features_list), len(feature_names)))
    for k, pf in enumerate(pair_features_list):
        for fi, fn in enumerate(feature_names):
            X_pred[k, fi] = pf.get(fn, 0)

    # 予測
    preds = model.predict(X_pred)
    scores = np.zeros(6)
    for k, (i, j) in enumerate(pair_ij):
        scores[i] += preds[k]

    ranked = np.argsort(-scores)
    wakus = [bd["waku"] for bd in boat_data]

    # 温度スケーリングで確率を計算
    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    win_probs = exp_scores / exp_scores.sum()

    # 3連単確率を近似計算（Bradley-Terry）
    combos = []
    for p in permutations(range(6), 3):
        i1, i2, i3 = p
        # P(i1が1着) × P(i2が2着|i1除外) × P(i3が3着|i1,i2除外)
        remaining1 = list(range(6))
        p1 = win_probs[i1] / sum(win_probs[r] for r in remaining1)

        remaining2 = [r for r in remaining1 if r != i1]
        p2 = win_probs[i2] / sum(win_probs[r] for r in remaining2)

        remaining3 = [r for r in remaining2 if r != i2]
        p3 = win_probs[i3] / sum(win_probs[r] for r in remaining3)

        combo_prob = p1 * p2 * p3
        combo_str = f"{wakus[i1]}-{wakus[i2]}-{wakus[i3]}"
        combos.append({
            "combo": combo_str,
            "prob": combo_prob,
            "w1": wakus[i1], "w2": wakus[i2], "w3": wakus[i3]
        })

    combos.sort(key=lambda x: -x["prob"])

    # 確信度スコア = Top1確率 - Top2確率
    conf_score = combos[0]["prob"] - combos[1]["prob"] if len(combos) >= 2 else 0

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
        "top1_prob": combos[0]["prob"] if combos else 0,
        "top1_combo": combos[0]["combo"] if combos else "",
    }


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="ボートレース AI 予測", layout="wide", page_icon="🚤")

st.title("🚤 ボートレース AI 予測 v6")
st.caption("LightGBM ペアワイズランキングモデル + Temperature Scaling (T=5.0)")

# モデル読み込み
model, boat_features, feature_names, place_stats, temperature = load_model()
st.sidebar.success(f"✅ モデル読み込み完了 ({len(boat_features)} 特徴量, T={temperature})")

# ============================================================
# サイドバー
# ============================================================
st.sidebar.header("📅 設定")
today = datetime.date.today()
selected_date = st.sidebar.date_input("日付", value=today)
hd = selected_date.strftime("%Y%m%d")

st.sidebar.markdown("---")
mode = st.sidebar.radio("モード選択", ["📊 全場一括予測", "🎯 個別レース予測"])

# ============================================================
# 全場一括予測モード
# ============================================================
if mode == "📊 全場一括予測":
    st.header("📊 全場全レース一括予測")
    st.write(f"日付: **{hd}**")

    if st.button("🔍 一括予測を開始", type="primary"):
        # 開催場を検出
        with st.spinner("開催場を検出中..."):
            venues = scrape_today_venues(hd)

        if not venues:
            st.error("本日の開催場が見つかりません。日付を確認してください。")
        else:
            st.info(f"開催場: {', '.join(v['name'] for v in venues)} ({len(venues)}場)")

            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_races = len(venues) * 12

            for vi, venue in enumerate(venues):
                for rno in range(1, 13):
                    status_text.text(f"予測中... {venue['name']} {rno}R ({vi*12+rno}/{total_races})")
                    result = predict_race(
                        venue["jcd"], hd, rno,
                        model, boat_features, feature_names, place_stats, temperature
                    )
                    if result:
                        all_results.append(result)
                    time.sleep(0.5)  # サーバー負荷軽減
                    progress_bar.progress((vi * 12 + rno) / total_races)

            progress_bar.progress(1.0)
            status_text.text(f"完了！{len(all_results)}レースの予測が完了")

            if all_results:
                # 確信度順にソート
                all_results.sort(key=lambda x: -x["conf_score"])

                # === 今日の勝負レース TOP20 ===
                st.subheader("🏆 今日の勝負レース（確信度順 TOP20）")

                top_data = []
                for i, r in enumerate(all_results[:20]):
                    top_data.append({
                        "順位": i + 1,
                        "場": r["place"],
                        "R": f"{r['rno']}R",
                        "予測 1位": r["top1_combo"],
                        "確率": f"{r['top1_prob']*100:.1f}%",
                        "確信度": f"{r['conf_score']*100:.2f}%",
                    })

                st.dataframe(
                    pd.DataFrame(top_data),
                    use_container_width=True,
                    hide_index=True
                )

                # === 全レース結果 ===
                st.subheader("📋 全レース予測一覧")

                for venue_name in dict.fromkeys(PLACE_MAP.values()):
                    venue_results = [r for r in all_results if r["place"] == venue_name]
                    if not venue_results:
                        continue

                    with st.expander(f"🏟️ {venue_name} ({len(venue_results)}R)"):
                        for r in sorted(venue_results, key=lambda x: x["rno"]):
                            cols = st.columns([1, 3, 2])
                            with cols[0]:
                                st.markdown(f"**{r['rno']}R**")
                            with cols[1]:
                                top3 = r["combos"][:3]
                                for c in top3:
                                    st.text(f"  {c['combo']}  ({c['prob']*100:.1f}%)")
                            with cols[2]:
                                st.text(f"確信度: {r['conf_score']*100:.2f}%")
                            st.divider()

                # セッションに保存
                st.session_state["all_results"] = all_results


# ============================================================
# 個別レース予測モード
# ============================================================
elif mode == "🎯 個別レース予測":
    st.header("🎯 個別レース予測")

    col1, col2 = st.columns(2)
    with col1:
        place_options = {v: k for k, v in PLACE_MAP.items()}
        selected_place = st.selectbox("場を選択", list(place_options.keys()))
        jcd = place_options[selected_place]
    with col2:
        rno = st.selectbox("レース番号", list(range(1, 13)), format_func=lambda x: f"{x}R")

    if st.button("🔮 予測する", type="primary"):
        with st.spinner(f"{selected_place} {rno}R を予測中..."):
            result = predict_race(
                jcd, hd, rno,
                model, boat_features, feature_names, place_stats, temperature
            )

        if result is None:
            st.error("レースデータを取得できませんでした。開催日・場・レース番号を確認してください。")
        else:
            st.success(f"{result['place']} {result['rno']}R 予測完了")

            # === 選手情報 & スコア ===
            st.subheader("🚤 選手情報 & AIスコア")

            boat_df_data = []
            for i, bd in enumerate(result["boat_data"]):
                idx = i  # boat_data のインデックス
                boat_df_data.append({
                    "枠": bd["waku"],
                    "選手名": bd["_name"],
                    "級別": bd["_grade"],
                    "全国勝率": bd.get("national_win_rate", 0),
                    "展示T": bd.get("exhibition_time", 0),
                    "進入C": bd.get("entry_course", bd["waku"]),
                    "AIスコア": f"{result['scores'][idx]:.2f}",
                    "1着確率": f"{result['win_probs'][idx]*100:.1f}%",
                })

            st.dataframe(
                pd.DataFrame(boat_df_data),
                use_container_width=True,
                hide_index=True
            )

            # === 3連単予測 TOP10 ===
            st.subheader("🎯 3連単予測 TOP10")

            combo_data = []
            for i, c in enumerate(result["combos"][:10]):
                combo_data.append({
                    "順位": i + 1,
                    "3連単": c["combo"],
                    "確率": f"{c['prob']*100:.2f}%",
                    "期待倍率": f"{1/c['prob']:.0f}倍" if c["prob"] > 0 else "-",
                })

            st.dataframe(
                pd.DataFrame(combo_data),
                use_container_width=True,
                hide_index=True
            )

            # === 確信度 ===
            st.metric(
                label="確信度スコア",
                value=f"{result['conf_score']*100:.2f}%",
                help="Top1確率 - Top2確率。大きいほど予測に自信あり。"
            )

            # === 補足情報 ===
            with st.expander("📊 詳細スコア分布"):
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.rcParams['font.family'] = 'DejaVu Sans'

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # スコア棒グラフ
                wakus = [bd["waku"] for bd in result["boat_data"]]
                colors = ["#FF4444", "#000000", "#008800", "#FF8800", "#006688", "#00AA00"]
                axes[0].bar(wakus, result["scores"], color=colors[:6])
                axes[0].set_xlabel("Waku")
                axes[0].set_ylabel("Score")
                axes[0].set_title("Pairwise Score")

                # 確率棒グラフ
                axes[1].bar(wakus, result["win_probs"] * 100, color=colors[:6])
                axes[1].set_xlabel("Waku")
                axes[1].set_ylabel("Win Prob (%)")
                axes[1].set_title(f"Win Probability (T={temperature})")

                st.pyplot(fig)


# ============================================================
# フッター
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
**モデル情報**
- LightGBM v6 (pairwise lambdarank)
- 特徴量: 展示タイム, 進入コース, 勝率, モーター等
- 実ST除外 / 選手平均ST使用
- 温度スケーリング校正済み
- バックテスト: Top1 10.0%, ROI 105.7%
""")
st.sidebar.caption("※ 予測は参考情報です。投票は自己責任でお願いします。")
