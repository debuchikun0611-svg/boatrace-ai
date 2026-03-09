import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb
import json, os, re
import requests
from bs4 import BeautifulSoup
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ============================================================
# パス設定
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Streamlit Cloud対応: ファイルが見つからない場合のフォールバック
if not os.path.exists(os.path.join(SCRIPT_DIR, "ensemble_model_0.txt")):
    # カレントディレクトリを試す
    if os.path.exists("ensemble_model_0.txt"):
        SCRIPT_DIR = "."
    elif os.path.exists("/mount/src/boatrace-ai/ensemble_model_0.txt"):
        SCRIPT_DIR = "/mount/src/boatrace-ai"

# v6 ペアワイズ（マトリクス表示用）
MODEL_V6_PATH = os.path.join(SCRIPT_DIR, "pairwise_model_v6.txt")
BOAT_FEATURES_V6_PATH = os.path.join(SCRIPT_DIR, "boat_features_v6.json")
COLUMN_MAPPING_V6_PATH = os.path.join(SCRIPT_DIR, "column_mapping_v6.json")

# アンサンブルモデル（10体）
ENSEMBLE_MODEL_PATHS = [
    os.path.join(SCRIPT_DIR, f"ensemble_model_{i}.txt") for i in range(10)
]
ENSEMBLE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "ensemble_config.json")
LR2_FEATURES_PATH = os.path.join(SCRIPT_DIR, "lambdarank_features_v2.json")

# 共通設定
PLACE_STATS_PATH = os.path.join(SCRIPT_DIR, "place_stats_v4.json")
TEMPERATURE_PATH = os.path.join(SCRIPT_DIR, "temperature_v6.json")

# ============================================================
# 定数
# ============================================================
PLACE_MAP = {
    "01":"桐生","02":"戸田","03":"江戸川","04":"平和島","05":"多摩川",
    "06":"浜名湖","07":"蒲郡","08":"常滑","09":"津","10":"三国",
    "11":"びわこ","12":"住之江","13":"尼崎","14":"鳴門","15":"丸亀",
    "16":"児島","17":"宮島","18":"徳山","19":"下関","20":"若松",
    "21":"芦屋","22":"福岡","23":"唐津","24":"大村"
}
PLACE_NAME_TO_CODE = {v: int(k) for k, v in PLACE_MAP.items()}
WAKU_WIN_HIST = {1:55.4, 2:14.8, 3:12.1, 4:10.4, 5:5.5, 6:1.8}
PERMS_120 = list(permutations(range(1,7), 3))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

WEATHER_MAP = {"晴":0,"曇り":1,"曇":1,"雨":2,"雪":3,"霧":4}
WIND_DIR_MAP = {
    "北":0,"北北東":1,"北東":2,"東北東":3,"東":4,"東南東":5,
    "南東":6,"南南東":7,"南":8,"南南西":9,"南西":10,"西南西":11,
    "西":12,"西北西":13,"北西":14,"北北西":15,"無風":0
}
WIND_EFFECT = {
    0:0.5,1:0.3,2:0.0,3:-0.3,4:-0.5,5:-0.3,6:0.0,7:0.3,
    8:0.5,9:0.3,10:0.0,11:-0.3,12:-0.5,13:-0.3,14:0.0,15:0.3
}

# ============================================================
# モデル読み込み
# ============================================================
@st.cache_resource
def load_models():
    # v6 ペアワイズ
    model_v6 = lgb.Booster(model_file=MODEL_V6_PATH)
    with open(BOAT_FEATURES_V6_PATH, "r") as f:
        v6_boat_features = json.load(f)
    with open(COLUMN_MAPPING_V6_PATH, "r") as f:
        v6_col_map = json.load(f)

    # アンサンブル10モデル
    ensemble_models = []
    for i, path in enumerate(ENSEMBLE_MODEL_PATHS):
        try:
            if os.path.exists(path):
                m = lgb.Booster(model_file=path)
                ensemble_models.append(m)
            else:
                # ファイルが見つからない場合、同じディレクトリを検索
                alt_path = os.path.join(SCRIPT_DIR, f"ensemble_model_{i}.txt")
                if os.path.exists(alt_path):
                    m = lgb.Booster(model_file=alt_path)
                    ensemble_models.append(m)
        except Exception as e:
            pass  # 壊れたモデルはスキップ
    
    with open(LR2_FEATURES_PATH, "r") as f:
        lr2_features = json.load(f)

    # 設定
    with open(PLACE_STATS_PATH, "r") as f:
        ps_raw = json.load(f)
    place_w1 = {}
    if "win_rate" in ps_raw:
        for k, v in ps_raw["win_rate"].items():
            place_w1[int(k)] = float(v)

    with open(TEMPERATURE_PATH, "r") as f:
        t_cfg = json.load(f)
    temperature = float(t_cfg.get("temperature", t_cfg.get("T", 5.0)))

    # アンサンブル設定
    ens_config = {}
    if os.path.exists(ENSEMBLE_CONFIG_PATH):
        with open(ENSEMBLE_CONFIG_PATH, "r") as f:
            ens_config = json.load(f)

    return (model_v6, v6_boat_features, v6_col_map,
            ensemble_models, lr2_features,
            place_w1, temperature, ens_config)

# ============================================================
# スクレイピング
# ============================================================
def get_text(el):
    return el.get_text(strip=True) if el else ""

def safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except:
        return default

def scrape_racelist(date_str):
    """日付のレース一覧を取得"""
    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={date_str}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
    except:
        return {}

    venues = {}
    # 開催場を検出
    table = soup.select("div.table1")
    for div in table:
        links = div.select("a[href*='raceindex']")
        for a in links:
            href = a.get("href", "")
            m = re.search(r"jcd=(\\d+)", href)
            if m:
                jcd = m.group(1).zfill(2)
                name = PLACE_MAP.get(jcd, jcd)
                venues[jcd] = name

    if not venues:
        for jcd, name in PLACE_MAP.items():
            try:
                check_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd={jcd}&hd={date_str}"
                cr = requests.get(check_url, headers=HEADERS, timeout=5)
                if cr.status_code == 200 and "出走表" in cr.text:
                    venues[jcd] = name
            except:
                pass

    return venues

def scrape_beforeinfo(jcd, race_num, date_str):
    """直前情報をスクレイピング"""
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_num}&jcd={jcd}&hd={date_str}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
    except:
        return None

    result = {
        "weather": "", "wind_dir": "", "wind_speed": 0, "wave": 0,
        "boats": []
    }

    # 天候情報
    weather_div = soup.select_one("div.weather1")
    if weather_div:
        w_body = weather_div.select("div.weather1_body")
        for wb in w_body:
            label = get_text(wb.select_one("div.weather1_bodyUnitLabelTitle"))
            value = get_text(wb.select_one("div.weather1_bodyUnitLabelData"))
            if not value:
                img = wb.select_one("img")
                if img:
                    alt = img.get("alt", "")
                    value = alt
            if "天候" in label:
                result["weather"] = value
            elif "風向" in label:
                result["wind_dir"] = value
            elif "風速" in label:
                result["wind_speed"] = safe_float(re.sub(r"[^0-9.]", "", value))
            elif "波高" in label:
                result["wave"] = safe_float(re.sub(r"[^0-9.]", "", value))

    # 出走表
    racelist_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_num}&jcd={jcd}&hd={date_str}"
    try:
        r2 = requests.get(racelist_url, headers=HEADERS, timeout=10)
        soup2 = BeautifulSoup(r2.content, "html.parser")
    except:
        soup2 = None

    boats = []
    rows = soup.select("tbody.is-fs12")
    for i, row in enumerate(rows[:6]):
        waku = i + 1
        tds = row.select("td")

        boat = {"waku": waku}

        # 展示タイム
        et_el = row.select_one("td.is-boatColor1") or (tds[4] if len(tds) > 4 else None)
        boat["exhibition_time"] = safe_float(get_text(et_el)) if et_el else 0.0

        # スタート展示
        spans = row.select("span")
        for sp in spans:
            txt = get_text(sp)
            if txt.startswith("F") or txt.startswith(".") or txt.startswith("L"):
                boat["start_timing"] = safe_float(txt.replace("F", "-").replace("L", "0."))
                break
        if "start_timing" not in boat:
            boat["start_timing"] = 0.0

        boats.append(boat)

    # 進入コース
    course_div = soup.select_one("div.table1[class*='startCourse']")
    if not course_div:
        course_div = soup.select_one("div.startCourse")
    
    entry_courses = {}
    if course_div:
        course_spans = course_div.select("span")
        for ci, sp in enumerate(course_spans[:6]):
            txt = get_text(sp)
            w = safe_float(txt)
            if 1 <= w <= 6:
                entry_courses[ci+1] = int(w)
    
    # 出走表から選手情報
    racer_info = {}
    if soup2:
        r_rows = soup2.select("tbody.is-fs12")
        for i, rr in enumerate(r_rows[:6]):
            w = i + 1
            info = {}
            tds = rr.select("td")
            
            # 各種勝率
            rate_cells = rr.select("td")
            nums = []
            for td in rate_cells:
                txt = get_text(td)
                try:
                    nums.append(float(txt))
                except:
                    pass
            
            racer_info[w] = nums

    # ボート情報組み立て
    for b in boats:
        w = b["waku"]
        if w in entry_courses:
            b["entry_course"] = entry_courses[w]
        else:
            b["entry_course"] = w
        b["course_diff"] = b["entry_course"] - w

    result["boats"] = boats
    result["racer_info"] = racer_info

    # 出走表の詳細データ取得
    detail_boats = scrape_racelist_detail(jcd, race_num, date_str)
    if detail_boats:
        for db in detail_boats:
            w = db["waku"]
            for b in boats:
                if b["waku"] == w:
                    b.update(db)
                    break

    result["boats"] = boats
    return result

def scrape_racelist_detail(jcd, race_num, date_str):
    """出走表から選手成績を取得"""
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_num}&jcd={jcd}&hd={date_str}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
    except:
        return None

    boats = []
    rows = soup.select("tbody.is-fs12")
    for i, row in enumerate(rows[:6]):
        waku = i + 1
        b = {"waku": waku}

        tds = row.select("td")
        nums = []
        for td in tds:
            txt = get_text(td)
            txt_clean = txt.replace("R", "").replace("F", "").replace("L", "").strip()
            try:
                nums.append(float(txt_clean))
            except:
                pass

        # 一般的な出走表の列順:
        # 枠, 登番, 選手名, 年齢, 支部, 体重, 級別, 全国勝率, 全国2連率, 当地勝率, 当地2連率,
        # モーターNo, モーター2連率, ボートNo, ボート2連率
        if len(nums) >= 10:
            b["national_win_rate"] = nums[0] if len(nums) > 0 else 0
            b["national_2連rate"] = nums[1] if len(nums) > 1 else 0
            b["local_win_rate"] = nums[2] if len(nums) > 2 else 0
            b["local_2連rate"] = nums[3] if len(nums) > 3 else 0
            b["motor_2連rate"] = nums[5] if len(nums) > 5 else 0
            b["boat_2連rate"] = nums[7] if len(nums) > 7 else 0
        elif len(nums) >= 6:
            b["national_win_rate"] = nums[0]
            b["national_2連rate"] = nums[1]
            b["local_win_rate"] = nums[2]
            b["local_2連rate"] = nums[3]
            b["motor_2連rate"] = nums[4]
            b["boat_2連rate"] = nums[5]
        else:
            b["national_win_rate"] = 0
            b["national_2連rate"] = 0
            b["local_win_rate"] = 0
            b["local_2連rate"] = 0
            b["motor_2連rate"] = 0
            b["boat_2連rate"] = 0

        # グレード判定
        grade_el = row.select_one("td.is-gradeColor")
        grade_txt = get_text(grade_el) if grade_el else ""
        grade_map_local = {"A1":3, "A2":2, "B1":1, "B2":0}
        b["racer_grade"] = grade_txt

        b["national_3連rate"] = 0.0
        b["local_3連rate"] = 0.0
        b["motor_3連rate"] = 0.0
        b["boat_3連rate"] = 0.0
        b["flying_count"] = 0.0
        b["late_count"] = 0.0
        b["avg_st"] = 0.0
        b["machine_score"] = (b.get("motor_2連rate",0) + b.get("boat_2連rate",0)) / 2

        boats.append(b)

    return boats

def check_race_exists(jcd, race_num, date_str):
    """レースが存在するか確認"""
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_num}&jcd={jcd}&hd={date_str}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=5)
        return r.status_code == 200 and "出走表" in r.text
    except:
        return False

# ============================================================
# 予測
# ============================================================
def predict_race(boats_data, jcd_code, model_v6, v6_boat_features,
                 ensemble_models, lr2_features, place_w1, temperature):
    """10モデルアンサンブル予測"""
    try:
        jcd = int(jcd_code)
        n_feat = len(lr2_features)
        boats = boats_data.get("boats", [])

        if len(boats) < 6:
            return None

        # 天候特徴量
        weather_val = WEATHER_MAP.get(str(boats_data.get("weather", "")), 0)
        wind_dir_str = str(boats_data.get("wind_dir", ""))
        wind_dir_val = WIND_DIR_MAP.get(wind_dir_str, 0)
        wind_speed = safe_float(boats_data.get("wind_speed", 0))
        wind_eff = WIND_EFFECT.get(wind_dir_val, 0) * wind_speed

        pw1 = place_w1.get(jcd, 55.0)

        # 各艇の基本特徴量を確実にセット
        for b in boats:
            b["jcd"] = jcd
            b["grade"] = 0
            b["weather"] = weather_val
            b["wind_dir"] = wind_dir_val
            b["wind_speed"] = wind_speed
            b["wind_effect"] = wind_eff
            b["place_w1_winrate"] = pw1
            b["waku_win_hist"] = WAKU_WIN_HIST.get(b.get("waku", 1), 5.0)

            # 数値フィールドを確実にfloatに
            for key in ["national_win_rate", "national_2連rate", "national_3連rate",
                        "local_win_rate", "local_2連rate", "local_3連rate",
                        "motor_2連rate", "motor_3連rate", "boat_2連rate", "boat_3連rate",
                        "exhibition_time", "start_timing", "avg_st",
                        "entry_course", "course_diff", "flying_count", "late_count"]:
                b[key] = safe_float(b.get(key, 0))

            if b.get("entry_course", 0) == 0:
                b["entry_course"] = float(b.get("waku", 1))
            b["course_diff"] = b["entry_course"] - b.get("waku", 1)
            b["machine_score"] = (b["motor_2連rate"] + b["boat_2連rate"]) / 2

        # 相対特徴量
        stats_keys = ["national_win_rate", "national_2連rate",
                      "local_win_rate", "motor_2連rate",
                      "exhibition_time", "avg_st", "machine_score"]
        vals = {k: np.array([b[k] for b in boats]) for k in stats_keys}
        means = {k: float(np.mean(v)) for k, v in vals.items()}
        stds = {k: max(float(np.std(v)), 1e-6) for k, v in vals.items()}

        for i, b in enumerate(boats):
            for k in stats_keys:
                b[f"{k}_z"] = (b[k] - means[k]) / stds[k]
                b[f"{k}_rank"] = float(np.argsort(np.argsort(-vals[k]))[i] + 1)
            b["diff_w1_national_win_rate"] = b["national_win_rate"] - boats[0]["national_win_rate"]
            b["diff_w1_exhibition_time"] = b["exhibition_time"] - boats[0]["exhibition_time"]
            b["diff_w1_motor_2連rate"] = b["motor_2連rate"] - boats[0]["motor_2連rate"]
            b["diff_w1_machine_score"] = b["machine_score"] - boats[0]["machine_score"]
            b["et_diff_mean"] = b["exhibition_time"] - means["exhibition_time"]
            b["et_diff_best"] = b["exhibition_time"] - min(vals["exhibition_time"])

        # v6スコア計算
        v6_n_feat = model_v6.num_feature()
        raw_feats = []
        for b in boats:
            vec = [safe_float(b.get(f, 0)) for f in v6_boat_features]
            raw_feats.append(vec)

        pair_rows = []
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                row = list(raw_feats[i]) + list(raw_feats[j])
                row += [raw_feats[i][k] - raw_feats[j][k] for k in range(len(v6_boat_features))]
                row += [i + 1, j + 1, (i + 1) * (j + 1), abs(i - j)]
                pair_rows.append(row)

        X_pair = np.array(pair_rows, dtype=np.float32)
        if X_pair.shape[1] < v6_n_feat:
            X_pair = np.hstack([X_pair, np.zeros((X_pair.shape[0], v6_n_feat - X_pair.shape[1]), dtype=np.float32)])
        elif X_pair.shape[1] > v6_n_feat:
            X_pair = X_pair[:, :v6_n_feat]

        raw_pred = model_v6.predict(X_pair)
        sig = 1.0 / (1.0 + np.exp(-raw_pred / temperature))

        pw_matrix = np.zeros((6, 6))
        win_scores = np.zeros(6)
        idx = 0
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                pw_matrix[i][j] = sig[idx]
                win_scores[i] += sig[idx]
                idx += 1
        total = win_scores.sum()
        if total > 0:
            win_scores /= total

        for i, b in enumerate(boats):
            b["v6_score"] = float(win_scores[i])

        # 120通り候補の特徴量構築
        boat_keys = [
            "waku", "national_win_rate", "national_2連rate", "national_3連rate",
            "local_win_rate", "motor_2連rate", "motor_3連rate", "exhibition_time",
            "start_timing", "avg_st", "entry_course", "course_diff", "machine_score",
            "flying_count", "late_count", "waku_win_hist",
            "national_win_rate_z", "national_win_rate_rank",
            "national_2連rate_z",
            "motor_2連rate_z", "motor_2連rate_z",
            "exhibition_time_z", "exhibition_time_rank",
            "machine_score_z", "machine_score_rank",
            "diff_w1_national_win_rate", "diff_w1_exhibition_time",
            "diff_w1_motor_2連rate", "diff_w1_machine_score",
            "et_diff_mean", "et_diff_best",
            "v6_score"
        ]

        X_candidates = []
        for perm in PERMS_120:
            i1, i2, i3 = perm[0] - 1, perm[1] - 1, perm[2] - 1
            b1, b2, b3 = boats[i1], boats[i2], boats[i3]

            vec = []
            for b in [b1, b2, b3]:
                for k in boat_keys:
                    vec.append(safe_float(b.get(k, 0)))

            diff_keys = ["national_win_rate", "exhibition_time", "motor_2連rate",
                         "machine_score", "v6_score", "entry_course"]
            for k in diff_keys:
                vec.append(safe_float(b1.get(k, 0)) - safe_float(b2.get(k, 0)))
                vec.append(safe_float(b1.get(k, 0)) - safe_float(b3.get(k, 0)))
                vec.append(safe_float(b2.get(k, 0)) - safe_float(b3.get(k, 0)))

            trio_keys = ["national_win_rate", "motor_2連rate", "exhibition_time",
                         "machine_score", "v6_score"]
            for k in trio_keys:
                vs = [safe_float(b1.get(k, 0)), safe_float(b2.get(k, 0)), safe_float(b3.get(k, 0))]
                vec += [float(np.mean(vs)), float(np.std(vs)), max(vs) - min(vs), min(vs)]

            vec.append(perm[0] * 100 + perm[1] * 10 + perm[2])
            vec.append(perm[0])
            vec.append(perm[1])
            vec.append(perm[2])
            vec.append(perm[0] * perm[1] * perm[2])

            vec += [jcd, 0, weather_val, wind_speed, wind_eff, pw1]
            vec += [b1["v6_score"] + b2["v6_score"] + b3["v6_score"],
                    b1["v6_score"] - b2["v6_score"],
                    max(b1["v6_score"], b2["v6_score"], b3["v6_score"]),
                    1.0 if boats[i1]["v6_score"] == max(b["v6_score"] for b in boats) else 0.0]

            if len(vec) < n_feat:
                vec.extend([0.0] * (n_feat - len(vec)))
            elif len(vec) > n_feat:
                vec = vec[:n_feat]

            X_candidates.append(vec)

        X = np.array(X_candidates, dtype=np.float32)

        # 10モデルで予測
        all_scores = []
        for model in ensemble_models:
            pred = model.predict(X)
            all_scores.append(pred)

        if len(all_scores) == 0:
            return None

        all_scores = np.array(all_scores)
        ensemble_score = all_scores.mean(axis=0)

        # 多数決
        votes = {}
        for mi in range(len(ensemble_models)):
            best_idx = int(np.argmax(all_scores[mi]))
            perm = PERMS_120[best_idx]
            combo = f"{perm[0]}-{perm[1]}-{perm[2]}"
            votes[combo] = votes.get(combo, 0) + 1

        # ランキング
        ranking = np.argsort(-ensemble_score)

        top_combos = []
        for rank_i in range(min(20, len(ensemble_score))):
            idx = int(ranking[rank_i])
            perm = PERMS_120[idx]
            combo = f"{perm[0]}-{perm[1]}-{perm[2]}"
            score = float(ensemble_score[idx])
            n_votes = votes.get(combo, 0)
            top_combos.append({
                "rank": rank_i + 1,
                "combo": combo,
                "score": score,
                "votes": n_votes
            })

        if len(ensemble_score) >= 2:
            confidence = float(ensemble_score[int(ranking[0])] - ensemble_score[int(ranking[1])])
        else:
            confidence = 0.0

        top1_votes = top_combos[0]["votes"] if top_combos else 0

        # 勝率
        boat_win_scores = np.zeros(6)
        for pi, perm in enumerate(PERMS_120):
            if pi < len(ensemble_score):
                boat_win_scores[perm[0] - 1] += ensemble_score[pi]
        exp_s = np.exp(boat_win_scores - boat_win_scores.max())
        win_probs = exp_s / exp_s.sum()

        return {
            "top_combos": top_combos,
            "confidence": confidence,
            "top1_votes": top1_votes,
            "total_models": len(ensemble_models),
            "win_probs": win_probs,
            "pw_matrix": pw_matrix,
            "votes": votes,
            "boats": boats
        }

    except Exception as e:
        st.error(f"予測エラー: {str(e)}")
        return None


# ============================================================
# 表示ヘルパー
# ============================================================
def format_st(val):
    try:
        v = float(val)
        if v == 0:
            return "-"
        return f"{v:.2f}"
    except:
        return str(val)

def confidence_label(conf, votes, total):
    """確信度と票数から信頼レベルを判定"""
    vote_ratio = votes / total if total > 0 else 0
    if vote_ratio >= 0.8 and conf >= 0.10:
        return "🔴 超高信頼", "red"
    elif vote_ratio >= 0.6 and conf >= 0.05:
        return "🟠 高信頼", "orange"
    elif vote_ratio >= 0.4:
        return "🟡 中信頼", "goldenrod"
    else:
        return "⚪ 低信頼", "gray"

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="ボートレース AI v8", layout="wide")
st.title("🚤 ボートレース AI 予測 v8")
st.caption("10モデル アンサンブル ｜ LambdaRank × 多数決 ｜ ペアワイズv6 マトリクス表示")

# モデル読み込み
(model_v6, v6_boat_features, v6_col_map,
 ensemble_models, lr2_features,
 place_w1, temperature, ens_config) = load_models()

n_models = len(ensemble_models)
st.sidebar.success(
    f"モデル読込完了\\n"
    f"- アンサンブル: {n_models}モデル\\n"
    f"- 特徴量数: {len(lr2_features)}\\n"
    f"- バックテスト ROI: 135.9%"
)

# サイドバー設定
import datetime
st.sidebar.header("設定")
today = datetime.date.today()
sel_date = st.sidebar.date_input("日付", today)
date_str = sel_date.strftime("%Y%m%d")

mode = st.sidebar.radio("モード", ["全場一括予測", "個別レース予測"])

# ============================================================
# 全場一括予測
# ============================================================
if mode == "全場一括予測":
    if st.sidebar.button("🚀 全場予測開始"):
        with st.spinner("開催場を検索中..."):
            venues = scrape_racelist(date_str)

        if not venues:
            st.error("開催場が見つかりません")
        else:
            st.info(f"開催場: {', '.join(venues.values())} ({len(venues)}場)")

            all_results = []
            progress = st.progress(0)
            status = st.empty()

            total_races = len(venues) * 12
            done = 0

            for jcd, venue_name in venues.items():
                for rno in range(1, 13):
                    status.text(f"{venue_name} {rno}R を予測中...")
                    try:
                        data = scrape_beforeinfo(jcd, rno, date_str)
                        if data and len(data.get("boats", [])) >= 6:
                            result = predict_race(
                                data, jcd, model_v6, v6_boat_features,
                                ensemble_models, lr2_features, place_w1, temperature
                            )
                            if result:
                                top1 = result["top_combos"][0]
                                label, color = confidence_label(
                                    result["confidence"], 
                                    result["top1_votes"],
                                    result["total_models"]
                                )
                                all_results.append({
                                    "場": venue_name,
                                    "R": rno,
                                    "予測": top1["combo"],
                                    "票数": f"{result['top1_votes']}/{result['total_models']}",
                                    "確信度": result["confidence"],
                                    "信頼": label,
                                    "jcd": jcd
                                })
                    except Exception as e:
                        pass

                    done += 1
                    progress.progress(done / total_races)

            progress.empty()
            status.empty()

            if all_results:
                df_results = pd.DataFrame(all_results)

                # 高信頼レース（8票以上）
                high_conf = df_results[df_results["票数"].apply(
                    lambda x: int(x.split("/")[0]) >= 8)]

                if len(high_conf) > 0:
                    st.subheader(f"🔴 高信頼レース（8票以上）: {len(high_conf)}件")
                    st.dataframe(
                        high_conf[["場","R","予測","票数","信頼"]].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True
                    )

                st.subheader(f"📋 全レース結果 ({len(df_results)}件)")
                st.dataframe(
                    df_results[["場","R","予測","票数","確信度","信頼"]].sort_values(
                        "確信度", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("予測可能なレースがありません")

# ============================================================
# 個別レース予測
# ============================================================
else:
    col1, col2 = st.sidebar.columns(2)
    sel_venue = col1.selectbox("場", list(PLACE_MAP.values()))
    sel_race = col2.selectbox("レース", list(range(1, 13)))

    jcd = str(PLACE_NAME_TO_CODE.get(sel_venue, 1)).zfill(2)

    if st.sidebar.button("🔍 予測実行"):
        with st.spinner(f"{sel_venue} {sel_race}R を予測中..."):
            data = scrape_beforeinfo(jcd, sel_race, date_str)

        if not data or len(data.get("boats", [])) < 6:
            st.error("レースデータを取得できません")
        else:
            result = predict_race(
                data, jcd, model_v6, v6_boat_features,
                ensemble_models, lr2_features, place_w1, temperature
            )

            if result is None:
                st.error("予測に失敗しました")
            else:
                st.header(f"🏁 {sel_venue} {sel_race}R")

                # 天候情報
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                col_w1.metric("天候", data.get("weather", "-"))
                col_w2.metric("風向", data.get("wind_dir", "-"))
                col_w3.metric("風速", f"{data.get('wind_speed', 0)}m")
                col_w4.metric("波高", f"{data.get('wave', 0)}cm")

                st.divider()

                # 信頼度メトリクス
                label, color = confidence_label(
                    result["confidence"],
                    result["top1_votes"],
                    result["total_models"]
                )
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("🎯 予測1位", result["top_combos"][0]["combo"])
                col_m2.metric("📊 モデル合意",
                              f"{result['top1_votes']}/{result['total_models']}票")
                col_m3.metric("確信度", f"{result['confidence']:.4f}")

                st.markdown(f"**信頼レベル: <span style='color:{color};font-size:1.3em'>"
                            f"{label}</span>**", unsafe_allow_html=True)

                st.divider()

                # 選手情報テーブル
                st.subheader("🚤 出走情報")
                boat_df = pd.DataFrame([{
                    "枠": b["waku"],
                    "全国勝率": b.get("national_win_rate", 0),
                    "全国2連率": b.get("national_2連rate", 0),
                    "当地勝率": b.get("local_win_rate", 0),
                    "モーター2連率": b.get("motor_2連rate", 0),
                    "ボート2連率": b.get("boat_2連rate", 0),
                    "展示T": b.get("exhibition_time", 0),
                    "進入C": int(b.get("entry_course", b["waku"])),
                    "ST": format_st(b.get("start_timing", 0)),
                    "v6勝率": f"{result['win_probs'][b['waku']-1]*100:.1f}%"
                } for b in result["boats"]])
                st.dataframe(boat_df, use_container_width=True, hide_index=True)

                st.divider()

                # Top-10 三連単予測
                st.subheader("🏆 三連単 Top-10 予測")
                top10_data = []
                for tc in result["top_combos"][:10]:
                    vote_bar = "🟢" * tc["votes"] + "⚫" * (result["total_models"] - tc["votes"])
                    top10_data.append({
                        "順位": tc["rank"],
                        "組合せ": tc["combo"],
                        "スコア": f"{tc['score']:.4f}",
                        "票数": f"{tc['votes']}/{result['total_models']}",
                        "投票": vote_bar
                    })
                st.dataframe(pd.DataFrame(top10_data),
                             use_container_width=True, hide_index=True)

                st.divider()

                # 多数決詳細
                st.subheader("🗳️ モデル投票分布")
                vote_sorted = sorted(result["votes"].items(), key=lambda x: -x[1])
                vote_data = []
                for combo, v in vote_sorted[:10]:
                    vote_data.append({"組合せ": combo, "票数": v,
                                      "割合": f"{v/result['total_models']*100:.0f}%"})
                st.dataframe(pd.DataFrame(vote_data),
                             use_container_width=True, hide_index=True)

                st.divider()

                # ペアワイズ勝率マトリクス
                st.subheader("📊 ペアワイズ勝率マトリクス")
                pw = result["pw_matrix"]
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(pw, cmap="RdYlGn", vmin=0.3, vmax=0.7)
                ax.set_xticks(range(6))
                ax.set_yticks(range(6))
                ax.set_xticklabels([f"Boat {i+1}" for i in range(6)])
                ax.set_yticklabels([f"Boat {i+1}" for i in range(6)])
                for i in range(6):
                    for j in range(6):
                        if i != j:
                            ax.text(j, i, f"{pw[i][j]:.2f}",
                                    ha="center", va="center", fontsize=9)
                        else:
                            ax.text(j, i, "-", ha="center", va="center", fontsize=9)
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title("Win Probability (row beats column)")
                st.pyplot(fig)
                plt.close()

                # 平均勝率ランキング
                avg_pw = pw.sum(axis=1) / 5
                rank_order = np.argsort(-avg_pw)
                st.subheader("📈 総合力ランキング")
                rank_data = []
                for ri, bi in enumerate(rank_order):
                    rank_data.append({
                        "順位": ri+1,
                        "艇": bi+1,
                        "平均勝率": f"{avg_pw[bi]:.3f}",
                        "1着確率": f"{result['win_probs'][bi]*100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(rank_data),
                             use_container_width=True, hide_index=True)

# フッター
st.divider()
st.caption(
    f"🤖 Boatrace AI v8 | {n_models}モデル アンサンブル LambdaRank | "
    f"{len(lr2_features)}特徴量 | "
    f"Back-test: Top-1 11.24%, ROI 135.9% | "
    f"高信頼(8票+): Top-1 15.0%, ROI 147.8%"
)
