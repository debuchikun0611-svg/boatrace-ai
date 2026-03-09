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
if not os.path.exists(os.path.join(SCRIPT_DIR, "ensemble_model_0.txt")):
    if os.path.exists("ensemble_model_0.txt"):
        SCRIPT_DIR = "."
    elif os.path.exists("/mount/src/boatrace-ai/ensemble_model_0.txt"):
        SCRIPT_DIR = "/mount/src/boatrace-ai"

MODEL_V6_PATH = os.path.join(SCRIPT_DIR, "pairwise_model_v6.txt")
BOAT_FEATURES_V6_PATH = os.path.join(SCRIPT_DIR, "boat_features_v6.json")
COLUMN_MAPPING_V6_PATH = os.path.join(SCRIPT_DIR, "column_mapping_v6.json")

ENSEMBLE_MODEL_PATHS = [
    os.path.join(SCRIPT_DIR, f"ensemble_model_{i}.txt") for i in range(10)
]
ENSEMBLE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "ensemble_config.json")
LR2_FEATURES_PATH = os.path.join(SCRIPT_DIR, "lambdarank_features_v2.json")

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
    model_v6 = lgb.Booster(model_file=MODEL_V6_PATH)
    with open(BOAT_FEATURES_V6_PATH, "r") as f:
        v6_boat_features = json.load(f)
    with open(COLUMN_MAPPING_V6_PATH, "r") as f:
        v6_col_map = json.load(f)

    ensemble_models = []
    for i, path in enumerate(ENSEMBLE_MODEL_PATHS):
        try:
            if os.path.exists(path):
                m = lgb.Booster(model_file=path)
                ensemble_models.append(m)
            else:
                alt_path = os.path.join(SCRIPT_DIR, f"ensemble_model_{i}.txt")
                if os.path.exists(alt_path):
                    m = lgb.Booster(model_file=alt_path)
                    ensemble_models.append(m)
        except Exception:
            pass

    with open(LR2_FEATURES_PATH, "r") as f:
        lr2_features = json.load(f)

    with open(PLACE_STATS_PATH, "r") as f:
        ps_raw = json.load(f)
    place_w1 = {}
    if "win_rate" in ps_raw:
        for k, v in ps_raw["win_rate"].items():
            place_w1[int(k)] = float(v)

    with open(TEMPERATURE_PATH, "r") as f:
        t_cfg = json.load(f)
    temperature = float(t_cfg.get("temperature", t_cfg.get("T", 5.0)))

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

def get_text_br(tag):
    """brタグを改行に変換してテキスト取得"""
    if tag is None:
        return ""
    for br in tag.find_all("br"):
        br.replace_with("\n")
    return tag.get_text(strip=False)

def parse_td_lines(td):
    """td内のbr区切りテキストを行ごとの数値リストとして返す"""
    raw = get_text_br(td)
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    nums = []
    for line in lines:
        try:
            nums.append(float(line))
        except ValueError:
            pass
    return nums

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
    table = soup.select("div.table1")
    for div in table:
        links = div.select("a[href*='raceindex']")
        for a in links:
            href = a.get("href", "")
            m = re.search(r"jcd=(\d+)", href)
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
    """出走表＋直前情報をスクレイピング"""
    result = {"weather": "", "wind_dir": "", "wind_speed": 0, "wave": 0, "boats": []}
    boats = []

    def parse_fl_st(text):
        """'F0L00.13' → (f_count, l_count, avg_st)"""
        f_count = 0
        l_count = 0
        avg_st = 0.0
        m_f = re.search(r'F(\d+)', text)
        if m_f:
            f_count = int(m_f.group(1))
        m_l = re.search(r'L(\d+)', text)
        if m_l:
            l_count = int(m_l.group(1))
        nums = re.findall(r'(\d+\.\d+)', text)
        if nums:
            avg_st = float(nums[-1])
        return f_count, l_count, avg_st

    # ========== 1. 出走表ページ ==========
    url_racelist = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_num}&jcd={jcd}&hd={date_str}"
    try:
        r1 = requests.get(url_racelist, headers=HEADERS, timeout=15)
        soup1 = BeautifulSoup(r1.content, "html.parser")
        tbody_list = soup1.select("tbody.is-fs12")

        for i, tbody in enumerate(tbody_list[:6]):
            waku = i + 1
            tds = tbody.select("td")
            b = {
                "waku": waku,
                "national_win_rate": 0, "national_2連rate": 0, "national_3連rate": 0,
                "local_win_rate": 0, "local_2連rate": 0, "local_3連rate": 0,
                "motor_2連rate": 0, "motor_3連rate": 0,
                "boat_2連rate": 0, "boat_3連rate": 0,
                "avg_st": 0, "flying_count": 0, "late_count": 0,
                "machine_score": 0,
                "exhibition_time": 0.0, "start_timing": 0.0,
                "entry_course": float(waku), "course_diff": 0.0,
            }

            if len(tds) >= 8:
                # td[3]: F数, L数, 平均ST
                text3 = get_text(tds[3])
                f_count, l_count, avg_st = parse_fl_st(text3)
                b["flying_count"] = f_count
                b["late_count"] = l_count
                b["avg_st"] = avg_st

                # td[4]: 全国勝率, 全国2連率, 全国3連率
                vals4 = parse_td_lines(tds[4])
                if len(vals4) >= 3:
                    b["national_win_rate"] = vals4[0]
                    b["national_2連rate"] = vals4[1]
                    b["national_3連rate"] = vals4[2]
                elif len(vals4) == 2:
                    b["national_win_rate"] = vals4[0]
                    b["national_2連rate"] = vals4[1]
                elif len(vals4) == 1:
                    b["national_win_rate"] = vals4[0]

                # td[5]: 当地勝率, 当地2連率, 当地3連率
                vals5 = parse_td_lines(tds[5])
                if len(vals5) >= 3:
                    b["local_win_rate"] = vals5[0]
                    b["local_2連rate"] = vals5[1]
                    b["local_3連rate"] = vals5[2]
                elif len(vals5) == 2:
                    b["local_win_rate"] = vals5[0]
                    b["local_2連rate"] = vals5[1]
                elif len(vals5) == 1:
                    b["local_win_rate"] = vals5[0]

                # td[6]: モーターNo, モーター2連率, モーター3連率
                vals6 = parse_td_lines(tds[6])
                if len(vals6) >= 3:
                    b["motor_2連rate"] = vals6[1]
                    b["motor_3連rate"] = vals6[2]
                elif len(vals6) >= 2:
                    b["motor_2連rate"] = vals6[1]

                # td[7]: ボートNo, ボート2連率, ボート3連率
                vals7 = parse_td_lines(tds[7])
                if len(vals7) >= 3:
                    b["boat_2連rate"] = vals7[1]
                    b["boat_3連rate"] = vals7[2]
                elif len(vals7) >= 2:
                    b["boat_2連rate"] = vals7[1]

            b["machine_score"] = (b["motor_2連rate"] + b["boat_2連rate"]) / 2
            boats.append(b)

    except Exception as e:
        st.warning(f"出走表取得エラー: {e}")
        for waku in range(1, 7):
            boats.append({
                "waku": waku, "national_win_rate": 0, "national_2連rate": 0, "national_3連rate": 0,
                "local_win_rate": 0, "local_2連rate": 0, "local_3連rate": 0,
                "motor_2連rate": 0, "motor_3連rate": 0, "boat_2連rate": 0, "boat_3連rate": 0,
                "avg_st": 0, "flying_count": 0, "late_count": 0, "machine_score": 0,
                "exhibition_time": 0.0, "start_timing": 0.0,
                "entry_course": float(waku), "course_diff": 0.0,
            })

    # ========== 2. 直前情報ページ ==========
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_num}&jcd={jcd}&hd={date_str}"
    try:
        r2 = requests.get(url_before, headers=HEADERS, timeout=15)
        soup2 = BeautifulSoup(r2.content, "html.parser")

        # --- 天候 ---
        weather_div = soup2.select_one("div.weather1")
        if weather_div:
            wp = weather_div.select_one("p[class*='is-weather']")
            if wp:
                for cls in wp.get("class", []):
                    if cls.startswith("is-weather"):
                        wcode = cls.replace("is-weather", "")
                        wmap = {"1": "晴", "2": "曇り", "3": "雨", "4": "雪", "5": "霧"}
                        result["weather"] = wmap.get(wcode, "")
            if not result["weather"]:
                for span in weather_div.select("span.weather1_bodyUnitLabelTitle"):
                    txt = get_text(span)
                    if txt in ["晴", "曇り", "雨", "雪", "霧"]:
                        result["weather"] = txt

            wind_p = weather_div.select_one("p[class*='is-wind']")
            if wind_p:
                for cls in wind_p.get("class", []):
                    m = re.search(r'is-wind(\d+)', cls)
                    if m:
                        wind_num = m.group(1)
                        wdmap = {
                            "1":"北","2":"北北東","3":"北東","4":"東北東",
                            "5":"東","6":"東南東","7":"南東","8":"南南東",
                            "9":"南","10":"南南西","11":"南西","12":"西南西",
                            "13":"西","14":"西北西","15":"北西","16":"北北西"
                        }
                        result["wind_dir"] = wdmap.get(wind_num, "")

            for unit in weather_div.select("div.weather1_bodyUnit"):
                title_span = unit.select_one("span.weather1_bodyUnitLabelTitle")
                data_span = unit.select_one("span.weather1_bodyUnitLabelData")
                if title_span and data_span:
                    title = get_text(title_span)
                    data = get_text(data_span)
                    num = safe_float(re.sub(r'[^0-9.]', '', data))
                    if "風速" in title:
                        result["wind_speed"] = num
                    elif "波高" in title:
                        result["wave"] = num

        # --- 展示タイム ---
        tbody_list2 = soup2.select("tbody.is-fs12")
        for i, tbody in enumerate(tbody_list2[:6]):
            if i >= len(boats):
                break
            tds = tbody.select("td")
            if len(tds) > 4:
                try:
                    et = float(get_text(tds[4]))
                    if 6.0 <= et <= 8.0:
                        boats[i]["exhibition_time"] = et
                except:
                    pass

        # --- スタート展示（進入コース・STタイミング）取得 ---
        start_table = soup2.select_one("div.table1")
        if start_table:
            course_spans = start_table.select("span")
            course_nums = []
            for sp in course_spans:
                txt = get_text(sp).strip()
                if txt.isdigit() and 1 <= int(txt) <= 6:
                    course_nums.append(int(txt))
            if len(course_nums) >= 6:
                for ci, waku_num in enumerate(course_nums[:6]):
                    for b in boats:
                        if b["waku"] == waku_num:
                            b["entry_course"] = float(ci + 1)
                            b["course_diff"] = b["entry_course"] - b["waku"]
                            break

            st_cells = start_table.select("td")
            st_vals = []
            for td in st_cells:
                txt = get_text(td).strip()
                m = re.match(r'^F\.(\d+)$', txt)
                if m:
                    st_vals.append(-float(f"0.{m.group(1)}"))
                    continue
                m2 = re.match(r'^\.(\d+)$', txt)
                if m2:
                    st_vals.append(float(f"0.{m2.group(1)}"))
                    continue
                m3 = re.match(r'^0\.(\d+)$', txt)
                if m3:
                    st_vals.append(float(txt))
            if len(st_vals) >= 6 and len(course_nums) >= 6:
                for ci, waku_num in enumerate(course_nums[:6]):
                    if ci < len(st_vals):
                        for b in boats:
                            if b["waku"] == waku_num:
                                b["start_timing"] = st_vals[ci]
                                break

    except Exception as e:
        st.warning(f"直前情報取得エラー: {e}")

    result["boats"] = boats
    return result

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

        weather_val = WEATHER_MAP.get(str(boats_data.get("weather", "")), 0)
        wind_dir_str = str(boats_data.get("wind_dir", ""))
        wind_dir_val = WIND_DIR_MAP.get(wind_dir_str, 0)
        wind_speed = safe_float(boats_data.get("wind_speed", 0))
        wind_eff = WIND_EFFECT.get(wind_dir_val, 0) * wind_speed

        pw1 = place_w1.get(jcd, 55.0)

        for b in boats:
            b["jcd"] = jcd
            b["grade"] = 0
            b["weather"] = weather_val
            b["wind_dir"] = wind_dir_val
            b["wind_speed"] = wind_speed
            b["wind_effect"] = wind_eff
            b["place_w1_winrate"] = pw1
            b["waku_win_hist"] = WAKU_WIN_HIST.get(b.get("waku", 1), 5.0)

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

        # v6スコア
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

        # 120通り候補
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

        votes = {}
        for mi in range(len(ensemble_models)):
            best_idx = int(np.argmax(all_scores[mi]))
            perm = PERMS_120[best_idx]
            combo = f"{perm[0]}-{perm[1]}-{perm[2]}"
            votes[combo] = votes.get(combo, 0) + 1

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

(model_v6, v6_boat_features, v6_col_map,
 ensemble_models, lr2_features,
 place_w1, temperature, ens_config) = load_models()

n_models = len(ensemble_models)
st.sidebar.success(
    f"モデル読込完了\n"
    f"- アンサンブル: {n_models}モデル\n"
    f"- 特徴量数: {len(lr2_features)}\n"
    f"- バックテスト ROI: 135.9%"
)

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
                            res = predict_race(
                                data, jcd, model_v6, v6_boat_features,
                                ensemble_models, lr2_features, place_w1, temperature
                            )
                            if res:
                                top1 = res["top_combos"][0]
                                label, color = confidence_label(
                                    res["confidence"],
                                    res["top1_votes"],
                                    res["total_models"]
                                )
                                all_results.append({
                                    "場": venue_name,
                                    "R": rno,
                                    "予測": top1["combo"],
                                    "票数": f"{res['top1_votes']}/{res['total_models']}",
                                    "確信度": res["confidence"],
                                    "信頼": label,
                                    "jcd": jcd
                                })
                    except Exception:
                        pass

                    done += 1
                    progress.progress(done / total_races)

            progress.empty()
            status.empty()

            if all_results:
                df_results = pd.DataFrame(all_results)

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

                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                col_w1.metric("天候", data.get("weather", "-"))
                col_w2.metric("風向", data.get("wind_dir", "-"))
                col_w3.metric("風速", f"{data.get('wind_speed', 0)}m")
                col_w4.metric("波高", f"{data.get('wave', 0)}cm")

                st.divider()

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

                # 出走情報
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

                # Top-10
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

                # 多数決
                st.subheader("🗳️ モデル投票分布")
                vote_sorted = sorted(result["votes"].items(), key=lambda x: -x[1])
                vote_data = []
                for combo, v in vote_sorted[:10]:
                    vote_data.append({"組合せ": combo, "票数": v,
                                      "割合": f"{v/result['total_models']*100:.0f}%"})
                st.dataframe(pd.DataFrame(vote_data),
                             use_container_width=True, hide_index=True)

                st.divider()

                # ペアワイズマトリクス
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

                # ランキング
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
