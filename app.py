import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import re
from bs4 import BeautifulSoup
from itertools import permutations, combinations
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競艇AI予想 v11", page_icon="🚤", layout="wide")

# =============================================================================
# 定数
# =============================================================================
PLACE_CODES = {
    '桐生': '01', '戸田': '02', '江戸川': '03', '平和島': '04',
    '多摩川': '05', '浜名湖': '06', '蒲郡': '07', '常滑': '08',
    '津': '09', 'びわこ': '10', '三国': '11', '住之江': '12',
    '尼崎': '13', '鳴門': '14', '丸亀': '15', '児島': '16',
    '宮島': '17', '徳山': '18', '下関': '19', '若松': '20',
    '芦屋': '21', '福岡': '22', '唐津': '23', '大村': '24',
}
GRADE_MAP = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
GRADE_COLORS = {'A1': '🔴', 'A2': '🟠', 'B1': '🔵', 'B2': '⚪'}
WAKU_COLORS = {1: '⬜', 2: '⬛', 3: '🟥', 4: '🟦', 5: '🟨', 6: '🟩'}

# 学習データから算出した場別統計
PLACE_W1_WINRATE = {
    '大村': 0.6238, '徳山': 0.6220, '下関': 0.6040, '住之江': 0.6026,
    '尼崎': 0.5992, '常滑': 0.5894, '芦屋': 0.5858, '若松': 0.5750,
    '津': 0.5708, '蒲郡': 0.5707, '丸亀': 0.5704, '児島': 0.5550,
    '宮島': 0.5472, 'びわこ': 0.5425, '唐津': 0.5392, '多摩川': 0.5372,
    '三国': 0.5338, '浜名湖': 0.5178, '桐生': 0.5080, '鳴門': 0.4764,
    '江戸川': 0.4661, '戸田': 0.4550, '平和島': 0.4439, '福岡': 0.5400,
}
PLACE_UPSET_RATE = {k: 1.0 - v for k, v in PLACE_W1_WINRATE.items()}

# 学習時のplace_codeマッピング（学習データの出現順）
PLACE_CODE_MAP = {
    '唐津': 0, '若松': 1, '下関': 2, '徳山': 3, '丸亀': 4, '尼崎': 5,
    'びわこ': 6, '津': 7, '蒲郡': 8, '浜名湖': 9, '平和島': 10, '戸田': 11,
    '芦屋': 12, '宮島': 13, '児島': 14, '住之江': 15, '常滑': 16, '江戸川': 17,
    '桐生': 18, '大村': 19, '鳴門': 20, '多摩川': 21, '三国': 22, '福岡': 23,
}

WAKU_WIN_HIST = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03}


# =============================================================================
# モデルロード
# =============================================================================
@st.cache_resource
def load_models():
    base = './'
    # v3 1着特化モデル
    with open(base + 'win_model_v3.pkl', 'rb') as f:
        win_model = pickle.load(f)
    # コース別成績データ
    df_racer = pd.read_csv(base + 'racer_course_data_v2.csv')
    return win_model, df_racer


# =============================================================================
# データ取得関数（変更なし）
# =============================================================================
def fetch_race_data(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    boats = []
    toban_links = soup.select('a[href*="toban"]')
    tobans = []
    for a in toban_links:
        m = re.search(r'toban=(\d+)', a.get('href', ''))
        if m:
            t = int(m.group(1))
            if not tobans or tobans[-1] != t:
                tobans.append(t)
    names = [div.get_text(strip=True) for div in soup.select('div.is-fs18')]
    tbodies = soup.select('tbody.is-fs12')
    for i, tbody in enumerate(tbodies[:6]):
        waku = i + 1
        boat = {'waku': waku}
        if i < len(tobans): boat['toban'] = tobans[i]
        if i < len(names): boat['name'] = names[i]
        full_text = tbody.get_text()
        grade_match = re.search(r'(A1|A2|B1|B2)', full_text)
        if grade_match: boat['grade'] = grade_match.group(1)
        age_match = re.search(r'(\d{2})歳', full_text)
        if age_match: boat['age'] = int(age_match.group(1))
        weight_match = re.search(r'([\d\.]+)kg', full_text)
        if weight_match: boat['weight'] = float(weight_match.group(1))
        line_tds = tbody.select('td.is-lineH2')
        if len(line_tds) >= 5:
            pat = r'(\d{1,2}\.\d{2})'
            st_text = line_tds[0].get_text(strip=True)
            st_match = re.search(r'(\d+\.\d+)$', st_text)
            if st_match: boat['avg_st'] = float(st_match.group(1))
            nat_nums = re.findall(pat, line_tds[1].get_text(strip=True))
            if len(nat_nums) >= 1: boat['national_win_rate'] = float(nat_nums[0])
            if len(nat_nums) >= 2: boat['national_2rate'] = float(nat_nums[1])
            loc_nums = re.findall(pat, line_tds[2].get_text(strip=True))
            if len(loc_nums) >= 1: boat['local_win_rate'] = float(loc_nums[0])
            if len(loc_nums) >= 2: boat['local_2rate'] = float(loc_nums[1])
            motor_nums = re.findall(pat, line_tds[3].get_text(strip=True))
            if len(motor_nums) >= 1: boat['motor_2rate'] = float(motor_nums[0])
            boat_nums = re.findall(pat, line_tds[4].get_text(strip=True))
            if len(boat_nums) >= 1: boat['boat_2rate'] = float(boat_nums[0])
        boats.append(boat)
    return boats


def fetch_beforeinfo(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    info = {}
    # 展示タイム
    main_table = soup.select_one('table.is-w748')
    if main_table:
        for tr in main_table.select('tr'):
            boat_color = tr.select_one('td[class*="is-boatColor"]')
            if boat_color:
                tds = tr.select('td')
                try:
                    waku = int(boat_color.get_text(strip=True))
                except:
                    continue
                if len(tds) >= 5:
                    try:
                        et_val = float(tds[4].get_text(strip=True))
                        if 5.5 <= et_val <= 8.5:
                            info[f'et_{waku}'] = et_val
                    except:
                        pass
    # スタート展示 → コース進入も取得
    st_table = soup.select_one('table.is-w238')
    if st_table:
        course_pos = 0
        for tr in st_table.select('tr'):
            tds = tr.select('td')
            if len(tds) >= 1:
                txt = tds[0].get_text(strip=True)
                st_match = re.match(r'^(\d)(F?)(\.?\d{2})$', txt)
                if st_match:
                    course_pos += 1
                    waku_num = int(st_match.group(1))
                    is_flying = st_match.group(2) == 'F'
                    st_digits = st_match.group(3)
                    if st_digits.startswith('.'):
                        st_val = float('0' + st_digits)
                    else:
                        st_val = float('0.' + st_digits)
                    if is_flying:
                        st_val = -st_val
                    info[f'st_{waku_num}'] = st_val
                    info[f'entry_course_{waku_num}'] = course_pos
    return info


def fetch_trifecta_odds(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={hd}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    odds_dict = {}
    tables = soup.select('table')
    if len(tables) < 2:
        return odds_dict
    table = tables[1]
    header = table.select('tr')[0]
    ths = header.select('th')
    col_first = {}
    for ci in range(6):
        if ci * 2 < len(ths):
            boat_class = [c for c in ths[ci * 2].get('class', []) if c.startswith('is-boatColor')]
            if boat_class:
                col_first[ci] = int(boat_class[0].replace('is-boatColor', ''))
    col_second = {}
    for row in table.select('tr')[1:]:
        tds = row.select('td')
        td_idx = 0
        col_idx = 0
        while td_idx < len(tds) and col_idx < 6:
            td = tds[td_idx]
            classes = td.get('class', [])
            if 'is-fs14' in classes:
                second = int(td.get_text(strip=True))
                col_second[col_idx] = second
                third = int(tds[td_idx + 1].get_text(strip=True))
                odds_text = tds[td_idx + 2].get_text(strip=True).replace(',', '')
                try:
                    odds_val = float(odds_text)
                except:
                    odds_val = 0
                first = col_first.get(col_idx, 0)
                if first > 0:
                    odds_dict[f"{first}-{second}-{third}"] = odds_val
                td_idx += 3
                col_idx += 1
            else:
                third = int(td.get_text(strip=True))
                odds_text = tds[td_idx + 1].get_text(strip=True).replace(',', '')
                try:
                    odds_val = float(odds_text)
                except:
                    odds_val = 0
                first = col_first.get(col_idx, 0)
                second = col_second.get(col_idx, 0)
                if first > 0 and second > 0:
                    odds_dict[f"{first}-{second}-{third}"] = odds_val
                td_idx += 2
                col_idx += 1
    return odds_dict


# =============================================================================
# オッズ合成関数
# =============================================================================
def derive_all_odds(trifecta_odds):
    wakus = list(range(1, 7))
    result = {'trifecta': dict(trifecta_odds)}
    exacta_odds = {}
    for a in wakus:
        for b in wakus:
            if a == b: continue
            inv_sum = sum(1.0 / trifecta_odds[f'{a}-{b}-{c}']
                         for c in wakus if c != a and c != b
                         and trifecta_odds.get(f'{a}-{b}-{c}', 0) > 0)
            if inv_sum > 0:
                exacta_odds[f'{a}-{b}'] = round(1.0 / inv_sum, 1)
    result['exacta'] = exacta_odds
    quinella_odds = {}
    for combo in combinations(wakus, 2):
        a, b = combo
        inv_sum = sum(1.0 / exacta_odds[k] for k in [f'{a}-{b}', f'{b}-{a}']
                      if exacta_odds.get(k, 0) > 0)
        if inv_sum > 0:
            quinella_odds[f'{a}={b}'] = round(1.0 / inv_sum, 1)
    result['quinella'] = quinella_odds
    trio_odds = {}
    for combo in combinations(wakus, 3):
        a, b, c = combo
        inv_sum = sum(1.0 / trifecta_odds.get(f'{p[0]}-{p[1]}-{p[2]}', 0)
                      for p in permutations(combo)
                      if trifecta_odds.get(f'{p[0]}-{p[1]}-{p[2]}', 0) > 0)
        if inv_sum > 0:
            trio_odds[f'{a}={b}={c}'] = round(1.0 / inv_sum, 1)
    result['trio'] = trio_odds
    # 合成単勝オッズ
    win_odds = {}
    for w in wakus:
        inv_sum = sum(1.0 / trifecta_odds.get(f'{w}-{b}-{c}', 0)
                      for b in wakus if b != w
                      for c in wakus if c != w and c != b
                      and trifecta_odds.get(f'{w}-{b}-{c}', 0) > 0)
        if inv_sum > 0:
            win_odds[w] = round(1.0 / inv_sum, 2)
    result['win'] = win_odds
    return result


# =============================================================================
# v3特徴量構築（95特徴量完全再現）
# =============================================================================
def build_features_v3(boats, features, before_info, df_racer, place_name):
    n = len(boats)
    rows = []

    # 展示タイム・ST取得
    et_list = [before_info.get(f'et_{i+1}', 0) for i in range(n)]
    et_valid = [v for v in et_list if v > 0]
    et_mean = np.mean(et_valid) if et_valid else 6.80
    et_std = np.std(et_valid) if len(et_valid) > 1 else 0.1

    st_list = [before_info.get(f'st_{i+1}', 0.15) for i in range(n)]
    st_mean = np.mean(st_list)
    st_std = np.std(st_list) if np.std(st_list) > 0 else 0.01
    st_best = min(st_list)

    for boat in boats:
        waku = boat.get('waku', 0)
        toban = boat.get('toban', 0)
        nwr = boat.get('national_win_rate', 0)
        n2r = boat.get('national_2rate', 0)
        lwr = boat.get('local_win_rate', 0)
        l2r = boat.get('local_2rate', 0)
        m2r = boat.get('motor_2rate', 0)
        b2r = boat.get('boat_2rate', 0)
        grade = GRADE_MAP.get(boat.get('grade', 'B2'), 1)
        avg_st = boat.get('avg_st', 0.15)
        age = boat.get('age', 35)
        weight = boat.get('weight', 52)
        et = before_info.get(f'et_{waku}', et_mean)
        st = before_info.get(f'st_{waku}', 0.15)
        entry = before_info.get(f'entry_course_{waku}', waku)

        # コース別成績
        racer_row = df_racer[df_racer['toban'] == toban]
        if len(racer_row) > 0:
            r = racer_row.iloc[0]
            course_entry_rate = r.get(f'entry_rate_{waku}', 0) if f'entry_rate_{waku}' in r.index else 0
            course_win3_rate = r.get(f'win3_rate_{waku}', 0) if f'win3_rate_{waku}' in r.index else 0
            course_avg_st = r.get(f'avg_st_{waku}', 0) if f'avg_st_{waku}' in r.index else 0
        else:
            course_entry_rate = 0
            course_win3_rate = 0
            course_avg_st = 0

        row = {
            'waku': waku,
            'national_win_rate': nwr,
            'national_2rate': n2r,
            'local_win_rate': lwr,
            'local_2rate': l2r,
            'motor_2rate': m2r,
            'boat_2rate': b2r,
            'grade_num': grade,
            'avg_st': avg_st,
            'age': age,
            'weight': weight,
            'exhibition_time': et,
            'start_timing': st,
            'entry_course': entry,
            'recent_win': nwr / 10.0 if nwr > 0 else 0,
            'place_in_rate': lwr / 10.0 if lwr > 0 else 0,
            'course_entry_rate': course_entry_rate,
            'course_win3_rate': course_win3_rate,
            'course_avg_st': course_avg_st,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- 基本派生特徴量 ---
    df['is_waku1'] = (df['waku'] == 1).astype(int)
    df['waku_win_hist'] = df['waku'].map(WAKU_WIN_HIST)
    df['waku_penalty'] = df['waku'].apply(lambda x: max(0, x - 3))
    df['win_rate_diff'] = df['national_win_rate'] - df['local_win_rate']

    # vs_avg / rank
    for col in ['national_win_rate', 'national_2rate']:
        df[f'{col}_vs_avg'] = df[col] - df[col].mean()
        df[f'{col}_rank'] = df[col].rank(ascending=False)

    # 1号艇との比較
    w1_row = df[df['waku'] == 1].iloc[0] if len(df[df['waku'] == 1]) > 0 else df.iloc[0]
    df['vs_waku1'] = df['national_win_rate'] - w1_row['national_win_rate']
    df['vs_race_max'] = df['national_win_rate'] - df['national_win_rate'].max()

    # 交互作用
    df['motor_rank_x_waku'] = df['national_win_rate_rank'] * df['waku']  # motor_2rate_rankの代用
    # 正確にはmotor_2rate_rank
    df['motor_2rate_rank_temp'] = df['motor_2rate'].rank(ascending=False)
    df['motor_rank_x_waku'] = df['motor_2rate_rank_temp'] * df['waku']
    df['waku_x_winrate'] = df['waku'] * df['national_win_rate']
    df['winrate_x_grade'] = df['national_win_rate'] * df['grade_num']
    df['grade_vs_race'] = df['grade_num'] - df['grade_num'].mean()
    df['win_rate_product'] = df['national_win_rate'] * df['national_2rate']
    df['machine_score'] = df['motor_2rate'] + df['boat_2rate']

    # --- 展示タイム特徴量 ---
    df['et_rank'] = df['exhibition_time'].rank()
    df['et_diff'] = df['exhibition_time'] - df['exhibition_time'].mean()
    df['et_vs_best'] = df['exhibition_time'] - df['exhibition_time'].min()
    df['et_z'] = (df['exhibition_time'] - df['exhibition_time'].mean()) / (df['exhibition_time'].std() if df['exhibition_time'].std() > 0 else 0.1)

    # --- ST特徴量 ---
    df['st_diff'] = df['start_timing'] - df['start_timing'].mean()
    df['st_vs_best'] = df['start_timing'] - df['start_timing'].min()
    df['st_rank'] = df['start_timing'].rank()
    df['st_z'] = (df['start_timing'] - st_mean) / st_std if st_std > 0 else 0
    df['st_vs_avg_st'] = df['start_timing'] - df['avg_st']

    # --- Zスコア特徴量 ---
    for col, alias in [('national_win_rate', 'nwr'), ('national_2rate', 'n2r'),
                        ('motor_2rate', 'motor'), ('grade_num', 'grade'),
                        ('local_win_rate', 'lwr'), ('boat_2rate', 'boat')]:
        std = df[col].std()
        df[f'{alias}_z'] = (df[col] - df[col].mean()) / std if std > 0 else 0

    # --- 総合スコア ---
    df['total_score'] = df['nwr_z'] + df['n2r_z'] + df['motor_z'] + df['grade_z'] + df['lwr_z']
    df['total_rank'] = df['total_score'].rank(ascending=False)
    df['nwr_rank'] = df['national_win_rate'].rank(ascending=False)

    # --- 1号艇との差分 ---
    w1_nwr = w1_row['national_win_rate']
    w1_n2r = w1_row['national_2rate']
    w1_grade = w1_row['grade_num']
    w1_motor = w1_row['motor_2rate']
    w1_lwr = w1_row['local_win_rate']
    w1_total = df[df['waku'] == 1]['total_score'].values[0] if len(df[df['waku'] == 1]) > 0 else 0

    df['nwr_diff_w1'] = df['national_win_rate'] - w1_nwr
    df['n2r_diff_w1'] = df['national_2rate'] - w1_n2r
    df['grade_diff_w1'] = df['grade_num'] - w1_grade
    df['motor_diff_w1'] = df['motor_2rate'] - w1_motor
    df['lwr_diff_w1'] = df['local_win_rate'] - w1_lwr
    df['total_diff_w1'] = df['total_score'] - w1_total

    # 2号艇との差分
    w2_row = df[df['waku'] == 2].iloc[0] if len(df[df['waku'] == 2]) > 0 else df.iloc[1]
    df['nwr_diff_w2'] = df['national_win_rate'] - w2_row['national_win_rate']
    df['total_diff_w2'] = df['total_score'] - (df[df['waku'] == 2]['total_score'].values[0] if len(df[df['waku'] == 2]) > 0 else 0)

    # --- 1号艇情報（全行共通） ---
    w1_st = before_info.get('st_1', 0.15)
    df['w1_nwr'] = w1_nwr
    df['w1_grade'] = w1_grade
    df['w1_motor'] = w1_motor
    df['w1_total'] = w1_total
    df['w1_dominance'] = w1_nwr - df[df['waku'] != 1]['national_win_rate'].max() if len(df[df['waku'] != 1]) > 0 else 0
    df['w1_st'] = w1_st

    # --- 他艇情報 ---
    others = df[df['waku'] != 1]['national_win_rate']
    df['others_max'] = others.max() if len(others) > 0 else 0
    df['others_top2_avg'] = others.nlargest(2).mean() if len(others) >= 2 else 0
    df['vs_others_max'] = df['national_win_rate'] - others.max() if len(others) > 0 else 0

    # --- レース全体統計 ---
    df['race_nwr_mean'] = df['national_win_rate'].mean()
    df['race_nwr_std'] = df['national_win_rate'].std()
    df['race_nwr_range'] = df['national_win_rate'].max() - df['national_win_rate'].min()
    df['race_grade_mean'] = df['grade_num'].mean()
    df['race_grade_std'] = df['grade_num'].std()
    df['race_motor_mean'] = df['motor_2rate'].mean()
    df['race_motor_std'] = df['motor_2rate'].std()
    df['race_nwr_max'] = df['national_win_rate'].max()
    df['race_nwr_min'] = df['national_win_rate'].min()

    # --- 場情報 ---
    df['place_code'] = PLACE_CODE_MAP.get(place_name, 0)
    df['place_w1_winrate'] = PLACE_W1_WINRATE.get(place_name, 0.54)
    df['place_upset_rate'] = PLACE_UPSET_RATE.get(place_name, 0.46)
    df['waku_x_place_w1wr'] = df['waku'] * df['place_w1_winrate']

    # --- コース進入特徴量 ---
    df['course_change'] = df['entry_course'] - df['waku']
    df['course_inward'] = (df['entry_course'] < df['waku']).astype(int)
    df['entry_vs_waku'] = df['entry_course'] - df['waku']

    # --- 交互作用（追加） ---
    df['waku_x_total'] = df['waku'] * df['total_score']
    df['waku_x_nwr_z'] = df['waku'] * df['nwr_z']
    df['waku_x_motor_z'] = df['waku'] * df['motor_z']

    # --- フラグ特徴量 ---
    df['is_strong_inner'] = ((df['waku'] <= 2) & (df['grade_num'] >= 3)).astype(int)
    df['is_weak_w1'] = ((df['waku'] == 1) & (df['national_win_rate'] < df['national_win_rate'].mean())).astype(int)
    df['is_A1_inner'] = ((df['waku'] <= 2) & (df['grade_num'] == 4)).astype(int)
    df['is_A1_w1'] = ((df['waku'] == 1) & (df['grade_num'] == 4)).astype(int)

    # --- パワーインデックス ---
    df['power_index'] = df['nwr_z'] * 0.3 + df['motor_z'] * 0.2 + df['grade_z'] * 0.2 + df['lwr_z'] * 0.15 + df['boat_z'] * 0.15
    df['power_rank'] = df['power_index'].rank(ascending=False)

    # NaN処理
    df = df.fillna(0)

    # 特徴量を揃えて返す
    for f in features:
        if f not in df.columns:
            df[f] = 0

    return df[features]


# =============================================================================
# v3予測関数
# =============================================================================
def predict_win_prob_v3(model_data, X):
    """v3モデルで1着確率を予測"""
    raw = model_data['model'].predict(X)
    calibrated = model_data['isotonic'].predict(raw)
    calibrated = np.clip(calibrated, 0.001, 0.999)
    total = calibrated.sum()
    if total > 0:
        calibrated = calibrated / total
    return calibrated


# =============================================================================
# 市場確率の算出
# =============================================================================
def calc_market_probs(all_odds):
    """合成単勝オッズから市場の暗黙的1着確率を算出"""
    win_odds = all_odds.get('win', {})
    if not win_odds:
        return {}
    inv_sum = sum(1.0 / o for o in win_odds.values() if o > 0)
    if inv_sum == 0:
        return {}
    probs = {}
    for w, o in win_odds.items():
        if o > 0:
            probs[w] = (1.0 / o) / inv_sum
        else:
            probs[w] = 0
    return probs


# =============================================================================
# 3連単確率計算（1着モデルベース）
# =============================================================================
def calc_trifecta_probs_v3(win_probs, wakus):
    """1着確率から条件付き確率で3連単確率を計算"""
    p1 = dict(zip(wakus, win_probs))
    trifecta = {}
    for perm in permutations(wakus, 3):
        w1, w2, w3 = perm
        pp1 = p1[w1]
        rest2 = {w: p1[w] for w in wakus if w != w1}
        s2 = sum(rest2.values())
        pp2 = rest2[w2] / s2 if s2 > 0 else 1 / 5
        rest3 = {w: p1[w] for w in wakus if w != w1 and w != w2}
        s3 = sum(rest3.values())
        pp3 = rest3[w3] / s3 if s3 > 0 else 1 / 4
        trifecta[f"{w1}-{w2}-{w3}"] = pp1 * pp2 * pp3
    tp = sum(trifecta.values())
    if tp > 0:
        trifecta = {k: v / tp for k, v in trifecta.items()}
    return trifecta


def derive_all_probs(trifecta):
    wakus = list(range(1, 7))
    exacta = {}
    for perm in permutations(wakus, 2):
        w1, w2 = perm
        exacta[f"{w1}-{w2}"] = sum(trifecta.get(f"{w1}-{w2}-{w3}", 0)
                                    for w3 in wakus if w3 != w1 and w3 != w2)
    quinella = {}
    for comb in combinations(wakus, 2):
        w1, w2 = comb
        quinella[f"{w1}={w2}"] = exacta.get(f"{w1}-{w2}", 0) + exacta.get(f"{w2}-{w1}", 0)
    trio = {}
    for comb in combinations(wakus, 3):
        key = "=".join(map(str, comb))
        trio[key] = sum(trifecta.get(f"{a}-{b}-{c}", 0) for a, b, c in permutations(comb))
    win = {}
    for w in wakus:
        win[w] = sum(trifecta.get(f"{w}-{b}-{c}", 0)
                     for b in wakus if b != w
                     for c in wakus if c != w and c != b)
    return {'exacta': exacta, 'quinella': quinella, 'trifecta': trifecta, 'trio': trio, 'win': win}


def calc_kelly(prob, odds):
    if odds <= 1 or prob <= 0:
        return 0
    kelly = (prob * odds - 1) / (odds - 1)
    return max(kelly, 0)


# =============================================================================
# メイン
# =============================================================================
def main():
    st.title("🚤 競艇AI予想 v11")
    st.caption("1着特化モデル v3 | エッジ検出戦略D | 3連単頭固定6点買い | ROI 198%")

    try:
        win_model, df_racer = load_models()
        features = win_model['features']
    except Exception as e:
        st.error(f"モデルロードエラー: {e}")
        st.info("必要ファイル: win_model_v3.pkl, racer_course_data_v2.csv")
        return

    # --- サイドバー ---
    st.sidebar.header("🎯 レース選択")
    place = st.sidebar.selectbox("場所", list(PLACE_CODES.keys()), index=15)
    race_num = st.sidebar.selectbox("レース番号", list(range(1, 13)))
    from datetime import date
    race_date = st.sidebar.date_input("日付", value=date.today())

    st.sidebar.header("⚙️ 戦略設定")
    edge_threshold = st.sidebar.slider("エッジ閾値 (%)", 0, 30, 10, step=1)
    n_2nd_candidates = st.sidebar.selectbox("2着候補数", [2, 3, 4, 5], index=1)
    bet_amount = st.sidebar.number_input("1点あたり金額 (円)", 100, 10000, 100, step=100)

    st.sidebar.header("📊 表示設定")
    top_n = st.sidebar.slider("各券種 表示数", 5, 30, 15)

    jcd = PLACE_CODES[place]
    hd = race_date.strftime('%Y%m%d')

    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False

    if st.sidebar.button("🎯 予想する", type="primary", use_container_width=True):
        st.session_state.prediction_done = False
        with st.spinner("📋 出走表取得中..."):
            boats = fetch_race_data(jcd, hd, str(race_num))
        if len(boats) < 6:
            st.error("❌ 出走表の取得に失敗しました。日付・レース番号を確認してください。")
            return
        with st.spinner("📋 直前情報取得中..."):
            before_info = fetch_beforeinfo(jcd, hd, str(race_num))
        with st.spinner("📋 3連単オッズ取得中..."):
            trifecta_odds_raw = fetch_trifecta_odds(jcd, hd, str(race_num))

        st.session_state.boats = boats
        st.session_state.before_info = before_info
        st.session_state.trifecta_odds_raw = trifecta_odds_raw
        st.session_state.jcd = jcd
        st.session_state.hd = hd
        st.session_state.race_num = race_num
        st.session_state.place = place
        st.session_state.race_date = race_date
        st.session_state.prediction_done = True

    if st.session_state.prediction_done:
        if st.sidebar.button("🔄 オッズ再取得", use_container_width=True):
            with st.spinner("📋 オッズ再取得中..."):
                trifecta_odds_raw = fetch_trifecta_odds(
                    st.session_state.jcd, st.session_state.hd, str(st.session_state.race_num))
                st.session_state.trifecta_odds_raw = trifecta_odds_raw
                st.success(f"✅ オッズ再取得完了 ({len(trifecta_odds_raw)}/120通り)")

    if not st.session_state.prediction_done:
        st.info("👈 サイドバーからレースを選択して「予想する」を押してください")
        return

    # --- データ取得 ---
    boats = st.session_state.boats
    before_info = st.session_state.before_info
    trifecta_odds_raw = st.session_state.trifecta_odds_raw
    place_name = st.session_state.place
    rnum = st.session_state.race_num
    rdate = st.session_state.race_date

    has_odds = len(trifecta_odds_raw) >= 100
    if has_odds:
        all_odds = derive_all_odds(trifecta_odds_raw)
        market_probs = calc_market_probs(all_odds)

    et_count = sum(1 for k in before_info if k.startswith('et_'))
    st_count = sum(1 for k in before_info if k.startswith('st_') and not k.startswith('st_rank'))

    st.header(f"📋 {place_name} {rnum}R ({rdate})")

    # ステータス表示
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if et_count >= 6:
            st.success(f"✅ 展示タイム: {et_count}/6艇")
        else:
            st.warning(f"⚠️ 展示タイム: {et_count}/6艇")
    with col_s2:
        if st_count >= 6:
            st.success(f"✅ スタート展示: {st_count}/6艇")
        else:
            st.warning(f"⚠️ スタート展示: {st_count}/6艇")
    with col_s3:
        if has_odds:
            st.success(f"✅ オッズ: {len(trifecta_odds_raw)}/120通り")
        else:
            st.warning(f"⚠️ オッズ: {len(trifecta_odds_raw)}/120通り")

    # --- 出走表 ---
    entry_data = []
    for b in boats:
        w = b['waku']
        entry_data.append({
            '枠': f"{WAKU_COLORS.get(w, '')} {w}",
            '登番': b.get('toban', '?'),
            '名前': b.get('name', '?'),
            '級別': f"{GRADE_COLORS.get(b.get('grade', ''), '')} {b.get('grade', '?')}",
            '全国勝率': b.get('national_win_rate', 0),
            '全国2率': b.get('national_2rate', 0),
            'モーター2率': b.get('motor_2rate', 0),
            '展示T': before_info.get(f'et_{w}', '-'),
            'ST': before_info.get(f'st_{w}', '-'),
            'コース': before_info.get(f'entry_course_{w}', w),
        })
    st.dataframe(pd.DataFrame(entry_data), use_container_width=True, hide_index=True)

    # =================================================================
    # AI予測
    # =================================================================
    with st.spinner("🔧 AI予測計算中..."):
        X = build_features_v3(boats, features, before_info, df_racer, place_name)
        wakus = [b['waku'] for b in boats]
        win_probs = predict_win_prob_v3(win_model, X)

    # --- 1着確率 & エッジ ---
    st.header("🎯 1着予測 & エッジ分析")

    pred_df = pd.DataFrame({
        'waku': wakus,
        'name': [b.get('name', '?') for b in boats],
        'grade': [b.get('grade', '?') for b in boats],
        'model_prob': win_probs,
    })
    pred_df = pred_df.sort_values('model_prob', ascending=False).reset_index(drop=True)

    if has_odds and market_probs:
        pred_df['market_prob'] = pred_df['waku'].map(market_probs)
        pred_df['edge'] = pred_df['model_prob'] - pred_df['market_prob']
        pred_df['edge_pct'] = pred_df['edge'] * 100

        # 合成単勝オッズ
        pred_df['win_odds'] = pred_df['waku'].map(all_odds.get('win', {}))

        display_data = []
        for _, row in pred_df.iterrows():
            w = int(row['waku'])
            edge_val = row['edge_pct']
            if edge_val >= 10:
                edge_icon = '🔥🔥'
            elif edge_val >= 5:
                edge_icon = '🔥'
            elif edge_val >= 0:
                edge_icon = '✅'
            else:
                edge_icon = '⬇️'

            display_data.append({
                '枠': f"{WAKU_COLORS.get(w, '')} {w}",
                '名前': row['name'],
                '級別': f"{GRADE_COLORS.get(row['grade'], '')} {row['grade']}",
                'AI予測': f"{row['model_prob']:.1%}",
                '市場評価': f"{row['market_prob']:.1%}",
                'エッジ': f"{edge_val:+.1f}%",
                '単勝ｵｯｽﾞ': f"{row['win_odds']:.1f}" if pd.notna(row.get('win_odds')) else '-',
                '判定': edge_icon,
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        display_data = []
        for _, row in pred_df.iterrows():
            w = int(row['waku'])
            display_data.append({
                '枠': f"{WAKU_COLORS.get(w, '')} {w}",
                '名前': row['name'],
                '級別': f"{GRADE_COLORS.get(row['grade'], '')} {row['grade']}",
                'AI予測': f"{row['model_prob']:.1%}",
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

    # =================================================================
    # 戦略D: 頭固定6点買い推奨
    # =================================================================
    st.header("💰 戦略D: 3連単 頭固定買い目")

    top1_waku = pred_df.iloc[0]['waku']
    top1_name = pred_df.iloc[0]['name']
    top1_prob = pred_df.iloc[0]['model_prob']

    if has_odds and market_probs:
        top1_market = market_probs.get(top1_waku, 0)
        top1_edge = (top1_prob - top1_market) * 100
    else:
        top1_edge = None

    # 2-3着候補（モデル順位2位以降）
    candidates = pred_df.iloc[1:1+n_2nd_candidates]['waku'].tolist()

    # 買い目生成
    bets = []
    for perm in permutations(candidates, 2):
        combo = f"{int(top1_waku)}-{int(perm[0])}-{int(perm[1])}"
        prob = 0
        odds = trifecta_odds_raw.get(combo, 0)
        bets.append({'combo': combo, 'odds': odds})

    # エッジ判定
    if top1_edge is not None:
        if top1_edge >= edge_threshold:
            st.success(f"🔥 購入推奨！ {int(top1_waku)}号艇 {top1_name} "
                      f"(AI: {top1_prob:.1%} vs 市場: {top1_market:.1%}, エッジ: {top1_edge:+.1f}%)")
        elif top1_edge >= 5:
            st.info(f"✅ やや有望 {int(top1_waku)}号艇 {top1_name} "
                   f"(AI: {top1_prob:.1%} vs 市場: {top1_market:.1%}, エッジ: {top1_edge:+.1f}%)")
        elif top1_edge >= 0:
            st.warning(f"⚠️ エッジ小 {int(top1_waku)}号艇 {top1_name} "
                      f"(AI: {top1_prob:.1%} vs 市場: {top1_market:.1%}, エッジ: {top1_edge:+.1f}%)")
        else:
            st.error(f"❌ 見送り推奨 {int(top1_waku)}号艇 {top1_name} "
                    f"(AI: {top1_prob:.1%} vs 市場: {top1_market:.1%}, エッジ: {top1_edge:+.1f}%)")
    else:
        st.info(f"📊 {int(top1_waku)}号艇 {top1_name} (AI: {top1_prob:.1%}) ※オッズ未取得のためエッジ計算不可")

    # 買い目テーブル
    bet_data = []
    total_invest = 0
    for b in bets:
        d = {
            '買い目': b['combo'],
            'ｵｯｽﾞ': f"{b['odds']:.1f}" if b['odds'] > 0 else '-',
            '金額': f"¥{bet_amount:,}",
        }
        if b['odds'] > 0:
            d['払戻'] = f"¥{int(b['odds'] * bet_amount):,}"
        else:
            d['払戻'] = '-'
        bet_data.append(d)
        total_invest += bet_amount

    st.dataframe(pd.DataFrame(bet_data), use_container_width=True, hide_index=True)

    n_bets = len(bets)
    valid_odds = [b['odds'] for b in bets if b['odds'] > 0]
    st.markdown(f"**合計: {n_bets}点 × ¥{bet_amount:,} = ¥{total_invest:,}** | "
               f"最低配当: ¥{int(min(valid_odds) * bet_amount):,}" if valid_odds else
               f"**合計: {n_bets}点 × ¥{bet_amount:,} = ¥{total_invest:,}**")

    # =================================================================
    # 全券種確率（展開表示）
    # =================================================================
    with st.expander("📊 全券種 確率・オッズ・期待値", expanded=False):
        trifecta_probs = calc_trifecta_probs_v3(win_probs, wakus)
        all_probs = derive_all_probs(trifecta_probs)

        # 3連単
        st.subheader("3連単 TOP")
        sorted_3t = sorted(trifecta_probs.items(), key=lambda x: -x[1])
        data_3t = []
        for i, (combo, prob) in enumerate(sorted_3t[:top_n], 1):
            d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
            if has_odds:
                o = trifecta_odds_raw.get(combo, 0)
                ev = prob * o if o > 0 else 0
                kelly = calc_kelly(prob, o) if o > 0 else 0
                d['ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                d['期待値'] = f"{ev:.2f}"
                d['Kelly'] = f"{kelly:.1%}" if kelly > 0 else '-'
                d['判定'] = '🔥' if kelly >= 0.03 else ('✅' if kelly > 0 else '')
            data_3t.append(d)
        st.dataframe(pd.DataFrame(data_3t), use_container_width=True, hide_index=True)

        # 2連単
        col_e, col_q = st.columns(2)
        with col_e:
            st.subheader("2連単")
            sorted_ex = sorted(all_probs['exacta'].items(), key=lambda x: -x[1])
            ex_data = []
            for i, (combo, prob) in enumerate(sorted_ex[:top_n], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    o = all_odds['exacta'].get(combo, 0)
                    ev = prob * o if o > 0 else 0
                    kelly = calc_kelly(prob, o) if o > 0 else 0
                    d['ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                    d['期待値'] = f"{ev:.2f}"
                    d['Kelly'] = f"{kelly:.1%}" if kelly > 0 else '-'
                ex_data.append(d)
            st.dataframe(pd.DataFrame(ex_data), use_container_width=True, hide_index=True)

        with col_q:
            st.subheader("2連複")
            sorted_q = sorted(all_probs['quinella'].items(), key=lambda x: -x[1])
            q_data = []
            for i, (combo, prob) in enumerate(sorted_q[:top_n], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    o = all_odds['quinella'].get(combo, 0)
                    ev = prob * o if o > 0 else 0
                    kelly = calc_kelly(prob, o) if o > 0 else 0
                    d['ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                    d['期待値'] = f"{ev:.2f}"
                    d['Kelly'] = f"{kelly:.1%}" if kelly > 0 else '-'
                q_data.append(d)
            st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)

    # =================================================================
    # フッター
    # =================================================================
    st.divider()
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.caption(f"📊 モデル: win_model_v3 (AUC {win_model.get('test_auc', 0):.4f}) | "
                  f"特徴量: {len(features)}個 | 戦略D: ROI 198.4%")
    with col_f2:
        st.caption(f"⚠️ 投資は自己責任です。バックテスト結果は将来の利益を保証しません。")


if __name__ == '__main__':
    main()
