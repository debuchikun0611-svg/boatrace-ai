# app.py - 競艇AI予想 v9
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

st.set_page_config(page_title="競艇AI予想 v9", page_icon="🚤", layout="wide")

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


# ============================================
# モデル読み込み (v9: 全体Platt)
# ============================================
@st.cache_resource
def load_models():
    base = './'
    models = {}
    for name in ['1着', '2連対', '3連対']:
        with open(base + f'boatrace_model_{name}_v9.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    df_racer = pd.read_csv(base + 'racer_course_data.csv')
    return models, df_racer


# ============================================
# 出走表取得
# ============================================
def fetch_race_data(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    resp = requests.get(url, timeout=15)
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
        if i < len(tobans):
            boat['toban'] = tobans[i]
        if i < len(names):
            boat['name'] = names[i]
        full_text = tbody.get_text()
        grade_match = re.search(r'(A1|A2|B1|B2)', full_text)
        if grade_match:
            boat['grade'] = grade_match.group(1)
        age_match = re.search(r'(\d{2})歳', full_text)
        if age_match:
            boat['age'] = int(age_match.group(1))
        weight_match = re.search(r'([\d\.]+)kg', full_text)
        if weight_match:
            boat['weight'] = float(weight_match.group(1))
        line_tds = tbody.select('td.is-lineH2')
        if len(line_tds) >= 5:
            pat = r'(\d{1,2}\.\d{2})'
            st_text = line_tds[0].get_text(strip=True)
            st_match = re.search(r'(\d+\.\d+)$', st_text)
            if st_match:
                boat['avg_st'] = float(st_match.group(1))
            nat_nums = re.findall(pat, line_tds[1].get_text(strip=True))
            if len(nat_nums) >= 1:
                boat['national_win_rate'] = float(nat_nums[0])
            if len(nat_nums) >= 2:
                boat['national_2rate'] = float(nat_nums[1])
            loc_nums = re.findall(pat, line_tds[2].get_text(strip=True))
            if len(loc_nums) >= 1:
                boat['local_win_rate'] = float(loc_nums[0])
            if len(loc_nums) >= 2:
                boat['local_2rate'] = float(loc_nums[1])
            motor_nums = re.findall(pat, line_tds[3].get_text(strip=True))
            if len(motor_nums) >= 1:
                boat['motor_2rate'] = float(motor_nums[0])
            boat_nums = re.findall(pat, line_tds[4].get_text(strip=True))
            if len(boat_nums) >= 1:
                boat['boat_2rate'] = float(boat_nums[0])
        boats.append(boat)
    return boats


# ============================================
# 直前情報取得
# ============================================
def fetch_beforeinfo(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    info = {}
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
    st_table = soup.select_one('table.is-w238')
    if st_table:
        for tr in st_table.select('tr'):
            tds = tr.select('td')
            if len(tds) >= 1:
                txt = tds[0].get_text(strip=True)
                st_match = re.match(r'^(\d)(F?)(\.?\d{2})$', txt)
                if st_match:
                    course = int(st_match.group(1))
                    is_flying = st_match.group(2) == 'F'
                    st_digits = st_match.group(3)
                    if st_digits.startswith('.'):
                        st_val = float('0' + st_digits)
                    else:
                        st_val = float('0.' + st_digits)
                    if is_flying:
                        st_val = -st_val
                    info[f'st_{course}'] = st_val
    return info


# ============================================
# 3連単オッズ取得
# ============================================
def fetch_trifecta_odds(jcd, hd, rno):
    """3連単オッズを取得。キー: '1-2-3' 等、値: オッズ(float)"""
    url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={hd}"
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    odds_dict = {}

    # 3連単オッズページの解析
    # 1着ごとのタブがあり、各タブ内に2着-3着のテーブルがある
    odds_tables = soup.select('table.is-w495')
    
    # 方式1: テーブルから直接取得
    all_tds = soup.select('td.oddsPoint')
    
    if not all_tds:
        # 方式2: oddstf のページ構造に合わせて解析
        # 各1着ごとにテーブルがある
        for first in range(1, 7):
            # 1着=first のオッズテーブルを探す
            tables = soup.select(f'div#odds3t{first} table')
            if not tables:
                continue
            for table in tables:
                for tr in table.select('tr'):
                    tds = tr.select('td')
                    for td in tds:
                        # テキストにオッズ値とIDがある
                        pass

    # 汎用的なパース: ページ全体からオッズデータを抽出
    # boatrace.jp の3連単ページはJSで描画されることがある
    # テキストベースで抽出を試みる
    text = resp.text
    
    # パターン: "1-2-3" のような組み合わせとオッズの対
    # odds3t ページの構造を解析
    for first in range(1, 7):
        # 各1着番号に対応するセクション
        section_pattern = f'odds3t_{first}'
        
    # 別アプローチ: oddsページから直接JSONを取得
    odds_url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={hd}"
    
    # HTMLテーブルから直接パース
    all_tables = soup.select('table')
    for table in all_tables:
        rows = table.select('tr')
        for row in rows:
            cells = row.select('td')
            for cell in cells:
                txt = cell.get_text(strip=True)
                # オッズ値のパターン（数字.数字）
                if re.match(r'^\d+\.\d+$', txt):
                    pass
    
    # 最終手段: 個別の1着別オッズページから取得
    for first in range(1, 7):
        url_f = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={hd}&kession={first}"
        try:
            resp_f = requests.get(url_f, timeout=10)
            soup_f = BeautifulSoup(resp_f.content, 'html.parser')
            
            # テーブル内のオッズセルを取得
            odds_cells = soup_f.select('td.is-p3-0')
            if not odds_cells:
                odds_cells = soup_f.select('td.oddsPoint')
            
            idx = 0
            for second in range(1, 7):
                if second == first:
                    continue
                for third in range(1, 7):
                    if third == first or third == second:
                        continue
                    if idx < len(odds_cells):
                        try:
                            odds_val = float(odds_cells[idx].get_text(strip=True).replace(',', ''))
                            odds_dict[f"{first}-{second}-{third}"] = odds_val
                        except:
                            pass
                        idx += 1
        except:
            continue

    return odds_dict


def fetch_trifecta_odds_v2(jcd, hd, rno):
    """3連単オッズ取得（確実版）: 1着番号ごとにページ取得"""
    odds_dict = {}
    
    for first in range(1, 7):
        url = (f"https://www.boatrace.jp/owpc/pc/race/oddstf?"
               f"rno={rno}&jcd={jcd}&hd={hd}")
        try:
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # 全テーブルを走査してオッズを取得
            # 3連単オッズは is-p3-0 クラスか、数値のみのセル
            tables = soup.select('table.is-w495')
            
            if tables:
                for t_idx, table in enumerate(tables):
                    first_boat = t_idx + 1
                    odds_cells = table.select('td.oddsPoint, td.is-p3-0')
                    
                    if not odds_cells:
                        # テーブル内の全tdからオッズっぽい値を抽出
                        all_tds = table.select('td')
                        odds_cells = []
                        for td in all_tds:
                            txt = td.get_text(strip=True)
                            if re.match(r'^\d{1,5}\.\d$', txt.replace(',', '')):
                                odds_cells.append(td)
                    
                    cell_idx = 0
                    for second in range(1, 7):
                        if second == first_boat:
                            continue
                        for third in range(1, 7):
                            if third == first_boat or third == second:
                                continue
                            if cell_idx < len(odds_cells):
                                try:
                                    val = odds_cells[cell_idx].get_text(strip=True).replace(',', '')
                                    odds_dict[f"{first_boat}-{second}-{third}"] = float(val)
                                except:
                                    pass
                                cell_idx += 1
                
                if odds_dict:
                    break  # 1回のリクエストで全部取れた
            
        except:
            continue
    
    # 取れなかった場合、別の方法を試す
    if len(odds_dict) < 100:
        try:
            url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={hd}"
            resp = requests.get(url, timeout=15)
            text = resp.text
            
            # scriptタグ内のオッズデータを探す
            odds_pattern = re.findall(r'"(\d-\d-\d)"\s*:\s*([\d.]+)', text)
            for combo, val in odds_pattern:
                odds_dict[combo] = float(val)
                
        except:
            pass
    
    return odds_dict


# ============================================
# 特徴量作成 (v9用)
# ============================================
def build_features(boats, features, before_info, df_racer):
    racer_num_cols = [c for c in df_racer.columns if c not in ['toban', 'class_rank']]

    rows = []
    n = len(boats)
    et_list = [before_info.get(f'et_{i+1}', 0) for i in range(n)]
    et_mean = np.mean([v for v in et_list if v > 0]) if any(v > 0 for v in et_list) else 6.8

    for boat in boats:
        waku = boat.get('waku', 0)
        toban = boat.get('toban', 0)
        row = {
            'waku': waku,
            'age': boat.get('age', 35),
            'weight': boat.get('weight', 52),
            'national_win_rate': boat.get('national_win_rate', 0),
            'national_2rate': boat.get('national_2rate', 0),
            'local_win_rate': boat.get('local_win_rate', 0),
            'local_2rate': boat.get('local_2rate', 0),
            'motor_2rate': boat.get('motor_2rate', 0),
            'boat_2rate': boat.get('boat_2rate', 0),
            'grade_num': GRADE_MAP.get(boat.get('grade', 'B2'), 1),
            'exhibition_time': before_info.get(f'et_{waku}', et_mean),
            'st_time': before_info.get(f'st_{waku}', 0.15),
        }

        # racer_course_data
        racer_row = df_racer[df_racer['toban'] == toban]
        if len(racer_row) > 0:
            r = racer_row.iloc[0]
            row['course_entry_rate'] = r.get(f'entry_rate_{waku}', 0) if f'entry_rate_{waku}' in r.index else 0
            row['course_win3_rate'] = r.get(f'win3_rate_{waku}', 0) if f'win3_rate_{waku}' in r.index else 0
            row['course_avg_st'] = r.get(f'avg_st_{waku}', 0) if f'avg_st_{waku}' in r.index else 0
        else:
            row['course_entry_rate'] = 0
            row['course_win3_rate'] = 0
            row['course_avg_st'] = 0

        for k in row:
            if pd.isna(row[k]):
                row[k] = 0
        rows.append(row)

    df = pd.DataFrame(rows)

    # レース内統計量
    for col in ['national_win_rate', 'national_2rate', 'motor_2rate', 'boat_2rate']:
        df[f'{col}_vs_avg'] = df[col] - df[col].mean()
        df[f'{col}_rank'] = df[col].rank(ascending=False)

    df['weight_vs_avg'] = df['weight'] - df['weight'].mean()
    df['is_waku1'] = (df['waku'] == 1).astype(int)
    df['is_waku2'] = (df['waku'] == 2).astype(int)
    df['is_waku3'] = (df['waku'] == 3).astype(int)
    df['win_rate_diff'] = df['national_win_rate'] - df['local_win_rate']
    df['machine_score'] = df['motor_2rate'] + df['boat_2rate']
    df['waku_penalty'] = df['waku'].apply(lambda x: max(0, x - 3))
    df['waku_win_hist'] = df['waku'].map({1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03})
    df['motor_rank_x_waku'] = df['motor_2rate_rank'] * df['waku']
    df['waku_x_winrate'] = df['waku'] * df['national_win_rate']
    df['winrate_x_grade'] = df['national_win_rate'] * df['grade_num']
    df['grade_vs_race'] = df['grade_num'] - df['grade_num'].mean()
    df['win_rate_product'] = df['national_win_rate'] * df['national_2rate']
    df['race_grade_level'] = df['grade_num'].mean()
    df['vs_race_max'] = df['national_win_rate'] - df['national_win_rate'].max()

    # 1号艇情報
    waku1 = df[df['waku'] == 1].iloc[0] if len(df[df['waku'] == 1]) > 0 else df.iloc[0]
    df['waku1_win_rate'] = waku1['national_win_rate']
    df['vs_waku1'] = df['national_win_rate'] - waku1['national_win_rate']

    # ET/ST
    df['et_rank'] = df['exhibition_time'].rank()
    df['et_diff'] = df['exhibition_time'] - df['exhibition_time'].mean()
    df['et_best_diff'] = df['exhibition_time'] - df['exhibition_time'].min()
    df['et_waku'] = df['exhibition_time'] * df['waku']
    df['st_rank'] = df['st_time'].rank()
    df['st_diff'] = df['st_time'] - df['st_time'].mean()
    df['st_best_diff'] = df['st_time'] - df['st_time'].min()
    df['st_waku'] = df['st_time'] * df['waku']
    df['et_st_combined'] = df['et_rank'] + df['st_rank']

    # コース vs_avg
    for col in ['course_entry_rate', 'course_win3_rate', 'course_avg_st']:
        df[f'{col}_vs_avg'] = df[col] - df[col].mean()

    # 不足列を0埋め
    for f in features:
        if f not in df.columns:
            df[f] = 0

    return df[features]


# ============================================
# 予測 (v9: 全体Platt + 合計制約正規化)
# ============================================
def predict_race(X, wakus, models):
    results = pd.DataFrame({'waku': wakus})

    for target_name, target_sum in [('1着', 1.0), ('2連対', 2.0), ('3連対', 3.0)]:
        md = models[target_name]
        raw = md['model'].predict(X)
        platt = md['platt'].predict_proba(raw.reshape(-1, 1))[:, 1]

        # レース内正規化（合計 = target_sum）
        s = platt.sum()
        if s > 0:
            normed = platt / s * target_sum
        else:
            normed = np.full(len(platt), target_sum / 6)
        results[f'p_{target_name}'] = normed

    # 単調性の強制: 1着 <= 2連対 <= 3連対
    results['p_2連対'] = results[['p_1着', 'p_2連対']].max(axis=1)
    results['p_3連対'] = results[['p_2連対', 'p_3連対']].max(axis=1)

    # 再正規化
    for name, ts in [('2連対', 2.0), ('3連対', 3.0)]:
        s = results[f'p_{name}'].sum()
        if s > 0:
            results[f'p_{name}'] = results[f'p_{name}'] / s * ts

    # 着順別確率
    results['p_2着'] = (results['p_2連対'] - results['p_1着']).clip(lower=0)
    results['p_3着'] = (results['p_3連対'] - results['p_2連対']).clip(lower=0)

    return results


# ============================================
# 全券種の確率計算
# ============================================
def calc_all_combinations(results):
    """3連単確率から全券種の確率を計算"""
    wakus = results['waku'].values
    p1 = dict(zip(wakus, results['p_1着'].values))
    p12 = dict(zip(wakus, results['p_2連対'].values))
    p123 = dict(zip(wakus, results['p_3連対'].values))

    # --- 3連単 (120通り) ---
    trifecta = {}
    for perm in permutations(wakus, 3):
        w1, w2, w3 = perm

        p_w1 = p1[w1]

        remaining2 = [w for w in wakus if w != w1]
        s2 = sum(p12[w] for w in remaining2)
        p_w2 = p12[w2] / s2 if s2 > 0 else 1 / 5

        remaining3 = [w for w in wakus if w != w1 and w != w2]
        s3 = sum(p123[w] for w in remaining3)
        p_w3 = p123[w3] / s3 if s3 > 0 else 1 / 4

        trifecta[f"{w1}-{w2}-{w3}"] = p_w1 * p_w2 * p_w3

    # 正規化
    tp = sum(trifecta.values())
    if tp > 0:
        trifecta = {k: v / tp for k, v in trifecta.items()}

    # --- 3連単から他の券種を導出 ---

    # 2連単 (30通り): w1が1着、w2が2着
    exacta = {}
    for perm in permutations(wakus, 2):
        w1, w2 = perm
        key = f"{w1}-{w2}"
        exacta[key] = sum(trifecta.get(f"{w1}-{w2}-{w3}", 0)
                          for w3 in wakus if w3 != w1 and w3 != w2)

    # 2連複 (15通り): w1,w2が1-2着（順不同）
    quinella = {}
    for comb in combinations(sorted(wakus), 2):
        w1, w2 = comb
        key = f"{w1}={w2}"
        quinella[key] = exacta.get(f"{w1}-{w2}", 0) + exacta.get(f"{w2}-{w1}", 0)

    # 3連複 (20通り): w1,w2,w3が1-3着（順不同）
    trio = {}
    for comb in combinations(sorted(wakus), 3):
        key = "=".join(map(str, comb))
        trio[key] = sum(trifecta.get(f"{a}-{b}-{c}", 0)
                        for a, b, c in permutations(comb))

    # 単勝 (6通り): wが1着
    win = {}
    for w in wakus:
        win[str(w)] = sum(trifecta.get(f"{w}-{w2}-{w3}", 0)
                          for w2 in wakus if w2 != w
                          for w3 in wakus if w3 != w and w3 != w2)

    # 複勝 (6通り): wが1着 or 2着
    place = {}
    for w in wakus:
        # 1着の確率
        p_1st = win[str(w)]
        # 2着の確率 = wが2着になる全3連単の合計
        p_2nd = sum(trifecta.get(f"{w1}-{w}-{w3}", 0)
                    for w1 in wakus if w1 != w
                    for w3 in wakus if w3 != w and w3 != w1)
        place[str(w)] = p_1st + p_2nd

    return {
        'trifecta': trifecta,   # 3連単
        'trio': trio,            # 3連複
        'exacta': exacta,       # 2連単
        'quinella': quinella,    # 2連複
        'win': win,              # 単勝
        'place': place,          # 複勝
    }


# ============================================
# 3連単オッズから合成オッズを計算
# ============================================
def calc_synthetic_odds(trifecta_odds):
    """
    3連単オッズのみから全券種の合成オッズを計算。
    合成オッズ = 1 / Σ(1/各3連単オッズ)
    """
    wakus = list(range(1, 7))
    result = {}

    # --- 単勝の合成オッズ ---
    # 例: 2の単勝 = 2が1着の全3連単の合成オッズ
    win_odds = {}
    for w in wakus:
        inv_sum = 0
        for w2 in wakus:
            if w2 == w:
                continue
            for w3 in wakus:
                if w3 == w or w3 == w2:
                    continue
                key = f"{w}-{w2}-{w3}"
                odds = trifecta_odds.get(key, 0)
                if odds > 0:
                    inv_sum += 1 / odds
        win_odds[str(w)] = 1 / inv_sum if inv_sum > 0 else 0
    result['win'] = win_odds

    # --- 複勝の合成オッズ ---
    # 例: 2の複勝 = 2が1着or2着の全3連単の合成オッズ
    place_odds = {}
    for w in wakus:
        inv_sum = 0
        for w1 in wakus:
            for w2 in wakus:
                if w1 == w2:
                    continue
                for w3 in wakus:
                    if w3 == w1 or w3 == w2:
                        continue
                    if w1 == w or w2 == w:
                        key = f"{w1}-{w2}-{w3}"
                        odds = trifecta_odds.get(key, 0)
                        if odds > 0:
                            inv_sum += 1 / odds
        place_odds[str(w)] = 1 / inv_sum if inv_sum > 0 else 0
    result['place'] = place_odds

    # --- 2連単の合成オッズ ---
    # 例: 1-2 = 1着1号,2着2号の全3連単の合成オッズ
    exacta_odds = {}
    for perm in permutations(wakus, 2):
        w1, w2 = perm
        inv_sum = 0
        for w3 in wakus:
            if w3 == w1 or w3 == w2:
                continue
            key = f"{w1}-{w2}-{w3}"
            odds = trifecta_odds.get(key, 0)
            if odds > 0:
                inv_sum += 1 / odds
        exacta_odds[f"{w1}-{w2}"] = 1 / inv_sum if inv_sum > 0 else 0
    result['exacta'] = exacta_odds

    # --- 2連複の合成オッズ ---
    # 例: 1=2 = 1,2が1-2着（順不同）の全3連単の合成オッズ
    quinella_odds = {}
    for comb in combinations(wakus, 2):
        w1, w2 = sorted(comb)
        inv_sum = 0
        for w3 in wakus:
            if w3 == w1 or w3 == w2:
                continue
            for key in [f"{w1}-{w2}-{w3}", f"{w2}-{w1}-{w3}"]:
                odds = trifecta_odds.get(key, 0)
                if odds > 0:
                    inv_sum += 1 / odds
        quinella_odds[f"{w1}={w2}"] = 1 / inv_sum if inv_sum > 0 else 0
    result['quinella'] = quinella_odds

    # --- 3連複の合成オッズ ---
    # 例: 1=2=3 = 1,2,3が1-3着（順不同）の全3連単の合成オッズ
    trio_odds = {}
    for comb in combinations(wakus, 3):
        w1, w2, w3 = sorted(comb)
        inv_sum = 0
        for a, b, c in permutations(comb):
            key = f"{a}-{b}-{c}"
            odds = trifecta_odds.get(key, 0)
            if odds > 0:
                inv_sum += 1 / odds
        trio_odds[f"{w1}={w2}={w3}"] = 1 / inv_sum if inv_sum > 0 else 0
    result['trio'] = trio_odds

    # --- 3連単の合成オッズ（そのまま）---
    result['trifecta'] = trifecta_odds

    return result


# ============================================
# 期待値計算
# ============================================
def calc_expected_values(probs, synthetic_odds):
    """確率 × 合成オッズ = 期待値"""
    ev = {}
    for bet_type in ['win', 'place', 'exacta', 'quinella', 'trifecta', 'trio']:
        ev[bet_type] = {}
        prob_dict = probs.get(bet_type, {})
        odds_dict = synthetic_odds.get(bet_type, {})
        for key in prob_dict:
            p = prob_dict[key]
            o = odds_dict.get(key, 0)
            ev[bet_type][key] = p * o if o > 0 else 0
    return ev


# ============================================
# メイン
# ============================================
def main():
    st.title("🚤 競艇AI予想 v9")
    st.caption("1着・2連対・3連対 LightGBM × 全体Plattキャリブレーション | 全券種確率＋期待値")

    try:
        models, df_racer = load_models()
        features = models['1着']['features']
    except Exception as e:
        st.error(f"モデルロードエラー: {e}")
        st.info("必要ファイル: boatrace_model_1着_v9.pkl, "
                "boatrace_model_2連対_v9.pkl, "
                "boatrace_model_3連対_v9.pkl, "
                "racer_course_data.csv")
        return

    st.sidebar.header("🎯 レース選択")
    place = st.sidebar.selectbox("場所", list(PLACE_CODES.keys()), index=15)
    race_num = st.sidebar.selectbox("レース番号", list(range(1, 13)))
    from datetime import date
    race_date = st.sidebar.date_input("日付", value=date.today())

    st.sidebar.header("⚙️ 表示設定")
    top_n_3t = st.sidebar.slider("3連単 表示数", 5, 30, 20)
    top_n_2t = st.sidebar.slider("2連単 表示数", 5, 15, 10)
    top_n_3f = st.sidebar.slider("3連複 表示数", 5, 15, 10)

    jcd = PLACE_CODES[place]
    hd = race_date.strftime('%Y%m%d')

    if st.sidebar.button("🎯 予想する", type="primary", use_container_width=True):
        # 出走表取得
        with st.spinner("📋 出走表取得中..."):
            boats = fetch_race_data(jcd, hd, str(race_num))
        if len(boats) < 6:
            st.error("❌ 出走表の取得に失敗しました。")
            return

        # 直前情報取得
        with st.spinner("📋 直前情報取得中..."):
            before_info = fetch_beforeinfo(jcd, hd, str(race_num))

        # 3連単オッズ取得
        with st.spinner("📋 3連単オッズ取得中..."):
            trifecta_odds_raw = fetch_trifecta_odds_v2(jcd, hd, str(race_num))
            odds_count = len(trifecta_odds_raw)

        et_count = sum(1 for k in before_info if k.startswith('et_'))

        st.header(f"📋 {place} {race_num}R ({race_date})")

        if et_count < 6:
            st.warning(f"⚠️ 展示タイム未取得（{et_count}/6艇）。直前情報公開前の可能性があります。")
        if odds_count < 100:
            st.warning(f"⚠️ 3連単オッズ取得: {odds_count}/120通り。オッズ未発表の可能性があります。")

        # 出走表表示
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
                'ボート2率': b.get('boat_2rate', 0),
                '展示T': before_info.get(f'et_{w}', '-'),
                'ST': before_info.get(f'st_{w}', '-'),
            })
        st.dataframe(pd.DataFrame(entry_data), use_container_width=True, hide_index=True)

        # AI予測
        with st.spinner("🔧 AI予測計算中..."):
            X = build_features(boats, features, before_info, df_racer)
            results = predict_race(X, [b['waku'] for b in boats], models)
            all_probs = calc_all_combinations(results)

            # 合成オッズ＆期待値
            if odds_count >= 100:
                synthetic_odds = calc_synthetic_odds(trifecta_odds_raw)
                expected_values = calc_expected_values(all_probs, synthetic_odds)
                has_odds = True
            else:
                has_odds = False

        # ==========================================
        # 着順別確率テーブル
        # ==========================================
        st.header("🎯 着順別確率")
        prob_data = []
        for _, row in results.iterrows():
            w = int(row['waku'])
            name = boats[w - 1].get('name', '?')
            d = {
                '枠': f"{WAKU_COLORS.get(w, '')} {w}",
                '名前': name,
                '1着率': f"{row['p_1着']:.1%}",
                '2着率': f"{row['p_2着']:.1%}",
                '3着率': f"{row['p_3着']:.1%}",
                '2連対率': f"{row['p_2連対']:.1%}",
                '3連対率': f"{row['p_3連対']:.1%}",
            }
            prob_data.append(d)
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

        # ==========================================
        # 単勝・複勝
        # ==========================================
        st.header("🏆 単勝・複勝")
        col_w, col_p = st.columns(2)

        with col_w:
            st.subheader("単勝")
            win_data = []
            for w in sorted(all_probs['win'].keys(), key=lambda x: -all_probs['win'][x]):
                name = boats[int(w) - 1].get('name', '?')
                d = {
                    '枠': f"{WAKU_COLORS.get(int(w), '')} {w}",
                    '名前': name,
                    '確率': f"{all_probs['win'][w]:.1%}",
                }
                if has_odds:
                    odds_val = synthetic_odds['win'].get(w, 0)
                    ev_val = expected_values['win'].get(w, 0)
                    d['合成オッズ'] = f"{odds_val:.1f}" if odds_val > 0 else '-'
                    d['期待値'] = f"{ev_val:.2f}"
                    d['判定'] = '🔥' if ev_val >= 1.2 else ('✅' if ev_val >= 1.0 else '❌')
                win_data.append(d)
            st.dataframe(pd.DataFrame(win_data), use_container_width=True, hide_index=True)

        with col_p:
            st.subheader("複勝")
            place_data = []
            for w in sorted(all_probs['place'].keys(), key=lambda x: -all_probs['place'][x]):
                name = boats[int(w) - 1].get('name', '?')
                d = {
                    '枠': f"{WAKU_COLORS.get(int(w), '')} {w}",
                    '名前': name,
                    '確率': f"{all_probs['place'][w]:.1%}",
                }
                if has_odds:
                    odds_val = synthetic_odds['place'].get(w, 0)
                    ev_val = expected_values['place'].get(w, 0)
                    d['合成オッズ'] = f"{odds_val:.1f}" if odds_val > 0 else '-'
                    d['期待値'] = f"{ev_val:.2f}"
                    d['判定'] = '🔥' if ev_val >= 1.2 else ('✅' if ev_val >= 1.0 else '❌')
                place_data.append(d)
            st.dataframe(pd.DataFrame(place_data), use_container_width=True, hide_index=True)

        # ==========================================
        # 2連単・2連複
        # ==========================================
        st.header("🥈 2連単・2連複")
        col_e, col_q = st.columns(2)

        with col_e:
            st.subheader("2連単")
            sorted_exacta = sorted(all_probs['exacta'].items(), key=lambda x: -x[1])
            ex_data = []
            for i, (combo, prob) in enumerate(sorted_exacta[:top_n_2t], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    odds_val = synthetic_odds['exacta'].get(combo, 0)
                    ev_val = expected_values['exacta'].get(combo, 0)
                    d['合成オッズ'] = f"{odds_val:.1f}" if odds_val > 0 else '-'
                    d['期待値'] = f"{ev_val:.2f}"
                    d['判定'] = '🔥' if ev_val >= 1.2 else ('✅' if ev_val >= 1.0 else '❌')
                ex_data.append(d)
            st.dataframe(pd.DataFrame(ex_data), use_container_width=True, hide_index=True)

        with col_q:
            st.subheader("2連複")
            sorted_quinella = sorted(all_probs['quinella'].items(), key=lambda x: -x[1])
            q_data = []
            for i, (combo, prob) in enumerate(sorted_quinella[:top_n_2t], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    odds_val = synthetic_odds['quinella'].get(combo, 0)
                    ev_val = expected_values['quinella'].get(combo, 0)
                    d['合成オッズ'] = f"{odds_val:.1f}" if odds_val > 0 else '-'
                    d['期待値'] = f"{ev_val:.2f}"
                    d['判定'] = '🔥' if ev_val >= 1.2 else ('✅' if ev_val >= 1.0 else '❌')
                q_data.append(d)
            st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)

        # ==========================================
        # 3連単・3連複
        # ==========================================
        st.header("🥇 3連単・3連複")

        # 信頼度表示
        sorted_3t = sorted(all_probs['trifecta'].items(), key=lambda x: -x[1])
        top1_prob = sorted_3t[0][1] if sorted_3t else 0
        if top1_prob >= 0.15:
            st.success(f"🔥 高確信レース！ TOP1確率: {top1_prob:.1%}")
        elif top1_prob >= 0.10:
            st.info(f"✅ 有望レース TOP1確率: {top1_prob:.1%}")
        elif top1_prob >= 0.08:
            st.warning(f"⚠️ やや不確実 TOP1確率: {top1_prob:.1%}")
        else:
            st.error(f"❌ 荒れ予想 TOP1確率: {top1_prob:.1%}")

        col_3t, col_3f = st.columns(2)

        with col_3t:
            st.subheader("3連単")
            data_3t = []
            for i, (combo, prob) in enumerate(sorted_3t[:top_n_3t], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    odds_val = trifecta_odds_raw.get(combo, 0)
                    ev_val = prob * odds_val if odds_val > 0 else 0
                    d['オッズ'] = f"{odds_val:.1f}" if odds_val > 0 else '-'
                    d['期待値'] = f"{ev_val:.2f}"
                    d['判定'] = '🔥' if ev_val >= 1.2 else ('✅' if ev_val >= 1.0 else '❌')
                data_3t.append(d)
            st.dataframe(pd.DataFrame(data_3t), use_container_width=True, hide_index=True)

        with col_3f:
            st.subheader("3連複")
            sorted_3f = sorted(all_probs['trio'].items(), key=lambda x: -x[1])
            data_3f = []
            for i, (combo, prob) in enumerate(sorted_3f[:top_n_3f], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    odds_val = synthetic_odds['trio'].get(combo, 0)
                    ev_val = expected_values['trio'].get(combo, 0)
                    d['合成オッズ'] = f"{odds_val:.1f}" if odds_val > 0 else '-'
                    d['期待値'] = f"{ev_val:.2f}"
                    d['判定'] = '🔥' if ev_val >= 1.2 else ('✅' if ev_val >= 1.0 else '❌')
                data_3f.append(d)
            st.dataframe(pd.DataFrame(data_3f), use_container_width=True, hide_index=True)

        # ==========================================
        # 期待値ランキング（全券種横断）
        # ==========================================
        if has_odds:
            st.header("💰 期待値ランキング（全券種横断 TOP20）")
            all_ev = []
            bet_type_labels = {
                'win': '単勝', 'place': '複勝', 'exacta': '2連単',
                'quinella': '2連複', 'trifecta': '3連単', 'trio': '3連複'
            }
            for bet_type, label in bet_type_labels.items():
                ev_dict = expected_values.get(bet_type, {})
                prob_dict = all_probs.get(bet_type, {})
                if bet_type == 'trifecta':
                    odds_dict = trifecta_odds_raw
                else:
                    odds_dict = synthetic_odds.get(bet_type, {})
                for key, ev_val in ev_dict.items():
                    if ev_val > 0:
                        all_ev.append({
                            '券種': label,
                            '組み合わせ': key,
                            '確率': f"{prob_dict.get(key, 0):.2%}",
                            'オッズ': f"{odds_dict.get(key, 0):.1f}",
                            '期待値': ev_val,
                        })

            all_ev.sort(key=lambda x: -x['期待値'])
            top_ev = all_ev[:20]
            for item in top_ev:
                item['判定'] = '🔥' if item['期待値'] >= 1.2 else ('✅' if item['期待値'] >= 1.0 else '❌')
                item['期待値'] = f"{item['期待値']:.2f}"
            st.dataframe(pd.DataFrame(top_ev), use_container_width=True, hide_index=True)

        st.divider()
        st.caption(
            f"📊 モデル: LightGBM v9 (1着/2連対/3連対) × 全体Platt | "
            f"特徴量: {len(features)}個 | "
            f"バックテスト: 9,847レース TOP1的中率 9.8% | "
            f"オッズ: 3連単オッズから全券種合成"
        )


if __name__ == '__main__':
    main()
