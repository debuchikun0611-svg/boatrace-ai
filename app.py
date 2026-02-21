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


@st.cache_resource
def load_models():
    base = './'
    models = {}
    for name in ['1着', '2連対', '3連対']:
        with open(base + f'boatrace_model_{name}_v9.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    df_racer = pd.read_csv(base + 'racer_course_data.csv')
    return models, df_racer


def fetch_race_data(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={hd}"
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    boats = []
    toban_links = soup.select('a[href*="toban"]')
    tobans = []
    for a in toban_links:
        m = re.search(r'toban=(\\d+)', a.get('href', ''))
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
        age_match = re.search(r'(\\d{2})歳', full_text)
        if age_match: boat['age'] = int(age_match.group(1))
        weight_match = re.search(r'([\\d\\.]+)kg', full_text)
        if weight_match: boat['weight'] = float(weight_match.group(1))
        line_tds = tbody.select('td.is-lineH2')
        if len(line_tds) >= 5:
            pat = r'(\\d{1,2}\\.\\d{2})'
            st_text = line_tds[0].get_text(strip=True)
            st_match = re.search(r'(\\d+\\.\\d+)$', st_text)
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
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    info = {}
    main_table = soup.select_one('table.is-w748')
    if main_table:
        for tr in main_table.select('tr'):
            boat_color = tr.select_one('td[class*="is-boatColor"]')
            if boat_color:
                tds = tr.select('td')
                try: waku = int(boat_color.get_text(strip=True))
                except: continue
                if len(tds) >= 5:
                    try:
                        et_val = float(tds[4].get_text(strip=True))
                        if 5.5 <= et_val <= 8.5: info[f'et_{waku}'] = et_val
                    except: pass
    st_table = soup.select_one('table.is-w238')
    if st_table:
        for tr in st_table.select('tr'):
            tds = tr.select('td')
            if len(tds) >= 1:
                txt = tds[0].get_text(strip=True)
                st_match = re.match(r'^(\\d)(F?)(\\.?\\d{2})$', txt)
                if st_match:
                    course = int(st_match.group(1))
                    is_flying = st_match.group(2) == 'F'
                    st_digits = st_match.group(3)
                    st_val = float('0' + st_digits) if st_digits.startswith('.') else float('0.' + st_digits)
                    if is_flying: st_val = -st_val
                    info[f'st_{course}'] = st_val
    return info


def fetch_trifecta_odds(jcd, hd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={hd}"
    resp = requests.get(url, timeout=15)
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
            boat_class = [c for c in ths[ci*2].get('class', []) if c.startswith('is-boatColor')]
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
                try: odds_val = float(odds_text)
                except: odds_val = 0
                first = col_first.get(col_idx, 0)
                if first > 0:
                    odds_dict[f"{first}-{second}-{third}"] = odds_val
                td_idx += 3
                col_idx += 1
            else:
                third = int(td.get_text(strip=True))
                odds_text = tds[td_idx + 1].get_text(strip=True).replace(',', '')
                try: odds_val = float(odds_text)
                except: odds_val = 0
                first = col_first.get(col_idx, 0)
                second = col_second.get(col_idx, 0)
                if first > 0 and second > 0:
                    odds_dict[f"{first}-{second}-{third}"] = odds_val
                td_idx += 2
                col_idx += 1
    return odds_dict


def build_features(boats, features, before_info, df_racer):
    rows = []
    n = len(boats)
    et_list = [before_info.get(f'et_{i+1}', 0) for i in range(n)]
    et_mean = np.mean([v for v in et_list if v > 0]) if any(v > 0 for v in et_list) else 6.8
    for boat in boats:
        waku = boat.get('waku', 0)
        toban = boat.get('toban', 0)
        row = {
            'waku': waku, 'age': boat.get('age', 35), 'weight': boat.get('weight', 52),
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
            if pd.isna(row[k]): row[k] = 0
        rows.append(row)
    df = pd.DataFrame(rows)
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
    df['waku_win_hist'] = df['waku'].map({1:0.55,2:0.14,3:0.12,4:0.10,5:0.06,6:0.03})
    df['motor_rank_x_waku'] = df['motor_2rate_rank'] * df['waku']
    df['waku_x_winrate'] = df['waku'] * df['national_win_rate']
    df['winrate_x_grade'] = df['national_win_rate'] * df['grade_num']
    df['grade_vs_race'] = df['grade_num'] - df['grade_num'].mean()
    df['win_rate_product'] = df['national_win_rate'] * df['national_2rate']
    df['race_grade_level'] = df['grade_num'].mean()
    df['vs_race_max'] = df['national_win_rate'] - df['national_win_rate'].max()
    waku1 = df[df['waku']==1].iloc[0] if len(df[df['waku']==1]) > 0 else df.iloc[0]
    df['waku1_win_rate'] = waku1['national_win_rate']
    df['vs_waku1'] = df['national_win_rate'] - waku1['national_win_rate']
    df['et_rank'] = df['exhibition_time'].rank()
    df['et_diff'] = df['exhibition_time'] - df['exhibition_time'].mean()
    df['et_best_diff'] = df['exhibition_time'] - df['exhibition_time'].min()
    df['et_waku'] = df['exhibition_time'] * df['waku']
    df['st_rank'] = df['st_time'].rank()
    df['st_diff'] = df['st_time'] - df['st_time'].mean()
    df['st_best_diff'] = df['st_time'] - df['st_time'].min()
    df['st_waku'] = df['st_time'] * df['waku']
    df['et_st_combined'] = df['et_rank'] + df['st_rank']
    for col in ['course_entry_rate', 'course_win3_rate', 'course_avg_st']:
        df[f'{col}_vs_avg'] = df[col] - df[col].mean()
    for f in features:
        if f not in df.columns: df[f] = 0
    return df[features]


def predict_race(X, wakus, models):
    results = pd.DataFrame({'waku': wakus})
    for target_name, target_sum in [('1着', 1.0), ('2連対', 2.0), ('3連対', 3.0)]:
        md = models[target_name]
        raw = md['model'].predict(X)
        platt = md['platt'].predict_proba(raw.reshape(-1, 1))[:, 1]
        s = platt.sum()
        normed = platt / s * target_sum if s > 0 else np.full(len(platt), target_sum / 6)
        results[f'p_{target_name}'] = normed
    results['p_2連対'] = results[['p_1着', 'p_2連対']].max(axis=1)
    results['p_3連対'] = results[['p_2連対', 'p_3連対']].max(axis=1)
    for name, ts in [('2連対', 2.0), ('3連対', 3.0)]:
        s = results[f'p_{name}'].sum()
        if s > 0: results[f'p_{name}'] = results[f'p_{name}'] / s * ts
    results['p_2着'] = (results['p_2連対'] - results['p_1着']).clip(lower=0)
    results['p_3着'] = (results['p_3連対'] - results['p_2連対']).clip(lower=0)
    return results


def calc_trifecta_probs(results):
    wakus = results['waku'].values
    p1 = dict(zip(wakus, results['p_1着'].values))
    p12 = dict(zip(wakus, results['p_2連対'].values))
    p123 = dict(zip(wakus, results['p_3連対'].values))
    trifecta = {}
    for perm in permutations(wakus, 3):
        w1, w2, w3 = perm
        p_w1 = p1[w1]
        remaining2 = [w for w in wakus if w != w1]
        s2 = sum(p12[w] for w in remaining2)
        p_w2 = p12[w2] / s2 if s2 > 0 else 1/5
        remaining3 = [w for w in wakus if w != w1 and w != w2]
        s3 = sum(p123[w] for w in remaining3)
        p_w3 = p123[w3] / s3 if s3 > 0 else 1/4
        trifecta[f"{w1}-{w2}-{w3}"] = p_w1 * p_w2 * p_w3
    tp = sum(trifecta.values())
    if tp > 0:
        trifecta = {k: v/tp for k, v in trifecta.items()}
    return trifecta


def derive_all_probs(trifecta, results):
    wakus = results['waku'].values
    win = {}
    for _, row in results.iterrows():
        win[str(int(row['waku']))] = row['p_1着']
    place = {}
    for _, row in results.iterrows():
        place[str(int(row['waku']))] = row['p_2連対'] / 2
    exacta = {}
    for perm in permutations(wakus, 2):
        w1, w2 = perm
        key = f"{w1}-{w2}"
        exacta[key] = sum(trifecta.get(f"{w1}-{w2}-{w3}", 0)
                          for w3 in wakus if w3 != w1 and w3 != w2)
    quinella = {}
    for comb in combinations(sorted(wakus), 2):
        w1, w2 = comb
        key = f"{w1}={w2}"
        quinella[key] = exacta.get(f"{w1}-{w2}", 0) + exacta.get(f"{w2}-{w1}", 0)
    trio = {}
    for comb in combinations(sorted(wakus), 3):
        key = "=".join(map(str, comb))
        trio[key] = sum(trifecta.get(f"{a}-{b}-{c}", 0) for a, b, c in permutations(comb))
    return {'win': win, 'place': place, 'exacta': exacta,
            'quinella': quinella, 'trifecta': trifecta, 'trio': trio}


def calc_synthetic_odds(trifecta_odds):
    wakus = list(range(1, 7))
    result = {}
    win_odds = {}
    for w in wakus:
        inv_sum = sum(1/trifecta_odds[f"{w}-{w2}-{w3}"]
                      for w2 in wakus if w2 != w
                      for w3 in wakus if w3 != w and w3 != w2
                      if trifecta_odds.get(f"{w}-{w2}-{w3}", 0) > 0)
        win_odds[str(w)] = 1/inv_sum if inv_sum > 0 else 0
    result['win'] = win_odds
    place_odds = {}
    for w in wakus:
        inv_sum = 0
        for w1 in wakus:
            for w2 in wakus:
                if w1 == w2: continue
                if w1 != w and w2 != w: continue
                for w3 in wakus:
                    if w3 == w1 or w3 == w2: continue
                    o = trifecta_odds.get(f"{w1}-{w2}-{w3}", 0)
                    if o > 0: inv_sum += 1/o
        place_odds[str(w)] = 1/inv_sum if inv_sum > 0 else 0
    result['place'] = place_odds
    exacta_odds = {}
    for w1 in wakus:
        for w2 in wakus:
            if w1 == w2: continue
            inv_sum = sum(1/trifecta_odds[f"{w1}-{w2}-{w3}"]
                          for w3 in wakus if w3 != w1 and w3 != w2
                          if trifecta_odds.get(f"{w1}-{w2}-{w3}", 0) > 0)
            exacta_odds[f"{w1}-{w2}"] = 1/inv_sum if inv_sum > 0 else 0
    result['exacta'] = exacta_odds
    quinella_odds = {}
    for comb in combinations(wakus, 2):
        w1, w2 = sorted(comb)
        inv_sum = 0
        for w3 in wakus:
            if w3 == w1 or w3 == w2: continue
            for key in [f"{w1}-{w2}-{w3}", f"{w2}-{w1}-{w3}"]:
                o = trifecta_odds.get(key, 0)
                if o > 0: inv_sum += 1/o
        quinella_odds[f"{w1}={w2}"] = 1/inv_sum if inv_sum > 0 else 0
    result['quinella'] = quinella_odds
    trio_odds = {}
    for comb in combinations(wakus, 3):
        w1, w2, w3 = sorted(comb)
        inv_sum = sum(1/trifecta_odds.get(f"{a}-{b}-{c}", 0)
                      for a, b, c in permutations(comb)
                      if trifecta_odds.get(f"{a}-{b}-{c}", 0) > 0)
        trio_odds[f"{w1}={w2}={w3}"] = 1/inv_sum if inv_sum > 0 else 0
    result['trio'] = trio_odds
    result['trifecta'] = trifecta_odds
    return result


def main():
    st.title("🚤 競艇AI予想 v9")
    st.caption("1着・2連対・3連対 LightGBM × 全体Platt | 全券種確率＋3連単オッズ合成期待値")
    try:
        models, df_racer = load_models()
        features = models['1着']['features']
    except Exception as e:
        st.error(f"モデルロードエラー: {e}")
        st.info("必要ファイル: boatrace_model_1着_v9.pkl, boatrace_model_2連対_v9.pkl, "
                "boatrace_model_3連対_v9.pkl, racer_course_data.csv")
        return
    st.sidebar.header("🎯 レース選択")
    place = st.sidebar.selectbox("場所", list(PLACE_CODES.keys()), index=15)
    race_num = st.sidebar.selectbox("レース番号", list(range(1, 13)))
    from datetime import date
    race_date = st.sidebar.date_input("日付", value=date.today())
    st.sidebar.header("⚙️ 表示設定")
    top_n = st.sidebar.slider("各券種 表示数", 5, 30, 15)
    jcd = PLACE_CODES[place]
    hd = race_date.strftime('%Y%m%d')
    if st.sidebar.button("🎯 予想する", type="primary", use_container_width=True):
        with st.spinner("📋 出走表取得中..."):
            boats = fetch_race_data(jcd, hd, str(race_num))
        if len(boats) < 6:
            st.error("❌ 出走表の取得に失敗しました。")
            return
        with st.spinner("📋 直前情報取得中..."):
            before_info = fetch_beforeinfo(jcd, hd, str(race_num))
        with st.spinner("📋 3連単オッズ取得中..."):
            trifecta_odds_raw = fetch_trifecta_odds(jcd, hd, str(race_num))
            odds_count = len(trifecta_odds_raw)
            has_odds = odds_count >= 100
            if has_odds:
                synthetic_odds = calc_synthetic_odds(trifecta_odds_raw)
        et_count = sum(1 for k in before_info if k.startswith('et_'))
        st.header(f"📋 {place} {race_num}R ({race_date})")
        if et_count < 6:
            st.warning(f"⚠️ 展示タイム未取得（{et_count}/6艇）")
        if not has_odds:
            st.warning(f"⚠️ 3連単オッズ: {odds_count}/120通り取得")
        entry_data = []
        for b in boats:
            w = b['waku']
            entry_data.append({
                '枠': f"{WAKU_COLORS.get(w,'')} {w}", '登番': b.get('toban','?'),
                '名前': b.get('name','?'),
                '級別': f"{GRADE_COLORS.get(b.get('grade',''),'')} {b.get('grade','?')}",
                '全国勝率': b.get('national_win_rate',0), '全国2率': b.get('national_2rate',0),
                'モーター2率': b.get('motor_2rate',0), 'ボート2率': b.get('boat_2rate',0),
                '展示T': before_info.get(f'et_{w}','-'), 'ST': before_info.get(f'st_{w}','-'),
            })
        st.dataframe(pd.DataFrame(entry_data), use_container_width=True, hide_index=True)
        with st.spinner("🔧 AI予測計算中..."):
            X = build_features(boats, features, before_info, df_racer)
            results = predict_race(X, [b['waku'] for b in boats], models)
            trifecta = calc_trifecta_probs(results)
            all_probs = derive_all_probs(trifecta, results)
        st.header("🎯 着順別確率・単勝・複勝")
        main_data = []
        for _, row in results.iterrows():
            w = int(row['waku'])
            name = boats[w-1].get('name', '?')
            d = {'枠': f"{WAKU_COLORS.get(w,'')} {w}", '名前': name,
                 '単勝(=1着率)': f"{row['p_1着']:.1%}"}
            if has_odds:
                wo = synthetic_odds['win'].get(str(w), 0)
                ev_w = row['p_1着'] * wo if wo > 0 else 0
                d['単勝合成ｵｯｽﾞ'] = f"{wo:.1f}" if wo > 0 else '-'
                d['単勝期待値'] = f"{ev_w:.2f}"
            d['複勝(=2連対率)'] = f"{row['p_2連対']/2:.1%}"
            if has_odds:
                po = synthetic_odds['place'].get(str(w), 0)
                ev_p = (row['p_2連対']/2) * po if po > 0 else 0
                d['複勝合成ｵｯｽﾞ'] = f"{po:.1f}" if po > 0 else '-'
                d['複勝期待値'] = f"{ev_p:.2f}"
            d['2着率'] = f"{row['p_2着']:.1%}"
            d['3着率'] = f"{row['p_3着']:.1%}"
            d['3連対率'] = f"{row['p_3連対']:.1%}"
            main_data.append(d)
        st.dataframe(pd.DataFrame(main_data), use_container_width=True, hide_index=True)
        st.header("🥈 2連単・2連複")
        col_e, col_q = st.columns(2)
        with col_e:
            st.subheader("2連単")
            sorted_ex = sorted(all_probs['exacta'].items(), key=lambda x: -x[1])
            ex_data = []
            for i, (combo, prob) in enumerate(sorted_ex[:top_n], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    o = synthetic_odds['exacta'].get(combo, 0)
                    ev = prob * o if o > 0 else 0
                    d['合成ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                    d['期待値'] = f"{ev:.2f}"
                    d[''] = '🔥' if ev >= 1.2 else ('✅' if ev >= 1.0 else '')
                ex_data.append(d)
            st.dataframe(pd.DataFrame(ex_data), use_container_width=True, hide_index=True)
        with col_q:
            st.subheader("2連複")
            sorted_q = sorted(all_probs['quinella'].items(), key=lambda x: -x[1])
            q_data = []
            for i, (combo, prob) in enumerate(sorted_q[:top_n], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    o = synthetic_odds['quinella'].get(combo, 0)
                    ev = prob * o if o > 0 else 0
                    d['合成ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                    d['期待値'] = f"{ev:.2f}"
                    d[''] = '🔥' if ev >= 1.2 else ('✅' if ev >= 1.0 else '')
                q_data.append(d)
            st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)
        st.header("🥇 3連単・3連複")
        sorted_3t = sorted(trifecta.items(), key=lambda x: -x[1])
        top1_prob = sorted_3t[0][1] if sorted_3t else 0
        if top1_prob >= 0.15: st.success(f"🔥 高確信レース！ TOP1確率: {top1_prob:.1%}")
        elif top1_prob >= 0.10: st.info(f"✅ 有望レース TOP1確率: {top1_prob:.1%}")
        elif top1_prob >= 0.08: st.warning(f"⚠️ やや不確実 TOP1確率: {top1_prob:.1%}")
        else: st.error(f"❌ 荒れ予想 TOP1確率: {top1_prob:.1%}")
        col_3t, col_3f = st.columns(2)
        with col_3t:
            st.subheader("3連単")
            data_3t = []
            for i, (combo, prob) in enumerate(sorted_3t[:top_n], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    o = trifecta_odds_raw.get(combo, 0)
                    ev = prob * o if o > 0 else 0
                    d['ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                    d['期待値'] = f"{ev:.2f}"
                    d[''] = '🔥' if ev >= 1.2 else ('✅' if ev >= 1.0 else '')
                data_3t.append(d)
            st.dataframe(pd.DataFrame(data_3t), use_container_width=True, hide_index=True)
        with col_3f:
            st.subheader("3連複")
            sorted_3f = sorted(all_probs['trio'].items(), key=lambda x: -x[1])
            data_3f = []
            for i, (combo, prob) in enumerate(sorted_3f[:top_n], 1):
                d = {'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"}
                if has_odds:
                    o = synthetic_odds['trio'].get(combo, 0)
                    ev = prob * o if o > 0 else 0
                    d['合成ｵｯｽﾞ'] = f"{o:.1f}" if o > 0 else '-'
                    d['期待値'] = f"{ev:.2f}"
                    d[''] = '🔥' if ev >= 1.2 else ('✅' if ev >= 1.0 else '')
                data_3f.append(d)
            st.dataframe(pd.DataFrame(data_3f), use_container_width=True, hide_index=True)
        if has_odds:
            st.header("💰 期待値ランキング TOP20")
            all_ev = []
            bet_labels = {'win':'単勝','place':'複勝','exacta':'2連単',
                          'quinella':'2連複','trifecta':'3連単','trio':'3連複'}
            for bt, label in bet_labels.items():
                prob_dict = all_probs[bt]
                odds_dict = trifecta_odds_raw if bt == 'trifecta' else synthetic_odds[bt]
                for key, prob in prob_dict.items():
                    o = odds_dict.get(key, 0)
                    if o > 0 and prob > 0:
                        ev = prob * o
                        all_ev.append({'券種': label, '組み合わせ': key,
                                       '確率': f"{prob:.2%}", 'ｵｯｽﾞ': f"{o:.1f}",
                                       '期待値': ev})
            all_ev.sort(key=lambda x: -x['期待値'])
            for item in all_ev[:20]:
                item[''] = '🔥' if item['期待値'] >= 1.2 else ('✅' if item['期待値'] >= 1.0 else '')
                item['期待値'] = f"{item['期待値']:.2f}"
            st.dataframe(pd.DataFrame(all_ev[:20]), use_container_width=True, hide_index=True)
        st.divider()
        st.caption(f"📊 モデル: LightGBM v9 (1着/2連対/3連対) x 全体Platt | "
                   f"特徴量: {len(features)}個 | バックテスト: 9,847R TOP1的中率9.8% | "
                   f"3連単オッズ: {odds_count}/120通り取得")

if __name__ == '__main__':
    main()
