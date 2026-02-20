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

st.set_page_config(page_title="競艇AI予想", page_icon="🚤", layout="wide")

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
    with open(base + 'boatrace_model_1着_independent_full.pkl', 'rb') as f:
        m_1st = pickle.load(f)
    with open(base + 'boatrace_model_2着_independent_full.pkl', 'rb') as f:
        m_2nd = pickle.load(f)
    with open(base + 'boatrace_model_3着_independent_full.pkl', 'rb') as f:
        m_3rd = pickle.load(f)
    df_racer = pd.read_csv(base + 'racer_course_data.csv')
    return m_1st, m_2nd, m_3rd, df_racer

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
            def parse_rates(text):
                return re.findall(r'(\d{1,2}\.\d{2})', text)
            st_text = line_tds[0].get_text(strip=True)
            st_match = re.search(r'(\d+\.\d+)$', st_text)
            if st_match:
                boat['avg_st'] = float(st_match.group(1))
            nat_text = line_tds[1].get_text(strip=True)
            nat_match = re.match(r'(\d\.\d{2})(.*)', nat_text)
            if nat_match:
                boat['national_win_rate'] = float(nat_match.group(1))
                rest_nums = parse_rates(nat_match.group(2))
                if len(rest_nums) >= 1:
                    boat['national_2rate'] = float(rest_nums[0])
            loc_text = line_tds[2].get_text(strip=True)
            loc_match = re.match(r'(\d\.\d{2})(.*)', loc_text)
            if loc_match:
                boat['local_win_rate'] = float(loc_match.group(1))
                rest_nums = parse_rates(loc_match.group(2))
                if len(rest_nums) >= 1:
                    boat['local_2rate'] = float(rest_nums[0])
            else:
                loc_nums = parse_rates(loc_text)
                if len(loc_nums) >= 2:
                    boat['local_win_rate'] = float(loc_nums[0])
                    boat['local_2rate'] = float(loc_nums[1])
            motor_text = line_tds[3].get_text(strip=True)
            motor_nums = parse_rates(motor_text)
            if len(motor_nums) >= 1:
                boat['motor_2rate'] = float(motor_nums[0])
            boat_text = line_tds[4].get_text(strip=True)
            boat_nums = parse_rates(boat_text)
            if len(boat_nums) >= 1:
                boat['boat_2rate'] = float(boat_nums[0])
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

def build_features(boats, features, before_info, df_racer):
    rows = []
    n = len(boats)
    et_list = [before_info.get(f'et_{i+1}', 0) for i in range(n)]
    et_mean = np.mean(et_list) if any(v > 0 for v in et_list) else 6.8
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
            'is_waku1': 1 if waku == 1 else 0,
            'exhibition_time': before_info.get(f'et_{waku}', et_mean),
            'st_time': before_info.get(f'st_{waku}', 0.15),
            'waku_penalty': max(0, waku - 3) * 0.5,
        }
        racer_row = df_racer[df_racer['toban'] == toban]
        if len(racer_row) > 0:
            r = racer_row.iloc[0]
            for col_prefix, key in [('entry_rate', 'course_entry_rate'),
                                     ('win3_rate', 'course_win3_rate'),
                                     ('avg_st', 'course_avg_st'),
                                     ('start_order', 'course_start_order')]:
                col = f'{col_prefix}_{waku}'
                val = r.get(col, 0) if col in r.index else 0
                row[key] = val if not pd.isna(val) else 0
        else:
            row['course_entry_rate'] = 0
            row['course_win3_rate'] = 0
            row['course_avg_st'] = 0
            row['course_start_order'] = 0
        for k in row:
            if pd.isna(row[k]):
                row[k] = 0
        rows.append(row)
    df = pd.DataFrame(rows)
    df['is_waku12'] = df['waku'].apply(lambda x: 1 if x <= 2 else 0)
    df['is_waku56'] = df['waku'].apply(lambda x: 1 if x >= 5 else 0)
    for col_base, col in [('national_win_rate', 'national_win_rate_vs_avg'),
                           ('national_2rate', 'national_2rate_vs_avg'),
                           ('motor_2rate', 'motor_2rate_vs_avg'),
                           ('boat_2rate', 'boat_2rate_vs_avg')]:
        df[col] = df[col_base] - df[col_base].mean()
    df['et_rank'] = df['exhibition_time'].rank(ascending=True)
    df['et_vs_avg'] = df['exhibition_time'] - df['exhibition_time'].mean()
    df['et_vs_best'] = df['exhibition_time'] - df['exhibition_time'].min()
    df['et_waku'] = df['exhibition_time'] * df['waku']
    df['st_rank'] = df['st_time'].rank(ascending=True)
    df['st_vs_avg'] = df['st_time'] - df['st_time'].mean()
    df['st_waku'] = df['st_time'] * df['waku']
    df['et_st_combined'] = df['exhibition_time'] + df['st_time']
    df['in_win3_rate'] = df['course_win3_rate']
    df['in_entry_rate'] = df['course_entry_rate']
    df['avg_win3_all'] = df['course_win3_rate'].mean()
    df['course_win3_rate_vs_avg'] = df['course_win3_rate'] - df['course_win3_rate'].mean()
    df['course_avg_st_vs_avg'] = df['course_avg_st'] - df['course_avg_st'].mean()
    df['avg_win3_all_vs_avg'] = df['course_win3_rate'] - df['course_win3_rate'].mean()
    df['course_win3_rate_rank'] = df['course_win3_rate'].rank(ascending=False)
    df['class_num'] = df['grade_num']
    df['class_num_vs_avg'] = df['class_num'] - df['class_num'].mean()
    df['win3_x_waku'] = df['course_win3_rate'] * df['waku']
    df['st_x_win3'] = df['course_avg_st'] * df['course_win3_rate']
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]

def predict_race(X, wakus, m_1st, m_2nd, m_3rd):
    """独立3モデル（1着・2着・3着）で予測"""
    results = pd.DataFrame({'waku': wakus})
    
    for target, model_dict, col in [('1着', m_1st, 'p_1st'),
                                      ('2着', m_2nd, 'p_2nd'),
                                      ('3着', m_3rd, 'p_3rd')]:
        raw = model_dict['model'].predict(X)
        results[col] = 0.0
        for w in range(1, 7):
            mask = results['waku'] == w
            if mask.sum() == 0:
                continue
            results.loc[mask, col] = model_dict['platt_models'][w].predict_proba(
                raw[mask.values].reshape(-1, 1))[:, 1]
    
    # 各着順を正規化（合計=1）
    for col in ['p_1st', 'p_2nd', 'p_3rd']:
        s = results[col].sum()
        if s > 0:
            results[col] /= s
    
    # 3連対率 = 1着率 + 2着率 + 3着率
    results['p_top3'] = results['p_1st'] + results['p_2nd'] + results['p_3rd']
    
    return results

def calc_combinations(results):
    """独立3モデルに対応した組み合わせ確率計算"""
    wakus = results['waku'].values
    p1 = dict(zip(wakus, results['p_1st'].values))
    p2 = dict(zip(wakus, results['p_2nd'].values))
    p3 = dict(zip(wakus, results['p_3rd'].values))
    
    # 各着順を再正規化
    for d in [p1, p2, p3]:
        s = sum(d.values())
        if s > 0:
            for k in d:
                d[k] /= s
    
    # 3連単: 条件付き確率で計算
    trifecta = {}
    for perm in permutations(wakus, 3):
        w1, w2, w3 = perm
        prob_1 = p1[w1]
        rem2 = {k: v for k, v in p2.items() if k != w1}
        s2 = sum(rem2.values())
        prob_2 = rem2[w2] / s2 if s2 > 0 else 0
        rem3 = {k: v for k, v in p3.items() if k != w1 and k != w2}
        s3 = sum(rem3.values())
        prob_3 = rem3[w3] / s3 if s3 > 0 else 0
        trifecta[f"{w1}-{w2}-{w3}"] = prob_1 * prob_2 * prob_3
    total = sum(trifecta.values())
    if total > 0:
        trifecta = {k: v / total for k, v in trifecta.items()}
    
    # 2連単: 3連単から集約
    exacta = {}
    for perm in permutations(wakus, 2):
        w1, w2 = perm
        exacta[f"{w1}-{w2}"] = sum(trifecta.get(f"{w1}-{w2}-{w3}", 0)
                                   for w3 in wakus if w3 != w1 and w3 != w2)
    
    # 3連複: 3連単から集約
    trio = {}
    for comb in combinations(wakus, 3):
        key = "-".join(map(str, sorted(comb)))
        trio[key] = sum(trifecta.get(f"{a}-{b}-{c}", 0)
                       for a, b, c in permutations(comb))
    
    return trifecta, exacta, trio

def main():
    st.title("🚤 競艇AI予想")
    st.caption("独立3モデル体制（1着・2着・3着）× Plattキャリブレーション")
    try:
        m_1st, m_2nd, m_3rd, df_racer = load_models()
        features = m_1st['features']
    except Exception as e:
        st.error(f"モデルロードエラー: {e}")
        st.info("必要なファイル: boatrace_model_1着_independent_full.pkl, "
                "boatrace_model_2着_independent_full.pkl, "
                "boatrace_model_3着_independent_full.pkl, "
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
        with st.spinner("📋 出走表取得中..."):
            boats = fetch_race_data(jcd, hd, str(race_num))
        if len(boats) < 6:
            st.error("❌ 出走表の取得に失敗しました。日付・場所・レース番号を確認してください。")
            return
        with st.spinner("📋 直前情報取得中..."):
            before_info = fetch_beforeinfo(jcd, hd, str(race_num))
        et_count = sum(1 for k in before_info if k.startswith('et_'))
        st.header(f"📋 {place} {race_num}R ({race_date})")
        if et_count < 6:
            st.warning(f"⚠️ 展示タイム未取得（{et_count}/6艇）。直前情報公開前の可能性があります。")
        entry_data = []
        for b in boats:
            w = b['waku']
            entry_data.append({
                '枠': f"{WAKU_COLORS.get(w, '')} {w}",
                '登番': b.get('toban', '?'),
                '名前': b.get('name', '?'),
                '級別': f"{GRADE_COLORS.get(b.get('grade',''), '')} {b.get('grade', '?')}",
                '全国勝率': b.get('national_win_rate', 0),
                '全国2率': b.get('national_2rate', 0),
                'モーター2率': b.get('motor_2rate', 0),
                'ボート2率': b.get('boat_2rate', 0),
                '展示T': before_info.get(f'et_{w}', '-'),
                'ST': before_info.get(f'st_{w}', '-'),
            })
        st.dataframe(pd.DataFrame(entry_data), use_container_width=True, hide_index=True)
        with st.spinner("🔧 AI予測計算中..."):
            X = build_features(boats, features, before_info, df_racer)
            results = predict_race(X, [b['waku'] for b in boats], m_1st, m_2nd, m_3rd)
            trifecta, exacta, trio = calc_combinations(results)
        st.header("🎯 着順別確率")
        prob_data = []
        for _, row in results.iterrows():
            w = int(row['waku'])
            name = boats[w-1].get('name', '?')
            prob_data.append({
                '枠': f"{WAKU_COLORS.get(w, '')} {w}",
                '名前': name,
                '1着率': f"{row['p_1st']:.1%}",
                '2着率': f"{row['p_2nd']:.1%}",
                '3着率': f"{row['p_3rd']:.1%}",
                '3連対率': f"{row['p_top3']:.1%}",
            })
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)
        sorted_3t = sorted(trifecta.items(), key=lambda x: -x[1])
        top1_prob = sorted_3t[0][1]
        if top1_prob >= 0.15:
            st.success(f"🔥 高確信レース！ TOP1確率: {top1_prob:.1%}")
        elif top1_prob >= 0.10:
            st.info(f"✅ 有望レース TOP1確率: {top1_prob:.1%}")
        elif top1_prob >= 0.08:
            st.warning(f"⚠️ やや不確実 TOP1確率: {top1_prob:.1%}")
        else:
            st.error(f"❌ 荒れ予想 TOP1確率: {top1_prob:.1%}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🥇 3連単")
            data_3t = []
            for i, (combo, prob) in enumerate(sorted_3t[:top_n_3t], 1):
                data_3t.append({'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"})
            st.dataframe(pd.DataFrame(data_3t), use_container_width=True, hide_index=True)
        with col2:
            st.subheader("🥈 2連単")
            sorted_2t = sorted(exacta.items(), key=lambda x: -x[1])
            data_2t = []
            for i, (combo, prob) in enumerate(sorted_2t[:top_n_2t], 1):
                data_2t.append({'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"})
            st.dataframe(pd.DataFrame(data_2t), use_container_width=True, hide_index=True)
        with col3:
            st.subheader("🥉 3連複")
            sorted_3f = sorted(trio.items(), key=lambda x: -x[1])
            data_3f = []
            for i, (combo, prob) in enumerate(sorted_3f[:top_n_3f], 1):
                data_3f.append({'順位': i, '組み合わせ': combo, '確率': f"{prob:.2%}"})
            st.dataframe(pd.DataFrame(data_3f), use_container_width=True, hide_index=True)
        st.divider()
        st.caption("📊 モデル: LightGBM 独立3モデル (43特徴量) | "
                   "キャリブレーション: Platt (枠別) | "
                   f"バックテスト: 10,151レース 3連単TOP1≥10% ROI 141.3%")

if __name__ == '__main__':
    main()
