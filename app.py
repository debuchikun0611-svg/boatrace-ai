import streamlit as st
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import io, os, urllib.request, pathlib

# ============================================================
# フォント設定
# ============================================================
FONT_PATH = None

@st.cache_resource
def setup_font():
    global FONT_PATH
    font_dir = pathlib.Path("fonts")
    font_dir.mkdir(exist_ok=True)
    font_file = font_dir / "NotoSansJP-Regular.ttf"
    if not font_file.exists():
        for url in [
            "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
            "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
        ]:
            try:
                urllib.request.urlretrieve(url, str(font_file))
                break
            except:
                continue
    if font_file.exists():
        FONT_PATH = str(font_file)
        font_manager.fontManager.addfont(FONT_PATH)
        prop = font_manager.FontProperties(fname=FONT_PATH)
        matplotlib.rcParams['font.family'] = prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
    return FONT_PATH

FONT_PATH = setup_font()


# ============================================================
# エージェントクラス
# ============================================================

@dataclass
class BoatAgent:
    lane: int
    name: str
    rank: str
    win_rate: float
    avg_st: float
    st_stability: float
    lane_win_rate: float
    top2_rate: float
    top3_rate: float
    ability: int
    motor_contribution: float
    exhibition_time: float
    lap_time: float
    turn_time: float
    weight: float
    wins_nige: int = 0
    wins_sashi: int = 0
    wins_makuri: int = 0
    wins_makuri_sashi: int = 0
    wins_nuki: int = 0
    comment: str = ""
    actual_st: float = 0.0

    def calculate_start_timing(self) -> float:
        instability = (100 - self.st_stability) / 100.0
        noise_std = 0.015 + 0.05 * instability
        noise = np.random.normal(0, noise_std)
        self.actual_st = max(0.01, round(self.avg_st + noise, 2))
        return self.actual_st

    def get_power_score(self) -> float:
        wr = np.clip((self.win_rate - 2.0) / 6.0, 0, 1)
        ab = np.clip((self.ability - 38) / 20.0, 0, 1)
        mt = np.clip((self.motor_contribution + 1.0) / 2.0, 0, 1)
        ex = np.clip((7.05 - self.exhibition_time) / 0.20, 0, 1)
        lt = np.clip((38.5 - self.lap_time) / 2.0, 0, 1)
        tt = np.clip((12.0 - self.turn_time) / 0.8, 0, 1)
        t3 = np.clip(self.top3_rate / 70.0, 0, 1)
        return (wr*0.25 + ab*0.15 + mt*0.10 + ex*0.10 + lt*0.10 + tt*0.10 + t3*0.20)


# ============================================================
# レースシミュレーター
# ============================================================

@dataclass
class RaceCondition:
    temperature: float = 15.0
    wind_speed: int = 2
    wave_height: int = 2
    weather: str = ""

class RaceSimulator:
    def __init__(self, agents, condition):
        self.agents = sorted(agents, key=lambda a: a.lane)
        self.condition = condition

    def _compute_race_weights(self):
        weights = {}
        for a in self.agents:
            lane_win = a.lane_win_rate
            general_win_est = max(0, (a.win_rate - 2.0) * 7.5)
            if lane_win > 0:
                base_prob = lane_win * 0.6 + general_win_est * 0.4
            else:
                base_prob = general_win_est * 0.7 + 2.0
            power = a.get_power_score()
            power_adjust = 0.8 + power * 0.4
            st_quality = max(0, (0.22 - a.avg_st) / 0.10)
            st_adjust = 0.9 + st_quality * 0.1
            weights[a.lane] = max(0.5, base_prob * power_adjust * st_adjust)
        return weights

    def simulate_race(self):
        for a in self.agents:
            a.calculate_start_timing()
        start_timings = {a.lane: a.actual_st for a in self.agents}
        base_weights = self._compute_race_weights()
        st_mean = np.mean([a.actual_st for a in self.agents])

        adjusted_weights = {}
        for a in self.agents:
            st_diff = st_mean - a.actual_st
            st_bonus = np.clip(1.0 + st_diff * 3.0, 0.5, 1.8)
            total_wins = a.wins_nige + a.wins_sashi + a.wins_makuri + a.wins_makuri_sashi + a.wins_nuki
            turn_skill = ((a.wins_makuri + a.wins_makuri_sashi + a.wins_sashi) / total_wins) if total_wins > 0 else 0.2
            chaos = np.random.exponential(0.15)
            adjusted_weights[a.lane] = base_weights[a.lane] * st_bonus * (1.0 + turn_skill * chaos * 2.0)

        finish_order = []
        remaining = [a.lane for a in self.agents]
        temp_w = dict(adjusted_weights)
        for pos in range(len(self.agents)):
            w = np.array([temp_w[l] for l in remaining])
            if pos <= 1: w = w ** 1.5
            elif pos <= 3: w = w ** 1.0
            else: w = w ** 0.5
            w = w / w.sum()
            chosen = np.random.choice(remaining, p=w)
            finish_order.append(chosen)
            remaining.remove(chosen)

        history = self._gen_trajectory(finish_order, start_timings)
        return {"finish_order": finish_order, "history": history, "start_timings": start_timings}

    def _gen_trajectory(self, finish_order, start_timings):
        steps = 300
        history = {a.lane: [] for a in self.agents}
        rank_map = {lane: rank for rank, lane in enumerate(finish_order)}
        base_speeds = {lane: 6.05 - rank_map[lane] * 0.10 for lane in rank_map}
        positions = {a.lane: max(0, (0.25 - start_timings[a.lane]) * 30) for a in self.agents}
        for step in range(steps):
            progress = step / steps
            for a in self.agents:
                speed = base_speeds[a.lane]
                pos_in_lap = positions[a.lane] % 600
                is_turn = (130 < pos_in_lap < 170) or (430 < pos_in_lap < 470)
                if is_turn:
                    tq = (12.0 - a.turn_time) / 0.8
                    speed += tq * 0.3 + np.random.normal(0, 0.2)
                    if rank_map[a.lane] >= 3 and random.random() < 0.15:
                        speed += np.random.uniform(0.3, 0.8)
                else:
                    speed += np.random.normal(0, 0.08)
                if progress > 0.3:
                    ts = base_speeds[a.lane]
                    c = min(1.0, (progress - 0.3) * 1.5)
                    speed = speed * (1 - c * 0.3) + ts * c * 0.3
                positions[a.lane] += max(3.5, speed)
                history[a.lane].append(positions[a.lane])
        return history


# ============================================================
# ★ 改良パーサー（ボートレース日和フォーマット完全対応）
# ============================================================

def parse_race_data_v2(text: str) -> Tuple[List[BoatAgent], RaceCondition, dict]:
    """
    ボートレース日和のコピペデータを正確にパースする。
    改行で分断されたパーセント＋(サンプル数)のフォーマットに対応。
    """
    lines = text.strip().split('\n')
    lines_raw = [l.rstrip() for l in lines]  # 右側の空白だけ除去
    lines = [l.strip() for l in lines if l.strip()]

    parsed = {}  # 解析結果を辞書に格納

    # ==== 1. 全テキストを結合してパターン検索もできるようにする ====
    full_text = '\n'.join(lines)

    # ==== 2. 選手基本情報 ====
    # 登録番号 (4桁×6)
    numbers = None
    for line in lines:
        fours = re.findall(r'\b(\d{4})\b', line)
        if len(fours) >= 6:
            vals = [int(n) for n in fours[:6]]
            if all(1000 <= v <= 9999 for v in vals):
                numbers = vals
                break

    # 選手名 (タブ/スペース区切りの漢字2-6文字×6)
    names = None
    name_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,6}')
    for line in lines:
        parts = re.split(r'[\t]+', line)
        if len(parts) >= 6:
            cands = [p.strip() for p in parts if name_pattern.fullmatch(p.strip()) and
                     p.strip() not in ['基本情報','枠別情報','今節成績','直前情報','モータ情報',
                                        'オッズ検索','オッズ一覧','出目ランク','一般戦','選手']]
            if len(cands) >= 6:
                names = cands[:6]
                break
    if not names:
        for line in lines:
            parts = re.split(r'[\t　 ]+', line)
            cands = [p for p in parts if name_pattern.fullmatch(p) and len(p) >= 2
                     and p not in ['基本情報','枠別情報','今節成績','直前情報','モータ情報',
                                    'オッズ検索','オッズ一覧','出目ランク','一般戦','選手']]
            if len(cands) >= 6:
                names = cands[:6]
                break

    # 級別
    ranks = None
    for line in lines:
        found = re.findall(r'\b(A1|A2|B1|B2)\b', line)
        if len(found) >= 6:
            ranks = found[:6]
            break

    # ==== 3. 数値データ抽出（基本情報セクション） ====

    def find_6_values_after_keyword(keyword, value_pattern, validator=None, search_range=3):
        """キーワードを含む行の同じ行 or 続く行から6個の値を探す"""
        for i, line in enumerate(lines):
            if keyword in line:
                for j in range(i, min(i + search_range, len(lines))):
                    nums = re.findall(value_pattern, lines[j])
                    if len(nums) >= 6:
                        vals = [float(n) if '.' in str(n) else n for n in nums[:6]]
                        if validator is None or validator(vals):
                            return vals
        return None

    def find_section_6floats(section_keyword, row_keyword, search_after=True):
        """セクションキーワードを見つけてから、行キーワードの行で6つのfloatを探す"""
        in_section = False
        for i, line in enumerate(lines):
            if section_keyword in line:
                in_section = True
                continue
            if in_section and row_keyword in line:
                nums = re.findall(r'(\d+\.\d+)', line)
                if len(nums) >= 6:
                    return [float(n) for n in nums[:6]]
                # 次の行も見る
                if i + 1 < len(lines):
                    nums = re.findall(r'(\d+\.\d+)', lines[i+1])
                    if len(nums) >= 6:
                        return [float(n) for n in nums[:6]]
        return None

    # ★★★ 枠別パーセント抽出（改行で分断されるパターンに対応）★★★
    def find_lane_percentages(section_keyword, row_keyword):
        """
        枠別情報の「93.1%\n(29)\t7.7%\n(13)...」のような改行分断フォーマットに対応

        戦略: セクションキーワードの後、行キーワードを見つけたら
        そこから下方向に全テキストを結合してパーセントを抽出
        """
        in_section = False
        for i, line in enumerate(lines):
            if section_keyword in line:
                in_section = True
                continue
            if in_section and row_keyword in line:
                # この行と続く数行を結合してパーセントを探す
                combined = ' '.join(lines[i:i+12])  # 十分な行数を結合
                pcts = re.findall(r'(\d+\.?\d*)%', combined)
                if len(pcts) >= 6:
                    return [float(p) for p in pcts[:6]]
                # フォールバック: 同じ行のみ
                pcts = re.findall(r'(\d+\.?\d*)%', line)
                if len(pcts) >= 6:
                    return [float(p) for p in pcts[:6]]
        return None

    def find_first_6pcts_after(keyword):
        """キーワードの後、最初に6個以上のパーセントが揃う箇所を探す（複数行結合）"""
        for i, line in enumerate(lines):
            if keyword in line:
                # ここから下を結合しながらパーセントを収集
                collected_pcts = []
                for j in range(i, min(i + 20, len(lines))):
                    pcts_in_line = re.findall(r'(\d+\.?\d*)%', lines[j])
                    collected_pcts.extend([float(p) for p in pcts_in_line])
                    if len(collected_pcts) >= 6:
                        return collected_pcts[:6]
        return None

    # --- 平均ST ---
    st_6m = find_section_6floats('平均ST', '直近6ヶ月')
    st_3m = find_section_6floats('平均ST', '直近3ヶ月')
    st_1m = find_section_6floats('平均ST', '直近1ヶ月')
    st_first = find_section_6floats('平均ST', '初日')

    # 平均STが「基本情報」セクションで見つからない場合のフォールバック
    if not st_6m:
        for i, line in enumerate(lines):
            if '直近6ヶ月' in line and i > 0:
                prev_lines = lines[max(0,i-5):i]
                if any('平均ST' in pl or 'ST' in pl for pl in prev_lines):
                    nums = re.findall(r'0\.\d+', line)
                    if len(nums) >= 6:
                        st_6m = [float(n) for n in nums[:6]]
                        break

    # --- 勝率 ---
    win_rate_6m = None
    for i, line in enumerate(lines):
        if '勝率' in line and '枠' not in line and '連対' not in line and '1着' not in line:
            for j in range(i+1, min(i+8, len(lines))):
                if '直近6ヶ月' in lines[j]:
                    nums = re.findall(r'(\d+\.\d{2})', lines[j])
                    if len(nums) >= 6:
                        vals = [float(n) for n in nums[:6]]
                        if all(1.0 <= v <= 10.0 for v in vals):
                            win_rate_6m = vals
                            break
            if win_rate_6m:
                break

    # --- 2連対率（総合 = 基本情報セクション）---
    top2_6m = None
    for i, line in enumerate(lines):
        if '2連対率' in line and '総合' not in line and '枠' not in line:
            for j in range(i+1, min(i+5, len(lines))):
                if '直近6ヶ月' in lines[j]:
                    pcts = re.findall(r'(\d+\.?\d*)%', lines[j])
                    if len(pcts) >= 6:
                        top2_6m = [float(p) for p in pcts[:6]]
                        break
            if top2_6m:
                break

    # --- 3連対率（総合 = 基本情報セクション）---
    top3_6m = None
    for i, line in enumerate(lines):
        if '3連対率' in line and '総合' not in line and '枠' not in line:
            for j in range(i+1, min(i+5, len(lines))):
                if '直近6ヶ月' in lines[j]:
                    pcts = re.findall(r'(\d+\.?\d*)%', lines[j])
                    if len(pcts) >= 6:
                        top3_6m = [float(p) for p in pcts[:6]]
                        break
            if top3_6m:
                break

    # --- ★ 枠別1着率（直近6ヶ月）★ ---
    lane_win_6m = find_lane_percentages('1着率(総合)', '直近6ヶ月')
    if not lane_win_6m:
        lane_win_6m = find_first_6pcts_after('1着率(総合)')

    # --- 枠別2連対率（直近6ヶ月）---
    lane_top2_6m = find_lane_percentages('2連対率(総合)', '直近6ヶ月')
    if not lane_top2_6m:
        lane_top2_6m = find_first_6pcts_after('2連対率(総合)')

    # --- 枠別3連対率（直近6ヶ月）---
    lane_top3_6m = find_lane_percentages('3連対率(総合)', '直近6ヶ月')
    if not lane_top3_6m:
        lane_top3_6m = find_first_6pcts_after('3連対率(総合)')

    # --- 決まり手 ---
    def find_kimarite(keyword):
        for i, line in enumerate(lines):
            if line.startswith(keyword) or line == keyword:
                nums = re.findall(r'\b(\d+)\b', line)
                if len(nums) >= 6:
                    return [int(n) for n in nums[:6]]
                if i+1 < len(lines):
                    combined = line + '\t' + lines[i+1]
                    nums = re.findall(r'\b(\d+)\b', combined)
                    if len(nums) >= 6:
                        return [int(n) for n in nums[:6]]
        return None

    nige = find_kimarite('逃げ')
    sashi = find_kimarite('差し')
    makuri = find_kimarite('捲り')
    m_sashi = find_kimarite('捲差')
    nuki = find_kimarite('抜き')

    # --- 能力値 ---
    ability_vals = None
    for i, line in enumerate(lines):
        if '能力値' in line:
            for j in range(i, min(i+3, len(lines))):
                nums = re.findall(r'\b(\d{2})\b', lines[j])
                if len(nums) >= 6:
                    vals = [int(n) for n in nums[:6]]
                    if all(20 <= v <= 70 for v in vals):
                        ability_vals = vals
                        break
            if ability_vals:
                break

    # --- ST安定率 ---
    stability = None
    for i, line in enumerate(lines):
        if '安定率' in line and 'ST' not in line:
            pcts = re.findall(r'(\d+\.?\d*)%', line)
            if len(pcts) >= 6:
                stability = [float(p) for p in pcts[:6]]
                break

    # --- モーター貢献P ---
    motor = None
    for i, line in enumerate(lines):
        if '貢献P' in line or '貢献' in line:
            nums = re.findall(r'(-?\d+\.\d+)', line)
            if len(nums) >= 6:
                motor = [float(n) for n in nums[:6]]
            elif i+1 < len(lines):
                nums = re.findall(r'(-?\d+\.\d+)', lines[i+1])
                if len(nums) >= 6:
                    motor = [float(n) for n in nums[:6]]
            if motor:
                break

    # --- 展示タイム ---
    exhibition = None
    for i, line in enumerate(lines):
        # 「展示」で始まり、6.xxまたは7.xxが6個ある行
        if re.match(r'^展示\s', line) or line.startswith('展示\t'):
            nums = re.findall(r'([67]\.\d{2})', line)
            if len(nums) >= 6:
                exhibition = [float(n) for n in nums[:6]]
                break

    # --- 周回タイム ---
    lap = None
    for i, line in enumerate(lines):
        if re.match(r'^周回\s', line) or line.startswith('周回\t'):
            if '足' not in line:
                nums = re.findall(r'(\d{2}\.\d{2})', line)
                if len(nums) >= 6:
                    lap = [float(n) for n in nums[:6]]
                    break

    # --- 周り足 ---
    turn = None
    for i, line in enumerate(lines):
        if '周り足' in line:
            nums = re.findall(r'(\d{2}\.\d{2})', line)
            if len(nums) >= 6:
                turn = [float(n) for n in nums[:6]]
                break

    # --- 体重 ---
    weight = None
    for i, line in enumerate(lines):
        if re.match(r'^体重\s', line) or line.startswith('体重\t'):
            nums = re.findall(r'(\d{2}\.\d)', line)
            if len(nums) >= 6:
                weight = [float(n) for n in nums[:6]]
                break

    # --- コメント ---
    comments = [''] * 6
    for i, line in enumerate(lines):
        if 'コメント' in line:
            c_idx = 0
            for j in range(i+1, min(i+20, len(lines))):
                cl = lines[j].strip()
                if cl and len(cl) >= 3 and not cl.startswith('展示') and not cl.startswith('体重'):
                    if any(k in cl for k in ['。','感じ','ない','ある','った','した','思う','思った',
                                               '弱い','良い','悪い','まずまず','違和感']):
                        if c_idx < 6:
                            comments[c_idx] = cl[:40]
                            c_idx += 1
                if c_idx >= 6:
                    break

    # --- 天候 ---
    temp = 15.0; wind = 2; wave = 2; weather = ""
    for line in lines:
        t_match = re.findall(r'(\d+\.?\d*)℃', line)
        if t_match and '気温' in line: temp = float(t_match[0])
        w_match = re.findall(r'(\d+)m', line)
        if w_match and '風速' in line: wind = int(w_match[0])
        wv_match = re.findall(r'(\d+)cm', line)
        if wv_match and '波高' in line: wave = int(wv_match[0])
        for w_name in ['晴れ','曇り','雨','雪','霧']:
            if w_name in line: weather = w_name

    # ==== デフォルト値設定 ====
    d = lambda v, default: v if v else default
    numbers = d(numbers, [0]*6)
    names = d(names, [f'選手{i+1}' for i in range(6)])
    ranks = d(ranks, ['B1']*6)
    st_6m = d(st_6m, [0.17]*6)
    st_1m = d(st_1m, st_6m)
    st_first = d(st_first, st_6m)
    win_rate_6m = d(win_rate_6m, [4.0]*6)
    stability = d(stability, [50.0]*6)
    ability_vals = d(ability_vals, [45]*6)
    motor = d(motor, [0.0]*6)
    exhibition = d(exhibition, [6.95]*6)
    lap = d(lap, [37.5]*6)
    turn = d(turn, [11.6]*6)
    weight = d(weight, [52.0]*6)
    top2_6m = d(top2_6m, [20.0]*6)
    top3_6m = d(top3_6m, [35.0]*6)
    nige = d(nige, [2]*6)
    sashi = d(sashi, [1]*6)
    makuri = d(makuri, [1]*6)
    m_sashi = d(m_sashi, [0]*6)
    nuki = d(nuki, [0]*6)

    # 枠別1着率: データが取れなかった場合は勝率から推定
    if not lane_win_6m:
        lane_win_6m = [0.0] * 6
        for k in range(6):
            if k == 0:
                lane_win_6m[k] = min(85, max(15, (win_rate_6m[k] - 3.0) * 12.5 + 10))
            else:
                lane_win_6m[k] = max(0, min(30, (win_rate_6m[k] - 3.0) * 5.0))

    # エージェント生成
    agents = []
    for i in range(6):
        agents.append(BoatAgent(
            lane=i+1, name=names[i], rank=ranks[i],
            win_rate=win_rate_6m[i], avg_st=st_1m[i] if st_1m else st_6m[i],
            st_stability=stability[i], lane_win_rate=lane_win_6m[i],
            top2_rate=top2_6m[i], top3_rate=top3_6m[i],
            ability=ability_vals[i], motor_contribution=motor[i],
            exhibition_time=exhibition[i], lap_time=lap[i],
            turn_time=turn[i], weight=weight[i],
            wins_nige=nige[i], wins_sashi=sashi[i],
            wins_makuri=makuri[i], wins_makuri_sashi=m_sashi[i],
            wins_nuki=nuki[i], comment=comments[i]
        ))

    condition = RaceCondition(temperature=temp, wind_speed=wind, wave_height=wave, weather=weather)

    # デバッグ用: パース結果の概要
    parse_info = {
        'names_found': names is not None,
        'ranks_found': ranks is not None,
        'st_found': st_6m is not None,
        'win_rate_found': win_rate_6m is not None,
        'lane_win_found': lane_win_6m is not None,
        'ability_found': ability_vals is not None,
        'motor_found': motor is not None,
        'exhibition_found': exhibition is not None,
        'lap_found': lap is not None,
        'turn_found': turn is not None,
        'stability_found': stability is not None,
    }

    return agents, condition, parse_info


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="🚤 ボートレースシミュレーター", layout="wide", page_icon="🚤")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 20px;
    }
    .main-header h1 { color: #00d4ff; margin: 0; font-size: 32px; }
    .main-header p { color: #aaa; margin: 8px 0 0 0; }
    .result-box {
        background: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 5px solid #00d4ff; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚤 ボートレース レースシミュレーター v5.0</h1>
    <p>ボートレース日和のデータを貼り付けるだけ！モンテカルロ法で確率を算出</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# タブで入力方式を切り替え
# ============================================================

tab_paste, tab_form = st.tabs(["📋 テキスト貼り付け", "✏️ フォーム入力"])

agents = None
condition = RaceCondition()

# ==================== 貼り付けタブ ====================
with tab_paste:
    st.markdown("""
    **使い方**: ボートレース日和の出走表ページで、画面全体を `Ctrl+A` → `Ctrl+C` でコピーして下に貼り付けてください。
    基本情報・枠別情報・モータ情報・直前情報のすべてのタブのデータをまとめて貼り付けるのがベストです。
    """)

    text_input = st.text_area("📋 レースデータを貼り付け", value="", height=350,
                               placeholder="ボートレース日和のデータをここに貼り付け...")

    col_parse1, col_parse2 = st.columns([1, 3])
    with col_parse1:
        parse_btn = st.button("🔍 データ解析", type="secondary", use_container_width=True)

    if parse_btn and text_input.strip():
        try:
            agents_parsed, condition_parsed, parse_info = parse_race_data_v2(text_input)
            st.session_state['agents'] = agents_parsed
            st.session_state['condition'] = condition_parsed
            st.session_state['parse_info'] = parse_info
        except Exception as e:
            st.error(f"❌ 解析エラー: {e}")

    if 'agents' in st.session_state and st.session_state.get('agents'):
        agents = st.session_state['agents']
        condition = st.session_state.get('condition', RaceCondition())
        parse_info = st.session_state.get('parse_info', {})

        # 解析結果表示
        st.success("✅ データ解析完了")

        # パース結果の詳細
        with st.expander("🔎 解析結果の詳細（どのデータが取得できたか）"):
            items = {
                'names_found': '選手名', 'ranks_found': '級別', 'st_found': '平均ST',
                'win_rate_found': '勝率', 'lane_win_found': '枠別1着率',
                'ability_found': '能力値', 'motor_found': 'モーター貢献P',
                'exhibition_found': '展示タイム', 'lap_found': '周回タイム',
                'turn_found': '周り足', 'stability_found': 'ST安定率',
            }
            for key, label in items.items():
                status = "✅" if parse_info.get(key) else "⚠️ デフォルト値使用"
                st.write(f"  {status} {label}")

        # エージェント一覧
        for a in agents:
            st.write(f"  **{a.lane}号艇**: {a.name}({a.rank}) 勝率{a.win_rate:.2f} "
                     f"枠別1着率**{a.lane_win_rate:.1f}%** ST{a.avg_st:.2f} 能力{a.ability}")

        # 修正用エキスパンダー
        with st.expander("✏️ 解析結果を手動で修正"):
            st.warning("枠別1着率が0%の場合は、ここで手動入力してください")
            cols = st.columns(6)
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"**{agents[i].lane}号艇 {agents[i].name}**")
                    agents[i].lane_win_rate = st.number_input(
                        f"枠別1着率(%)", value=agents[i].lane_win_rate,
                        format="%.1f", step=1.0, key=f"fix_lw_{i}")
                    agents[i].win_rate = st.number_input(
                        f"勝率", value=agents[i].win_rate,
                        format="%.2f", step=0.1, key=f"fix_wr_{i}")
                    agents[i].avg_st = st.number_input(
                        f"平均ST", value=agents[i].avg_st,
                        format="%.2f", step=0.01, key=f"fix_st_{i}")
                    agents[i].ability = st.number_input(
                        f"能力値", value=agents[i].ability,
                        step=1, key=f"fix_ab_{i}")

# ==================== フォーム入力タブ ====================
with tab_form:
    st.info("💡 各艇のデータを直接入力できます。")

    lane_colors_hex = ["#FF3333", "#444444", "#CC3333", "#3333FF", "#CCAA00", "#008800"]
    lane_text_colors = ["white", "white", "white", "white", "black", "white"]

    # セッションステートにフォームデータを保存
    if 'form_agents' not in st.session_state:
        st.session_state['form_agents'] = None

    st.markdown("### 👤 選手情報 & データ")
    form_agents = []
    cols = st.columns(6)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"<div style='background:{lane_colors_hex[i]}; color:{lane_text_colors[i]}; "
                       f"padding:6px; border-radius:6px; text-align:center; font-weight:bold; font-size:14px;'>"
                       f"{i+1}号艇</div>", unsafe_allow_html=True)
            nm = st.text_input("名前", value="", key=f"fn_{i}")
            rk = st.selectbox("級別", ["A1","A2","B1","B2"], index=2, key=f"fr_{i}")
            wr = st.number_input("勝率", value=4.00, format="%.2f", step=0.1, key=f"fwr_{i}")
            ast = st.number_input("平均ST", value=0.16, format="%.2f", step=0.01, key=f"fst_{i}")
            stab = st.number_input("安定率(%)", value=50.0, format="%.1f", key=f"fstab_{i}")
            lwr = st.number_input("枠別1着率(%)", value=0.0, format="%.1f", key=f"flw_{i}")
            t2 = st.number_input("2連対率(%)", value=20.0, format="%.1f", key=f"ft2_{i}")
            t3 = st.number_input("3連対率(%)", value=35.0, format="%.1f", key=f"ft3_{i}")
            ab = st.number_input("能力値", value=45, step=1, key=f"fab_{i}")
            mt = st.number_input("ﾓｰﾀｰ貢献P", value=0.0, format="%.2f", step=0.1, key=f"fmt_{i}")
            ex = st.number_input("展示ﾀｲﾑ", value=6.95, format="%.2f", step=0.01, key=f"fex_{i}")
            lp = st.number_input("周回ﾀｲﾑ", value=37.50, format="%.2f", step=0.1, key=f"flp_{i}")
            tn = st.number_input("周り足", value=11.60, format="%.2f", step=0.1, key=f"ftn_{i}")
            wt = st.number_input("体重", value=52.0, format="%.1f", key=f"fwt_{i}")

            actual_lwr = lwr
            if actual_lwr == 0:
                if i == 0 and wr > 5.0:
                    actual_lwr = min(80, max(15, (wr - 3.0) * 12.5 + 10))
                else:
                    actual_lwr = max(0, (wr - 3.0) * 5.0)

            form_agents.append(BoatAgent(
                lane=i+1, name=nm or f'選手{i+1}', rank=rk,
                win_rate=wr, avg_st=ast, st_stability=stab,
                lane_win_rate=actual_lwr, top2_rate=t2, top3_rate=t3,
                ability=ab, motor_contribution=mt, exhibition_time=ex,
                lap_time=lp, turn_time=tn, weight=wt
            ))

    if st.button("📝 フォームデータを確定", key="form_confirm"):
        agents = form_agents
        st.session_state['agents'] = agents
        st.success("✅ フォームデータを確定しました")


# ============================================================
# シミュレーション実行
# ============================================================

st.markdown("---")

if agents is None and 'agents' in st.session_state:
    agents = st.session_state['agents']
    condition = st.session_state.get('condition', RaceCondition())

col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
with col_s1:
    n_sims = st.slider("シミュレーション回数", 1000, 20000, 5000, 1000)
with col_s2:
    st.markdown(f"**天候**: {condition.weather} {condition.temperature}℃ 風{condition.wind_speed}m 波{condition.wave_height}cm")

run_clicked = st.button("🚀 シミュレーション実行", type="primary", use_container_width=True)

if run_clicked:
    if not agents:
        st.error("⚠️ 先にデータを入力・解析してください！")
    else:
        # エージェント情報テーブル
        st.markdown("### 📋 エージェント一覧")
        df_ag = pd.DataFrame({
            '枠': [f'{a.lane}号艇' for a in agents],
            '選手名': [a.name for a in agents],
            '級別': [a.rank for a in agents],
            '勝率': [f'{a.win_rate:.2f}' for a in agents],
            '枠別1着率': [f'{a.lane_win_rate:.1f}%' for a in agents],
            'ST': [f'{a.avg_st:.2f}' for a in agents],
            '安定率': [f'{a.st_stability:.0f}%' for a in agents],
            '能力': [a.ability for a in agents],
            '展示': [a.exhibition_time for a in agents],
            'ﾓｰﾀｰ': [f'{a.motor_contribution:+.2f}' for a in agents],
            'ﾊﾟﾜｰ': [f'{a.get_power_score():.3f}' for a in agents],
        })
        st.dataframe(df_ag, hide_index=True, use_container_width=True)

        # コメント
        if any(a.comment for a in agents):
            with st.expander("💬 選手コメント"):
                for a in agents:
                    if a.comment:
                        st.write(f"**{a.lane}号艇 {a.name}**: 「{a.comment}」")

        # --- 単発シミュレーション ---
        st.markdown("### 🏁 単発シミュレーション（3回）")
        colors_plt = {1:'#FF0000', 2:'#444444', 3:'#E83030', 4:'#0000FF', 5:'#DAA520', 6:'#008800'}

        for trial in range(1, 4):
            sim = RaceSimulator(agents, condition)
            result = sim.simulate_race()
            o = result['finish_order']
            sts = result['start_timings']
            wl = o[0]

            if wl == 1: km = "逃げ"
            elif wl == 2: km = "差し" if sts[2] <= sts[1]+0.02 else "捲り"
            else:
                inner = [sts[l] for l in range(1, wl)]
                km = "捲り" if sts[wl] <= min(inner)+0.02 else "捲り差し"

            with st.expander(f"🏁 第{trial}回: **{o[0]}-{o[1]}-{o[2]}** 決まり手:{km}", expanded=(trial==1)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    medals = {0:"🥇", 1:"🥈", 2:"🥉"}
                    for rank, lane in enumerate(o):
                        a = [ag for ag in agents if ag.lane == lane][0]
                        m = medals.get(rank, f"  {rank+1}着")
                        st.write(f"{m} **{lane}号艇 {a.name}** ({a.rank}) ST:{sts[lane]:.2f}")
                    st.markdown(f"**3連単: {o[0]}-{o[1]}-{o[2]}** | 2連単: {o[0]}-{o[1]}")

                with c2:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                    for a in agents:
                        h = result['history'][a.lane]
                        lw = 3.5 if a.lane == wl else 1.2
                        alpha = 1.0 if a.lane == wl else 0.5
                        ax.plot(range(len(h)), h, color=colors_plt[a.lane], linewidth=lw,
                                label=f'{a.lane} {a.name}', alpha=alpha)
                    for lp in range(1, 4):
                        ax.axhline(y=lp*600, color='gray', linestyle='--', alpha=0.2)
                    ax.axhline(y=1800, color='red', linestyle='-', alpha=0.4)
                    ax.set_xlabel('Step'); ax.set_ylabel('m')
                    ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.2)
                    ax.set_title(f'Race #{trial}')
                    st.pyplot(fig); plt.close(fig)

        # --- モンテカルロ ---
        st.markdown(f"### 📊 モンテカルロシミュレーション（{n_sims}回）")
        progress = st.progress(0)

        win_c = {a.lane:0 for a in agents}
        top2_c = {a.lane:0 for a in agents}
        top3_c = {a.lane:0 for a in agents}
        tri_c = {}; exa_c = {}
        km_c = {"逃げ":0,"差し":0,"捲り":0,"捲り差し":0,"抜き":0}

        for si in range(n_sims):
            if si % max(1, n_sims//50) == 0:
                progress.progress(si / n_sims)
            sim = RaceSimulator(agents, condition)
            r = sim.simulate_race()
            oo = r['finish_order']; ss = r['start_timings']
            win_c[oo[0]] += 1
            for l in oo[:2]: top2_c[l] += 1
            for l in oo[:3]: top3_c[l] += 1
            tri_c[f"{oo[0]}-{oo[1]}-{oo[2]}"] = tri_c.get(f"{oo[0]}-{oo[1]}-{oo[2]}", 0) + 1
            exa_c[f"{oo[0]}-{oo[1]}"] = exa_c.get(f"{oo[0]}-{oo[1]}", 0) + 1
            wwl = oo[0]
            if wwl == 1: kk = "逃げ"
            elif wwl == 2: kk = "差し" if ss[2] <= ss[1]+0.02 else "捲り"
            else:
                inn = [ss[ll] for ll in range(1, wwl)]
                kk = "捲り" if ss[wwl] <= min(inn)+0.02 else "捲り差し"
            km_c[kk] += 1

        progress.progress(1.0)

        # グラフ
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ln = [f"{a.lane} {a.name}" for a in agents]
        bc = ['#FF0000', '#555555', '#E83030', '#0000FF', '#DAA520', '#008800']
        for idx, (title, cnt) in enumerate([('1着率', win_c), ('2連対率', top2_c), ('3連対率', top3_c)]):
            rates = [cnt[a.lane]/n_sims*100 for a in agents]
            axes[idx].bar(ln, rates, color=bc, edgecolor='black', linewidth=0.5)
            axes[idx].set_title(title, fontsize=13, fontweight='bold')
            axes[idx].set_ylabel('%')
            axes[idx].set_ylim(0, max(max(rates)*1.3, 5))
            for j, v in enumerate(rates):
                axes[idx].text(j, v+0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # テーブル
        st.markdown("#### 📊 各艇の確率")
        df_p = pd.DataFrame({
            '枠': [f'{a.lane}号艇' for a in agents],
            '選手名': [a.name for a in agents],
            '1着率': [f"{win_c[a.lane]/n_sims*100:.1f}%" for a in agents],
            '2連対率': [f"{top2_c[a.lane]/n_sims*100:.1f}%" for a in agents],
            '3連対率': [f"{top3_c[a.lane]/n_sims*100:.1f}%" for a in agents],
        })
        st.dataframe(df_p, hide_index=True, use_container_width=True)

        c_e, c_t = st.columns(2)
        with c_e:
            st.markdown("#### 🎯 2連単 TOP10")
            s_exa = sorted(exa_c.items(), key=lambda x:x[1], reverse=True)[:10]
            df_e = pd.DataFrame({'順位': range(1, len(s_exa)+1),
                                  '組合せ': [c for c,_ in s_exa],
                                  '確率': [f"{cnt/n_sims*100:.1f}%" for _,cnt in s_exa],
                                  '回数': [cnt for _,cnt in s_exa]})
            st.dataframe(df_e, hide_index=True, use_container_width=True)

        with c_t:
            st.markdown("#### 🎯 3連単 TOP15")
            s_tri = sorted(tri_c.items(), key=lambda x:x[1], reverse=True)[:15]
            df_t = pd.DataFrame({'順位': range(1, len(s_tri)+1),
                                  '組合せ': [c for c,_ in s_tri],
                                  '確率': [f"{cnt/n_sims*100:.1f}%" for _,cnt in s_tri],
                                  '回数': [cnt for _,cnt in s_tri]})
            st.dataframe(df_t, hide_index=True, use_container_width=True)

        # 決まり手
        st.markdown("#### 🎯 決まり手分布")
        km_s = sorted(km_c.items(), key=lambda x:x[1], reverse=True)
        fig_k, ax_k = plt.subplots(figsize=(8, 3))
        ax_k.barh([k for k,_ in km_s], [v/n_sims*100 for _,v in km_s],
                  color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'])
        for bar, (k, v) in zip(ax_k.patches, km_s):
            ax_k.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                     f'{v/n_sims*100:.1f}%', va='center', fontsize=11)
        ax_k.set_xlabel('%'); ax_k.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_k); plt.close(fig_k)

        st.warning("⚠️ このシミュレーションは統計モデルに基づく参考値です。実際のレース結果を保証するものではありません。")
        # ============================================================
# ★★★ オッズ入力 ＆ 合成オッズ ＆ 期待値 計算セクション ★★★
# ============================================================

st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #1a472a 0%, #2d5a27 100%);
            color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 20px 0;'>
    <h2 style='margin:0; color:#7dff7d;'>💰 オッズ分析 & 期待値計算</h2>
    <p style='color:#aaa; margin:5px 0 0 0;'>3連単オッズから全券種の合成オッズ・期待値を算出</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**使い方**: ボートレース公式サイトのオッズページ（3連単）を `Ctrl+A` → `Ctrl+C` でコピーして貼り付けてください。
または、手動で各組合せのオッズを入力できます。
""")

# ============================================================
# 3連単オッズ パーサー
# ============================================================

def parse_trifecta_odds(text: str) -> Dict[str, float]:
    """
    公式サイト/ボートレース日和の3連単オッズテキストを解析
    戻り値: {"1-2-3": 5.0, "1-2-4": 4.6, ...} の辞書
    """
    odds_dict = {}
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    # ---- 方式1: 公式サイトのフォーマット ----
    # "2   3   5.0" のようなパターン（1着は列ヘッダーから判定）
    current_first = None  # 1着艇番
    current_second = None  # 2着艇番

    for line in lines:
        # 1着が含まれる行: "1   黒野　　元基" -> 1着=1
        header_match = re.match(r'^(\d)\s+[\u4e00-\u9fff]', line)
        if header_match:
            current_first = int(header_match.group(1))
            continue

        # タブ/スペース区切りで数値ペアを抽出
        # "2   3   5.0     1   3   91.0   ..." のパターン
        # 各列は「2着 3着 オッズ」の3つ組
        tokens = re.split(r'[\t ]+', line.strip())

        # 全トークンから「整数 整数 小数」の3つ組を探す
        i = 0
        while i < len(tokens) - 2:
            try:
                a = tokens[i]
                b = tokens[i+1]
                c = tokens[i+2]

                if re.match(r'^\d$', a) and re.match(r'^\d$', b) and re.match(r'^\d+\.?\d*$', c):
                    second = int(a)
                    third = int(b)
                    odds_val = float(c)

                    if 1 <= second <= 6 and 1 <= third <= 6 and odds_val > 0:
                        # 1着の推定: current_firstがあればそれ、なければ列位置から推定
                        if current_first:
                            key = f"{current_first}-{second}-{third}"
                            if key not in odds_dict:
                                odds_dict[key] = odds_val
                        i += 3
                        continue
                i += 1
            except (ValueError, IndexError):
                i += 1

    # ---- 方式2: 直接的なパターン "1-2-3 5.0" ----
    if len(odds_dict) < 10:
        for line in lines:
            matches = re.findall(r'(\d)-(\d)-(\d)\s+(\d+\.?\d*)', line)
            for m in matches:
                key = f"{m[0]}-{m[1]}-{m[2]}"
                odds_dict[key] = float(m[3])

    # ---- 方式3: 公式サイトの列構造を再解析 ----
    if len(odds_dict) < 10:
        # 1着ごとのブロックで再解析
        for first in range(1, 7):
            for line in lines:
                tokens = re.split(r'[\t ]+', line.strip())
                i = 0
                while i < len(tokens) - 2:
                    try:
                        a, b, c = tokens[i], tokens[i+1], tokens[i+2]
                        if re.match(r'^\d$', a) and re.match(r'^\d$', b):
                            second, third = int(a), int(b)
                            odds_val = float(c)
                            if (1 <= second <= 6 and 1 <= third <= 6 and
                                second != third and odds_val > 0):
                                # 1着を推定 (second, thirdと異なる番号)
                                for f in range(1, 7):
                                    if f != second and f != third:
                                        key = f"{f}-{second}-{third}"
                                        if key not in odds_dict and odds_val > 1.0:
                                            # 重複チェック
                                            pass
                            i += 3
                            continue
                    except:
                        pass
                    i += 1

    return odds_dict


def parse_odds_from_official(text: str) -> Dict[str, float]:
    """
    公式サイトの3連単オッズテーブルをより正確にパース

    公式フォーマット:
    1着ヘッダー（選手名付き）の後に、
    2着  3着  オッズ  がブロックで並ぶ
    """
    odds_dict = {}
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    current_first = None

    for line in lines:
        # 1着のヘッダー行検出: "1   黒野　　元基" など
        m = re.match(r'^(\d)\s+[\u4e00-\u9fff\u3000-\u303f]', line)
        if m:
            current_first = int(m.group(1))
            continue

        if current_first is None:
            continue

        # この行からオッズデータを抽出
        # 複数列が並ぶ場合: "2   3   5.0     1   3   91.0    ..."
        # 2着から始まる場合: "4   4.6" (前行の2着を継続)
        tokens = re.split(r'\s+', line.strip())

        # 「整数 整数 数値」 の3つ組を全て探す
        i = 0
        temp_second = None
        while i < len(tokens):
            tok = tokens[i]

            # 2着番号と3着番号+オッズの組
            if re.match(r'^\d$', tok):
                val = int(tok)
                if 1 <= val <= 6:
                    if i + 2 < len(tokens):
                        next1 = tokens[i+1]
                        next2 = tokens[i+2]
                        # パターン: 2着 3着 オッズ
                        if (re.match(r'^\d$', next1) and
                            re.match(r'^\d+\.?\d*$', next2)):
                            second = val
                            third = int(next1)
                            odds_val = float(next2)
                            if (second != current_first and third != current_first
                                and second != third and odds_val > 0):
                                key = f"{current_first}-{second}-{third}"
                                odds_dict[key] = odds_val
                                temp_second = second
                            i += 3
                            continue
                    # パターン: 3着 オッズ (前の2着を継続)
                    if i + 1 < len(tokens) and temp_second is not None:
                        next1 = tokens[i+1]
                        if re.match(r'^\d+\.?\d*$', next1):
                            third = val
                            odds_val = float(next1)
                            if (third != current_first and third != temp_second
                                and odds_val > 0):
                                key = f"{current_first}-{temp_second}-{third}"
                                odds_dict[key] = odds_val
                            i += 2
                            continue
            i += 1

    return odds_dict


def merge_odds_parsers(text: str) -> Dict[str, float]:
    """複数パーサーの結果をマージ"""
    odds1 = parse_trifecta_odds(text)
    odds2 = parse_odds_from_official(text)
    # 多い方を採用、足りない分を補完
    if len(odds2) >= len(odds1):
        merged = dict(odds2)
        for k, v in odds1.items():
            if k not in merged:
                merged[k] = v
    else:
        merged = dict(odds1)
        for k, v in odds2.items():
            if k not in merged:
                merged[k] = v
    return merged


# ============================================================
# 合成オッズ計算
# ============================================================

def compute_synthetic_odds(trifecta_odds: Dict[str, float]) -> Dict:
    """
    3連単オッズから全券種の合成オッズを計算

    合成オッズの計算式:
    1 / 合成オッズ = Σ (1 / 各オッズ)
    → 合成オッズ = 1 / Σ(1/各オッズ)
    """
    results = {
        'trifecta': trifecta_odds,  # 3連単: そのまま
        'trio': {},       # 3連複
        'exacta': {},     # 2連単
        'quinella': {},   # 2連複
        'wide': {},       # 拡連複
    }

    # ---- 3連複 (1-2-3着を順不同) ----
    # 3連複 A-B-C = 3連単 A-B-C, A-C-B, B-A-C, B-C-A, C-A-B, C-B-A の合成
    from itertools import permutations
    trio_groups = {}  # key: "1-2-3" (ソート済み) -> [odds1, odds2, ...]
    for key, odds in trifecta_odds.items():
        parts = key.split('-')
        sorted_key = '-'.join(map(str, sorted(int(p) for p in parts)))
        if sorted_key not in trio_groups:
            trio_groups[sorted_key] = []
        trio_groups[sorted_key].append(odds)

    for key, odds_list in trio_groups.items():
        if odds_list:
            # 合成オッズ = 1 / Σ(1/odds)
            inv_sum = sum(1.0/o for o in odds_list if o > 0)
            if inv_sum > 0:
                results['trio'][key] = round(1.0 / inv_sum, 1)

    # ---- 2連単 (1着-2着を着順通り) ----
    # 2連単 A-B = 3連単 A-B-C (Cは3~6号艇のすべて) の合成
    exacta_groups = {}  # key: "1-2" -> [odds1, odds2, ...]
    for key, odds in trifecta_odds.items():
        parts = key.split('-')
        exacta_key = f"{parts[0]}-{parts[1]}"
        if exacta_key not in exacta_groups:
            exacta_groups[exacta_key] = []
        exacta_groups[exacta_key].append(odds)

    for key, odds_list in exacta_groups.items():
        if odds_list:
            inv_sum = sum(1.0/o for o in odds_list if o > 0)
            if inv_sum > 0:
                results['exacta'][key] = round(1.0 / inv_sum, 1)

    # ---- 2連複 (1着-2着を順不同) ----
    # 2連複 A=B = 2連単 A-B + 2連単 B-A の合成
    quinella_groups = {}
    for key, odds in results['exacta'].items():
        parts = key.split('-')
        sorted_key = '-'.join(map(str, sorted(int(p) for p in parts)))
        if sorted_key not in quinella_groups:
            quinella_groups[sorted_key] = []
        quinella_groups[sorted_key].append(odds)

    for key, odds_list in quinella_groups.items():
        if odds_list:
            inv_sum = sum(1.0/o for o in odds_list if o > 0)
            if inv_sum > 0:
                results['quinella'][key] = round(1.0 / inv_sum, 1)

    # ---- 拡連複 (3着以内の2艇を順不同) ----
    # 拡連複 A=B = 3連単で A,Bが共に1-3着に入る全パターンの合成
    wide_groups = {}
    for key, odds in trifecta_odds.items():
        parts = [int(p) for p in key.split('-')]
        top3 = parts[:3]  # 1着,2着,3着
        # 3着以内の2艇の全組合せ
        for ci in range(3):
            for cj in range(ci+1, 3):
                pair = tuple(sorted([top3[ci], top3[cj]]))
                pair_key = f"{pair[0]}-{pair[1]}"
                if pair_key not in wide_groups:
                    wide_groups[pair_key] = []
                wide_groups[pair_key].append(odds)

    for key, odds_list in wide_groups.items():
        if odds_list:
            inv_sum = sum(1.0/o for o in odds_list if o > 0)
            if inv_sum > 0:
                results['wide'][key] = round(1.0 / inv_sum, 1)

    return results


# ============================================================
# 期待値計算
# ============================================================

def compute_expected_values(synthetic_odds: Dict, sim_probabilities: Dict) -> Dict:
    """
    シミュレーション確率 × オッズ = 期待値 を全券種で計算
    期待値 > 1.0 なら理論上プラス
    """
    ev_results = {}

    for bet_type, odds_dict in synthetic_odds.items():
        ev_results[bet_type] = {}
        for combo, odds in odds_dict.items():
            prob = sim_probabilities.get(bet_type, {}).get(combo, 0)
            if prob > 0 and odds > 0:
                ev = prob * odds
                ev_results[bet_type][combo] = {
                    'odds': odds,
                    'prob': prob,
                    'ev': round(ev, 3),
                    'profitable': ev > 1.0
                }

    return ev_results


# ============================================================
# オッズ入力UI
# ============================================================

odds_tab_paste, odds_tab_manual = st.tabs(["📋 オッズ貼り付け", "✏️ 手動入力"])

trifecta_odds = {}

with odds_tab_paste:
    st.markdown("""
    公式サイトのオッズページ（3連単）→ `Ctrl+A` → `Ctrl+C` でコピーして貼り付け。
    または、3連単オッズを `1-2-3 5.0` の形式で1行1組合せで入力。
    """)

    odds_text = st.text_area("3連単オッズデータ", value="", height=300,
                              placeholder="公式サイトの3連単オッズをここに貼り付け...\n\nまたは:\n1-2-3 5.0\n1-2-4 4.6\n...")

    if st.button("🔍 オッズ解析", key="odds_parse"):
        if odds_text.strip():
            trifecta_odds = merge_odds_parsers(odds_text)
            if trifecta_odds:
                st.session_state['trifecta_odds'] = trifecta_odds
                st.success(f"✅ {len(trifecta_odds)}通りの3連単オッズを解析しました")
            else:
                st.warning("⚠️ オッズが解析できませんでした。手動入力を試してください。")

with odds_tab_manual:
    st.markdown("3連単オッズを直接入力できます。`1-2-3 5.0` の形式で1行ずつ入力してください。")

    manual_text = st.text_area("3連単オッズ（手動）", height=300,
                                placeholder="1-2-3 5.0\n1-2-4 4.6\n1-2-5 6.2\n1-2-6 13.9\n1-3-2 11.0\n...")

    if st.button("📝 手動オッズ確定", key="odds_manual"):
        if manual_text.strip():
            manual_odds = {}
            for line in manual_text.strip().split('\n'):
                line = line.strip()
                m = re.match(r'(\d)-(\d)-(\d)\s+(\d+\.?\d*)', line)
                if m:
                    key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    manual_odds[key] = float(m.group(4))
            if manual_odds:
                st.session_state['trifecta_odds'] = manual_odds
                trifecta_odds = manual_odds
                st.success(f"✅ {len(manual_odds)}通りを登録")

# セッションから復元
if 'trifecta_odds' in st.session_state and st.session_state['trifecta_odds']:
    trifecta_odds = st.session_state['trifecta_odds']


# ============================================================
# 期待値計算＆表示
# ============================================================

if trifecta_odds and agents:
    ev_btn = st.button("💰 期待値を計算", type="primary", use_container_width=True, key="ev_calc")

    if ev_btn:
        # --- 合成オッズ計算 ---
        synthetic = compute_synthetic_odds(trifecta_odds)

        # --- シミュレーション確率（モンテカルロ）---
        st.markdown("### 📊 確率計算中...")
        ev_n_sims = 10000
        ev_progress = st.progress(0)

        sim_probs = {
            'trifecta': {},
            'trio': {},
            'exacta': {},
            'quinella': {},
            'wide': {},
        }

        for si in range(ev_n_sims):
            if si % (ev_n_sims // 20) == 0:
                ev_progress.progress(si / ev_n_sims)

            sim = RaceSimulator(agents, condition)
            r = sim.simulate_race()
            o = r['finish_order']

            # 3連単
            tri_key = f"{o[0]}-{o[1]}-{o[2]}"
            sim_probs['trifecta'][tri_key] = sim_probs['trifecta'].get(tri_key, 0) + 1

            # 3連複
            trio_key = '-'.join(map(str, sorted(o[:3])))
            sim_probs['trio'][trio_key] = sim_probs['trio'].get(trio_key, 0) + 1

            # 2連単
            exa_key = f"{o[0]}-{o[1]}"
            sim_probs['exacta'][exa_key] = sim_probs['exacta'].get(exa_key, 0) + 1

            # 2連複
            qui_key = '-'.join(map(str, sorted(o[:2])))
            sim_probs['quinella'][qui_key] = sim_probs['quinella'].get(qui_key, 0) + 1

            # 拡連複 (3着以内の2艇の全組合せ)
            top3 = o[:3]
            for ci in range(3):
                for cj in range(ci+1, 3):
                    pair = tuple(sorted([top3[ci], top3[cj]]))
                    wide_key = f"{pair[0]}-{pair[1]}"
                    sim_probs['wide'][wide_key] = sim_probs['wide'].get(wide_key, 0) + 1

        ev_progress.progress(1.0)

        # 確率に変換
        for bet_type in sim_probs:
            for key in sim_probs[bet_type]:
                sim_probs[bet_type][key] /= ev_n_sims

        # --- 期待値計算 ---
        ev_results = compute_expected_values(synthetic, sim_probs)

        # ============================================================
        # 結果表示
        # ============================================================

        st.markdown("""
        <div style='background:#1a1a2e; color:white; padding:15px; border-radius:10px; margin:15px 0;'>
            <h3 style='color:#7dff7d; margin:0;'>💰 期待値分析結果</h3>
            <p style='color:#aaa; margin:5px 0 0 0;'>期待値 > 1.0 の買い目は理論上プラス収支（緑で表示）</p>
        </div>
        """, unsafe_allow_html=True)

        # 各券種のタブ
        tab_3t, tab_3f, tab_2t, tab_2f, tab_wide = st.tabs([
            "🎯 3連単", "🎯 3連複", "🎯 2連単", "🎯 2連複", "🎯 拡連複"
        ])

        def show_ev_table(bet_type, ev_data, title, show_top=30):
            if not ev_data.get(bet_type):
                st.warning(f"{title}のデータがありません")
                return

            items = ev_data[bet_type]
            # 期待値降順でソート
            sorted_items = sorted(items.items(), key=lambda x: x[1]['ev'], reverse=True)[:show_top]

            rows = []
            for combo, data in sorted_items:
                ev_color = "🟢" if data['ev'] > 1.0 else "🔴" if data['ev'] < 0.5 else "🟡"
                rows.append({
                    '': ev_color,
                    '組合せ': combo,
                    'オッズ': f"{data['odds']:.1f}",
                    '確率': f"{data['prob']*100:.2f}%",
                    '期待値': f"{data['ev']:.3f}",
                    '判定': '◎ 買い' if data['ev'] > 1.2 else '○ 妙味' if data['ev'] > 1.0 else '△' if data['ev'] > 0.7 else '✕'
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # 期待値 > 1.0 の数
            profitable = sum(1 for _, d in items.items() if d['ev'] > 1.0)
            total = len(items)
            st.info(f"📊 {title}: {total}通り中 **{profitable}通り** が期待値プラス（EV > 1.0）")

        with tab_3t:
            st.markdown(f"### 3連単 期待値ランキング（{len(ev_results.get('trifecta', {}))}通り）")
            show_ev_table('trifecta', ev_results, '3連単', show_top=30)

        with tab_3f:
            st.markdown(f"### 3連複 合成オッズ & 期待値（{len(ev_results.get('trio', {}))}通り）")
            show_ev_table('trio', ev_results, '3連複', show_top=20)

        with tab_2t:
            st.markdown(f"### 2連単 合成オッズ & 期待値（{len(ev_results.get('exacta', {}))}通り）")
            show_ev_table('exacta', ev_results, '2連単', show_top=30)

        with tab_2f:
            st.markdown(f"### 2連複 合成オッズ & 期待値（{len(ev_results.get('quinella', {}))}通り）")
            show_ev_table('quinella', ev_results, '2連複', show_top=15)

        with tab_wide:
            st.markdown(f"### 拡連複 合成オッズ & 期待値（{len(ev_results.get('wide', {}))}通り）")
            show_ev_table('wide', ev_results, '拡連複', show_top=15)

        # ============================================================
        # 期待値サマリーグラフ
        # ============================================================

        st.markdown("### 📊 期待値サマリー")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- 左: 券種別の期待値プラス率 ---
        bet_names_jp = {'trifecta': '3連単', 'trio': '3連複', 'exacta': '2連単',
                        'quinella': '2連複', 'wide': '拡連複'}
        bet_types_order = ['trifecta', 'trio', 'exacta', 'quinella', 'wide']

        profit_rates = []
        avg_evs = []
        for bt in bet_types_order:
            items = ev_results.get(bt, {})
            if items:
                pr = sum(1 for d in items.values() if d['ev'] > 1.0) / len(items) * 100
                ae = np.mean([d['ev'] for d in items.values()])
            else:
                pr = 0; ae = 0
            profit_rates.append(pr)
            avg_evs.append(ae)

        x_labels = [bet_names_jp[bt] for bt in bet_types_order]
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        axes[0].bar(x_labels, profit_rates, color=colors_bar, edgecolor='black', linewidth=0.5)
        axes[0].set_title('券種別 期待値プラス率', fontweight='bold')
        axes[0].set_ylabel('EV > 1.0 の割合 (%)')
        for j, v in enumerate(profit_rates):
            axes[0].text(j, v+0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # --- 右: 2連単 期待値TOP10 ---
        exa_items = ev_results.get('exacta', {})
        if exa_items:
            top_exa = sorted(exa_items.items(), key=lambda x: x[1]['ev'], reverse=True)[:10]
            combos = [c for c, _ in top_exa]
            evs = [d['ev'] for _, d in top_exa]
            colors_ev = ['#2ecc71' if e > 1.0 else '#e74c3c' for e in evs]
            axes[1].barh(combos[::-1], evs[::-1], color=colors_ev[::-1], edgecolor='black', linewidth=0.5)
            axes[1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='期待値 = 1.0')
            axes[1].set_title('2連単 期待値TOP10', fontweight='bold')
            axes[1].set_xlabel('期待値')
            axes[1].legend()
            for bar, val in zip(axes[1].patches, evs[::-1]):
                axes[1].text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                            f'{val:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ============================================================
        # おすすめ買い目
        # ============================================================

        st.markdown("### 🏆 おすすめ買い目（期待値 > 1.0）")

        all_profitable = []
        for bt in bet_types_order:
            for combo, data in ev_results.get(bt, {}).items():
                if data['ev'] > 1.0:
                    all_profitable.append({
                        '券種': bet_names_jp[bt],
                        '組合せ': combo,
                        'オッズ': data['odds'],
                        '確率': f"{data['prob']*100:.2f}%",
                        '期待値': data['ev'],
                    })

        if all_profitable:
            all_profitable.sort(key=lambda x: x['期待値'], reverse=True)
            df_profit = pd.DataFrame(all_profitable[:30])
            st.dataframe(df_profit, hide_index=True, use_container_width=True)
            st.success(f"🎉 全{len(all_profitable)}通りの期待値プラス買い目が見つかりました！")
        else:
            st.warning("期待値 > 1.0 の買い目は見つかりませんでした。")

        st.markdown("---")
        st.warning("⚠️ 期待値はシミュレーション確率×オッズの理論値です。実際のレース結果を保証するものではありません。"
                   "オッズは変動します。余裕のある範囲でお楽しみください。")

elif not agents:
    st.info("☝️ 先にレースデータを入力してシミュレーションを実行してください。")
elif not trifecta_odds:
    st.info("☝️ 3連単オッズを貼り付けるか手動入力してください。")

