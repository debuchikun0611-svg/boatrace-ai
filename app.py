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
# 🎰 オッズ自動取得 & 期待値計算セクション (v2.0)
# ============================================================
import itertools
from urllib.request import urlopen, Request
from io import StringIO

st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 20px; border-radius: 15px; margin: 10px 0;'>
    <h2 style='color: #e94560; text-align: center;'>🎰 オッズ分析 & 期待値計算</h2>
    <p style='color: #eee; text-align: center;'>
        3連単オッズを自動取得し、全券種（2連単・2連複・3連複・拡連複）の合成オッズと期待値を計算します
    </p>
</div>
""", unsafe_allow_html=True)

# --- 会場コード辞書 ---
VENUE_CODES = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05",
    "浜名湖": "06", "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10",
    "びわこ": "11", "住之江": "12", "尼崎": "13", "鳴門": "14", "丸亀": "15",
    "児島": "16", "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

# =============================================================
# 3連単オッズ取得関数 (公式サイトからスクレイピング)
# =============================================================
def fetch_trifecta_odds_from_official(venue_code: str, date_str: str, race_no: int) -> dict:
    """
    boatrace.jp から3連単オッズを取得する。
    venue_code: "18" (徳山), date_str: "20260227", race_no: 1-12
    戻り値: {"1-2-3": 47.2, "1-2-4": 60.3, ...} の辞書 (120通り)
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={race_no}&jcd={venue_code}&hd={date_str}"
    
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html_content = urlopen(req, timeout=10).read()
    except Exception as e:
        st.error(f"オッズページの取得に失敗しました: {e}")
        return {}
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
    except ImportError:
        st.error("BeautifulSoup がインストールされていません。requirements.txt に beautifulsoup4 を追加してください。")
        return {}
    
    # テーブルを探す: class="is-p3-0" の table、または oddsPoint を含むテーブル
    odds_dict = {}
    
    # 方法1: oddsPoint クラスの td を持つ tbody を探す
    all_tables = soup.find_all('table')
    target_table = None
    for table in all_tables:
        if table.find('td', class_='oddsPoint'):
            target_table = table
            break
    
    if target_table is None:
        # 方法2: CSS selector で直接指定
        selectors = [
            'body > main > div > div > div > div.contentsFrame1_inner > div:nth-child(6) > table',
            'div.contentsFrame1_inner table',
            'table.is-w495'
        ]
        for sel in selectors:
            target_table = soup.select_one(sel)
            if target_table and target_table.find('td', class_='oddsPoint'):
                break
            target_table = None
    
    if target_table is None:
        # 方法3: テキストからパース (フォールバック)
        return _parse_odds_from_text(soup.get_text())
    
    # tbody の tr からオッズを抽出
    tbody = target_table.find('tbody')
    if tbody is None:
        tbody = target_table
    
    rows = tbody.find_all('tr')
    
    odds_values = []
    for row in rows:
        cells = row.find_all('td', class_='oddsPoint')
        for cell in cells:
            text = cell.get_text(strip=True)
            try:
                odds_values.append(float(text))
            except ValueError:
                odds_values.append(0.0)  # 特払い等
    
    # 120通りの3連単オッズを辞書に格納
    # テーブル構造: 1着固定で、2着-3着の組み合わせ順に並ぶ
    # 行の構造: 1着=1,2着=2,3着=3,4,5,6 / 1着=1,2着=3,3着=2,4,5,6 / ...
    # 各行に6個のオッズ (6つの3着候補 - ただし1着,2着を除くので4個)
    
    # 公式サイトの構造に合わせたマッピング
    # 20行 × 6列 = 120通り (ただし実際は 20行 × 可変列)
    
    if len(odds_values) == 120:
        # きれいに120個取れた場合
        idx = 0
        for first in range(1, 7):
            for second in range(1, 7):
                if second == first:
                    continue
                for third in range(1, 7):
                    if third == first or third == second:
                        continue
                    key = f"{first}-{second}-{third}"
                    if idx < len(odds_values):
                        odds_dict[key] = odds_values[idx]
                        idx += 1
    else:
        # テーブル構造に合わせた解析
        # 各1着固定で4行、各行4オッズ (他の5艇から2着を選び、残り4艇が3着)
        idx = 0
        for first in range(1, 7):
            others = [x for x in range(1, 7) if x != first]
            for second in others:
                thirds = [x for x in range(1, 7) if x != first and x != second]
                for third in thirds:
                    key = f"{first}-{second}-{third}"
                    if idx < len(odds_values):
                        odds_dict[key] = odds_values[idx]
                        idx += 1
    
    return odds_dict


def _parse_odds_from_text(text: str) -> dict:
    """テキストからオッズを抽出するフォールバック"""
    import re
    odds_dict = {}
    # "X-Y-Z  odds" パターン
    pattern = r'(\d)-(\d)-(\d)\s+([\d.]+)'
    matches = re.findall(pattern, text)
    for m in matches:
        key = f"{m[0]}-{m[1]}-{m[2]}"
        try:
            odds_dict[key] = float(m[3])
        except ValueError:
            pass
    return odds_dict


def parse_pasted_odds_text(text: str) -> dict:
    """
    公式サイトからコピペしたテキストを解析して3連単オッズ辞書を返す。
    対応フォーマット:
      - "1-2-3  5.0" 形式
      - 公式テーブルコピー形式 (数字が並ぶ)
    """
    import re
    odds_dict = {}
    
    # パターン1: "X-Y-Z  odds" 形式
    pattern1 = re.findall(r'(\d)\s*[-=]\s*(\d)\s*[-=]\s*(\d)\s+([\d,.]+)', text)
    if pattern1:
        for m in pattern1:
            key = f"{m[0]}-{m[1]}-{m[2]}"
            try:
                odds_dict[key] = float(m[3].replace(',', ''))
            except ValueError:
                pass
        if len(odds_dict) >= 60:
            return odds_dict
    
    # パターン2: crawlerで取得したテキストフォーマット
    # "2   3   6.2     1   3   108.8  ..." のような構造
    lines = text.strip().split('\n')
    
    # 各1着艇ごとのブロックを検出
    current_first = None
    current_second = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 数字とオッズの組み合わせを探す
        # "2   3   6.2" → 2着=2, 3着=3, オッズ=6.2
        tokens = line.split()
        
        # 各トークンペア (数字, 小数) をスキャン
        i = 0
        while i < len(tokens) - 1:
            try:
                boat_num = int(tokens[i])
                if 1 <= boat_num <= 6:
                    # 次のトークンがオッズか？
                    next_val = tokens[i + 1].replace(',', '')
                    odds_val = float(next_val)
                    if 1.0 <= odds_val <= 99999:
                        # これは 3着=boat_num, オッズ=odds_val と解釈
                        # ただし2着と1着の情報が必要
                        pass
                    i += 2
                    continue
            except (ValueError, IndexError):
                pass
            i += 1
    
    # パターン3: 数字のみの行を集めて120個のオッズとして解釈
    all_numbers = re.findall(r'(?<!\d)(\d{1,5}(?:\.\d+)?)(?!\d)', text)
    float_candidates = []
    for n in all_numbers:
        try:
            v = float(n)
            if 1.0 <= v <= 99999:
                float_candidates.append(v)
        except ValueError:
            pass
    
    if len(float_candidates) >= 120:
        # 先頭120個を使う
        idx = 0
        for first in range(1, 7):
            for second in range(1, 7):
                if second == first:
                    continue
                for third in range(1, 7):
                    if third == first or third == second:
                        continue
                    key = f"{first}-{second}-{third}"
                    odds_dict[key] = float_candidates[idx]
                    idx += 1
        return odds_dict
    
    return odds_dict


# =============================================================
# 合成オッズ計算
# =============================================================
def compute_all_synthetic_odds(trifecta_odds: dict) -> dict:
    """
    3連単オッズから全券種の合成オッズを計算する。
    合成オッズ = 1 / Σ(1/各該当オッズ)
    
    券種:
      - 3連単 (trifecta): そのまま
      - 3連複 (trio): 1-2-3 と同着順の全順列を合成
      - 2連単 (exacta): 1着-2着固定、3着は全通り
      - 2連複 (quinella): 1着-2着の順不問、3着は全通り
      - 拡連複 (wide/quinella_place): 2艇が3着以内に入る全組合せ
    """
    boats = [1, 2, 3, 4, 5, 6]
    result = {
        'trifecta': {},   # 3連単: 120通り
        'trio': {},       # 3連複: 20通り
        'exacta': {},     # 2連単: 30通り
        'quinella': {},   # 2連複: 15通り
        'wide': {},       # 拡連複: 15通り
    }
    
    # 3連単はそのまま
    result['trifecta'] = dict(trifecta_odds)
    
    # 3連複: 3艇の組み合わせ (順不問)
    for combo in itertools.combinations(boats, 3):
        a, b, c = combo
        key = f"{a}={b}={c}"
        inv_sum = 0.0
        count = 0
        for perm in itertools.permutations(combo):
            tkey = f"{perm[0]}-{perm[1]}-{perm[2]}"
            if tkey in trifecta_odds and trifecta_odds[tkey] > 0:
                inv_sum += 1.0 / trifecta_odds[tkey]
                count += 1
        if inv_sum > 0:
            result['trio'][key] = round(1.0 / inv_sum, 1)
        else:
            result['trio'][key] = 0.0
    
    # 2連単: 1着-2着固定、3着は残り4艇のどれでもOK
    for first in boats:
        for second in boats:
            if first == second:
                continue
            key = f"{first}-{second}"
            inv_sum = 0.0
            for third in boats:
                if third == first or third == second:
                    continue
                tkey = f"{first}-{second}-{third}"
                if tkey in trifecta_odds and trifecta_odds[tkey] > 0:
                    inv_sum += 1.0 / trifecta_odds[tkey]
            if inv_sum > 0:
                result['exacta'][key] = round(1.0 / inv_sum, 1)
            else:
                result['exacta'][key] = 0.0
    
    # 2連複: 2艇の組み合わせが1着-2着 (順不問)、3着は残り4艇
    for combo in itertools.combinations(boats, 2):
        a, b = combo
        key = f"{a}={b}"
        inv_sum = 0.0
        for perm in itertools.permutations(combo):
            for third in boats:
                if third in combo:
                    continue
                tkey = f"{perm[0]}-{perm[1]}-{third}"
                if tkey in trifecta_odds and trifecta_odds[tkey] > 0:
                    inv_sum += 1.0 / trifecta_odds[tkey]
        if inv_sum > 0:
            result['quinella'][key] = round(1.0 / inv_sum, 1)
        else:
            result['quinella'][key] = 0.0
    
    # 拡連複: 2艇が3着以内に入ればOK (順不問、3着目は残り4艇のどれか)
    for combo in itertools.combinations(boats, 2):
        a, b = combo
        key = f"{a}={b}"
        inv_sum = 0.0
        # a,bが共に3着以内に入る3連単の全パターン
        others = [x for x in boats if x not in combo]
        for third_boat in others:
            # 3艇 = {a, b, third_boat} の全順列
            for perm in itertools.permutations([a, b, third_boat]):
                tkey = f"{perm[0]}-{perm[1]}-{perm[2]}"
                if tkey in trifecta_odds and trifecta_odds[tkey] > 0:
                    inv_sum += 1.0 / trifecta_odds[tkey]
        if inv_sum > 0:
            result['wide'][key] = round(1.0 / inv_sum, 1)
        else:
            result['wide'][key] = 0.0
    
    return result


# =============================================================
# 期待値計算
# =============================================================
def compute_expected_values(synthetic_odds: dict, sim_probabilities: dict) -> dict:
    """
    期待値 = シミュレーション確率 × オッズ
    EV > 1.0 なら理論上プラス期待値
    """
    ev_results = {}
    for ticket_type in synthetic_odds:
        ev_results[ticket_type] = {}
        for key, odds in synthetic_odds[ticket_type].items():
            prob = sim_probabilities.get(ticket_type, {}).get(key, 0.0)
            ev = prob * odds
            ev_results[ticket_type][key] = {
                'odds': odds,
                'probability': prob,
                'expected_value': round(ev, 4),
                'profitable': ev > 1.0
            }
    return ev_results


def run_monte_carlo_for_ev(agents, conditions, n_sims=10000) -> dict:
    """
    モンテカルロシミュレーションを実行し、全券種の的中確率を算出する。
    """
    boats = [1, 2, 3, 4, 5, 6]
    
    # カウンタ初期化
    counts = {
        'trifecta': {},   # "1-2-3": count
        'trio': {},       # "1=2=3": count
        'exacta': {},     # "1-2": count
        'quinella': {},   # "1=2": count
        'wide': {},       # "1=2": count
    }
    
    # 全キーを初期化
    for perm in itertools.permutations(boats, 3):
        counts['trifecta'][f"{perm[0]}-{perm[1]}-{perm[2]}"] = 0
    for combo in itertools.combinations(boats, 3):
        counts['trio'][f"{combo[0]}={combo[1]}={combo[2]}"] = 0
    for perm in itertools.permutations(boats, 2):
        counts['exacta'][f"{perm[0]}-{perm[1]}"] = 0
    for combo in itertools.combinations(boats, 2):
        counts['quinella'][f"{combo[0]}={combo[1]}"] = 0
        counts['wide'][f"{combo[0]}={combo[1]}"] = 0
    
    sim = RaceSimulator(agents, conditions)
    
    progress_bar = st.progress(0)
    
    for i in range(n_sims):
        result = sim.simulate_race()
        order = result['finish_order']  # [lane1, lane2, lane3, lane4, lane5, lane6]
        
        top3 = order[:3]
        
        # 3連単
        tkey = f"{top3[0]}-{top3[1]}-{top3[2]}"
        if tkey in counts['trifecta']:
            counts['trifecta'][tkey] += 1
        
        # 3連複
        trio_sorted = sorted(top3)
        trio_key = f"{trio_sorted[0]}={trio_sorted[1]}={trio_sorted[2]}"
        if trio_key in counts['trio']:
            counts['trio'][trio_key] += 1
        
        # 2連単
        ekey = f"{top3[0]}-{top3[1]}"
        if ekey in counts['exacta']:
            counts['exacta'][ekey] += 1
        
        # 2連複
        quin_sorted = sorted(top3[:2])
        qkey = f"{quin_sorted[0]}={quin_sorted[1]}"
        if qkey in counts['quinella']:
            counts['quinella'][qkey] += 1
        
        # 拡連複 (3着以内の全2艇組合せ)
        for combo in itertools.combinations(top3, 2):
            wsorted = sorted(combo)
            wkey = f"{wsorted[0]}={wsorted[1]}"
            if wkey in counts['wide']:
                counts['wide'][wkey] += 1
        
        if (i + 1) % (n_sims // 20) == 0:
            progress_bar.progress((i + 1) / n_sims)
    
    progress_bar.progress(1.0)
    
    # 確率に変換
    probabilities = {}
    for ticket_type in counts:
        probabilities[ticket_type] = {}
        for key, count in counts[ticket_type].items():
            probabilities[ticket_type][key] = count / n_sims
    
    return probabilities


# =============================================================
# UI: オッズ取得・分析セクション
# =============================================================
odds_method = st.radio(
    "オッズ取得方法を選択:",
    ["🌐 自動取得 (公式サイトから)", "📋 テキスト貼り付け", "✏️ 手動入力"],
    horizontal=True,
    key="odds_method_radio"
)

trifecta_odds = {}

if odds_method == "🌐 自動取得 (公式サイトから)":
    st.info("💡 ボートレース公式サイト (boatrace.jp) から3連単オッズを自動取得します。レース締切後のオッズが取得されます。")
    
    col_v, col_d, col_r = st.columns(3)
    with col_v:
        venue_name = st.selectbox("会場", list(VENUE_CODES.keys()), 
                                   index=list(VENUE_CODES.keys()).index("徳山"),
                                   key="odds_venue")
    with col_d:
        from datetime import date
        odds_date = st.date_input("日付", value=date(2026, 2, 27), key="odds_date")
    with col_r:
        race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1, key="odds_race_no")
    
    if st.button("🔍 オッズを自動取得", key="btn_fetch_odds", type="primary"):
        venue_code = VENUE_CODES[venue_name]
        date_str = odds_date.strftime("%Y%m%d")
        
        with st.spinner(f"📡 {venue_name} {odds_date} {race_no}R のオッズを取得中..."):
            trifecta_odds = fetch_trifecta_odds_from_official(venue_code, date_str, race_no)
        
        if trifecta_odds and len(trifecta_odds) > 0:
            st.success(f"✅ {len(trifecta_odds)}通りの3連単オッズを取得しました！")
            st.session_state['trifecta_odds'] = trifecta_odds
            st.session_state['odds_source'] = f"{venue_name} {odds_date} {race_no}R (自動取得)"
            
            # オッズ表を表示
            with st.expander("📊 取得した3連単オッズ一覧", expanded=False):
                odds_df = pd.DataFrame([
                    {"買い目": k, "オッズ": v} for k, v in sorted(trifecta_odds.items())
                ])
                st.dataframe(odds_df, use_container_width=True, height=400)
        else:
            st.warning("⚠️ オッズを取得できませんでした。レース前またはデータ未公開の可能性があります。テキスト貼り付けをお試しください。")

elif odds_method == "📋 テキスト貼り付け":
    st.info("💡 公式サイトの3連単オッズページからテキストをコピーして貼り付けてください。\n\n"
            "**フォーマット例:**\n"
            "- `1-2-3  5.0` 形式\n"
            "- 公式ページのテーブルをそのままコピー")
    
    odds_text = st.text_area(
        "3連単オッズテキスト",
        height=300,
        placeholder="公式サイトの3連単オッズページのテキストを貼り付けてください...",
        key="odds_text_input"
    )
    
    if st.button("🔍 オッズを解析", key="btn_parse_odds"):
        if odds_text.strip():
            trifecta_odds = parse_pasted_odds_text(odds_text)
            if trifecta_odds and len(trifecta_odds) >= 60:
                st.success(f"✅ {len(trifecta_odds)}通りの3連単オッズを解析しました！")
                st.session_state['trifecta_odds'] = trifecta_odds
                st.session_state['odds_source'] = "テキスト貼り付け"
            else:
                st.warning(f"⚠️ {len(trifecta_odds)}通りしか解析できませんでした。フォーマットを確認してください。")
        else:
            st.warning("テキストを入力してください。")

elif odds_method == "✏️ 手動入力":
    st.info("💡 主要な買い目のオッズを手動で入力してください（入力しない組み合わせは0として扱います）")
    
    manual_text = st.text_area(
        "買い目とオッズ（1行1組）",
        height=200,
        placeholder="1-2-3 5.0\n1-3-2 8.5\n1-2-4 6.0\n...",
        key="manual_odds_input"
    )
    
    if st.button("📝 手動オッズ確定", key="btn_manual_odds"):
        if manual_text.strip():
            import re
            trifecta_odds = {}
            for line in manual_text.strip().split('\n'):
                match = re.match(r'(\d)\s*-\s*(\d)\s*-\s*(\d)\s+([\d,.]+)', line.strip())
                if match:
                    key = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                    trifecta_odds[key] = float(match.group(4).replace(',', ''))
            if trifecta_odds:
                st.success(f"✅ {len(trifecta_odds)}通りのオッズを登録しました。")
                st.session_state['trifecta_odds'] = trifecta_odds
                st.session_state['odds_source'] = "手動入力"


# =============================================================
# 期待値計算実行
# =============================================================
if 'trifecta_odds' in st.session_state and st.session_state['trifecta_odds']:
    st.markdown("---")
    st.markdown("### 💰 期待値計算")
    
    if 'agents' not in st.session_state or not st.session_state.get('agents'):
        st.warning("⚠️ 先にレースデータを入力してエージェントを作成してください（上部のデータ入力セクション）。")
    else:
        agents = st.session_state['agents']
        trifecta_odds = st.session_state['trifecta_odds']
        source_info = st.session_state.get('odds_source', '不明')
        
        st.info(f"📊 オッズソース: **{source_info}** ({len(trifecta_odds)}通り)")
        
        ev_n_sims = st.slider("期待値計算用シミュレーション回数", 
                               min_value=1000, max_value=50000, value=10000, step=1000,
                               key="ev_sims_slider")
        
        if st.button("💰 期待値を計算する", key="btn_calc_ev", type="primary"):
            # 気象条件 (session_stateから取得またはデフォルト)
            conditions = st.session_state.get('conditions', 
                RaceCondition(temperature=12, wind_speed=3, wave_height=3, weather="曇り"))
            
            # Step 1: 合成オッズ計算
            with st.spinner("📐 合成オッズを計算中..."):
                synthetic_odds = compute_all_synthetic_odds(trifecta_odds)
            
            # Step 2: モンテカルロシミュレーション
            st.write(f"🎲 モンテカルロシミュレーション ({ev_n_sims}回) 実行中...")
            sim_probs = run_monte_carlo_for_ev(agents, conditions, ev_n_sims)
            
            # Step 3: 期待値計算
            with st.spinner("💹 期待値を算出中..."):
                ev_results = compute_expected_values(synthetic_odds, sim_probs)
            
            st.success("✅ 計算完了！")
            
            # =============================================================
            # 結果表示
            # =============================================================
            ticket_names = {
                'trifecta': '3連単',
                'trio': '3連複',
                'exacta': '2連単',
                'quinella': '2連複',
                'wide': '拡連複'
            }
            
            # タブで各券種を表示
            tabs = st.tabs([f"🎯 {name}" for name in ticket_names.values()])
            
            for (ticket_type, tab_name), tab in zip(ticket_names.items(), tabs):
                with tab:
                    ev_data = ev_results[ticket_type]
                    
                    # DataFrameに変換
                    rows = []
                    for key, info in ev_data.items():
                        if info['odds'] > 0:
                            rows.append({
                                '買い目': key,
                                'オッズ': info['odds'],
                                '的中確率': f"{info['probability']*100:.2f}%",
                                '確率(raw)': info['probability'],
                                '期待値': info['expected_value'],
                                '判定': '🟢 買い' if info['profitable'] else '🔴'
                            })
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        df = df.sort_values('期待値', ascending=False)
                        
                        # プラス期待値の数
                        profitable_count = sum(1 for r in rows if r['判定'] == '🟢 買い')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("総買い目数", len(rows))
                        with col2:
                            st.metric("プラス期待値", f"{profitable_count}件")
                        with col3:
                            if rows:
                                max_ev = max(r['期待値'] for r in rows)
                                st.metric("最大期待値", f"{max_ev:.2f}")
                        
                        # 期待値上位表示
                        st.markdown(f"#### 📈 {tab_name} 期待値ランキング (上位20)")
                        display_df = df.head(20)[['買い目', 'オッズ', '的中確率', '期待値', '判定']].reset_index(drop=True)
                        display_df.index = display_df.index + 1
                        
                        # 期待値>1.0を強調
                        def highlight_ev(row):
                            if row['期待値'] > 1.0:
                                return ['background-color: #d4edda'] * len(row)
                            elif row['期待値'] > 0.8:
                                return ['background-color: #fff3cd'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(
                            display_df.style.apply(highlight_ev, axis=1),
                            use_container_width=True
                        )
                        
                        # グラフ: 上位15の期待値棒グラフ
                        top15 = df.head(15)
                        fig_ev, ax_ev = plt.subplots(figsize=(10, 5))
                        colors = ['#27ae60' if ev > 1.0 else '#e74c3c' if ev < 0.5 else '#f39c12' 
                                  for ev in top15['期待値']]
                        bars = ax_ev.barh(range(len(top15)), top15['期待値'].values, color=colors)
                        ax_ev.set_yticks(range(len(top15)))
                        ax_ev.set_yticklabels(top15['買い目'].values)
                        ax_ev.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='損益分岐 (EV=1.0)')
                        ax_ev.set_xlabel('期待値')
                        ax_ev.set_title(f'{tab_name} 期待値 Top15')
                        ax_ev.legend()
                        ax_ev.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig_ev)
                        plt.close()
                    else:
                        st.info("該当するオッズデータがありません。")
            
            # =============================================================
            # おすすめ買い目サマリー
            # =============================================================
            st.markdown("---")
            st.markdown("### 🏆 おすすめ買い目 (期待値 > 1.0)")
            
            all_profitable = []
            for ticket_type, name in ticket_names.items():
                for key, info in ev_results[ticket_type].items():
                    if info['profitable'] and info['odds'] > 0:
                        all_profitable.append({
                            '券種': name,
                            '買い目': key,
                            'オッズ': info['odds'],
                            '的中確率': f"{info['probability']*100:.2f}%",
                            '期待値': info['expected_value']
                        })
            
            if all_profitable:
                prof_df = pd.DataFrame(all_profitable)
                prof_df = prof_df.sort_values('期待値', ascending=False)
                st.dataframe(prof_df, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 💡 買い目ガイド")
                
                # 券種別サマリー
                for name in ticket_names.values():
                    subset = [r for r in all_profitable if r['券種'] == name]
                    if subset:
                        top3 = sorted(subset, key=lambda x: x['期待値'], reverse=True)[:3]
                        recs = " / ".join([f"**{r['買い目']}** (EV={r['期待値']:.2f}, オッズ{r['オッズ']})" for r in top3])
                        st.write(f"**{name}**: {recs}")
            else:
                st.info("現在のシミュレーション結果では、期待値 > 1.0 の買い目はありません。\n\n"
                        "💡 これはオッズとシミュレーション確率のバランスによるものです。"
                        "期待値 0.8 以上の買い目も検討の価値があります。")
                
                # 期待値0.8以上を表示
                near_profitable = []
                for ticket_type, name in ticket_names.items():
                    for key, info in ev_results[ticket_type].items():
                        if info['expected_value'] > 0.8 and info['odds'] > 0:
                            near_profitable.append({
                                '券種': name,
                                '買い目': key,
                                'オッズ': info['odds'],
                                '的中確率': f"{info['probability']*100:.2f}%",
                                '期待値': info['expected_value']
                            })
                
                if near_profitable:
                    st.markdown("#### 📊 準おすすめ (期待値 0.8 以上)")
                    near_df = pd.DataFrame(near_profitable).sort_values('期待値', ascending=False)
                    st.dataframe(near_df.head(20), use_container_width=True)

            st.markdown("---")
            st.caption("⚠️ 注意: 期待値はシミュレーション結果に基づく推定値です。実際のレース結果を保証するものではありません。"
                       "また、オッズはレース締切まで変動します。投票は自己責任でお願いします。")
