"""先行研究(コンペ論文)の5条件に基づく「ドリブルイベント」抽出。

DataStadiumのプレーデータが持つ「ボール保持選手が得た/失った時刻」に相当する情報が
idsse-dataには無いため、trackingデータのball_owning_team(チーム単位、25Hz)と、
ボールに最も近い自チーム選手、という2つの手がかりから個人単位の保持区間を近似する。

5条件:
  (1) シーン内で最近接相手選手(ディフェンダー)が変化しない
  (2) アタッカーのボール保持時間が0.5秒以上
  (3) シーン内のアタッカー・ディフェンダーの直線移動距離(開始→終了の変位)がともに5m以上
  (4) ボール保持選手がGKではない
  (5) シーン開始時、アタッカー→ゴール方向とアタッカー→ディフェンダー方向のなす角が30度以内
"""

from dataclasses import dataclass

import numpy as np
from kloppy import sportec
from kloppy.domain import AttackingDirection

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FRAME_RATE = 25
DT = 1.0 / FRAME_RATE

CARRY_THRESHOLD_M = 3.0
MIN_DURATION_S = 0.5
MIN_DISPLACEMENT_M = 5.0
ANGLE_THRESHOLD_DEG = 60.0  # Yamazaki et al. (2026) Methods B, condition (iv)


@dataclass
class DribbleEvent:
    match_id: str
    period_id: int
    frames: list
    attacker_id: str
    defender_id: str
    duration_s: float
    attacker_disp_m: float
    defender_disp_m: float
    start_angle_deg: float


def to_xy(coord):
    return coord.x * PITCH_LENGTH, coord.y * PITCH_WIDTH


def goal_position(attacking_direction):
    # attacking team's target goal, normalized coords -> meters, y at pitch center
    if attacking_direction == AttackingDirection.LTR:
        return PITCH_LENGTH, PITCH_WIDTH / 2
    else:
        return 0.0, PITCH_WIDTH / 2


def nearest_teammate_to_ball(frame, team, ball_xy):
    best_player, best_dist = None, float("inf")
    for player in team.players:
        pd_ = frame.players_data.get(player)
        if pd_ is None or pd_.coordinates is None:
            continue
        px, py = to_xy(pd_.coordinates)
        dist = ((px - ball_xy[0]) ** 2 + (py - ball_xy[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_player = dist, player
    return best_player, best_dist


def nearest_opponent(frame, opponents, attacker_xy):
    best_player, best_dist = None, float("inf")
    for player in opponents:
        pd_ = frame.players_data.get(player)
        if pd_ is None or pd_.coordinates is None:
            continue
        px, py = to_xy(pd_.coordinates)
        dist = ((px - attacker_xy[0]) ** 2 + (py - attacker_xy[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_player = dist, player
    return best_player, best_dist


def is_goalkeeper(player):
    return str(player.starting_position) == "Goalkeeper"


def build_carrier_runs(frames, teams):
    """フレーム列から (carrier_player, team, run_frames) の連続区間を抽出する。"""
    team_by_name = {t.name: t for t in teams}
    runs = []
    current_player = None
    current_team = None
    current_frames = []

    for f in frames:
        if str(f.ball_state) != "BallState.ALIVE" or f.ball_owning_team is None:
            player = None
        else:
            team = team_by_name.get(f.ball_owning_team.name)
            if team is None:
                player = None
            else:
                ball_xy = to_xy(f.ball_coordinates)
                player, dist = nearest_teammate_to_ball(f, team, ball_xy)
                if player is None or dist > CARRY_THRESHOLD_M:
                    player = None

        if player is not None and player == current_player:
            current_frames.append(f)
        else:
            if current_player is not None and len(current_frames) >= 2:
                runs.append((current_player, current_team, current_frames))
            current_player = player
            current_team = team_by_name.get(f.ball_owning_team.name) if player is not None else None
            current_frames = [f] if player is not None else []

    if current_player is not None and len(current_frames) >= 2:
        runs.append((current_player, current_team, current_frames))
    return runs


def evaluate_run(match_id, attacker, attacker_team, run_frames, teams):
    if is_goalkeeper(attacker):
        return None

    opponent_team = next(t for t in teams if t.name != attacker_team.name)
    opponents = opponent_team.players

    duration_s = (len(run_frames) - 1) * DT
    if duration_s < MIN_DURATION_S:
        return None

    defender_ids = []
    defender_by_frame = []
    for f in run_frames:
        pd_ = f.players_data.get(attacker)
        if pd_ is None or pd_.coordinates is None:
            return None
        attacker_xy = to_xy(pd_.coordinates)
        defender, _ = nearest_opponent(f, opponents, attacker_xy)
        if defender is None:
            return None
        defender_ids.append(defender.player_id)
        defender_by_frame.append(defender)

    if len(set(defender_ids)) != 1:
        return None  # 条件(1)
    defender = defender_by_frame[0]

    a0 = run_frames[0].players_data.get(attacker).coordinates
    a1 = run_frames[-1].players_data.get(attacker).coordinates
    d0 = run_frames[0].players_data.get(defender).coordinates
    d1 = run_frames[-1].players_data.get(defender).coordinates
    if None in (a0, a1, d0, d1):
        return None

    ax0, ay0 = to_xy(a0)
    ax1, ay1 = to_xy(a1)
    dx0, dy0 = to_xy(d0)
    dx1, dy1 = to_xy(d1)

    attacker_disp = ((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2) ** 0.5
    defender_disp = ((dx1 - dx0) ** 2 + (dy1 - dy0) ** 2) ** 0.5
    if attacker_disp < MIN_DISPLACEMENT_M or defender_disp < MIN_DISPLACEMENT_M:
        return None  # 条件(3)

    gx, gy = goal_position(run_frames[0].attacking_direction)
    v_goal = np.array([gx - ax0, gy - ay0])
    v_def = np.array([dx0 - ax0, dy0 - ay0])
    if np.linalg.norm(v_goal) < 1e-6 or np.linalg.norm(v_def) < 1e-6:
        return None
    cos_angle = np.dot(v_goal, v_def) / (np.linalg.norm(v_goal) * np.linalg.norm(v_def))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle_deg > ANGLE_THRESHOLD_DEG:
        return None  # 条件(5)

    return DribbleEvent(
        match_id=match_id,
        period_id=run_frames[0].period.id,
        frames=run_frames,
        attacker_id=attacker.player_id,
        defender_id=defender.player_id,
        duration_s=duration_s,
        attacker_disp_m=attacker_disp,
        defender_disp_m=defender_disp,
        start_angle_deg=angle_deg,
    )


def extract_dribble_events(match_id, return_context=False):
    tracking = sportec.load_open_tracking_data(match_id=match_id)
    teams = tracking.metadata.teams

    frames_by_period = {}
    for f in tracking.frames:
        frames_by_period.setdefault(f.period.id, []).append(f)
    for pid in frames_by_period:
        frames_by_period[pid].sort(key=lambda fr: fr.timestamp)

    events = []
    n_runs = 0
    for pid, frames in frames_by_period.items():
        runs = build_carrier_runs(frames, teams)
        n_runs += len(runs)
        for attacker, team, run_frames in runs:
            ev = evaluate_run(match_id, attacker, team, run_frames, teams)
            if ev is not None:
                events.append(ev)

    print(f"{match_id}: {n_runs} candidate possession runs -> {len(events)} dribble events (5 conditions)")
    if return_context:
        return events, frames_by_period, teams
    return events


if __name__ == "__main__":
    for match_id in ["J03WPY"]:
        evs = extract_dribble_events(match_id)
        durations = [e.duration_s for e in evs]
        if durations:
            print(f"duration: mean={np.mean(durations):.2f}s, median={np.median(durations):.2f}s, "
                  f"min={min(durations):.2f}s, max={max(durations):.2f}s")
