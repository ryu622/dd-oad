"""判定②: θ_dの「瞬間値」を推定する手段があるかを確認する。

判定①で抽出した1v1イベント(TacklingGame, ground, WinnerRole=withBallControl)のうち
少数サンプルについて、ディフェンダー(Loser)の加速度ベクトルを

  1) 生の2階差分
  2) Savitzky-Golayフィルタによる平滑化

の両方で算出し、ノイズの大きさを比較する。さらに算出した加速度ベクトルを
e1(アタッカー→ディフェンダー方向), e2(その90度回転)に射影し、
瞬間的なθ_d(t)相当の量(駆動方向の角度)を定義できるかを確認する。

注意: v_d/τ_d の抗力項は、τ_dが未推定(フェーズ1で実施)のため、
本判定では簡略化のため無視した近似(a_d ≈ f_d[-cosθ_d e1 + sinθ_d e2])を用いる。
あくまで「ノイズに埋もれず定性的な傾向が取れるか」の実現可能性チェックが目的。
"""

from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from kloppy import sportec
from scipy.signal import savgol_filter

MATCH_ID = "J03WPY"
WINDOW_SECONDS = 2.5
PAD_SECONDS = 0.5
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FRAME_RATE = 25
DT = 1.0 / FRAME_RATE
SG_WINDOW = 11  # ~0.44s
SG_POLYORDER = 3
N_EXAMPLES = 8


def extract_duel_events(events):
    duels = []
    for e in events.events:
        if e.event_name != "TacklingGame":
            continue
        raw = e.raw_event
        if raw.get("Type") != "ground":
            continue
        if raw.get("WinnerRole") != "withBallControl":
            continue
        if raw.get("LoserRole") != "withoutBallControl":
            continue
        if raw.get("GoalKeeperInvolved") == "true":
            continue
        duels.append(
            {
                "period_id": e.period.id,
                "timestamp": e.timestamp,
                "winner_id": raw["Winner"],
                "loser_id": raw["Loser"],
            }
        )
    return duels


def build_frame_index(tracking):
    frames_by_period = {}
    for f in tracking.frames:
        frames_by_period.setdefault(f.period.id, []).append(f)
    for pid in frames_by_period:
        frames_by_period[pid].sort(key=lambda fr: fr.timestamp)
    return frames_by_period


def player_lookup(tracking):
    lookup = {}
    for team in tracking.metadata.teams:
        for p in team.players:
            lookup[p.player_id] = p
    return lookup


def extract_xy(frames_window, player):
    ts, xs, ys = [], [], []
    for f in frames_window:
        pd_ = f.players_data.get(player)
        if pd_ is None or pd_.coordinates is None:
            continue
        ts.append(f.timestamp.total_seconds())
        xs.append(pd_.coordinates.x * PITCH_LENGTH)
        ys.append(pd_.coordinates.y * PITCH_WIDTH)
    return np.array(ts), np.array(xs), np.array(ys)


def raw_second_diff(x, dt):
    v = np.gradient(x, dt)
    a = np.gradient(v, dt)
    return v, a


def sg_derivatives(x, dt, window, polyorder):
    x_s = savgol_filter(x, window_length=window, polyorder=polyorder)
    v_s = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=1, delta=dt)
    a_s = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=2, delta=dt)
    return x_s, v_s, a_s


def main():
    events = sportec.load_open_event_data(match_id=MATCH_ID)
    tracking = sportec.load_open_tracking_data(match_id=MATCH_ID)

    duels = extract_duel_events(events)
    frames_by_period = build_frame_index(tracking)
    players = player_lookup(tracking)

    examples = []
    raw_acc_all = []
    sg_acc_all = []

    for duel in duels:
        winner_player = players.get(duel["winner_id"])
        loser_player = players.get(duel["loser_id"])
        if winner_player is None or loser_player is None:
            continue

        period_frames = frames_by_period.get(duel["period_id"], [])
        t_end = duel["timestamp"] + timedelta(seconds=PAD_SECONDS)
        t_start = duel["timestamp"] - timedelta(seconds=WINDOW_SECONDS + PAD_SECONDS)

        window = [f for f in period_frames if t_start <= f.timestamp <= t_end]
        if len(window) < SG_WINDOW + 5:
            continue

        t_a, xa, ya = extract_xy(window, winner_player)
        t_d, xd, yd = extract_xy(window, loser_player)
        if len(t_d) < SG_WINDOW + 5 or len(t_a) != len(t_d):
            continue

        # raw second difference (defender)
        _, ax_raw = raw_second_diff(xd, DT)
        _, ay_raw = raw_second_diff(yd, DT)
        acc_raw = np.sqrt(ax_raw**2 + ay_raw**2)

        # Savitzky-Golay smoothed
        xd_s, vxd_s, axd_s = sg_derivatives(xd, DT, SG_WINDOW, SG_POLYORDER)
        yd_s, vyd_s, ayd_s = sg_derivatives(yd, DT, SG_WINDOW, SG_POLYORDER)
        acc_sg = np.sqrt(axd_s**2 + ayd_s**2)

        xa_s, _, _ = sg_derivatives(xa, DT, SG_WINDOW, SG_POLYORDER)
        ya_s, _, _ = sg_derivatives(ya, DT, SG_WINDOW, SG_POLYORDER)

        # e1: attacker -> defender unit vector (smoothed positions), e2: 90deg rotation
        ex = xd_s - xa_s
        ey = yd_s - ya_s
        norm = np.sqrt(ex**2 + ey**2)
        valid = norm > 1e-6
        e1x, e1y = np.zeros_like(ex), np.zeros_like(ey)
        e1x[valid] = ex[valid] / norm[valid]
        e1y[valid] = ey[valid] / norm[valid]
        e2x, e2y = -e1y, e1x

        a_dot_e1 = axd_s * e1x + ayd_s * e1y
        a_dot_e2 = axd_s * e2x + ayd_s * e2y
        # a_d ~ f_d[-cos(theta_d) e1 + sin(theta_d) e2]  =>  theta_d = atan2(a.e2, -a.e1)
        theta_d = np.degrees(np.arctan2(a_dot_e2, -a_dot_e1))

        # trim padding for reporting/plotting (keep core 2.5s window)
        core_mask = (t_d >= duel["timestamp"].total_seconds() - WINDOW_SECONDS) & (
            t_d <= duel["timestamp"].total_seconds()
        )
        t_rel = t_d[core_mask] - duel["timestamp"].total_seconds()

        raw_acc_all.append(acc_raw[core_mask])
        sg_acc_all.append(acc_sg[core_mask])

        if len(examples) < N_EXAMPLES:
            examples.append(
                {
                    "t_rel": t_rel,
                    "acc_raw": acc_raw[core_mask],
                    "acc_sg": acc_sg[core_mask],
                    "theta_d": theta_d[core_mask],
                }
            )

    raw_acc_flat = np.concatenate(raw_acc_all)
    sg_acc_flat = np.concatenate(sg_acc_all)

    print(f"n events used: {len(raw_acc_all)}")
    print(
        f"raw 2nd-diff accel [m/s^2]: mean={raw_acc_flat.mean():.2f}, "
        f"median={np.median(raw_acc_flat):.2f}, p95={np.percentile(raw_acc_flat,95):.2f}, "
        f"max={raw_acc_flat.max():.2f}"
    )
    print(
        f"SG-smoothed accel [m/s^2]: mean={sg_acc_flat.mean():.2f}, "
        f"median={np.median(sg_acc_flat):.2f}, p95={np.percentile(sg_acc_flat,95):.2f}, "
        f"max={sg_acc_flat.max():.2f}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    ax = axes[0]
    for ex in examples:
        ax.plot(ex["t_rel"], ex["acc_raw"], alpha=0.6, linewidth=1)
    ax.set_ylabel("raw 2nd-diff |accel| [m/s^2]")
    ax.set_title(f"defender acceleration magnitude: raw vs SG-smoothed (n_examples={len(examples)})")

    ax = axes[1]
    for ex in examples:
        ax.plot(ex["t_rel"], ex["acc_sg"], alpha=0.7, linewidth=1.3)
    ax.set_ylabel("SG-smoothed |accel| [m/s^2]")

    ax = axes[2]
    for ex in examples:
        ax.plot(ex["t_rel"], ex["theta_d"], alpha=0.7, linewidth=1.3, marker=".", markersize=2)
    ax.set_ylabel("proxy theta_d(t) [deg]")
    ax.set_xlabel("time relative to tackle event [s]")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(180, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(-180, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/judgement2_instantaneous_theta.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
