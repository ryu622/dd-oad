"""判定③: 間合いd(t)と瞬間theta_d(t)の間に、目視で分かる相関があるかを確認する。

判定①②のロジックを再利用し、3試合分の1v1イベント(TacklingGame, ground,
WinnerRole=withBallControl)について d(t) と 瞬間theta_d(t)プロキシ を算出、
判定②で見えた「加速度が小さいときの角度不安定性」を簡易な大きさ閾値フィルタで
除去した上で、全イベントをプールした散布図を作る。
"""

from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from kloppy import sportec
from scipy.signal import savgol_filter

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1"]
WINDOW_SECONDS = 2.5
PAD_SECONDS = 0.5
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FRAME_RATE = 25
DT = 1.0 / FRAME_RATE
SG_WINDOW = 11
SG_POLYORDER = 3
ACCEL_MAG_THRESHOLD = 1.5  # m/s^2, judgement2で見えた角度不安定性への簡易フィルタ


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


def sg_derivatives(x, dt, window, polyorder):
    x_s = savgol_filter(x, window_length=window, polyorder=polyorder)
    a_s = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=2, delta=dt)
    return x_s, a_s


def process_match(match_id):
    events = sportec.load_open_event_data(match_id=match_id)
    tracking = sportec.load_open_tracking_data(match_id=match_id)

    duels = extract_duel_events(events)
    frames_by_period = build_frame_index(tracking)
    players = player_lookup(tracking)

    d_all, theta_all = [], []

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

        xd_s, axd_s = sg_derivatives(xd, DT, SG_WINDOW, SG_POLYORDER)
        yd_s, ayd_s = sg_derivatives(yd, DT, SG_WINDOW, SG_POLYORDER)
        xa_s, _ = sg_derivatives(xa, DT, SG_WINDOW, SG_POLYORDER)
        ya_s, _ = sg_derivatives(ya, DT, SG_WINDOW, SG_POLYORDER)

        ex = xd_s - xa_s
        ey = yd_s - ya_s
        d_t = np.sqrt(ex**2 + ey**2)
        norm = d_t.copy()
        valid_dir = norm > 1e-6
        e1x, e1y = np.zeros_like(ex), np.zeros_like(ey)
        e1x[valid_dir] = ex[valid_dir] / norm[valid_dir]
        e1y[valid_dir] = ey[valid_dir] / norm[valid_dir]
        e2x, e2y = -e1y, e1x

        a_mag = np.sqrt(axd_s**2 + ayd_s**2)
        a_dot_e1 = axd_s * e1x + ayd_s * e1y
        a_dot_e2 = axd_s * e2x + ayd_s * e2y
        theta_d = np.degrees(np.arctan2(a_dot_e2, -a_dot_e1))

        core_mask = (t_d >= duel["timestamp"].total_seconds() - WINDOW_SECONDS) & (
            t_d <= duel["timestamp"].total_seconds()
        )
        keep = core_mask & (a_mag > ACCEL_MAG_THRESHOLD) & valid_dir

        d_all.append(d_t[keep])
        theta_all.append(theta_d[keep])

    return (
        np.concatenate(d_all) if d_all else np.array([]),
        np.concatenate(theta_all) if theta_all else np.array([]),
    )


def main():
    d_pool, theta_pool = [], []
    for match_id in MATCH_IDS:
        print(f"=== {match_id} ===")
        d_m, theta_m = process_match(match_id)
        print(f"{match_id}: {len(d_m)} frames retained after accel-magnitude filter")
        d_pool.append(d_m)
        theta_pool.append(theta_m)

    d_pool = np.concatenate(d_pool)
    theta_pool = np.concatenate(theta_pool)
    print(f"\ntotal pooled frames: {len(d_pool)}")

    # spearman-style monotonic check without extra deps: rank correlation via numpy
    order_d = np.argsort(d_pool)
    rank_d = np.empty_like(order_d, dtype=float)
    rank_d[order_d] = np.arange(len(d_pool))
    order_t = np.argsort(theta_pool)
    rank_t = np.empty_like(order_t, dtype=float)
    rank_t[order_t] = np.arange(len(theta_pool))
    spearman = np.corrcoef(rank_d, rank_t)[0, 1]
    print(f"Spearman rank correlation(d, theta_d): {spearman:.3f}")

    # binned median trend for visual aid
    bins = np.linspace(d_pool.min(), min(d_pool.max(), 20), 15)
    bin_idx = np.digitize(d_pool, bins)
    bin_centers, bin_medians, bin_counts = [], [], []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() < 20:
            continue
        bin_centers.append((bins[i - 1] + bins[i]) / 2)
        bin_medians.append(np.median(theta_pool[mask]))
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d_pool, theta_pool, s=4, alpha=0.15, color="C0")
    ax.plot(bin_centers, bin_medians, color="C3", linewidth=2, marker="o", label="binned median")
    ax.set_xlabel("distance d(t) [m]")
    ax.set_ylabel("proxy theta_d(t) [deg]")
    ax.set_title(f"theta_d(t) vs d(t), pooled n={len(d_pool)} (accel filter > {ACCEL_MAG_THRESHOLD} m/s^2)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/judgement3_correlation.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
