"""|a_d(t)|とd(t)の相関(-0.39)が、v_d(t)(ディフェンダーの速度の大きさ)の変化による
アーティファクトでないかを確認する。

運動方程式 v̇_d = -v_d/τ_d + f_d[...] より、観測される加速度|a_d(t)|はf_dそのものではなく、
抗力項(-v_d/τ_d、速度に依存)と駆動項(f_d)の合成である。間合いが縮まるとv_d(t)自体が
変化している(例えば減速して身構える)なら、|a_d(t)|とd(t)の相関はv_dの変化を拾っている
だけで、f_dの変化を意味しない可能性がある。

ここでは:
  1. Spearman(d, v_d) を見て、速度自体が間合いに依存しているかを確認
  2. v_dを統制した上でのSpearman(d, |a_d|)(ランクに基づく偏相関)を計算し、
     元の相関(-0.39)がv_dの影響を除いても残るかを確認する
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from extract_dribble_events import extract_dribble_events, to_xy

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1"]
FRAME_RATE = 25
DT = 1.0 / FRAME_RATE
SG_WINDOW = 11
SG_POLYORDER = 3
PAD_FRAMES = 8


def get_padded_frames(period_frames, idx_by_id, run_frames):
    idx0 = idx_by_id[id(run_frames[0])]
    idx1 = idx_by_id[id(run_frames[-1])]
    lo = max(0, idx0 - PAD_FRAMES)
    hi = min(len(period_frames), idx1 + PAD_FRAMES + 1)
    return period_frames[lo:hi], idx0 - lo, idx1 - lo


def series_for_player(frames, player):
    xs, ys = [], []
    for f in frames:
        pd_ = f.players_data.get(player)
        x, y = to_xy(pd_.coordinates)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def sg_all_derivatives(x, dt, window, polyorder):
    n = len(x)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return None, None, None
    x_s = savgol_filter(x, window_length=w, polyorder=polyorder)
    v_s = savgol_filter(x, window_length=w, polyorder=polyorder, deriv=1, delta=dt)
    a_s = savgol_filter(x, window_length=w, polyorder=polyorder, deriv=2, delta=dt)
    return x_s, v_s, a_s


def rank(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x))
    return r


def spearman(x, y):
    return np.corrcoef(rank(x), rank(y))[0, 1]


def partial_spearman(x, y, z):
    """rank変換した上でのPearson偏相関(x,yの相関からzの効果を除く)。"""
    rx, ry, rz = rank(x), rank(y), rank(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


def main():
    all_events = []
    frames_by_period_all = {}
    teams_by_match = {}
    for match_id in MATCH_IDS:
        events, frames_by_period, teams = extract_dribble_events(match_id, return_context=True)
        all_events.extend(events)
        frames_by_period_all[match_id] = frames_by_period
        teams_by_match[match_id] = teams

    player_lookup_by_match = {}
    for match_id, teams in teams_by_match.items():
        lookup = {}
        for t in teams:
            for p in t.players:
                lookup[p.player_id] = p
        player_lookup_by_match[match_id] = lookup

    idx_by_id_lookup = {}
    for match_id, frames_by_period in frames_by_period_all.items():
        for period_id, period_frames in frames_by_period.items():
            idx_by_id_lookup[(match_id, period_id)] = {id(f): i for i, f in enumerate(period_frames)}

    d_pool, amag_pool, vmag_pool = [], [], []

    for ev in all_events:
        players = player_lookup_by_match[ev.match_id]
        attacker = players[ev.attacker_id]
        defender = players[ev.defender_id]
        period_frames = frames_by_period_all[ev.match_id][ev.period_id]
        idx_by_id = idx_by_id_lookup[(ev.match_id, ev.period_id)]

        padded_frames, core_lo, core_hi = get_padded_frames(period_frames, idx_by_id, ev.frames)

        xa, ya = series_for_player(padded_frames, attacker)
        xd, yd = series_for_player(padded_frames, defender)
        d_t_full = np.sqrt((xa - xd) ** 2 + (ya - yd) ** 2)

        xd_s, vxd_s, axd_s = sg_all_derivatives(xd, DT, SG_WINDOW, SG_POLYORDER)
        yd_s, vyd_s, ayd_s = sg_all_derivatives(yd, DT, SG_WINDOW, SG_POLYORDER)
        if xd_s is None:
            continue

        a_mag = np.sqrt(axd_s**2 + ayd_s**2)
        v_mag = np.sqrt(vxd_s**2 + vyd_s**2)

        core_mask = np.zeros(len(xd_s), dtype=bool)
        core_mask[core_lo : core_hi + 1] = True

        d_pool.append(d_t_full[core_mask])
        amag_pool.append(a_mag[core_mask])
        vmag_pool.append(v_mag[core_mask])

    d_pool = np.concatenate(d_pool)
    amag_pool = np.concatenate(amag_pool)
    vmag_pool = np.concatenate(vmag_pool)

    print(f"total dribble events: {len(all_events)}, pooled frames: {len(d_pool)}")
    print(f"Spearman(d, |a_d|)          = {spearman(d_pool, amag_pool):.3f}  (元の結果の再掲)")
    print(f"Spearman(d, v_d)            = {spearman(d_pool, vmag_pool):.3f}")
    print(f"Spearman(|a_d|, v_d)        = {spearman(amag_pool, vmag_pool):.3f}")
    print(f"Partial Spearman(d, |a_d| | v_d) = {partial_spearman(d_pool, amag_pool, vmag_pool):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.scatter(d_pool, vmag_pool, s=4, alpha=0.1, color="C4")
    bins = np.linspace(d_pool.min(), min(d_pool.max(), 25), 15)
    bin_idx = np.digitize(d_pool, bins)
    centers, medians = [], []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() < 20:
            continue
        centers.append((bins[i - 1] + bins[i]) / 2)
        medians.append(np.median(vmag_pool[mask]))
    ax.plot(centers, medians, color="C3", linewidth=2, marker="o", label="binned median")
    ax.set_xlabel("distance d(t) [m]")
    ax.set_ylabel("v_d(t) [m/s]")
    ax.set_title(f"defender speed vs d(t), spearman={spearman(d_pool, vmag_pool):.3f}")
    ax.legend()

    ax = axes[1]
    ax.scatter(vmag_pool, amag_pool, s=4, alpha=0.08, color="C5")
    ax.set_xlabel("v_d(t) [m/s]")
    ax.set_ylabel("|a_d(t)| [m/s^2]")
    ax.set_title(f"|a_d| vs v_d, spearman={spearman(amag_pool, vmag_pool):.3f}")

    ax = axes[2]
    sc = ax.scatter(d_pool, amag_pool, s=4, alpha=0.08, c=vmag_pool, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="v_d(t) [m/s]")
    ax.set_xlabel("distance d(t) [m]")
    ax.set_ylabel("|a_d(t)| [m/s^2]")
    ax.set_title("|a_d| vs d(t), colored by v_d")

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/judgement_v2_speed_control.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
