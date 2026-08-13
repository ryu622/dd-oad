"""新方式(先行研究5条件準拠のドリブルイベント抽出)による判定①・③の再実施。

TacklingGame起点(判定①〜③, v1)で見えた選択バイアス(タックルという結果に条件付けられた
サンプルしか集まらない)を、ボール保持の開始を起点とする抽出に切り替えることで解消できるかを検証する。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from extract_dribble_events import extract_dribble_events, to_xy, PITCH_LENGTH, PITCH_WIDTH

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1"]
FRAME_RATE = 25
DT = 1.0 / FRAME_RATE
SG_WINDOW = 11
SG_POLYORDER = 3
PAD_FRAMES = 8  # ~0.32s、フィルタの端点安定化用
ACCEL_MAG_THRESHOLD = 1.5


def get_padded_frames(period_frames, idx_by_id, run_frames):
    idx0 = idx_by_id[id(run_frames[0])]
    idx1 = idx_by_id[id(run_frames[-1])]
    lo = max(0, idx0 - PAD_FRAMES)
    hi = min(len(period_frames), idx1 + PAD_FRAMES + 1)
    return period_frames[lo:hi], idx0 - lo, idx1 - lo  # padded frames, core start/end idx within padded list


def series_for_player(frames, player):
    xs, ys = [], []
    for f in frames:
        pd_ = f.players_data.get(player)
        x, y = to_xy(pd_.coordinates)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def sg_derivatives(x, dt, window, polyorder):
    n = len(x)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return None, None
    x_s = savgol_filter(x, window_length=w, polyorder=polyorder)
    a_s = savgol_filter(x, window_length=w, polyorder=polyorder, deriv=2, delta=dt)
    return x_s, a_s


def main():
    all_events = []
    frames_by_period_all = {}
    teams_by_match = {}
    for match_id in MATCH_IDS:
        events, frames_by_period, teams = extract_dribble_events(match_id, return_context=True)
        all_events.extend(events)
        frames_by_period_all[match_id] = frames_by_period
        teams_by_match[match_id] = teams

    print(f"\ntotal dribble events (3 matches): {len(all_events)}")
    durations = np.array([e.duration_s for e in all_events])
    print(f"duration: mean={durations.mean():.2f}s, median={np.median(durations):.2f}s, "
          f"min={durations.min():.2f}s, max={durations.max():.2f}s")

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
            idx_by_id_lookup[(match_id, period_id)] = {
                id(f): i for i, f in enumerate(period_frames)
            }

    # 判定①: d(t)の変動幅
    d_ranges = []
    all_series = []

    # 判定③: プールしたd(t) vs theta_d(t)
    d_pool, theta_pool = [], []

    # 追加検証: プールしたd(t) vs |a_d(t)|(f_dのプロキシ)。
    # 角度の分析とは違い、大きさそのものを見たいので加速度閾値フィルタは適用しない。
    d_pool_mag, mag_pool = [], []

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
        d_core = d_t_full[core_lo : core_hi + 1]
        d_ranges.append(d_core.max() - d_core.min())
        t_rel = np.arange(len(d_core)) * DT
        t_rel = t_rel - t_rel[-1]
        all_series.append((t_rel, d_core))

        xd_s, axd_s = sg_derivatives(xd, DT, SG_WINDOW, SG_POLYORDER)
        yd_s, ayd_s = sg_derivatives(yd, DT, SG_WINDOW, SG_POLYORDER)
        xa_s, _ = sg_derivatives(xa, DT, SG_WINDOW, SG_POLYORDER)
        ya_s, _ = sg_derivatives(ya, DT, SG_WINDOW, SG_POLYORDER)
        if xd_s is None or xa_s is None:
            continue

        ex = xd_s - xa_s
        ey = yd_s - ya_s
        norm = np.sqrt(ex**2 + ey**2)
        valid = norm > 1e-6
        e1x, e1y = np.zeros_like(ex), np.zeros_like(ey)
        e1x[valid] = ex[valid] / norm[valid]
        e1y[valid] = ey[valid] / norm[valid]
        e2x, e2y = -e1y, e1x

        a_mag = np.sqrt(axd_s**2 + ayd_s**2)
        a_dot_e1 = axd_s * e1x + ayd_s * e1y
        a_dot_e2 = axd_s * e2x + ayd_s * e2y
        theta_d = np.degrees(np.arctan2(a_dot_e2, -a_dot_e1))

        core_mask = np.zeros(len(xd_s), dtype=bool)
        core_mask[core_lo : core_hi + 1] = True

        keep = core_mask & (a_mag > ACCEL_MAG_THRESHOLD) & valid
        d_pool.append(d_t_full[keep])
        theta_pool.append(theta_d[keep])

        d_pool_mag.append(d_t_full[core_mask])
        mag_pool.append(a_mag[core_mask])

    d_ranges = np.array(d_ranges)
    print(f"\n[judgement1] d(t) range within event: mean={d_ranges.mean():.2f}m, "
          f"median={np.median(d_ranges):.2f}m, min={d_ranges.min():.2f}m, max={d_ranges.max():.2f}m")

    d_pool = np.concatenate(d_pool)
    theta_pool = np.concatenate(theta_pool)
    print(f"[judgement3] pooled frames after accel filter: {len(d_pool)}")

    order_d = np.argsort(d_pool)
    rank_d = np.empty_like(order_d, dtype=float)
    rank_d[order_d] = np.arange(len(d_pool))
    order_t = np.argsort(theta_pool)
    rank_t = np.empty_like(order_t, dtype=float)
    rank_t[order_t] = np.arange(len(theta_pool))
    spearman = np.corrcoef(rank_d, rank_t)[0, 1]
    print(f"[judgement3] Spearman rank correlation(d, theta_d): {spearman:.3f}")

    d_pool_mag = np.concatenate(d_pool_mag)
    mag_pool = np.concatenate(mag_pool)
    order_dm = np.argsort(d_pool_mag)
    rank_dm = np.empty_like(order_dm, dtype=float)
    rank_dm[order_dm] = np.arange(len(d_pool_mag))
    order_m = np.argsort(mag_pool)
    rank_m = np.empty_like(order_m, dtype=float)
    rank_m[order_m] = np.arange(len(mag_pool))
    spearman_mag = np.corrcoef(rank_dm, rank_m)[0, 1]
    print(f"[magnitude check] pooled frames (no accel filter): {len(d_pool_mag)}")
    print(f"[magnitude check] Spearman rank correlation(d, |a_d|): {spearman_mag:.3f}")

    # プロット
    fig, axes = plt.subplots(1, 4, figsize=(23, 5))

    ax = axes[0]
    for t_rel, ds in all_series:
        ax.plot(t_rel, ds, alpha=0.3, linewidth=0.8, color="C0")
    ax.set_xlabel("time relative to scene end [s]")
    ax.set_ylabel("distance d(t) [m]")
    ax.set_title(f"d(t) overlay (new extraction), n={len(all_series)} events")

    ax = axes[1]
    ax.hist(d_ranges, bins=30, color="C1")
    ax.set_xlabel("range of d(t) within event [m]")
    ax.set_ylabel("count")
    ax.set_title("distribution of within-event distance variation")

    ax = axes[2]
    ax.scatter(d_pool, theta_pool, s=4, alpha=0.15, color="C0")
    bins = np.linspace(d_pool.min(), min(d_pool.max(), 25), 15)
    bin_idx = np.digitize(d_pool, bins)
    centers, medians = [], []
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() < 20:
            continue
        centers.append((bins[i - 1] + bins[i]) / 2)
        medians.append(np.median(theta_pool[mask]))
    ax.plot(centers, medians, color="C3", linewidth=2, marker="o", label="binned median")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("distance d(t) [m]")
    ax.set_ylabel("proxy theta_d(t) [deg]")
    ax.set_title(f"theta_d(t) vs d(t), n={len(d_pool)}")
    ax.legend()

    ax = axes[3]
    ax.scatter(d_pool_mag, mag_pool, s=4, alpha=0.1, color="C2")
    bins_m = np.linspace(d_pool_mag.min(), min(d_pool_mag.max(), 25), 15)
    bin_idx_m = np.digitize(d_pool_mag, bins_m)
    centers_m, medians_m = [], []
    for i in range(1, len(bins_m)):
        mask = bin_idx_m == i
        if mask.sum() < 20:
            continue
        centers_m.append((bins_m[i - 1] + bins_m[i]) / 2)
        medians_m.append(np.median(mag_pool[mask]))
    ax.plot(centers_m, medians_m, color="C3", linewidth=2, marker="o", label="binned median")
    ax.set_xlabel("distance d(t) [m]")
    ax.set_ylabel("|a_d(t)| [m/s^2]  (f_d proxy)")
    ax.set_title(f"accel magnitude vs d(t), n={len(d_pool_mag)}, spearman={spearman_mag:.3f}")
    ax.legend()

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/judgement_v2_results.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
