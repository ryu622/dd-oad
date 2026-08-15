"""動的f_d(d)モデルを30件のサブサンプルにフィットし、
f_d0 vs f_d_inf の向きに一貫した傾向があるかを確認する。

見るべきは「誤差が改善するか」(パラメータが増えるので当然改善する)ではなく、
「f_d0 > f_d_inf(近いほど強い = H1'の向き)のイベントが多数派か」である。
"""

import csv
import time

import numpy as np

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import prepare_event, fit_event as fit_event_const
from fit_oad_dynamic_fd import fit_event_dynamic

MATCH_IDS = ["J03WPY", "J03WMX"]
N_SUBSAMPLE = 30
OUT_CSV = "/Users/ryuya/dev/dd-oad/documents/dynamic_fd_subsample.csv"


def event_distance_stats(prepped):
    xd_obs, yd_obs = prepped["xd_obs"], prepped["yd_obs"]
    t_core = prepped["t_core"]
    xa = np.array([prepped["ax_interp"](t) for t in t_core])
    ya = np.array([prepped["ay_interp"](t) for t in t_core])
    d = np.hypot(xd_obs - xa, yd_obs - ya)
    return d.mean(), d.min(), d.max()


def main():
    # 再現性のため、全イベントを集めてから固定シードでサブサンプリングする
    all_events = []
    context = {}
    for match_id in MATCH_IDS:
        events, frames_by_period, teams = extract_dribble_events(match_id, return_context=True)
        player_lookup = {p.player_id: p for t in teams for p in t.players}
        idx_by_id_lookup = {
            period_id: {id(f): i for i, f in enumerate(period_frames)}
            for period_id, period_frames in frames_by_period.items()
        }
        context[match_id] = (frames_by_period, player_lookup, idx_by_id_lookup)
        for i, ev in enumerate(events):
            all_events.append((match_id, i, ev))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_events), size=min(N_SUBSAMPLE, len(all_events)), replace=False)
    sample = [all_events[i] for i in idx]
    print(f"total events: {len(all_events)}, subsample: {len(sample)}\n")

    rows = []
    t_start = time.time()

    for k, (match_id, i, ev) in enumerate(sample):
        frames_by_period, player_lookup, idx_by_id_lookup = context[match_id]
        attacker = player_lookup[ev.attacker_id]
        defender = player_lookup[ev.defender_id]
        period_frames = frames_by_period[ev.period_id]
        idx_by_id = idx_by_id_lookup[ev.period_id]

        prepped = prepare_event(ev, period_frames, idx_by_id, attacker, defender)
        if prepped is None:
            continue

        t0 = time.time()
        const_params, const_err = fit_event_const(prepped, seed=k)
        dyn_params, dyn_err = fit_event_dynamic(prepped, seed=k, const_params=const_params)
        tau_d, f_d0, f_d_inf, d0, theta_d = dyn_params
        d_mean, d_min, d_max = event_distance_stats(prepped)

        direction = "H1'" if f_d0 > f_d_inf else "opposite"
        rows.append(
            {
                "match_id": match_id,
                "event_idx": i,
                "duration_s": ev.duration_s,
                "const_f_d": const_params[1],
                "const_err": const_err,
                "tau_d": tau_d,
                "f_d0": f_d0,
                "f_d_inf": f_d_inf,
                "d0": d0,
                "theta_d_deg": np.degrees(theta_d),
                "dyn_err": dyn_err,
                "improvement_pct": (const_err - dyn_err) / const_err * 100,
                "direction": direction,
                "d_mean": d_mean,
                "d_min": d_min,
                "d_max": d_max,
            }
        )
        print(
            f"[{k+1}/{len(sample)}] {match_id} ev{i}: const_err={const_err:.3f} dyn_err={dyn_err:.3f} "
            f"f_d0={f_d0:.2f} f_d_inf={f_d_inf:.2f} d0={d0:.2f} -> {direction} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_h1 = sum(1 for r in rows if r["direction"] == "H1'")
    n_opp = len(rows) - n_h1
    print(f"\n=== summary (n={len(rows)}, {(time.time()-t_start)/60:.1f} min) ===")
    print(f"H1' direction (f_d0 > f_d_inf): {n_h1}")
    print(f"opposite direction:             {n_opp}")

    # 境界張り付きの割合(結果の信頼性チェック)
    n_bound = sum(
        1 for r in rows
        if r["f_d0"] <= 0.02 or r["f_d0"] >= 11.9 or r["f_d_inf"] <= 0.02 or r["f_d_inf"] >= 11.9
        or r["d0"] >= 11.9
    )
    print(f"events with a parameter at a bound: {n_bound}/{len(rows)}")
    print(f"saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
