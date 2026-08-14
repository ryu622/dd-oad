"""fit_oad_parameters.pyのパイロット結果を踏まえ、3試合・224イベント全体に
(tau_d, f_d, theta_d)推定を展開する。結果はCSVに保存し、後続の
「f_dと間合いの相関」分析(フェーズ2)に使う。
"""

import csv
import time

import numpy as np

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import fit_event, prepare_event, simulate

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1"]
OUT_CSV = "/Users/ryuya/dev/dd-oad/documents/oad_fit_results.csv"


def event_distance_stats(prepped):
    xd_obs, yd_obs = prepped["xd_obs"], prepped["yd_obs"]
    t_core = prepped["t_core"]
    xa = np.array([prepped["ax_interp"](t) for t in t_core])
    ya = np.array([prepped["ay_interp"](t) for t in t_core])
    d = np.hypot(xd_obs - xa, yd_obs - ya)
    return d[0], d[-1], d.mean(), d.min(), d.max()


def main():
    rows = []
    t_start_all = time.time()

    for match_id in MATCH_IDS:
        events, frames_by_period, teams = extract_dribble_events(match_id, return_context=True)
        player_lookup = {p.player_id: p for t in teams for p in t.players}
        idx_by_id_lookup = {
            period_id: {id(f): i for i, f in enumerate(period_frames)}
            for period_id, period_frames in frames_by_period.items()
        }

        for i, ev in enumerate(events):
            t0 = time.time()
            attacker = player_lookup[ev.attacker_id]
            defender = player_lookup[ev.defender_id]
            period_frames = frames_by_period[ev.period_id]
            idx_by_id = idx_by_id_lookup[ev.period_id]

            prepped = prepare_event(ev, period_frames, idx_by_id, attacker, defender)
            if prepped is None:
                continue

            params, err = fit_event(prepped, seed=hash((match_id, i)) % (2**31))
            tau_d, f_d, theta_d = params
            d_start, d_end, d_mean, d_min, d_max = event_distance_stats(prepped)

            rows.append(
                {
                    "match_id": match_id,
                    "event_idx": i,
                    "attacker_id": ev.attacker_id,
                    "defender_id": ev.defender_id,
                    "duration_s": ev.duration_s,
                    "n_frames": len(prepped["t_core"]),
                    "tau_d": tau_d,
                    "f_d": f_d,
                    "theta_d_deg": np.degrees(theta_d),
                    "fit_error": err,
                    "d_start": d_start,
                    "d_end": d_end,
                    "d_mean": d_mean,
                    "d_min": d_min,
                    "d_max": d_max,
                }
            )
            elapsed = time.time() - t0
            print(
                f"[{match_id} {i+1}/{len(events)}] tau_d={tau_d:.2f} f_d={f_d:.2f} "
                f"theta_d={np.degrees(theta_d):.1f} err={err:.3f} d_mean={d_mean:.2f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.time() - t_start_all
    print(f"\ndone: {len(rows)} events fit in {total_elapsed/60:.1f} min, saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
