"""階層モデル(プーリング)を実データのサブサンプルに適用する。

追補6の復元テストで、プーリングにより現実的なノイズ水準でも
(f_d0, f_d_inf, d0) の識別が回復することを確認した。ここではその設定を
実データ(idsse-data、複数試合からのドリブルイベント)に適用し、
H1'(間合いが縮まるほどf_dが増加する、つまりf_d0 > f_d_inf)を検証する。

計算コストが大きいため、Google Colab等での実行を想定している
(ローカルでの目安: 10イベントで約20分、50イベントで概ね1.5〜2時間)。
"""

import time

import numpy as np

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import prepare_event
from fit_pooled_dynamic_fd import fit_pooled, pooled_objective, fit_inner

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1"]
N_SUBSAMPLE = 50
SEED = 42
OUT_JSON = "/Users/ryuya/dev/dd-oad/documents/pooled_real_result.json"


def main():
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

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_events), size=min(N_SUBSAMPLE, len(all_events)), replace=False)
    sample = [all_events[i] for i in idx]
    print(f"total events: {len(all_events)}, subsample: {len(sample)}")

    prepped_list = []
    d_means = []
    for match_id, i, ev in sample:
        frames_by_period, player_lookup, idx_by_id_lookup = context[match_id]
        attacker = player_lookup[ev.attacker_id]
        defender = player_lookup[ev.defender_id]
        period_frames = frames_by_period[ev.period_id]
        idx_by_id = idx_by_id_lookup[ev.period_id]
        prepped = prepare_event(ev, period_frames, idx_by_id, attacker, defender)
        if prepped is None:
            continue
        prepped_list.append(prepped)

        t_core = prepped["t_core"]
        xa = np.array([prepped["ax_interp"](t) for t in t_core])
        ya = np.array([prepped["ay_interp"](t) for t in t_core])
        d = np.hypot(prepped["xd_obs"] - xa, prepped["yd_obs"] - ya)
        d_means.append(d.mean())

    print(f"usable events: {len(prepped_list)}")
    print(f"d_mean range across events: {min(d_means):.2f} - {max(d_means):.2f} m\n")

    t0 = time.time()
    (f_d0, f_d_inf, d0), err = fit_pooled(prepped_list, seed=0, verbose=True)
    elapsed = time.time() - t0

    print(f"\n=== POOLED FIT ON REAL DATA (n={len(prepped_list)}, {elapsed/60:.1f} min) ===")
    print(f"  f_d0    = {f_d0:.3f}")
    print(f"  f_d_inf = {f_d_inf:.3f}")
    print(f"  d0      = {d0:.3f}")
    print(f"  mean fitting error = {err:.4f}")
    direction = "H1' supported (f_d0 > f_d_inf: closer -> stronger)" if f_d0 > f_d_inf else \
                "opposite of H1' (f_d0 < f_d_inf: farther -> stronger)"
    print(f"  direction: {direction}")

    # 個々のイベントのtau_d, theta_dも参考として推定しておく
    per_event = []
    for prepped, (match_id, i, ev) in zip(prepped_list, sample):
        (tau_d, theta_d), e = fit_inner(prepped, f_d0, f_d_inf, d0, seed=0)
        per_event.append(
            {"match_id": match_id, "event_idx": i, "tau_d": float(tau_d),
             "theta_d_deg": float(np.degrees(theta_d)), "err": float(e)}
        )

    import json
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "n_events": len(prepped_list),
                "f_d0": float(f_d0),
                "f_d_inf": float(f_d_inf),
                "d0": float(d0),
                "mean_error": float(err),
                "elapsed_min": elapsed / 60,
                "per_event": per_event,
            },
            f,
            indent=2,
        )
    print(f"saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
