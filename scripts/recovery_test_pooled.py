"""プーリングで識別可能性が回復するかの復元テスト。

追補5の復元テスト(1イベント単独)では、0.1mのノイズで (f_d0, f_d_inf, d0) の推定が
崩壊した。ここでは同じ真値・同じノイズ水準で、複数イベントに共有パラメータを課した
プーリング推定を行い、真値を復元できるかを確認する。

これが復元できなければ、プーリングでも救えないということであり、実データに適用する意味はない。
"""

import time

import numpy as np

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import prepare_event
from fit_oad_dynamic_fd import simulate_dynamic
from fit_pooled_dynamic_fd import fit_pooled, pooled_objective

MATCH_ID = "J03WPY"
N_EVENTS = 10
NOISE_STD = 0.1  # m、追補5で1イベント推定が崩壊した水準

# 真値: 形状パラメータは全イベント共通、tau_d/theta_dはイベントごとに変える
TRUE_FD0 = 8.0
TRUE_FDINF = 2.0
TRUE_D0 = 3.0


def main():
    events, frames_by_period, teams = extract_dribble_events(MATCH_ID, return_context=True)
    player_lookup = {p.player_id: p for t in teams for p in t.players}
    idx_by_id_lookup = {
        period_id: {id(f): i for i, f in enumerate(period_frames)}
        for period_id, period_frames in frames_by_period.items()
    }

    rng = np.random.default_rng(0)
    prepped_list = []

    # 継続時間が標準的なイベントを土台に選ぶ
    candidates = sorted(events, key=lambda e: abs(e.duration_s - 2.5))
    for ev in candidates:
        if len(prepped_list) >= N_EVENTS:
            break
        attacker = player_lookup[ev.attacker_id]
        defender = player_lookup[ev.defender_id]
        period_frames = frames_by_period[ev.period_id]
        idx_by_id = idx_by_id_lookup[ev.period_id]
        prepped = prepare_event(ev, period_frames, idx_by_id, attacker, defender)
        if prepped is None:
            continue

        # イベントごとに異なる tau_d, theta_d の真値を割り当てる
        tau_true = rng.uniform(1.0, 5.0)
        theta_true = rng.uniform(-np.pi, np.pi)
        true_params = (tau_true, TRUE_FD0, TRUE_FDINF, TRUE_D0, theta_true)

        result = simulate_dynamic(true_params, prepped)
        if result is None:
            continue
        xd_true, yd_true = result

        synth = dict(prepped)
        synth["xd_obs"] = xd_true + rng.normal(0, NOISE_STD, size=len(xd_true))
        synth["yd_obs"] = yd_true + rng.normal(0, NOISE_STD, size=len(yd_true))
        synth["state0"] = [synth["xd_obs"][0], synth["yd_obs"][0], prepped["state0"][2], prepped["state0"][3]]
        prepped_list.append(synth)

    print(f"synthetic events: {len(prepped_list)}, noise={NOISE_STD}m")
    print(f"TRUE shared params: f_d0={TRUE_FD0}, f_d_inf={TRUE_FDINF}, d0={TRUE_D0}\n")

    t0 = time.time()
    est, err = fit_pooled(prepped_list, seed=0, verbose=True)
    f_d0_e, f_d_inf_e, d0_e = est

    print(f"\n=== POOLED RECOVERY RESULT ({(time.time()-t0)/60:.1f} min) ===")
    print(f"  f_d0   : true={TRUE_FD0:5.2f}  est={f_d0_e:5.2f}")
    print(f"  f_d_inf: true={TRUE_FDINF:5.2f}  est={f_d_inf_e:5.2f}")
    print(f"  d0     : true={TRUE_D0:5.2f}  est={d0_e:5.2f}")
    direction_ok = (f_d0_e > f_d_inf_e) == (TRUE_FD0 > TRUE_FDINF)
    print(f"  direction: {'OK' if direction_ok else 'WRONG'}")

    # 決定的な診断: 真値での目的関数値と比較する
    #   est_err < true_err -> ノイズのせいで「誤った答えの方がよく当てはまる」= 識別不能
    #   est_err > true_err -> 真値の方が良いのに見つけられていない = 最適化の失敗
    true_err = pooled_objective((TRUE_FD0, TRUE_FDINF, TRUE_D0), prepped_list, seed=0)
    print(f"\n  objective at ESTIMATE: {err:.4f}")
    print(f"  objective at TRUTH   : {true_err:.4f}")
    if err < true_err:
        print("  -> 推定解の方が真値より当てはまりが良い = ノイズによる構造的識別不能")
    else:
        print("  -> 真値の方が当てはまりが良い = 最適化が探索しきれていない(手法改善の余地あり)")


if __name__ == "__main__":
    main()
