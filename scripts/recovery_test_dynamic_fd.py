"""識別可能性の確認: 合成データによるパラメータ復元テスト。

既知の真値 (tau_d, f_d0, f_d_inf, d0, theta_d) から人工的なディフェンダー軌跡を生成し、
それを実データと同じフィッティングパイプラインに通して、元のパラメータを復元できるかを見る。

- 復元できる  -> 端点張り付きは最適化の問題。手法改善に投資する価値がある
- 復元できない -> 2.5秒程度の1軌跡から5パラメータを決めるだけの情報がない(構造的識別不能)。
                  最適化をいくら改善しても解決しない

アタッカーの軌跡は実データのものをそのまま使い、ディフェンダーだけを合成する。
実測誤差(TRACAB相当、±ノイズ)を加えた条件も試す。
"""

import numpy as np

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import prepare_event, fit_event as fit_event_const
from fit_oad_dynamic_fd import fit_event_dynamic, simulate_dynamic

MATCH_ID = "J03WPY"
NOISE_LEVELS = [0.0, 0.1, 0.3]  # m, TRACAB系の測位誤差スケールを意識

TRUE_PARAM_SETS = [
    # (tau_d, f_d0, f_d_inf, d0, theta_d)
    (2.0, 8.0, 2.0, 3.0, np.radians(-30)),   # H1'の向き(近いほど強い)、はっきりした差
    (2.0, 3.0, 3.0, 3.0, np.radians(-30)),   # f_d一定(縮退ケース)
    (1.5, 9.0, 1.5, 6.0, np.radians(45)),    # H1'の向き、特性距離が大きめ
]


def make_synthetic(prepped, true_params, noise_std, rng):
    """真値パラメータでディフェンダー軌跡を生成し、prepped を差し替えたコピーを返す。"""
    result = simulate_dynamic(true_params, prepped)
    if result is None:
        return None
    xd_true, yd_true = result

    xd_obs = xd_true + rng.normal(0, noise_std, size=len(xd_true))
    yd_obs = yd_true + rng.normal(0, noise_std, size=len(yd_true))

    synth = dict(prepped)
    synth["xd_obs"] = xd_obs
    synth["yd_obs"] = yd_obs
    # 初期状態も合成軌跡に合わせる(位置は合成値、速度は真値の初期速度をそのまま流用)
    synth["state0"] = [xd_obs[0], yd_obs[0], prepped["state0"][2], prepped["state0"][3]]
    return synth


def main():
    events, frames_by_period, teams = extract_dribble_events(MATCH_ID, return_context=True)
    player_lookup = {p.player_id: p for t in teams for p in t.players}
    idx_by_id_lookup = {
        period_id: {id(f): i for i, f in enumerate(period_frames)}
        for period_id, period_frames in frames_by_period.items()
    }

    # 継続時間が中央値付近の、素直なイベントを土台に使う
    ev = sorted(events, key=lambda e: abs(e.duration_s - 2.5))[0]
    attacker = player_lookup[ev.attacker_id]
    defender = player_lookup[ev.defender_id]
    period_frames = frames_by_period[ev.period_id]
    idx_by_id = idx_by_id_lookup[ev.period_id]
    prepped = prepare_event(ev, period_frames, idx_by_id, attacker, defender)
    print(f"base event: duration={ev.duration_s:.2f}s, n_frames={len(prepped['t_core'])}\n")

    rng = np.random.default_rng(0)

    for true_params in TRUE_PARAM_SETS:
        tau_t, f0_t, finf_t, d0_t, th_t = true_params
        print(f"=== TRUE: tau_d={tau_t}, f_d0={f0_t}, f_d_inf={finf_t}, d0={d0_t}, "
              f"theta_d={np.degrees(th_t):.0f}deg ===")

        for noise in NOISE_LEVELS:
            synth = make_synthetic(prepped, true_params, noise, rng)
            if synth is None:
                print(f"  noise={noise}: synthetic generation failed")
                continue

            const_params, const_err = fit_event_const(synth, seed=0)
            est, err = fit_event_dynamic(synth, seed=0, const_params=const_params)
            tau_e, f0_e, finf_e, d0_e, th_e = est

            direction_ok = (f0_e > finf_e) == (f0_t > finf_t)
            print(
                f"  noise={noise:.1f}m -> tau_d={tau_e:6.2f} f_d0={f0_e:6.2f} f_d_inf={finf_e:6.2f} "
                f"d0={d0_e:6.2f} theta_d={np.degrees(th_e):7.1f}deg  err={err:.4f}  "
                f"direction {'OK' if direction_ok else 'WRONG'}"
            )
        print()


if __name__ == "__main__":
    main()
