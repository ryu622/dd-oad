"""f_dを間合いd(t)の関数として直接フィッティングする(H1'の本命モデル、計画書4.1節)。

    f_d(d) = f_d_inf + (f_d0 - f_d_inf) * exp(-d/d0)

    v̇_d = -v_d/tau_d + f_d(d(t))[-cos(theta_d) e1 + sin(theta_d) e2]

定数f_dモデル(fit_oad_parameters.py)は f_d0 = f_d_inf の特殊ケースにあたる
(ネストしたモデル)。同じイベントに両方をフィットし、誤差がどれだけ改善するかを
比較する。d(t)はRHS内で毎ステップ計算されるe1のnormそのものを使う。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import (
    prepare_event,
    fitting_error as fitting_error_const,
    fit_event as fit_event_const,
    simulate as simulate_const,
    rotate_cw,
    TAU_BOUNDS,
    F_BOUNDS,
    THETA_BOUNDS,
    DT,
)

# 駆動加速度の上限は、人間の走行加速度として現実的な範囲に制限する
# (上限20 m/s^2 は非現実的で、最適化が無意味な端点に逃げる原因になっていた)
FD0_BOUNDS = (0.01, 12.0)
FDINF_BOUNDS = (0.01, 12.0)
# d0(特性距離)の上限も、1v1の間合いスケール(実測でおおむね20m以内)に対して
# 大きすぎると exp(-d/d0) がほぼ一定になり f_d が実質定数化してしまうため抑える
D0_BOUNDS = (0.3, 12.0)


def defender_rhs_dynamic(t, state, tau_d, f_d0, f_d_inf, d0, theta_d, ax_interp, ay_interp):
    xd, yd, vxd, vyd = state
    xa, ya = ax_interp(t), ay_interp(t)
    ex, ey = xd - xa, yd - ya
    d_now = np.hypot(ex, ey)
    if d_now < 1e-6:
        e1x, e1y = 0.0, 0.0
    else:
        e1x, e1y = ex / d_now, ey / d_now
    e2x, e2y = rotate_cw(e1x, e1y)

    f_d = f_d_inf + (f_d0 - f_d_inf) * np.exp(-d_now / d0)

    drive_x = f_d * (-np.cos(theta_d) * e1x + np.sin(theta_d) * e2x)
    drive_y = f_d * (-np.cos(theta_d) * e1y + np.sin(theta_d) * e2y)

    return [vxd, vyd, -vxd / tau_d + drive_x, -vyd / tau_d + drive_y]


def simulate_dynamic(params, prepped):
    tau_d, f_d0, f_d_inf, d0, theta_d = params
    t_core = prepped["t_core"]
    sol = solve_ivp(
        defender_rhs_dynamic,
        t_span=(t_core[0], t_core[-1]),
        y0=prepped["state0"],
        t_eval=t_core,
        args=(tau_d, f_d0, f_d_inf, d0, theta_d, prepped["ax_interp"], prepped["ay_interp"]),
        method="RK45",
        max_step=DT,
    )
    if not sol.success:
        return None
    return sol.y[0], sol.y[1]


def fitting_error_dynamic(params, prepped):
    result = simulate_dynamic(params, prepped)
    if result is None:
        return 1e3
    xd_model, yd_model = result
    xd_obs, yd_obs = prepped["xd_obs"], prepped["yd_obs"]
    numerator = np.sum(np.hypot(xd_obs - xd_model, yd_obs - yd_model))
    denom = np.sum(np.hypot(np.diff(xd_obs), np.diff(yd_obs)))
    if denom < 1e-9:
        return 1e3
    return numerator / denom


def fit_event_dynamic(prepped, seed=0, const_params=None, popsize=30, maxiter=150):
    """動的モデル(5パラメータ)をフィットする。

    const_params (tau_d, f_d, theta_d) が渡された場合、それを f_d0=f_d_inf=f_d
    に対応する点として初期集団の1個体に"種"として仕込む。これにより、動的モデルの
    最終結果が定数モデルの結果より悪くなることは(理論上)起きなくなる
    (定数モデルは動的モデルの特殊ケースであるため)。
    """
    bounds = [TAU_BOUNDS, FD0_BOUNDS, FDINF_BOUNDS, D0_BOUNDS, THETA_BOUNDS]
    n_params = len(bounds)
    pop_size_total = popsize * n_params

    rng = np.random.default_rng(seed)
    init_pop = np.array(
        [rng.uniform(lo, hi, size=pop_size_total) for lo, hi in bounds]
    ).T  # shape (pop_size_total, n_params)

    if const_params is not None:
        tau_d, f_d, theta_d = const_params
        seed_point = np.array([tau_d, f_d, f_d, 5.0, theta_d])
        # bounds内にクリップしてから注入
        for i, (lo, hi) in enumerate(bounds):
            seed_point[i] = np.clip(seed_point[i], lo, hi)
        init_pop[0] = seed_point

    res = differential_evolution(
        fitting_error_dynamic,
        bounds,
        args=(prepped,),
        seed=seed,
        init=init_pop,
        maxiter=maxiter,
        tol=1e-4,
        polish=True,
    )
    return res.x, res.fun


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    match_id = "J03WPY"
    events, frames_by_period, teams = extract_dribble_events(match_id, return_context=True)
    player_lookup = {p.player_id: p for t in teams for p in t.players}

    idx_by_id_lookup = {
        period_id: {id(f): i for i, f in enumerate(period_frames)}
        for period_id, period_frames in frames_by_period.items()
    }

    N_TEST = 6
    test_events = events[:N_TEST]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, ev in enumerate(test_events):
        attacker = player_lookup[ev.attacker_id]
        defender = player_lookup[ev.defender_id]
        period_frames = frames_by_period[ev.period_id]
        idx_by_id = idx_by_id_lookup[ev.period_id]

        prepped = prepare_event(ev, period_frames, idx_by_id, attacker, defender)
        if prepped is None:
            print(f"event {i}: skipped (too short)")
            continue

        const_params, const_err = fit_event_const(prepped, seed=i)
        dyn_params, dyn_err = fit_event_dynamic(prepped, seed=i, const_params=const_params)

        tau_d, f_d0, f_d_inf, d0, theta_d = dyn_params
        print(
            f"event {i}: duration={ev.duration_s:.2f}s  "
            f"CONST err={const_err:.4f} (f_d={const_params[1]:.2f})  "
            f"DYNAMIC err={dyn_err:.4f} (tau_d={tau_d:.2f} f_d0={f_d0:.2f} "
            f"f_d_inf={f_d_inf:.2f} d0={d0:.2f} theta_d={np.degrees(theta_d):.1f}deg)  "
            f"improvement={ (const_err-dyn_err)/const_err*100 :.1f}%"
        )

        xd_model, yd_model = simulate_dynamic(dyn_params, prepped)
        xd_model_c, yd_model_c = simulate_const(const_params, prepped)
        ax = axes[i]
        ax.plot(prepped["xd_obs"], prepped["yd_obs"], "o-", color="cyan", label="defender (empirical)", markersize=3)
        ax.plot(xd_model_c, yd_model_c, "--", color="gray", label="model (const f_d)", linewidth=1.5)
        ax.plot(xd_model, yd_model, "-", color="blue", label="model (dynamic f_d(d))", linewidth=2)
        t_core = prepped["t_core"]
        xa_plot = [prepped["ax_interp"](t) for t in t_core]
        ya_plot = [prepped["ay_interp"](t) for t in t_core]
        ax.plot(xa_plot, ya_plot, "o-", color="orange", label="attacker (empirical)", markersize=3)
        ax.set_title(f"event {i}: const={const_err:.3f} -> dyn={dyn_err:.3f}\nd0={d0:.1f}, f_d0={f_d0:.1f}, f_d_inf={f_d_inf:.1f}")
        ax.set_aspect("equal")
        if i == 0:
            ax.legend(fontsize=6)

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/fit_oad_dynamic_fd_pilot.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
