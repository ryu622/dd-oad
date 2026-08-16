"""OADモデルのディフェンダー側パラメータ (tau_d, f_d, theta_d) を、
先行研究(Yamazaki et al. 2026)Methods C節の手続きに準拠してイベントごとに推定する。

アタッカーの実測(平滑化)軌跡を既知として与え、ディフェンダーの運動方程式

    v̇_d = -v_d/tau_d + f_d[-cos(theta_d) e1 + sin(theta_d) e2]

を数値積分してモデル軌跡を得る。モデル軌跡と実測軌跡の誤差(論文式10、
平均点群距離を実測経路長で正規化したもの)をscipy.optimizeで最小化し、
(tau_d, f_d, theta_d) を推定する。

注意(座標系): 論文本文(Model節)は e2 を「e1を時計回りに90度回転したもの」と
定義している。時計回りの90度回転は (x,y) -> (y,-x)。これまでのスクリプト
(judgement1〜3, judgement_v2*)では反時計回り (x,y)->(-y,x) を使っていたため
符号が逆だった(相関の有無という結論には影響しないが、角度の値そのものを
解釈する本スクリプトでは論文の定義に合わせて修正した)。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

from extract_dribble_events import extract_dribble_events, to_xy

FRAME_RATE = 25
DT = 1.0 / FRAME_RATE
SG_WINDOW = 11
SG_POLYORDER = 3
PAD_FRAMES = 8

TAU_BOUNDS = (0.1, 20.0)
F_BOUNDS = (0.01, 20.0)
THETA_BOUNDS = (-np.pi, np.pi)


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


def sg_smooth_and_velocity(x, dt, window, polyorder):
    n = len(x)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return None, None
    x_s = savgol_filter(x, window_length=w, polyorder=polyorder)
    v_s = savgol_filter(x, window_length=w, polyorder=polyorder, deriv=1, delta=dt)
    return x_s, v_s


def rotate_cw(ex, ey):
    """e2 := e1を時計回りに90度回転(論文Model節の定義)。"""
    return ey, -ex


def defender_rhs(t, state, tau_d, f_d, theta_d, ax_interp, ay_interp):
    xd, yd, vxd, vyd = state
    xa, ya = ax_interp(t), ay_interp(t)
    ex, ey = xd - xa, yd - ya
    norm = np.hypot(ex, ey)
    if norm < 1e-6:
        e1x, e1y = 0.0, 0.0
    else:
        e1x, e1y = ex / norm, ey / norm
    e2x, e2y = rotate_cw(e1x, e1y)

    drive_x = f_d * (-np.cos(theta_d) * e1x + np.sin(theta_d) * e2x)
    drive_y = f_d * (-np.cos(theta_d) * e1y + np.sin(theta_d) * e2y)

    return [vxd, vyd, -vxd / tau_d + drive_x, -vyd / tau_d + drive_y]


class ShiftedInterp:
    """t_shift分ずらして評価するCubicSplineのラッパー。

    ラムダだとmultiprocessingでpickle化できないため、複数プロセスに
    prepped辞書を渡して並列処理するには、この形の(picklableな)クラスが必要。
    """

    def __init__(self, spline, shift):
        self.spline = spline
        self.shift = shift

    def __call__(self, t):
        return self.spline(t + self.shift)


def prepare_event(ev, period_frames, idx_by_id, attacker, defender):
    padded_frames, core_lo, core_hi = get_padded_frames(period_frames, idx_by_id, ev.frames)

    xa, ya = series_for_player(padded_frames, attacker)
    xd, yd = series_for_player(padded_frames, defender)

    xa_s, _ = sg_smooth_and_velocity(xa, DT, SG_WINDOW, SG_POLYORDER)
    ya_s, _ = sg_smooth_and_velocity(ya, DT, SG_WINDOW, SG_POLYORDER)
    xd_s, vxd_s = sg_smooth_and_velocity(xd, DT, SG_WINDOW, SG_POLYORDER)
    yd_s, vyd_s = sg_smooth_and_velocity(yd, DT, SG_WINDOW, SG_POLYORDER)
    if xa_s is None or xd_s is None:
        return None

    t_full = np.arange(len(xa_s)) * DT
    ax_interp_full = CubicSpline(t_full, xa_s)
    ay_interp_full = CubicSpline(t_full, ya_s)
    t_shift = t_full[core_lo]

    ax_interp = ShiftedInterp(ax_interp_full, t_shift)
    ay_interp = ShiftedInterp(ay_interp_full, t_shift)

    t_core = t_full[core_lo : core_hi + 1] - t_shift
    xd_obs = xd_s[core_lo : core_hi + 1]
    yd_obs = yd_s[core_lo : core_hi + 1]

    if len(t_core) < 5:
        return None

    state0 = [xd_obs[0], yd_obs[0], vxd_s[core_lo], vyd_s[core_lo]]

    return {
        "t_core": t_core,
        "xd_obs": xd_obs,
        "yd_obs": yd_obs,
        "state0": state0,
        "ax_interp": ax_interp,
        "ay_interp": ay_interp,
        # 生の配列も保持しておく。scipyのCubicSplineは内部にモジュール参照を持ち
        # pickle化できないため(multiprocessingで使えない)、プロセス境界を越える際は
        # これらの配列だけを渡し、ワーカー側でスプラインを再構築する(to_picklable/rebuild_interp参照)。
        "t_full": t_full,
        "xa_s": xa_s,
        "ya_s": ya_s,
        "t_shift": t_shift,
    }


def to_picklable(prepped):
    """prepped辞書からpickle不能なCubicSpline系オブジェクトを除いた版を返す
    (multiprocessingでワーカーに渡す用)。"""
    return {k: v for k, v in prepped.items() if k not in ("ax_interp", "ay_interp")}


def rebuild_interp(prepped_picklable):
    """to_picklableで落とした ax_interp/ay_interp を、ワーカープロセス内で再構築する。"""
    p = dict(prepped_picklable)
    ax_full = CubicSpline(p["t_full"], p["xa_s"])
    ay_full = CubicSpline(p["t_full"], p["ya_s"])
    p["ax_interp"] = ShiftedInterp(ax_full, p["t_shift"])
    p["ay_interp"] = ShiftedInterp(ay_full, p["t_shift"])
    return p


def simulate(params, prepped):
    tau_d, f_d, theta_d = params
    t_core = prepped["t_core"]
    sol = solve_ivp(
        defender_rhs,
        t_span=(t_core[0], t_core[-1]),
        y0=prepped["state0"],
        t_eval=t_core,
        args=(tau_d, f_d, theta_d, prepped["ax_interp"], prepped["ay_interp"]),
        method="RK45",
        max_step=DT,
    )
    if not sol.success:
        return None
    return sol.y[0], sol.y[1]


def fitting_error(params, prepped):
    result = simulate(params, prepped)
    if result is None:
        return 1e3
    xd_model, yd_model = result
    xd_obs, yd_obs = prepped["xd_obs"], prepped["yd_obs"]
    numerator = np.sum(np.hypot(xd_obs - xd_model, yd_obs - yd_model))
    denom = np.sum(np.hypot(np.diff(xd_obs), np.diff(yd_obs)))
    if denom < 1e-9:
        return 1e3
    return numerator / denom


def fit_event(prepped, seed=0):
    bounds = [TAU_BOUNDS, F_BOUNDS, THETA_BOUNDS]
    res = differential_evolution(
        fitting_error,
        bounds,
        args=(prepped,),
        seed=seed,
        maxiter=60,
        popsize=15,
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
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

        params, err = fit_event(prepped, seed=i)
        tau_d, f_d, theta_d = params
        print(
            f"event {i}: duration={ev.duration_s:.2f}s  "
            f"tau_d={tau_d:.2f} f_d={f_d:.2f} theta_d={np.degrees(theta_d):.1f}deg  "
            f"error={err:.4f}"
        )

        xd_model, yd_model = simulate(params, prepped)
        ax = axes[i]
        ax.plot(prepped["xd_obs"], prepped["yd_obs"], "o-", color="cyan", label="defender (empirical)", markersize=3)
        ax.plot(xd_model, yd_model, "-", color="blue", label="defender (model)", linewidth=2)
        t_core = prepped["t_core"]
        xa_plot = [prepped["ax_interp"](t) for t in t_core]
        ya_plot = [prepped["ay_interp"](t) for t in t_core]
        ax.plot(xa_plot, ya_plot, "o-", color="orange", label="attacker (empirical)", markersize=3)
        ax.set_title(f"event {i}: err={err:.3f}, f_d={f_d:.2f}")
        ax.set_aspect("equal")
        if i == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/fit_oad_parameters_pilot.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")
