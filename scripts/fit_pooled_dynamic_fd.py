"""階層(プーリング)モデル: f_d(d)の形状パラメータを全イベントで共有して同時推定する。

追補5で判明した通り、1イベント単独では (f_d0, f_d_inf, d0) は現実のノイズ水準で
識別不能である。そこで:

  - **共有パラメータ(全イベント共通)**: f_d0, f_d_inf, d0
      -> f_d(d) の「形」は選手・状況によらず共通の行動則である、という仮定
  - **イベント固有パラメータ**: tau_d, theta_d
      -> 各イベントの個別事情はこの2つに吸収させる

共有パラメータについては、1軌跡では埋もれる微弱な信号を多数のイベントで積み上げる形になり、
ノイズが打ち消し合って識別性が上がることを期待する。

最適化は入れ子構造:
  外側: (f_d0, f_d_inf, d0) を大域探索
  内側: 各イベントについて (tau_d, theta_d) だけを最適化し、その誤差の総和を外側の目的関数とする
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize

from fit_oad_dynamic_fd import fitting_error_dynamic, FD0_BOUNDS, FDINF_BOUNDS, D0_BOUNDS
from fit_oad_parameters import TAU_BOUNDS, THETA_BOUNDS

# 内側は2次元(tau_d, theta_d)なので、粗いグリッド探索 + 局所磨き上げで十分かつ高速。
# differential_evolutionを内側にも使うと1評価あたり数百回のODE積分が必要になり、
# 入れ子最適化全体が現実的な時間に収まらなかった。
#
# 復元テスト(recovery_test_pooled.py)では6x12グリッド・外側maxiter=20/popsize=8で
# 49分/10イベントかかったが、ログを見るとeval~600以降ほぼ結果が変わっておらず、
# 収束後も探索を続けて予算を浪費していた。グリッドと外側予算を絞って高速化する。
INNER_TAU_GRID = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
INNER_THETA_GRID = np.linspace(-np.pi, np.pi, 8, endpoint=False)

OUTER_MAXITER = 12
OUTER_POPSIZE = 6


def fit_inner(prepped, f_d0, f_d_inf, d0, seed=0):
    """共有パラメータを固定し、そのイベントの (tau_d, theta_d) だけを最適化する。

    粗いグリッドで最良点を見つけ、そこからNelder-Meadで磨き上げる。
    """

    def err(params_2d):
        tau_d, theta_d = params_2d
        tau_d = np.clip(tau_d, *TAU_BOUNDS)
        theta_d = np.clip(theta_d, *THETA_BOUNDS)
        return fitting_error_dynamic((tau_d, f_d0, f_d_inf, d0, theta_d), prepped)

    best_x, best_e = None, np.inf
    for tau in INNER_TAU_GRID:
        for theta in INNER_THETA_GRID:
            e = err((tau, theta))
            if e < best_e:
                best_e, best_x = e, (tau, theta)

    res = minimize(err, np.array(best_x), method="Nelder-Mead",
                   options={"maxiter": 60, "xatol": 1e-2, "fatol": 1e-4})
    if res.fun < best_e:
        return res.x, res.fun
    return np.array(best_x), best_e


def pooled_objective(shared_params, prepped_list, seed=0):
    """共有パラメータに対する目的関数: 各イベントの最良誤差の平均。"""
    f_d0, f_d_inf, d0 = shared_params
    total = 0.0
    for i, prepped in enumerate(prepped_list):
        _, e = fit_inner(prepped, f_d0, f_d_inf, d0, seed=seed + i)
        total += min(e, 10.0)  # 外れ値イベントに引きずられないよう上限を設ける
    return total / len(prepped_list)


def fit_pooled(prepped_list, seed=0, verbose=True):
    bounds = [FD0_BOUNDS, FDINF_BOUNDS, D0_BOUNDS]
    n_eval = {"count": 0}

    def obj(shared):
        n_eval["count"] += 1
        val = pooled_objective(shared, prepped_list, seed=seed)
        if verbose and n_eval["count"] % 10 == 0:
            print(
                f"    [eval {n_eval['count']}] f_d0={shared[0]:.2f} f_d_inf={shared[1]:.2f} "
                f"d0={shared[2]:.2f} -> mean_err={val:.4f}",
                flush=True,
            )
        return val

    res = differential_evolution(
        obj,
        bounds,
        seed=seed,
        maxiter=OUTER_MAXITER,
        popsize=OUTER_POPSIZE,
        tol=1e-3,
        polish=True,
    )
    return res.x, res.fun
