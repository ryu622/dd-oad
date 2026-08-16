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

import multiprocessing as mp
import os

import numpy as np
from scipy.optimize import differential_evolution, minimize

from fit_oad_dynamic_fd import fitting_error_dynamic, FD0_BOUNDS, FDINF_BOUNDS, D0_BOUNDS
from fit_oad_parameters import TAU_BOUNDS, THETA_BOUNDS, to_picklable, rebuild_interp

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


def _inner_worker(args):
    """multiprocessing用のトップレベル関数(spawnでpickleできるよう独立させる)。

    prepped_picklable は ax_interp/ay_interp(CubicSpline)を含まない版
    (to_picklableで除去済み)。ワーカープロセス内でスプラインを再構築してから使う。
    """
    prepped_picklable, f_d0, f_d_inf, d0, seed = args
    prepped = rebuild_interp(prepped_picklable)
    _, e = fit_inner(prepped, f_d0, f_d_inf, d0, seed=seed)
    return min(e, 10.0)  # 外れ値イベントに引きずられないよう上限を設ける


def pooled_objective(shared_params, prepped_list, seed=0, pool=None, prepped_list_picklable=None):
    """共有パラメータに対する目的関数: 各イベントの最良誤差の平均。

    pool(multiprocessing.Pool)を渡すと、イベントごとの内側最適化を並列実行する
    (このループが計算コストの支配項なので、並列化の効果が最も大きい)。
    並列時は、あらかじめto_picklableしておいたprepped_list_picklableを使う。
    """
    f_d0, f_d_inf, d0 = shared_params
    if pool is not None:
        args = [
            (p, f_d0, f_d_inf, d0, seed + i)
            for i, p in enumerate(prepped_list_picklable or [to_picklable(p) for p in prepped_list])
        ]
        errors = pool.map(_inner_worker, args)
    else:
        errors = []
        for i, prepped in enumerate(prepped_list):
            _, e = fit_inner(prepped, f_d0, f_d_inf, d0, seed=seed + i)
            errors.append(min(e, 10.0))
    return sum(errors) / len(errors)


def fit_pooled(prepped_list, seed=0, verbose=True, checkpoint_path=None, n_workers=None):
    """checkpoint_path を指定すると、世代ごとに現時点の最良推定値をJSONへ書き出す。
    実行が途中で中断されても、直前の世代までの最良値がファイルに残る。

    n_workers: イベントループの並列プロセス数。Noneの場合はCPUコア数-1を使う。
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    bounds = [FD0_BOUNDS, FDINF_BOUNDS, D0_BOUNDS]
    n_eval = {"count": 0}
    n_gen = {"count": 0}
    # ワーカーに渡す用のpickle可能な版を1回だけ作っておく(毎回作り直すと無駄)
    prepped_list_picklable = [to_picklable(p) for p in prepped_list]

    with mp.Pool(processes=n_workers) as pool:
        if verbose:
            print(f"  [pool] using {n_workers} worker processes", flush=True)

        def obj(shared):
            n_eval["count"] += 1
            val = pooled_objective(
                shared, prepped_list, seed=seed, pool=pool,
                prepped_list_picklable=prepped_list_picklable,
            )
            if verbose and n_eval["count"] % 10 == 0:
                print(
                    f"    [eval {n_eval['count']}] f_d0={shared[0]:.2f} f_d_inf={shared[1]:.2f} "
                    f"d0={shared[2]:.2f} -> mean_err={val:.4f}",
                    flush=True,
                )
            return val

        def save_checkpoint(xk, convergence):
            n_gen["count"] += 1
            if checkpoint_path is None:
                return
            import json
            f_d0, f_d_inf, d0 = xk
            err = pooled_objective(
                xk, prepped_list, seed=seed, pool=pool,
                prepped_list_picklable=prepped_list_picklable,
            )
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {
                        "generation": n_gen["count"],
                        "n_eval": n_eval["count"],
                        "n_events": len(prepped_list),
                        "f_d0": float(f_d0),
                        "f_d_inf": float(f_d_inf),
                        "d0": float(d0),
                        "mean_error": float(err),
                        "convergence": float(convergence),
                        "status": "in_progress",
                    },
                    f,
                    indent=2,
                )
            if verbose:
                print(f"  [checkpoint saved] generation={n_gen['count']} -> {checkpoint_path}", flush=True)

        res = differential_evolution(
            obj,
            bounds,
            seed=seed,
            maxiter=OUTER_MAXITER,
            popsize=OUTER_POPSIZE,
            tol=1e-3,
            callback=save_checkpoint if checkpoint_path else None,
            polish=True,
        )
    return res.x, res.fun
