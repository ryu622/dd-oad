"""tau_dが探索境界(20.0)に張り付く問題の診断。

tau_dを固定し、その各値についてf_d, theta_dだけを最適化した際の最良誤差を
プロットする(誤差のプロファイル)。境界に張り付いたイベントで、この誤差曲線が
tau_dを大きくするほど単調に下がり続ける(=境界の外側にも解がありそう、
tau_dが識別不能)のか、それとも境界の内側に本来の最小点があるのに最適化が
そこへ収束できていないだけなのかを見分ける。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from extract_dribble_events import extract_dribble_events
from fit_oad_parameters import prepare_event, fitting_error, F_BOUNDS, THETA_BOUNDS

TAU_GRID = np.concatenate([np.linspace(0.1, 5, 10), np.linspace(6, 20, 8), [30, 50, 100]])


def profile_error(prepped, tau_fixed):
    def err(params_2d):
        f_d, theta_d = params_2d
        return fitting_error((tau_fixed, f_d, theta_d), prepped)

    res = differential_evolution(
        err, [F_BOUNDS, THETA_BOUNDS], maxiter=40, popsize=12, tol=1e-4, polish=True, seed=0
    )
    return res.fun, res.x


def get_prepped(match_id, event_idx):
    events, frames_by_period, teams = extract_dribble_events(match_id, return_context=True)
    player_lookup = {p.player_id: p for t in teams for p in t.players}
    ev = events[event_idx]
    attacker = player_lookup[ev.attacker_id]
    defender = player_lookup[ev.defender_id]
    period_frames = frames_by_period[ev.period_id]
    idx_by_id = {id(f): i for i, f in enumerate(period_frames)}
    return prepare_event(ev, period_frames, idx_by_id, attacker, defender), ev


def main():
    df = pd.read_csv("documents/oad_fit_results.csv")

    bound_row = df[df["tau_d"] >= 19.5].iloc[0]
    good_row = df[df["fit_error"] < 0.15].sort_values("fit_error").iloc[0]

    print(f"bound-hitting example: {bound_row['match_id']} event {bound_row['event_idx']} "
          f"(fitted tau_d={bound_row['tau_d']:.2f}, f_d={bound_row['f_d']:.2f}, err={bound_row['fit_error']:.3f})")
    print(f"good-fit example: {good_row['match_id']} event {good_row['event_idx']} "
          f"(fitted tau_d={good_row['tau_d']:.2f}, f_d={good_row['f_d']:.2f}, err={good_row['fit_error']:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, row, label in [
        (axes[0], bound_row, "boundary-hitting event"),
        (axes[1], good_row, "well-fit event (for comparison)"),
    ]:
        prepped, ev = get_prepped(row["match_id"], int(row["event_idx"]))
        errors = []
        for tau in TAU_GRID:
            e, _ = profile_error(prepped, tau)
            errors.append(e)
            print(f"  tau_d={tau:.2f} -> best error (over f_d,theta_d) = {e:.4f}")
        ax.plot(TAU_GRID, errors, "o-", color="C0")
        ax.axvline(20.0, color="red", linestyle="--", label="original bound (20.0)")
        ax.set_xlabel("tau_d (fixed)")
        ax.set_ylabel("best fitting_error (optimized over f_d, theta_d)")
        ax.set_title(f"{label}\n{row['match_id']} event {int(row['event_idx'])}")
        ax.legend()

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/diagnose_tau_boundary.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
