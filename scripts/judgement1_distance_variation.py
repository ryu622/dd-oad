"""判定①: 1対1イベント内で間合い d(t) がどれだけ変動しているかを確認する。

TacklingGame(ground, WinnerRole=withBallControl)イベントを1対1の「決着時刻」とみなし、
決着前2.5秒間(先行研究のイベント平均継続時間)を疑似的な1対1イベント窓として、
Winner(アタッカー)-Loser(ディフェンダー)間の距離 d(t) をtrackingデータから算出する。

厳密な1対1イベント抽出(先行研究の移動距離・継続時間・ゴールとの位置関係による切り出し)
ではなく、判定①のための簡易プロキシであることに注意。
"""

from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from kloppy import sportec

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1"]
WINDOW_SECONDS = 2.5
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def extract_duel_events(events):
    duels = []
    for e in events.events:
        if e.event_name != "TacklingGame":
            continue
        raw = e.raw_event
        if raw.get("Type") != "ground":
            continue
        if raw.get("WinnerRole") != "withBallControl":
            continue
        if raw.get("LoserRole") != "withoutBallControl":
            continue
        if raw.get("GoalKeeperInvolved") == "true":
            continue
        duels.append(
            {
                "period_id": e.period.id,
                "timestamp": e.timestamp,
                "winner_id": raw["Winner"],
                "loser_id": raw["Loser"],
            }
        )
    return duels


def build_frame_index(tracking):
    frames_by_period = {}
    for f in tracking.frames:
        frames_by_period.setdefault(f.period.id, []).append(f)
    for pid in frames_by_period:
        frames_by_period[pid].sort(key=lambda fr: fr.timestamp)
    return frames_by_period


def player_lookup(tracking):
    lookup = {}
    for team in tracking.metadata.teams:
        for p in team.players:
            lookup[p.player_id] = p
    return lookup


def compute_d_series(frames_window, winner_player, loser_player):
    ts, ds = [], []
    for f in frames_window:
        wd = f.players_data.get(winner_player)
        ld = f.players_data.get(loser_player)
        if wd is None or ld is None or wd.coordinates is None or ld.coordinates is None:
            continue
        dx = (wd.coordinates.x - ld.coordinates.x) * PITCH_LENGTH
        dy = (wd.coordinates.y - ld.coordinates.y) * PITCH_WIDTH
        ts.append(f.timestamp.total_seconds())
        ds.append((dx**2 + dy**2) ** 0.5)
    return np.array(ts), np.array(ds)


def main():
    all_series = []
    summary_rows = []

    for match_id in MATCH_IDS:
        print(f"=== loading {match_id} ===")
        events = sportec.load_open_event_data(match_id=match_id)
        tracking = sportec.load_open_tracking_data(match_id=match_id)

        duels = extract_duel_events(events)
        print(f"{match_id}: {len(duels)} ground/withBallControl duels found")

        frames_by_period = build_frame_index(tracking)
        players = player_lookup(tracking)

        for duel in duels:
            winner_player = players.get(duel["winner_id"])
            loser_player = players.get(duel["loser_id"])
            if winner_player is None or loser_player is None:
                continue

            period_frames = frames_by_period.get(duel["period_id"], [])
            t_end = duel["timestamp"]
            t_start = t_end - timedelta(seconds=WINDOW_SECONDS)

            window = [f for f in period_frames if t_start <= f.timestamp <= t_end]
            if len(window) < 10:
                continue

            ts, ds = compute_d_series(window, winner_player, loser_player)
            if len(ds) < 10:
                continue

            t_rel = ts - ts[-1]
            all_series.append((t_rel, ds))
            summary_rows.append(
                {
                    "match_id": match_id,
                    "d_start": ds[0],
                    "d_end": ds[-1],
                    "d_min": ds.min(),
                    "d_max": ds.max(),
                    "range": ds.max() - ds.min(),
                    "n_frames": len(ds),
                }
            )

    print(f"\ntotal usable events: {len(all_series)}")

    ranges = np.array([r["range"] for r in summary_rows])
    d_starts = np.array([r["d_start"] for r in summary_rows])
    d_ends = np.array([r["d_end"] for r in summary_rows])
    monotone_decrease = np.mean(d_ends < d_starts)

    print(f"range(d) [m]: mean={ranges.mean():.2f}, median={np.median(ranges):.2f}, "
          f"min={ranges.min():.2f}, max={ranges.max():.2f}")
    print(f"d_start mean={d_starts.mean():.2f}m, d_end mean={d_ends.mean():.2f}m")
    print(f"fraction with d_end < d_start (approaching by tackle time): {monotone_decrease:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for t_rel, ds in all_series:
        ax.plot(t_rel, ds, alpha=0.25, linewidth=0.8, color="C0")
    ax.set_xlabel("time relative to tackle event [s]")
    ax.set_ylabel("distance d(t) [m]")
    ax.set_title(f"d(t) overlay, n={len(all_series)} events")
    ax.axhline(0, color="gray", linewidth=0.5)

    ax2 = axes[1]
    ax2.hist(ranges, bins=30, color="C1")
    ax2.set_xlabel("range of d(t) within window [m]  (max - min)")
    ax2.set_ylabel("count")
    ax2.set_title("distribution of within-event distance variation")

    fig.tight_layout()
    out_path = "/Users/ryuya/dev/dd-oad/documents/judgement1_distance_variation.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
