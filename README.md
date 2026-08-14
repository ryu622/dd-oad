# dd-oad

サッカーの1対1(アタッカー・ディフェンダー)局面をモデル化したOADモデル
(Yamazaki, Goto, Otoguro, Nishimori, Shiraishi & Narizuka, 2026, arXiv:2607.05845,
"A behavioral principle underlying attacker–defender interactions in soccer")を拡張する研究プロジェクト。

## 研究テーマ

OADモデルは、ディフェンダーの駆動力の大きさ $f_d$ を1イベント(平均約2.5秒)を通じて時間不変の定数として扱う。本研究はこの仮定を緩和し、$f_d$ を間合い $d(t)$ の関数として動的にモデル化する:

$$\dot{\mathbf{v}}_d = -\frac{\mathbf{v}_d}{\tau_d} + f_d(d(t))\left[-\cos\theta_d\,\mathbf{e}_1 + \sin\theta_d\,\mathbf{e}_2\right]$$

$$H_1:\ \text{間合い}d(t)\text{が縮まるにつれて、}f_d\text{(駆動力の大きさ)は増加する}$$

詳細な背景・仮説の経緯・novelty上の位置づけは [`documents/research_plan.md`](documents/research_plan.md) を参照。

当初は駆動角度 $\theta_d$ の動的化(平行戦略↔積極戦略の切り替え)を主眼としていたが、パイロット検証の結果 $\theta_d$ は間合いによらずほぼ一定であり、代わりに $f_d$ が間合いと相関することが分かったため主仮説を転換した。転換の経緯・検証結果は [`documents/judgement_results.md`](documents/judgement_results.md) にまとめている。

## セットアップ

Python 3.12、[`uv`](https://github.com/astral-sh/uv) で環境を管理している。

```bash
uv sync
uv run python scripts/<script_name>.py
```

`pip`/`python` を直接使わず、常に `uv run` を経由すること(詳細は `CLAUDE.md`)。

## データ

[spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data)(ブンデスリーガ1部・2部2022/23シーズン、Hugging Face `pysport/idsse-data` として再ホスト)を [`kloppy`](https://github.com/PySport/kloppy) 経由でロードする。先行研究本来のJ1リーグデータ(DataStadium提供)へのアクセスは未確認のため、公開データセットで代替している。

参照論文PDF(`confer/OAD.pdf`)は著作権の都合上リポジトリには含めていない(`.gitignore`で除外)。

## ディレクトリ構成

```
documents/
  research_plan.md        研究計画書(目的・仮説・手法・スケジュール)
  judgement_results.md    パイロット検証(判定①〜③)・パラメータ推定の結果まとめ
scripts/
  extract_dribble_events.py       先行研究Methods B節準拠の5条件による1v1(ドリブル)イベント抽出
  judgement1_distance_variation.py  判定①: 間合いd(t)の変動幅の確認(TacklingGame起点、旧版)
  judgement2_instantaneous_theta.py 判定②: 瞬間加速度・θ_d(t)推定の実現可能性(旧版)
  judgement3_correlation.py         判定③: d(t) vs θ_d(t)の相関(旧版)
  judgement_v2.py                   新方式抽出によるθ_d相関・|a_d|(f_dプロキシ)相関の再検証
  judgement_v2_speed_control.py     |a_d(t)|とd(t)の相関がv_d(t)の交絡でないかの偏相関チェック
  fit_oad_parameters.py             (τ_d, f_d, θ_d)同時フィッティング(先行研究Methods C節準拠)
  fit_oad_parameters_batch.py       全イベントへのバッチフィッティング
```

## 現在の状況

判定①〜③、および $(\tau_d,f_d,\theta_d)$ の同時フィッティング(フェーズ1)まで完了。$f_d$ と間合いの相関は方向として一貫して負(H1と整合)だが、フィッティング品質(約3割のイベントで $\tau_d$ が探索境界に張り付く)に課題が残っており、効果の大きさはまだ確定していない。詳細は `documents/judgement_results.md` の「追補3」を参照。
