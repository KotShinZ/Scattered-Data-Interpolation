# Scattered-Data-Interpolation

散布データ補間

## プロジェクト概要

外部ライブラリに依存せずに 3 次元散布データ補間手法を比較する Python スクリプトを収録しています。SciPy や scikit-learn を利用できない環境でも実行できるよう、線形代数計算から補間アルゴリズムまで標準ライブラリで実装しています。

## 使い方

```bash
python src/compare_interpolation.py
```

実行すると以下の成果物が生成されます。

- `data/dummy_3d_points.csv`: 乱数で生成した 3 次元散布データセット。
- `output/results_summary.csv`: 各補間法の RMSE、1 次滑らかさ、2 次滑らかさ指標。
- `output/interpolation_comparison.svg`: 上記指標を可視化した棒グラフ。

コンソールには各手法の要約が表示され、精度と滑らかさを比較できます。
