# Scattered-Data-Interpolation

散布データ補間

## プロジェクト概要

外部ライブラリに依存せずに 3 次元散布データ補間手法を比較する Python スクリプトを収録しています。SciPy や scikit-learn を利用できない環境でも実行できるよう、線形代数計算から補間アルゴリズムまで標準ライブラリで実装しています。

## 使い方

```bash
python src/compare_interpolation.py
```

実行すると以下の成果物が生成されます。

- `data/dummy_3d_points.csv`: sin 波ベースに小さなノイズを加えた 10 点の 3 次元散布データセット。
- `output/results_summary.csv`: 各補間法の RMSE、1 次滑らかさ、2 次滑らかさ指標。
- `output/interpolation_comparison.svg`: 上記指標を可視化した棒グラフ。

コンソールには各手法の要約が表示され、精度と滑らかさを比較できます。

## GitHub Pages での実行

リポジトリを GitHub に配置した状態で、ブラウザ上だけで補間処理を実行して結果を確認することもできます。

1. GitHub リポジトリの設定で **Pages** を有効化し、ビルド元に `main` ブランチの `docs/` ディレクトリを指定します。
2. デプロイ完了後、公開された GitHub Pages の URL にアクセスすると `docs/index.html` がロードされます。
3. ページ上の「結果を計算」ボタンを押すと Pyodide 上で Python が実行され、表・棒グラフに加えて各補間法ごとの 3D サーフェスが表示されます。
   ドロップダウンで押しつぶす軸を選べば、その軸方向を平均化した 2 次元サーフェス（例: Z 軸を押しつぶして X-Y 平面に展開）を切り替えられます。
4. 独自のデータを試したい場合は <code>x,y,z,value</code> 列を含む CSV ファイルをインポートしてください（ヘッダー行は任意）。
   ファイルを読み込む前は付属のダミーデータで計算が実行され、インポート後は選択したファイルが使われます。「ダミーデータを使用」ボタンで付属データに戻せます。

Pyodide は `docs/compare_interpolation.py`（ローカルの `src/compare_interpolation.py` と同一内容）をブラウザ内に読み込んで実行するため、GitHub Pages 上でもローカルと同じ純 Python 実装を利用できます。両者を更新した際は内容が一致するようにしてください。
