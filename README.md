# MeshMAE Zero-Shot – 化石メッシュ自動クラスタリングガイド

本リポジトリは、化石や骨格の3Dメッシュを対象としたゼロショット（学習済みモデルの再学習なし）クラスタリングパイプラインを提供します。公式 [MeshMAE](https://github.com/Maple728/MeshMAE) の学習ユーティリティをラップし、化石ドメイン向けの事前学習継続、特徴量抽出、クラスタ数の自動決定、レポート生成までを一貫して実行できるように構成されています。ここでは環境構築から各ステップの意味、関係する用語の詳細解説までを網羅的にまとめています。

## リポジトリ全体像

```
meshmae-zero-shot/
├── README.md
├── env/
│   └── requirements.txt         # Python 依存パッケージ一覧（MeshMAE + クラスタリング関連）
├── configs/
│   ├── pretrain_target.yaml     # 化石ドメインでの自己教師あり学習（SSL）継続設定
│   ├── extract.yaml             # 特徴量抽出の標準設定
│   └── cluster.yaml             # クラスタリング・自動クラスタ数決定・レポート設定
├── datasets/
│   ├── fossils_raw/             # ユーザーが用意する生のメッシュ（.ply / .stl / .obj など）
│   └── fossils_maps/            # 前処理済みメッシュ（多様体化 + 約500面 + MAPS）
├── checkpoints/
│   ├── shapenet_pretrain.pkl    # 公式 MeshMAE 事前学習済みモデル（入力）
│   └── fossils_target.pkl       # 化石向け継続学習後のモデル（出力例）
├── src/
│   ├── preprocess/
│   │   └── make_manifold_and_maps.py
│   ├── pretrain/
│   │   └── run_target_pretrain.sh
│   ├── embed/
│   │   └── extract_embeddings.py
│   ├── cluster/
│   │   ├── auto_k.py
│   │   ├── run_clustering.py
│   │   ├── plot_dendrogram.py
│   │   └── templates/
│   │       └── report.html
│   └── viz/
│       └── render_thumbs.py
├── out/                         # 実行結果（プロット、CSV、HTML レポートなど）
└── embeddings/                  # 抽出済み特徴量（.npy / .csv）
```

`.gitignore` により大容量成果物はバージョン管理対象外ですが、空ディレクトリを維持するために `.gitkeep` が配置されています。

## 背景とコンセプト

### ゼロショットクラスタリングとは？
学習済みモデルを追加の教師あり学習なしで別データに適用し、類似度に基づいてデータを自律的にグループ分けする手法です。本プロジェクトでは、ShapeNet 等で事前学習された MeshMAE エンコーダを利用して化石メッシュの特徴量を抽出し、教師ラベルなしでクラスタリングを行います。

### MeshMAE とは？
MeshMAE はマスク化自己エンコーダ（Masked Autoencoder, MAE）を三角メッシュ向けに拡張したフレームワークです。メッシュの局所パッチをマスクして再構成することで形状の潜在表現を学習します。本リポジトリでは公式実装を外部依存として呼び出し、化石ドメインに合わせて追加の自己教師あり学習を行うラッパースクリプトを提供しています。

### パイプライン概要
1. **前処理**: メッシュを多様体化し、面数を約500にリサンプリング。必要に応じて MAPS（Multiresolution Adaptive Parameterization of Surfaces）階層を生成。
2. **自己教師あり学習継続（ドメイン適応）**: 化石データで MeshMAE エンコーダを追加学習し、ドメイン固有の特徴を抽出可能に。
3. **特徴量抽出**: 学習済みエンコーダから固定長ベクトルを取得。MeshMAE が利用できない場合は幾何特徴量にフォールバック。
4. **クラスタリング**: PCA や正規化を行い、自動クラスタ数推定（エルボー法、シルエット係数、ギャップ統計、GMM-BIC）を用いて K-Means と HDBSCAN によるクラスタリングを実施。結果を HTML レポートとして整理。

## 環境構築

1. Python 3.9 以上を推奨します。任意の仮想環境（`venv`, `conda`, `mamba` 等）を作成してください。
2. 依存パッケージをインストールします。

```bash
pip install -r env/requirements.txt
```

この requirements には、PyTorch (>=1.11) および CUDA 11.1 以降、メッシュ処理ライブラリ（`trimesh`, `open3d`, `pyvista`）、クラスタリング・可視化系ライブラリ（`scikit-learn`, `hdbscan`, `umap-learn`, `matplotlib`, `seaborn`）などが含まれます。

> **補足:** 公式 MeshMAE リポジトリを隣接ディレクトリにクローンし、開発モードでインストールしておくとエンコーダ関連の参照がスムーズです。
>
> ```bash
> git clone https://github.com/Maple728/MeshMAE.git ../MeshMAE
> pip install -e ../MeshMAE
> ```

## データ要件

- `datasets/fossils_raw/`: ユーザーが用意する生データを配置します。PLY/STL/OBJ など一般的なポリゴンメッシュ形式に対応しています。
- `datasets/fossils_maps/`: 前処理済みデータを保存します。メッシュは多様体化され、面数が揃えられ、必要に応じて MAPS 階層が作成されます。
- 任意でメタデータ CSV（最低でも `sample_id` 列）を用意すると、クラスタリング後の分析が容易になります。

## ステップ別手順

### 1. メッシュ前処理
MeshMAE の実験は多様体メッシュで約500面、MAPS 階層が整備されていることを前提としています。本リポジトリのラッパースクリプトから公式ツールチェーンを呼び出し、必要な加工を自動化します。`--make_maps` を指定した場合は、公式 SubdivNet リポジトリの `datagen_maps.py` をそのまま `--maps_script` に渡してください（隣接フォルダに `../SubdivNet/datagen_maps.py` がある場合は自動検出されます）。

#### MAPS 生成の呼び出し方法（デモ実行を避ける）
SubdivNet の `datagen_maps.py` はそのまま実行すると `if __name__ == "__main__": MAPS_demo1()` が走り、同梱デモ `airplane.obj` を探しに行きます。本リポジトリでは、デモを起動せずに `make_MAPS_shape` 相当の処理のみ呼び出すため、`src.preprocess.run_subdivnet_maps` という薄い CLI ラッパーを別プロセスで起動します。ラッパーは次の動作を行います。

- `--subdivnet_root` で渡されたパスを `PYTHONPATH` に追加し、`datagen_maps` と `maps.MAPS` を import する（親プロセスに SubdivNet の依存を混ぜない）。
- 入力メッシュと出力パスを「絶対パス」で受け取り、`base_size` / `depth` / `max_base_size` を使って MAPS を生成する。
- 出力ファイルは拡張子付き（例: `<outdir>/<stem>_MAPS.obj`）で保存し、`trimesh.export` の exporter 解決エラーを防ぐ。

`make_manifold_and_maps.py` 側は常にこのラッパーを subprocess で呼び出します。前処理中に SubdivNet 側の `__main__` ブロックが実行されることはなくなり、`cwd` もいじらないため、入力/出力を絶対パスで渡せばパス解決事故を防げます（旧実装は SubdivNet を `cwd` にして相対パスで呼んでいたため、別の場所に MAPS が出力され `_maps` が空のまま残るケースがありました）。

- 面数削減後のメッシュは `<out>/<stem>.<ext>` に保存されます。500 面以下のメッシュは簡約処理をスキップします。
- MAPS 出力は `<out>/<stem>_maps/` 配下に `<stem>_MAPS<ext>` が生成されます（`<ext>` は入力メッシュに倣います）。
- MAPS が失敗した場合は空ディレクトリを残さず、`<stem>_maps/error.log` にコマンド・stdout・stderr が残ります。成功時のみ `_maps` ディレクトリが移動して確定します。

```bash
python -m src.preprocess.make_manifold_and_maps \
  --in datasets/fossils_raw \
  --out datasets/fossils_maps \
  --target_faces 500 \
  --num_workers 0 \
  --make_maps \
  --maps_script ../SubdivNet/datagen_maps.py \
  --metadata datasets/fossils_maps/processing_metadata.json \
  --maps_extra_args -- --base_size 96 --depth 3 --max_base_size 192
```

`--maps_script` を省略した場合は `../SubdivNet/datagen_maps.py` を自動的に探します（見つからない場合はエラーになります）。`--maps_extra_args` で渡したいオプションは、すべて末尾に置くか、`--maps_extra_args -- --foo bar` のように区切ってください（`argparse.REMAINDER` で受け取るため）。SubdivNet の依存（`maps` パッケージが要求する `triangle`, `sortedcollections`, `networkx`, `rtree` など）は本リポジトリの `env/requirements.txt` には含まれていないため、必要に応じて SubdivNet 側の仮想環境にインストールしてください。



前処理の並列化は `--num_workers` で制御します。0 または 1 を指定すると従来どおり逐次処理になり、2 以上でその数だけプロセスを立ててメッシュを並列処理します。I/O 帯域や MAPS 生成の外部スクリプトがボトルネックになる場合は、CPU コア数より少なめの値に絞ると安定します。

並列化なし（逐次処理）の例:

```bash
python -m src.preprocess.make_manifold_and_maps \
  --in datasets/fossils_raw \
  --out datasets/fossils_maps \
  --target_faces 500 \
  --num_workers 0
```

並列化ありの例（4 プロセス）:

```bash
python -m src.preprocess.make_manifold_and_maps \
  --in datasets/fossils_raw \
  --out datasets/fossils_maps \
  --target_faces 500 \
  --num_workers 4
```

処理後は `datasets/fossils_maps/` 以下に元ディレクトリ構造を保ったまま保存され、面数やスケール、MAPS 有無を記録した JSON マニフェストが出力されます。

#### 二次デシメーションに必要なライブラリが入っているか確認する

`trimesh` 4.10 以降の二次デシメーションは `simplify_quadric_decimation` という名称で、内部的に [`fast-simplification`](https://pypi.org/project/fast-simplification/) を利用する薄いラッパーになっています。以下のコマンドで依存関係が導入済みかを確認できます（`simplify_quadric_decimation` に目標面数を渡すときは `face_count=<目標>` というキーワード引数を使う必要があります。位置引数は 0〜1 の削減率 `percent` として解釈されるため注意してください）。

- Python パッケージの存在とバージョンを確認

  ```bash
  python -m pip show trimesh fast-simplification open3d
  ```

- `simplify_quadric_decimation` が呼べるかをスクリプトで確認

  ```bash
  python - <<'PY'
  import trimesh

  print("trimesh version:", trimesh.__version__)
  print("has simplify_quadric_decimation:", hasattr(trimesh.Trimesh(), "simplify_quadric_decimation"))
  PY
  ```

- `simplify_quadric_decimation` が存在しない場合は、`python -m pip install -U fast-simplification` を実行してから上記チェックを再実行してください。Open3D は別系統のメッシュ簡略化 API を提供しますが、trimesh のこのメソッド自体は fast-simplification をバックエンドとして使います（fast-simplification のビルドに必要なツールチェーンがない場合は `pip install fast-simplification --no-binary=:all:` などでソースビルドを試し、C/C++ コンパイラや cmake を用意してください）。

#### MAPS 生成手順（公式 SubdivNet をそのまま利用）

MAPS 自体は SubdivNet の `maps` 実装に依存します。改変版ではなく、公式リポジトリをそのまま隣接フォルダにクローンして利用してください。

```bash
git clone https://github.com/lzhengning/SubdivNet ../SubdivNet
python -m pip install -r ../SubdivNet/requirements.txt
python -m pip install -e ../SubdivNet  # maps モジュールを Python から参照できるようにする
```

`maps` モジュールとその依存パッケージを用意した上で、次のいずれかの方法で実行してください。`--maps_extra_args` は `argparse.REMAINDER` で受け取るため、前処理スクリプト側のオプション（`--metadata` など）は必ず `--maps_extra_args` より前に置き、その後に `--` を挟んで MAPS スクリプトへ渡したい引数を列挙します。

1. **単一メッシュ（または `make_manifold_and_maps.py` からの委譲）**

  ```bash
  python ../SubdivNet/datagen_maps.py --base_size 96 --depth 3 input.obj output.obj
  ```

   `base_size=96`, `depth=3`, `max_base_size=192` は化石データ向けの保守的な推奨値です。`make_manifold_and_maps.py` から呼ぶ場合は前述の例のように `--maps_extra_args` へ同じオプションを渡します。

2. **データセット全体（フォルダ構造が `<src>/<label>/<split>/*.obj` の場合）**

  ```bash
  python ../SubdivNet/datagen_maps.py --config FOSSILS --n_worker 8
  ```

   FOSSILS 設定は `src_root=../MeshMAE-Zero-Shot/datasets/fossils_raw`、`dst_root=../MeshMAE-Zero-Shot/datasets/fossils_maps`、`n_variation=1`、`base_size=96`、`max_base_size=192`、`depth=3` を既定にしています。ラベル/分割フォルダが存在しない場合は、`make_manifold_and_maps.py` 経由で単一メッシュモードを使ってください（フォルダ構造を気にせず MAPS 出力が得られます）。

3. **SubdivNet オリジナル手順を使いたい場合**

   ```bash
   git clone https://github.com/lzhengning/SubdivNet
   cd SubdivNet
   pip install -r requirements.txt  # triangle, pymeshlab, shapely, networkx, rtree など
   python datagen_maps.py  # 必要に応じて src_root/dst_root を編集
   ```

配布済み MAPS データをそのまま使うだけなら、生成処理を回す必要はありません。

> **Troubleshooting – MAPS ディレクトリが空に見える場合**
>
> 以前の実装では SubdivNet をカレントディレクトリにしたまま相対パスの入力/出力を渡していたため、MAPS の出力が別場所へ書き出され、`<stem>_maps/` が空のまま残る不具合がありました。現在は入力メッシュと出力先を絶対パスで渡し、`PYTHONPATH` に SubdivNet を追加することで、期待したフォルダに MAPS ファイルが生成されます。失敗時は `_maps/error.log` を確認してください。

生成された MAPS 出力フォルダを本リポジトリ直下に作成した `datasets/` 配下へ配置し、MeshMAE 実行時には `--dataroot` でそのフォルダを指します（例: `--dataroot ./datasets/Manifold40-MAPS-96-3/`）。

### 2. 自己教師あり学習の継続（ドメイン適応）
#### 継続SSLの実行方法
1. `checkpoints/shapenet_pretrain.pkl` に公式 MeshMAE の事前学習済みチェックポイントを配置します。
2. `configs/pretrain_target.yaml` を編集し、`dataroot`（前処理済み化石メッシュ）、`init_checkpoint`（初期重み）、`save_checkpoint`（出力先）などを必要に応じて書き換えます。
3. `bash src/pretrain/run_target_pretrain.sh --config configs/pretrain_target.yaml` を実行します。`dry_run: true` に設定するとコマンド内容とパスの検証のみ行います。
4. 学習完了後は `checkpoints/fossils_target.pkl`（`save_checkpoint` で指定したパス）に継続学習済みモデルが保存され、`out/pretrain_log.json` に実行コマンドと missing/unexpected keys の記録が残ります。

> **注意:** `init_checkpoint` を空にしたまま `checkpoints/shapenet_pretrain.pkl` が存在すると安全のためスクリプトが停止します。初期化ファイルを使わない場合は名称を変更するか、別パスに移動してください。

~~公式の MeshMAE 事前学習済みチェックポイント（例: `shapenet_pretrain.pkl`）を `checkpoints/` に配置します。ラッパースクリプトが `configs/pretrain_target.yaml` を読み込み、公式スクリプト `scripts/pretrain/train_pretrain.sh` にパラメータを橋渡しします。~~

~~```bash
~~# MeshMAE リポジトリが別ディレクトリにある場合は環境変数で指定
~~bash src/pretrain/run_target_pretrain.sh --config configs/pretrain_target.yaml
~~```~~

~~YAML には `--dataroot`, `--batch_size`, `--epochs` などの設定が含まれます。動作確認のみの場合は `--dry-run` を付与してください。~~

~~> **発展的な利用:** MeshMAE 本体に継続学習向けのパッチを適用している場合は、`configs/pretrain_target.yaml` 内の `resume_checkpoint` や `save_checkpoint` を `checkpoints/shapenet_pretrain.pkl` および出力ファイルへ設定することで、初期重みからの再開と保存を制御できます。パッチを適用していない場合は、標準的な自己教師あり学習として動作します。~~

### 3. 特徴量抽出
化石向け自己教師あり学習を行った後、またはベースモデルのみで構わない場合でも、前処理済みメッシュを固定長ベクトルへ変換します。

```bash
python -m src.embed.extract_embeddings \
  --config configs/extract.yaml \
  --model-factory meshmae.models_mae.mae_vit_base_patch16 \
  --normalize
```

抽出スクリプトには二つのモードがあります。

1. **MeshMAE エンコーダモード**: 公式パッケージを利用し、`--model-factory` でエンコーダ構築関数を指定します。チェックポイント読み込みは `configs/extract.yaml` の `input.checkpoint` で制御され、`forward_encoder(...)` や `encode(...)` を呼び出して CLS トークンまたはパッチ埋め込み平均 (`encoder.pool_strategy`) をプールします。
2. **幾何特徴量フォールバックモード**: MeshMAE が利用できない場合は `--force-geometry` を指定するか、自動検出で幾何ベース特徴量（バウンディングボックス、体積、表面積、慣性テンソル、平均曲率など）を算出します。煙試験や軽量検証に便利です。

出力は `embeddings/raw_embeddings.npy`（設定で変更可）とメタデータ CSV として保存されます。必要に応じて PCA/UMAP などの次元削減結果も同時に書き出されます。

### 4. クラスタ数自動決定とクラスタリング、レポート生成
クラスタリングパイプラインでは、特徴量の標準化・PCA を施した後に複数指標を用いてクラスタ数 `k` を自動推定し、K-Means と HDBSCAN によるクラスタリング、未知クラス検出、可視化プロット、HTML レポート生成まで行います。

```bash
python -m src.cluster.run_clustering \
  --emb embeddings/raw_embeddings.npy \
  --meta embeddings/meta.csv \
  --config configs/cluster.yaml \
  --out-dir out
```

主な成果物:

- `out/cluster/kmeans_assignments.csv`
- `out/cluster/hdbscan_assignments.csv`
- `out/cluster/consensus.csv`
- `out/cluster/summary.json`
- `out/plots/*.png`（エルボー法、シルエット、ギャップ統計、GMM BIC、UMAP など）
- `out/report.html`（テンプレートは `src/cluster/templates/report.html`）

### 5. 階層クラスタリング & デンドログラム

K-Means による平坦クラスタリングに加えて、PCA 後の特徴量を用いた Ward 法の階層クラスタリングとデンドログラム可視化を行えます。凝集過程に Optimal Leaf Ordering (OLO) を適用することで、似た試料が連続して並ぶ読みやすい樹形を得られます。さらに、K-Means のラベルと比較した Adjusted Rand Index (ARI) と Variation of Information (VI) を算出し、クラスタ構造の一致度を定量的に確認します。

```bash
python -m src.cluster.plot_dendrogram \
  --emb embeddings/pca_embeddings.npy \
  --kmeans out/cluster/kmeans_assignments.csv \
  --meta embeddings/meta.csv \
  --out-plot out/plots/dendrogram.png
```

主な出力物:

- `out/plots/dendrogram.png`: Ward×Euclid + OLO によるデンドログラム。葉ラベルは `meta.csv` の `sample_id` 列を使用（未指定時はインデックス）。
- `out/cluster/hier_labels_k*.csv`: 指定クラスタ数（既定は K-Means のクラスタ数）で平坦化した階層クラスタリングのラベル。
- `out/cluster/hier_vs_kmeans_metrics.json`: K-Means と階層クラスタリングの一致度メトリクス（`k_star`, `ARI_hier_vs_kmeans`, `VI_hier_vs_kmeans`）。

距離閾値でカットしたい場合は `--distance-threshold`、既存の K* とは異なるクラスタ数で比較したい場合は `--max-clusters` を指定してください。高さカットの目安として、デンドログラムの“U字”の高さ（コーフェネティック距離）を参照すると理解しやすくなります。

## トラブルシューティングとヒント

- **CUDA エラー**: `torch.cuda.is_available()` が False の場合は CPU 実行に切り替わりますが、学習時間が長くなります。CUDA ドライバと PyTorch のバージョン互換を確認してください。
- **メッシュの非多様体問題**: `make_manifold_and_maps.py` のログで警告が出た場合は、入力メッシュの自動修復が難しい可能性があります。メッシュ修復ツール（MeshLab など）で事前に問題箇所を修正してください。
- **クラスタ数のばらつき**: 自動推定指標が一致しない場合があります。`configs/cluster.yaml` の重み付けや `min_cluster_size` を調整し、専門家の知見で結果をレビューすることを推奨します。

## 用語解説

- **多様体 (Manifold)**: 各頂点近傍が 2D 平面と同相なメッシュ。非多様体要素（穴、自己交差）があると MeshMAE の畳み込みが破綻します。
- **MAPS (Multiresolution Adaptive Parameterization of Surfaces)**: メッシュ表面を多段階にパラメトライズする手法。粗〜細の階層表現が得られ、メッシュの逐次処理やマルチスケール学習に有効です。
- **自己教師あり学習 (Self-Supervised Learning; SSL)**: 正解ラベルを用いずにデータ自身から擬似ラベルを生成して学習する枠組み。MeshMAE ではメッシュパッチのマスク再構成が学習課題です。
- **ゼロショット (Zero-Shot)**: 目的ドメインのラベル付きデータを使わずに別ドメインで学習したモデルを適用する設定。本プロジェクトでは ShapeNet などで学んだ表現を化石データに転用します。
- **クラスタ数自動決定 (Auto-k)**: クラスタリングの際に適切なクラスタ数を指標に基づいて推定する手法。エルボー法、シルエット係数、ギャップ統計、ガウス混合モデルの BIC などの複数指標を組み合わせることで安定化しています。
- **HDBSCAN**: Density-Based Spatial Clustering of Applications with Noise の階層版。密度に基づくクラスタリング手法で、クラスタ境界が複雑な場合やノイズ点を自然に扱いたい場合に有効です。

## 参考情報

- 公式 MeshMAE: <https://github.com/Maple728/MeshMAE>
- MAPS 生成ツール（SubdivNet）: <https://github.com/zyq524/SubdivNet>
- HDBSCAN ドキュメント: <https://hdbscan.readthedocs.io>

この README が化石メッシュ解析ワークフローの全体像把握と運用の一助となれば幸いです。
