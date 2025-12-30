# MeshMAE Zero-Shot – 化石メッシュ自動クラスタリングガイド

本リポジトリは、化石や骨格の3Dメッシュを対象としたゼロショット（学習済みモデルの再学習なし）クラスタリングパイプラインを提供します。公式 [MeshMAE](https://github.com/Maple728/MeshMAE) の学習ユーティリティをラップし、化石ドメイン向けの事前学習継続、特徴量抽出、クラスタ数の自動決定、レポート生成までを一貫して実行できるように構成されています。ここでは環境構築から各ステップの意味、関係する用語の詳細解説までを網羅的にまとめています。

## まず最初に（クイックスタート）

最低限の流れだけ先に知りたい方向けに、典型的なコマンドの最短経路を示します。

1. 依存パッケージをインストール
   ```bash
   pip install -r env/requirements.txt
   ```
2. 生メッシュを配置
   ```
   datasets/fossils_raw/
   ├── sample_01.obj
   ├── sample_02.stl
   └── ...
   ```
3. メッシュ前処理（多様体化 + 500面化 + MAPS）
   ```bash
   python -m src.preprocess.make_manifold_and_maps \
     --in datasets/fossils_raw \
     --out datasets/fossils_maps \
     --target_faces 500 \
     --num_workers 0 \
     --make_maps \
     --subdivnet_root ../SubdivNet \
     --maps_extra_args -- --base_size 96 --depth 3 --max_base_size 192
   ```
4. 特徴量抽出（MeshMAE ベースモデル）
   ```bash
   python -m src.embed.extract_embeddings \
     --config configs/extract.yaml \
     --model-factory model.meshmae.Mesh_mae \
     --normalize
   ```
   > `liang3588/MeshMAE` を使う場合は `model.meshmae.Mesh_mae` のようにクラス指定します。\
   > 旧式の `meshmae.models_mae.mae_vit_base_patch16` 形式は、該当モジュールを提供する別実装向けです。
   >
   > **注意:** 特徴量抽出では `forward_encoder/encode/forward_features` などのエンコーダ系メソッドのみを利用します。\
   > `forward` は loss を返す実装が多く、埋め込みとして扱うと次元が潰れてクラスタリングが退化するため対象外です。\
   > もし抽出ベクトルが極端に小さい（例: 8 次元未満）場合は、モデルの組み合わせや入力形式を確認してください。
   >
   > **MeshMAE 公式 ckpt の入力について:** 公式 `shapenet_pretrain.pkl` は論文準拠の 10ch 特徴量
   > （area 1 + interior angles 3 + face normal 3 + face normal⋅vertex normals 3）を想定し、
   > 1 patch = 64 faces（3 回 subdivide）で学習されています。\
   > 本リポジトリのデフォルト `feature_mode=paper10` は公式 ckpt に合わせて 10ch を生成します。
   > 以前の 13ch（center を含む独自特徴）は `feature_mode=legacy13` に切り替え可能ですが、
   > 公式 ckpt とは互換性がありません。
5. クラスタリング + レポート生成
   ```bash
   python -m src.cluster.run_clustering \
     --emb embeddings/raw_embeddings.npy \
     --meta embeddings/meta.csv \
     --config configs/cluster.yaml \
     --out-dir out
   ```

> 継続SSL（ドメイン適応）を行う場合は、後述の「2. 自己教師あり学習の継続」を参照してください。

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

## どのファイルを編集すべきか（設定ファイルの役割）

| ファイル | 役割 | 代表的に変更する項目 |
| --- | --- | --- |
| `configs/pretrain_target.yaml` | 継続SSL（ドメイン適応）の設定 | `dataroot`, `init_checkpoint`, `save_checkpoint`, `epochs`, `batch_size` |
| `configs/extract.yaml` | 特徴量抽出の設定 | `input.dataroot`, `input.checkpoint`, `encoder.pool_strategy`, `output.*` |
| `configs/cluster.yaml` | クラスタリングと自動K推定 | `kmeans.max_k`, `hdbscan.min_cluster_size`, `auto_k.weights`, `pca.n_components` |

> 設定ファイルは YAML なので、インデントやスペースが崩れると読み込めません。編集後は `python -m src.embed.extract_embeddings --config configs/extract.yaml --dry-run` のような簡易実行で検証することを推奨します。

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

### 推奨ディレクトリ構造（実データと成果物）

```
datasets/
├── fossils_raw/        # 生メッシュ（入力）
└── fossils_maps/       # 前処理済みメッシュ（MAPS 付）

embeddings/
├── raw_embeddings.npy
└── meta.csv

out/
├── cluster/
├── plots/
└── report.html
```

## ステップ別手順

### 1. メッシュ前処理
MeshMAE の実験は多様体メッシュで約500面、MAPS 階層が整備されていることを前提としています。本リポジトリのラッパースクリプトから公式ツールチェーンを呼び出し、必要な加工を自動化します。`--make_maps` を指定した場合でも SubdivNet の `datagen_maps.py` を直接叩かず、必ず `python -m src.preprocess.run_subdivnet_maps` をサブプロセスで起動します。`--subdivnet_root` で SubdivNet リポジトリのルート（例: `../SubdivNet`）を渡してください。

#### MAPS 生成の呼び出し方法（デモ実行を避ける）
SubdivNet の `datagen_maps.py` には `if __name__ == "__main__": MAPS_demo1()` が含まれ、直接実行すると `airplane.obj` を探すデモが走って失敗します。本リポジトリでは `src.preprocess.run_subdivnet_maps` という薄い CLI ラッパーを別プロセスで起動し、demo を踏まずに MAPS を生成します。ラッパーは次の動作を行います。

- `--subdivnet_root` で渡されたパスを `PYTHONPATH` に追加し、`datagen_maps` と `maps.MAPS` を import する（親プロセスに SubdivNet の依存を混ぜない）。
- 入力メッシュは **修復後・簡略化前** の形状を一時ファイルとして保存して渡す。MAPS の位相前提を守るため、簡略化で manifold を壊さないようにする。
- `base_size` / `depth` / `max_base_size` は `--maps_extra_args` 経由で受け取り、拡張子付きの出力パス（例: `<outdir>/<stem>_MAPS.obj`）を必ず渡す。

`make_manifold_and_maps.py` 側は常にこのラッパーを subprocess で呼び出します。前処理中に SubdivNet 側の `__main__` ブロックが実行されることはなくなり、入力/出力を絶対パスで渡すことでパス解決事故を防ぎます。

- 簡略化後のメッシュは `<out>/<stem>.<ext>` に保存されます。500 面以下のメッシュは簡約処理をスキップします。
- MAPS 出力は `<out>/success/<relative>/<stem>_maps/` に `<stem>_MAPS.<ext>` が生成されます。`<relative>` は入力ルートからの相対パスで、元のフォルダ構造を保ったまま保存されます。失敗した場合はログが `<out>/failed/<relative>/<stem>_maps/error.log` に移動されるため、成功・失敗をディレクトリで分離したうえでトレースを確認できます。

> **修復の強度を切り替える `--aggressive-repair` オプション**<br>
> 自己交差や非多様体エッジが多いデータでは、MAPS の winding/edge チェックで弾かれることがあります。`--aggressive-repair` を付けると複数回の穴埋め、winding/inversion 修正、コンポーネント分割と watertight かつ winding-consistent な最大コンポーネントの選択を試行し、`failed/.../error.log` には修復後の面数・頂点数・watertight 状態が追記されます。軽量処理を優先したい場合はデフォルトのまま、MAPS 生成の失敗が頻発する場合は `--aggressive-repair` を有効化してください。

```bash
python -m src.preprocess.make_manifold_and_maps \
  --in datasets/fossils_raw \
  --out datasets/fossils_maps \
  --target_faces 500 \
  --num_workers 0 \
  --make_maps \
  --subdivnet_root ../SubdivNet \
  --metadata datasets/fossils_maps/processing_metadata.json \
  --maps_extra_args -- --base_size 96 --depth 3 --max_base_size 192
```

自己交差が多いデータセットで MAPS の winding / 非多様体チェックが頻繁に失敗する場合は、強化修復を有効化します（より多くの穴埋め・ winding 修正・コンポーネント分割を試みます）。

```bash
python -m src.preprocess.make_manifold_and_maps \
  --in datasets/fossils_raw \
  --out datasets/fossils_maps \
  --target_faces 500 \
  --num_workers 0 \
  --make_maps \
  --aggressive-repair \
  --subdivnet_root ../SubdivNet \
  --metadata datasets/fossils_maps/processing_metadata.json \
  --maps_extra_args -- --base_size 96 --depth 3 --max_base_size 192
```

`--subdivnet_root` を省略すると `../SubdivNet` を自動的に探します（`datagen_maps.py` が無い場合はエラーになります）。`--maps_extra_args` で渡したいオプションは、すべて末尾に置くか、`--maps_extra_args -- --foo bar` のように区切ってください（`argparse.REMAINDER` で受け取るため）。SubdivNet の依存（`maps` パッケージが要求する `triangle`, `sortedcollections`, `networkx`, `rtree` など）は本リポジトリの `env/requirements.txt` には含まれていないため、必要に応じて SubdivNet 側の仮想環境にインストールしてください。

出力物の意味:

- `<mesh>.<ext>`: 多様体化 + 簡略化後（target_faces=500）のメッシュ。
- `success/<relative>/<mesh>_maps/`: MAPS 生成結果。成功時は `<mesh>_MAPS.<ext>` を必ず含みます。
- `failed/<relative>/<mesh>_maps/error.log`: MAPS 生成失敗時のログ。`<relative>` は入力ルートからの相対パスで、成功時と同じ階層構造を維持します。

MAPS 生成で遭遇しやすいトラブルと回避策:

- `airplane.obj not found`: `datagen_maps.py` のデモが起動しています。`--subdivnet_root` を指定し、`datagen_maps.py` を直接叩かないようにします。
- `ValueError exporter not available`: 出力パスに拡張子が付いていません。`*_MAPS.<ext>` のようにファイル名を指定してください。
- `KeyError` や `cycle_basis` 周りの `IndexError`: 入力メッシュの位相が MAPS の前提を満たしていません。修復後のメッシュを簡略化せずに MAPS へ渡す（本スクリプトの既定動作）か、watertight/winding を満たすかを確認してください。

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

処理後は `datasets/fossils_maps/` 以下に元ディレクトリ構造を保ったまま保存され、面数やスケール、MAPS 有無、MAPS の格納先（成功/失敗のディレクトリを含む）を記録した JSON マニフェストが出力されます。

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

`maps` モジュールとその依存パッケージを用意した上で、`datagen_maps.py` を直接叩くのではなく、本リポジトリのラッパー経由で実行してください（`__main__` に仕込まれたデモを踏むのを防ぐため）。`--maps_extra_args` は `argparse.REMAINDER` で受け取るため、前処理スクリプト側のオプション（`--metadata` など）は必ず `--maps_extra_args` より前に置き、その後に `--` を挟んで MAPS スクリプトへ渡したい引数を列挙します。

1. **単一メッシュを手動で MAPS 生成する場合**

  ```bash
  python -m src.preprocess.run_subdivnet_maps \
    --subdivnet_root ../SubdivNet \
    --input input_repaired.obj \
    --out-dir output_maps_dir \
    --output-path output_maps_dir/input_repaired_MAPS.obj \
    --metadata output_maps_dir/run_metadata.json \ # オプション（推奨）
    --base_size 96 --depth 3 --max_base_size 192
  ```

   `base_size=96`, `depth=3`, `max_base_size=192` は化石データ向けの保守的な推奨値です。`make_manifold_and_maps.py` から呼ぶ場合は前述の例のように `--maps_extra_args` へ同じオプションを渡します。

   MAPS に渡す前にメッシュをクレンジングしたい場合は、`--clean-input` と `--clean-min-face-area 1e-9` のようなしきい値を追加すると、重複頂点/面の除去・ゼロ/極小面削除・法線再計算を行った上で、一時的にクリーンなメッシュを `out-dir` 配下へ書き出します（失敗時はそのクリーンメッシュのパスがメタデータに残ります）。

   `--metadata` を指定すると、以下の情報を JSON で保存します（親ディレクトリが無ければ自動作成されます）。

   - `input_faces` / `input_vertices`: 受け取った入力メッシュの面数・頂点数
   - `attempted_base_sizes`: 試行した base_size のリスト（降順）
   - `chosen_base_size`: 実際に採用した base_size（失敗時は `null`）
   - `actual_base_size`: MAPS 実装が報告した base_size（`maps.MAPS.base_size` を優先）
   - `success`: 成否フラグ
   - `output_path`: 出力メッシュの絶対パス（`output_path_relative` はフォルダ内の相対パス）
   - `cleaning`: `--clean-input` 指定時のクレンジング結果（削除した重複頂点/面数、ゼロ/極小面数、面・頂点数の推移、min_face_area）
   - `cleaned_input_path`: クリーン済みメッシュの絶対パス（`cleaned_input_relative` はフォルダ内の相対パス）
   - `failed_mesh_path`: MAPS 失敗時に保存した入力メッシュのパス（クリーン済みがあればそのコピーを指します）
   - `error`: 失敗時のエラーメッセージ

2. **データセット全体をまとめて処理する場合**

   `make_manifold_and_maps.py` に `--make_maps` を付けて実行してください（上記の例）。SubdivNet オリジナルの `--config FOSSILS` などを使う場合も、demo が走らないように `run_subdivnet_maps` へパラメータを渡す形を推奨します。MAPS 前にクレンジングしたい場合は `--clean_maps_input --clean_maps_min_face_area 1e-9` を併用すると、修復後メッシュから重複頂点/面や極小面を除去したクリーン版を MAPS に渡し、成功・失敗フォルダ内に残します。

配布済み MAPS データをそのまま使うだけなら、生成処理を回す必要はありません。

> **Troubleshooting – MAPS ディレクトリが空に見える場合**
>
> 以前の実装では SubdivNet をカレントディレクトリにしたまま相対パスの入力/出力を渡していたため、MAPS の出力が別場所へ書き出され、`<stem>_maps/` が空のまま残る不具合がありました。現在は入力メッシュと出力先を絶対パスで渡し、`PYTHONPATH` に SubdivNet を追加することで、期待したフォルダに MAPS ファイルが生成されます。失敗時は `failed/<relative>/<stem>_maps/error.log` を確認してください。

> **テストについて**  
> MAPS ラッパー CLI の `--metadata` 受け入れと JSON 出力を `src/preprocess/test_run_subdivnet_maps.py` でモック化して検証しています。`pytest src/preprocess/test_run_subdivnet_maps.py -q` で単体実行できます。

生成された MAPS 出力フォルダを本リポジトリ直下に作成した `datasets/` 配下へ配置し、MeshMAE 実行時には `--dataroot` でそのフォルダを指します（例: `--dataroot ./datasets/Manifold40-MAPS-96-3/`）。

#### 前処理で生成されるマニフェストの読み方（例）

`--metadata` を指定すると、処理済みメッシュごとのメタデータが JSON で保存されます。代表的な項目は以下です。

- `input_path` / `output_path`: 入力・出力メッシュのパス
- `faces_before` / `faces_after`: デシメーション前後の面数
- `manifold_repair`: 修復の有無や実行ログ
- `maps.success`: MAPS 成功 여부
- `maps.output_path`: MAPS 出力パス

失敗時は `failed/.../error.log` と合わせてこの JSON を確認すると、失敗原因の切り分けが容易になります。

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

#### 継続SSLなしで公式ベースモデルだけを使って特徴量抽出する手順
化石データでの追加学習を行わず、公式 MeshMAE の事前学習済みベースモデル（例: `mae_vit_base_patch16` の `shapenet_pretrain.pkl`）をそのまま用いて埋め込みを得たい場合は、以下の手順で継続SSLステップをスキップできます。

1. **前処理（必須）**  
   継続SSLをしない場合でも、MeshMAE のエンコーダは **多様体メッシュかつ MAPS 付き** の入力を前提としています。必ず「1. メッシュ前処理」のフローを実施してください。最小限は次のように `make_manifold_and_maps.py` を実行し、`datasets/fossils_maps/`（任意の出力先で可）に MAPS 付きメッシュを用意します。
   ```bash
   python -m src.preprocess.make_manifold_and_maps \
     --in datasets/fossils_raw \
     --out datasets/fossils_maps \
     --target_faces 500 \
     --num_workers 0 \
     --make_maps \
     --subdivnet_root ../SubdivNet \
     --maps_extra_args -- --base_size 96 --depth 3 --max_base_size 192
   ```
   MAPS を再利用できる配布データがある場合はこのステップを省略できますが、MeshMAE 用の MAPS 階層と約500面の多様体メッシュであることを確認してください。
2. **公式実装を用意する**  
   `MeshMAE` リポジトリをクローンして隣接ディレクトリに配置し、開発モードでインストールします。推論のみでも Python 側からモデル定義を参照できるようにする必要があります。
   ```bash
   git clone https://github.com/Maple728/MeshMAE.git ../MeshMAE
   pip install -e ../MeshMAE
   ```
3. **公式チェックポイントを配置する**  
   公式配布の事前学習済み重み（例: ShapeNet 事前学習の `shapenet_pretrain.pkl`）を `checkpoints/` に保存します。ダウンロード元が提供するファイル名と一致していればリネーム不要です。
   ```bash
   # 例: ダウンロード済みファイルを移動
   mv /path/to/shapenet_pretrain.pkl checkpoints/shapenet_pretrain.pkl
   ```
4. **`configs/extract.yaml` をベースモデル用に調整する**  
   - `input.checkpoint` を `./checkpoints/shapenet_pretrain.pkl` に変更する。  
   - `input.dataroot` を前処理済みメッシュのルート（例: `datasets/fossils_maps`）に合わせる。  
   - `encoder.pool_strategy`（`cls` または `mean`）や `encoder.normalize_embeddings` を必要に応じて設定する。
5. **埋め込みを抽出する**  
   継続SSLを行わないため追加の学習ジョブは不要です。以下のようにエンコーダのファクトリを公式実装から指定して抽出を実行します（CUDA が無い場合は `--device cpu` を指定）。
   ```bash
   python -m src.embed.extract_embeddings \
     --config configs/extract.yaml \
     --model-factory model.meshmae.Mesh_mae \
     --normalize
   ```
   実行後、`embeddings/raw_embeddings.npy` と `embeddings/meta.csv`（パスは YAML の `output.*` で変更可）に特徴量とメタデータが保存されます。`run_target_pretrain.sh` などの継続SSLスクリプトは実行不要です。

```bash
python -m src.embed.extract_embeddings \
  --config configs/extract.yaml \
  --model-factory model.meshmae.Mesh_mae \
  --normalize
```

抽出スクリプトには二つのモードがあります。

1. **MeshMAE エンコーダモード**: 公式パッケージを利用し、`--model-factory` でエンコーダ構築関数またはクラスを指定します。`liang3588/MeshMAE` 系の構成では `model.meshmae.Mesh_mae` のようにクラス指定が必要です（MeshMAE ルートを `PYTHONPATH` に追加）。チェックポイント読み込みは `configs/extract.yaml` の `input.checkpoint` で制御されます。`src/embed/meshmae_inputs.py` が MAPS メッシュから MeshMAE の入力テンソル（`faces`, `feats`, `centers`, `Fs`, `cordinates`）を構築し、`Mesh_mae.forward` 互換の形状（`patch_size=64`, `num_patches=256`, `channels=13`）でアダプタ経由で呼び出します。`forward_encoder(...)` や `encode(...)` が存在するモデルでは従来どおり CLS トークンまたはパッチ埋め込み平均 (`encoder.pool_strategy`) をプールします。
2. **幾何特徴量フォールバックモード**: MeshMAE が利用できない場合は `--force-geometry` を指定するか、自動検出で幾何ベース特徴量（バウンディングボックス、体積、表面積、慣性テンソル、平均曲率など）を算出します。煙試験や軽量検証に便利です。

出力は `embeddings/raw_embeddings.npy`（設定で変更可）とメタデータ CSV として保存されます。必要に応じて PCA/UMAP などの次元削減結果も同時に書き出されます。

#### 代表的な出力ファイル

| 出力 | 説明 |
| --- | --- |
| `embeddings/raw_embeddings.npy` | 生の埋め込みベクトル |
| `embeddings/meta.csv` | メッシュごとの ID・パス・ラベル等 |
| `embeddings/pca_embeddings.npy` | PCA 後の特徴量（設定次第） |
| `embeddings/umap_embeddings.npy` | UMAP 後の特徴量（設定次第） |

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

**診断ログ**: K-Means/HDBSCAN に渡す特徴量の平均・分散・ユニーク数をログ出力します。特徴量が全ゼロ・全同一・次元 0 の場合は例外で停止し、埋め込み抽出の設定や入力メッシュを見直してください。HDBSCAN は `min_cluster_size`/`min_samples`/`metric` をログし、スケール不整合が疑われる場合は追加の標準化を行います。ノイズ率が高いときはパラメータを緩和して再実行します。

#### 自動K推定の仕組み（概要）

`src/cluster/auto_k.py` では複数の評価指標（エルボー法、シルエット、ギャップ統計、GMM BIC など）を計算し、それぞれのスコアを重み付け平均して `k*` を推定します。`configs/cluster.yaml` の `auto_k.weights` を調整することで、どの指標を重視するかを制御できます。

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

**注意**: 階層クラスタリングでは特徴量の多様性（平均/分散/ユニーク数）と距離統計（最小・最大・平均）をログ出力します。距離が全て 0、もしくは K-Means が単一クラスタに潰れている場合は、まず埋め込み生成や K-Means の退化を解消してから再評価してください。

**大量の葉ラベル対策**: サンプル数が多いときは DPI を上げてもラベルが読めないため、`--label-mode` で挙動を切り替えてください。既定の `auto` は葉数が閾値を超えると `truncate` または `none` に切り替えます。`--format pdf/svg` のベクタ出力も推奨です。

```bash
python -m src.cluster.plot_dendrogram \
  --emb embeddings/pca_embeddings.npy \
  --kmeans out/cluster/kmeans_assignments.csv \
  --meta embeddings/meta.csv \
  --label-mode truncate \
  --truncate-mode lastp \
  --p 100 \
  --format pdf \
  --label-map-out out/plots/dendrogram_label_map.csv
```

## トラブルシューティングとヒント

- **CUDA エラー**: `torch.cuda.is_available()` が False の場合は CPU 実行に切り替わりますが、学習時間が長くなります。CUDA ドライバと PyTorch のバージョン互換を確認してください。
- **メッシュの非多様体問題**: `make_manifold_and_maps.py` のログで警告が出た場合は、入力メッシュの自動修復が難しい可能性があります。メッシュ修復ツール（MeshLab など）で事前に問題箇所を修正してください。
- **クラスタ数のばらつき**: 自動推定指標が一致しない場合があります。`configs/cluster.yaml` の重み付けや `min_cluster_size` を調整し、専門家の知見で結果をレビューすることを推奨します。
- **MAPS 生成が遅い/失敗する**: SubdivNet の依存が正しく入っているか（`triangle`, `rtree`, `sortedcollections` など）を確認し、`--aggressive-repair` や `--clean_maps_input` を試してください。
- **可視化が真っ白になる**: `pyvista` のバックエンドが headless 環境に非対応な場合があります。`PYVISTA_OFF_SCREEN=true` を設定し、仮想フレームバッファを用意してください。

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
