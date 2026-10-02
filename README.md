# 情報工学実験1A (JKJ1A)

機械学習を「自分で動かし、条件を変え、結果を説明する」ための実験教材です。
全4日構成の実験教材です。Day 1・Day 2のNotebookと、実験レポートのテンプレートを配布しています。

> DriveとColabは、大学アカウントの共有・管理制限を避けるため、個人のGoogleアカウントでの利用を推奨します。詳細は[環境構築ガイド](notebooks/day1/setup.md)を確認してください。

## 最初に読む

[環境構築とDrive同期の手順](notebooks/day1/setup.md)に従って準備してください。

- **編集・軽い実行**：Google Driveの同期フォルダをVS Code＋GitHub Copilot、またはCursorで開く。
- **ローカル環境**：配布の `environment.yml` でconda環境 `jkj1a` を作る。
- **GPUが必要な実行**：保存と同期を確認し、同じNotebookをブラウザ版Google Colabで開く。
- **Day 1はCPUで実行可能**。同じNotebookをローカルでもColabでも使えます。

## Day 1の順番

| Notebook | 内容 | 進め方 |
|---|---|---|
| [00 環境確認](notebooks/day1/00-environment.ipynb) | 実行環境、保存場所、GPU、ウィジェット、Hugging Faceの確認 | 各自実行 |
| [01 Python入門](notebooks/day1/01-python-demo.ipynb) | 変数、リスト、スライス、条件分岐、ループ、関数、クラス | 教員の投影デモ |
| [02 PyTorch入門](notebooks/day1/02-pytorch-demo.ipynb) | shape、抽出・代入、関数、argmax、バッチ、勾配 | 教員の投影デモ |
| [03 テンソル練習](notebooks/day1/03-tensor-exercises.ipynb) | 小問8問、ヒントと自己確認 | 各自演習 |
| [04 ReLUと表現力](notebooks/day1/04-relu-functions.ipynb) | 折れ曲がり、三角形の山、折れ線近似 | 実演と操作 |
| [05 ReLU関数の学習](notebooks/day1/05-relu-learning.ipynb) | ReLUネットワーク、損失・勾配・学習過程の観察 | 実演と実験 |
| [06 最終課題](notebooks/day1/06-assignment.ipynb) | 多層モデルの設計・学習・複雑な関数の実験 | 各自実験 |

Notebook内の案内に従い、**カーネルを再起動したら先頭から実行**します。
Notebook同士は別のカーネルでも動きます。


## Day 2の順番

Day 2は共通実習とチーム課題の着手まで進めます。発表・レポートはDay 3・4で行います。
CIFAR-10の学習とVLMの実行にはColabのGPUを推奨します。

| Notebook | 内容 |
|---|---|
| [01 CIFAR-10分類](notebooks/day2/01-cifar10-classification.ipynb) | 分類スコア、交差エントロピー、学習、評価、重みの保存 |
| [02 敵対的入力](notebooks/day2/02-adversarial-attack.ipynb) | FGSM、I-FGSM、攻撃強度・反復回数、ランダムノイズとの比較 |
| [03 学習済み分類器](notebooks/day2/03-pretrained-attack.ipynb) | ImageNet学習済みモデルへの攻撃 |
| [04 VLMの入力](notebooks/day2/04-vlm-inputs.ipynb) | 数え上げ、文字の追加、編集画像、実験記録 |
| [05 チーム課題](notebooks/day2/05-team-project.ipynb) | モデル選択、仮説、最初の比較実験、発表・レポートの準備 |

01で保存する`models/day2-cifar10.pt`を02で使用します。03・04はそれぞれ学習済みモデルを取得します。
既存のローカル環境では`python -m pip install -r requirements-local.txt`で依存関係を更新してください。
Colabですでにセットアップ済みのセッションは、ランタイムを再起動して先頭から実行してください。

## 旧教材

2021〜2025年度の教材は[2021-2025ブランチ](https://github.com/HiroshiKERA/JKJ1A/tree/2021-2025)に保存しています。

## 実験レポート

[レポートテンプレートと作成環境の設定手順](report-template/README.md)を参照してください。Overleaf、GitHub＋ローカルTeX環境、Overleaf＋GitHub（プレミアム機能）の3通りを紹介しています。
