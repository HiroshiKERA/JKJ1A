# 環境構築：Driveで編集し、必要に応じてColabで実行する

## Googleアカウントについて

DriveとColabで使うGoogleアカウントは、**個人のGoogleアカウント（個人Gmailなど）を推奨**します。
大学のGoogle Workspaceアカウントでは、管理者の設定により、大学外へのDrive共有、公開リンクの作成、Colabの利用や有料機能の購入が制限される場合があります。授業で使うNotebookは各自のDriveへコピーして使うため、教員のDriveを共有する必要はありません。

大学アカウントを使う場合は、Colabにログインできること、Notebookを自分のDriveへ保存できること、必要な共有操作ができることを事前に確認してください。GPUやColabの有料プランを授業の必須条件にはしません。

**初回のGoogle Drive for desktopの認証画面では、表示された権限を「すべて選択」して許可してください。** 権限を選択せずに続行すると、Driveの同期やColabからのファイルアクセスが失敗することがあります。確認画面のアカウントが授業用に用意した個人Googleアカウントであることと、アプリ名がGoogle Drive for desktopであることを確認してから進めます。

## 1. 編集環境を選ぶ

次のどちらかをインストールします。

- [Visual Studio Code](https://code.visualstudio.com/) ＋ [GitHub Copilot](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-extension)
- [Cursor](https://cursor.com/downloads)

どちらもPythonとJupyterの拡張機能を導入し、`.ipynb` を開ける状態にします。
VS Codeではリポジトリを開いたときに表示される推奨拡張機能から導入できます。
AI機能は各サービスへのログイン・利用条件の確認が必要です。
AIが提案した変更は実行結果と照らし合わせ、何を変えたか説明できるようにしましょう。

## 2. GitとGoogle Driveを準備する

1. [Git](https://git-scm.com/downloads) を導入します。WindowsではGit for Windows、macOSでは公式ページの案内に従ってください。
2. [Google Drive for desktop](https://support.google.com/drive/answer/10838124) を導入し、Colabでも使う**個人Googleアカウント**でログインします。
3. **マイドライブの中**に `Colab Notebooks` フォルダを作成します。「パソコン」のバックアップ領域とは異なります。
4. ストリーミング方式なら授業フォルダを「オフラインで使用可能」にします。ミラーリング方式でも構いません。[同期方式の説明](https://support.google.com/drive/answer/13401938)
5. ターミナルでそのフォルダへ移動し、以下を実行します。同期フォルダの実際のパスはOS・アカウントで異なります。

```sh
git --version
git clone https://github.com/HiroshiKERA/JKJ1A.git
cd JKJ1A
```

これでマイドライブに次の構成ができます。

```text
Colab Notebooks/JKJ1A/
  environment.yml
  notebooks/day1/
  jkj1a/
  results/             # 実験後に作成。重み・図など
```

エディタの「フォルダーを開く」で `JKJ1A` を開きます。

### 自分のパスを設定する

各Notebookの先頭セルにある`DRIVE_PROJECT`と`LOCAL_PROJECT`を、自分の環境の絶対パスに変更します。標準のDrive側は `/content/drive/MyDrive/Colab Notebooks/JKJ1A` です。


## 3. ローカルのconda環境 `jkj1a` を作る

condaが未導入なら [Miniforge](https://github.com/conda-forge/miniforge) をOS・CPUに合った方法で導入してください。
すでにcondaがあれば再導入は不要です。WindowsではMiniforge Promptなど、`conda` が使えるターミナルを開きます。

**`JKJ1A` フォルダで**以下を実行します。

```sh
conda env create -f environment.yml
conda activate jkj1a
python -m ipykernel install --user --name jkj1a --display-name "Python (jkj1a)"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

すでに `jkj1a` を作成済みなら、作成コマンドの代わりに `conda env update -n jkj1a -f environment.yml` を使います。
conda環境はcondaの通常の保存先に置き、Drive同期フォルダ内には置きません。
Notebook右上のカーネル選択で **Python (jkj1a)** を選び、`00-environment.ipynb` を実行します。

環境はPython 3.11、PyTorch、torchvision、NumPy、Matplotlib、scikit-learn、ipywidgets、ipykernelと、
Hugging Face用のtransformers・huggingface-hub・accelerate、Day 2の画像分類用のtimmを含みます。
これらは初回の環境構築でまとめて導入されます。既存環境では上記の更新コマンドを実行し、Notebookのカーネルを再起動してください。
大きなモデルの重みは環境作成時にはダウンロードしません。datasets・diffusersなどは後日の題材に応じて追加します。
ローカルはCPU実行を基本とします。GPUが必要な課題はColabで行います。
インストール先のOSによってPyTorchパッケージの大きさは異なります。
Intel Macなど配布wheelが対応しない端末では、ローカルは編集に使い、実行はColabで行ってください。

## 4. ブラウザ版Colabで同じNotebookを実行する

1. ローカルでNotebookを保存し、**エディタのNotebookを閉じます**。
2. Drive for desktopが同期完了したことを確認します。
3. ブラウザのGoogle Driveで対象の `.ipynb` を右クリックし、「アプリで開く → Google Colaboratory」を選びます。表示されなければ「アプリを追加」でColaboratoryを追加します。
4. GPUが必要な回はColabの「ランタイム → ランタイムのタイプを変更」でGPUを選択します。Day 1はCPUで十分です。
5. 各Notebook冒頭のセットアップセルを実行します。Driveへのアクセスを許可してください。
6. **Notebookを実行する前に、セル冒頭の `DRIVE_PROJECT` を自分のGoogle Drive上の絶対パスに合わせて変更します。** 標準値は `/content/drive/MyDrive/Colab Notebooks/JKJ1A` です。`LOCAL_PROJECT` は通常変更不要です。
7. 依存関係セルを実行します。Colabがランタイム再起動を求めた場合は再起動し、先頭から実行し直してください。

Colabではローカルのconda環境は使われません。配布の `requirements-colab.txt` で補助パッケージを導入し、
Colabに入っているPyTorchとCUDAを利用します。GPU割り当てはプラン・空き状況・利用制限によります。

**NotebookがDriveに保存されていても、実行中の作業ディレクトリは自動ではその場所になりません。**
セットアップセルが `JKJ1A` へ移動します。教材の保存先 `results/` はDrive上になり、ローカルでも同期されます。
`/content` 直下に保存しただけのファイルは一時ファイルです。

## 5. ローカル編集へ戻る

1. Colabで保存完了を確認し、Notebookのタブを閉じます。不要になったランタイムも終了します。
2. ローカルでDriveの同期完了を確認します。
3. VS Code／CursorでNotebookを開き直します。

**同じNotebookを両方で同時編集・保存しないでください。** また、別々の端末から同じ `.git` を同時に操作しません。
Notebookに保存した出力と、Pythonの変数・学習済みモデルは別物です。セッションを閉じる前に必要な重みを保存しましょう。

## うまく動かないとき

| 症状 | 確認すること |
|---|---|
| `conda` が見つからない | conda用のターミナルを使い、導入後はターミナルを開き直す |
| `ModuleNotFoundError` | ローカルならカーネルが `jkj1a` か、Colabなら依存関係セルを実行したか |
| Drive上のフォルダが見つからない | マイドライブか、`DRIVE_PROJECT` とフォルダ名が一致するか、同期済みか |
| スライダーが動かない | カーネルが接続中か、00の簡単なウィジェットが動くか。再接続後はセルを再実行 |
| GPUが表示されない | ColabでGPUを選び、00のCUDA確認を実行。Day 1はCPUのままでよい |
| 編集前の内容が見える | 両方の編集画面を閉じ、Driveの同期状態を確認する |

## 参考（2026-09-29確認）

- [PyTorchのインストール](https://pytorch.org/get-started/locally/)
- [Hugging Face Transformersのインストール](https://huggingface.co/docs/transformers/installation)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Google公式Colab拡張機能](https://marketplace.visualstudio.com/items?itemName=Google.colab)もありますが、本授業の標準はブラウザ版Colabです。
