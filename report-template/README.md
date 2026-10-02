# 情報工学実験1A レポートテンプレート

LaTeXで実験レポートを書くためのテンプレートです。本文・数式・図表・参考文献の例を、自分の実験内容に置き換えて使ってください。

テンプレートはこの`report-template/`フォルダに展開済みです。教材リポジトリをcloneして、このフォルダのファイルを使ってください。

[コンパイル済みPDFを確認する](main.pdf)

PDFには、AI使用の有無・内容と班員3名の感想・コメントを記載する独立した補足1ページを設けています。

## 作成環境を選ぶ

次の3通りから、自分に合った方法を選んでください。

| 方法 | 特徴 | 必要な準備 |
|---|---|---|
| Overleaf | ブラウザで編集・PDF生成ができる。環境構築を省きたい場合に便利 | Overleafアカウント |
| GitHub＋ローカルTeX環境 | VS CodeやCursorで編集し、GitHubで変更履歴を管理できる | GitHub、Git、TeX環境 |
| Overleaf＋GitHub | ブラウザでの編集とGitHubでの履歴管理を併用できる | 両方のアカウント、Overleafのプレミアム機能 |

## 1. Overleaf

1. [Overleaf](https://www.overleaf.com/)にログインします。
2. 「New project → Blank project」でプロジェクトを作成します。このフォルダの`.tex`・`.bib`ファイルをアップロードし、`figures/`フォルダを作って中の図もアップロードします。空のプロジェクトにある`main.tex`はテンプレートのものに置き換えます。
3. プロジェクトの設定で、Main documentを`main.tex`、Compilerを**pdfLaTeX**にします。
4. 「Recompile」でPDFを生成できることを確認します。
5. `main.tex`のタイトル・氏名・メールアドレス・学生証番号を自分のものに変更し、本文を執筆します。
6. PDFをダウンロードします。編集用のソースも、ときどきZIPでダウンロードして保存してください。

## 2. GitHub＋ローカルTeX環境

1. Windows・Linuxでは[TeX Live](https://tug.org/texlive/quickinstall.html)、macOSでは[MacTeX](https://tug.org/mactex/)をインストールします。日本語・数式関連のパッケージを使うため、フル構成が簡単です。
2. GitHubで、自分のレポート専用の**Privateリポジトリ**を作成し、ローカルにcloneします。
3. この`report-template/`フォルダの中身を、そのリポジトリの直下にコピーします。授業のリポジトリ全体をコピーする必要はありません。
4. VS CodeまたはCursorでフォルダを開き、`main.tex`を編集します。
5. `main.tex`があるフォルダで以下を実行します。`main.pdf`が生成されます。

```sh
latexmk -pdf main.tex
```

6. 本文・参考文献・図などのソースファイルをcommitし、GitHubへpushします。編集を始める前にはpullし、他の端末での変更を取り込みます。

`.aux`、`.log`、`.bbl`などの生成ファイルは履歴管理に不要です。このフォルダの[.gitignore](.gitignore)をレポート用リポジトリへコピーすると除外できます。図として使うPDFは必要なので削除しないでください。

## 3. Overleaf＋GitHub（プレミアム機能）

Overleafで執筆しながら、GitHubにも変更履歴を残したい場合に向いています。**GitHub直接同期はOverleafのプレミアム機能**です。所属機関の契約などで利用できる場合もあります。利用できない場合は、上の1または2を選んでください。

1. 方法1の手順で、テンプレートをアップロードしたOverleafプロジェクトを作成します。
2. Overleafのアカウント設定で、GitHubアカウントを連携します。
3. プロジェクトの「Integrations → GitHub」から、レポート用の新しいGitHubリポジトリを作成します。公開範囲は**Private**にします。
4. Overleafで編集したら、同じ画面の「Push Overleaf changes to GitHub」で変更を送ります。
5. GitHub側やローカルで編集した変更は、Overleaf側でpullして取り込みます。

同期は自動ではありません。編集場所を切り替えるときにpush・pullしてください。既存のOverleafプロジェクトと既存のGitHubリポジトリを後から直接結び付けることはできないため、上記ではOverleafから新しいリポジトリを作成します。

詳しくは[Overleaf公式：GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization)を参照してください。

## ファイル構成

- `main.tex`：本文。最初に編集するファイルです。
- `bibliography.bib`：参考文献。
- `figures/`：図。
- `prologue.tex`：パッケージと共通設定。
- 標準の`article`クラス（A4・11pt）を使用します。著者は氏名と学籍番号を組にして表示します。

このテンプレートは`bxcjkjatype`で日本語を扱います。コンパイラはpdfLaTeXを使ってください。

## レポートの提出要件

- 実験リポジトリで配布されている、このLaTeXテンプレートを使用してください。
- PDFは、**本文8ページ＋補足1ページ＋参考文献・補遺（ページ数無制限）**で構成してください。評価は本文8ページを主とします。
- 本文1ページ目に、班員全員の氏名・学籍番号を記載してください。追加の表紙は不要です。
- 各著者名に`\thanks{...}`で脚注を付け、その著者がどの実験・執筆・分析・議論などに貢献したかを記載してください。著者全員の貢献は、最後の著者にもう1つ`\thanks{...}`を付け、個人の貢献とは別の脚注に記載してください。
- 補足1ページには、**AI使用の有無・内容**と、**班員それぞれの感想・コメント**を記載してください。感想・コメントの内容は成績に影響しません。

テンプレートの著者3名と学籍番号は架空の例です。貢献の脚注も含め、実際の班員・分担に置き換えてください。本文の分量は執筆して調整し、補足は新しいページから始めて1ページにまとめてください。雛形自体を本文8ページに水増しするための空白ページは入れていません。
