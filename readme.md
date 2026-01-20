# MmEmojiManager
## 概要
Mattermostのカスタム絵文字マネージャです。

できること：
* フォルダ内の画像ファイルの読み込み、絵文字としての登録
* 自分が登録したカスタム絵文字の閲覧、削除

## 使い方

### 起動
* まず、MattermostのAPIトークンを取得してください
* MmEmojiManagerを実行し、URLとAPIトークンを設定してください
* 「接続・サーバキャッシュ更新」を一度押してください（後からでもOKです）

### 絵文字の追加
* 「フォルダ読み込み…」で登録したい画像が入ったフォルダを指定して、その中のファイルをリスト表示
* チェックを入れて、登録します
* 同名絵文字がある場合「BOTH_SAME_NAME」と表示され、登録はできません。
  * 登録名を変更してください

### 絵文字の削除
* 「サーバ絵文字読込」をして、自分の登録済み絵文字を表示
  * 自分以外のカスタム絵文字は表示されません。
* チェックを入れて、削除します


## ビルド手順

### 下記コマンドを実行してください
```sh
python -m venv buildenv
buildenv\Scripts\activate
python -m pip install -upgarde pip
python -m pip install pyinstaller pyside6 requests
pyinstaller --noconfirm --clean MmEmojiManager.spec
```

### ビルド成果物
#### 実行ファイル
dist/MmEmojiManager/MmEmojiManager.exe
#### ライブラリ
dist/MmEmojiManager/_internal

## 配布について
ビルド後、 deploy_zip.ps1 を実行すると MmEmojiManager.zip が作られます。
こちらのzipを配布してください
