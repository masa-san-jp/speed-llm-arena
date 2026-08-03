# speed-llm-arena

トランプゲーム「スピード」をLLM同士にリアルタイムでプレイさせ、勝敗からEloランキングを算出するアリーナ。反応速度は**実レイテンシ**(推論にかかった実時間)で競わせる。主対象はOllama上のローカルLLM。

## 特徴

- ルール判定は決定論的なPythonエンジン。LLMには局面をJSONで渡し、着手をJSONで返させる
- プレイヤーごとに独立スレッドで「観測 → 思考 → 着手」をループ。思考中に相手が先に出せば競合負けとして記録される
- モデルごとに平均レイテンシ・有効手・無効手・パース失敗を計測し、「速いが雑」「遅いが正確」を分析できる
- 総当たり戦でElo(K=32)を算出。先手を交互に入れ替えて座席バイアスを除去

## 必要環境

Python 3.10+ と requests のみ。

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# 1. エンジン検証(LLM不要のベースライン同士)
python speed_arena.py --mode selftest

# 2. ローカルLLM同士 (Ollama)
python speed_arena.py --mode ollama --models gpt-oss:20b gemma3:4b --games 10

# 3. モデルごとに別ホストを割り当てて公平にレイテンシ計測
python speed_arena.py --mode ollama --models gpt-oss:20b gemma3:4b \
    --hosts http://192.168.1.10:11434 http://192.168.1.11:11434

# 4. Claude APIでテスト(要 ANTHROPIC_API_KEY)
python speed_arena.py --mode anthropic --models claude-haiku-4-5-20251001 claude-sonnet-4-6
```

結果は `results.json` に出力される(schema_version付きJSON。詳細な契約は `docs/arena-spec.md` を参照)。自己対戦・本番対戦で生成した `results.json` はリポジトリにコミットしない(`.gitignore` 対象)。再現用fixtureは `tests/fixtures/` に別名で管理する。

詳しい仕様(ルール・計測定義・公平性・JSON契約)の正本は `docs/arena-spec.md` と対応するGitHub Issue。以下は入口としての概要のみ。

## ルール実装

各プレイヤー26枚(ランク1〜13を2枚ずつ)。手札4枚+山札。中央の台札2山に対し、ランク差1(KとAはループ)のカードを出せる。出したら山札から即補充。両者が出せない状態が続くと審判が「せーの」で台札を更新。全カードを出し切ったら勝ち。手詰まりとタイムアウト時は残枚数の少ない方が勝ち、同数は引き分け。

## リアルタイム設計の要点

- 各エージェントはブロッキングでLLMを呼び、その間に相手が先に出せる。出そうとしたカードが直前に無効化された場合は `invalid_moves` にカウントして即再観測
- 「せーの」は両者に合法手が無い状態が約0.2秒継続した場合のみ発動し、思考中の取りこぼしフリップを防ぐ
- 山札の並びはシード固定で再現可能

## LLM応答プロトコル(json-v1)

全Providerで同じ契約を使う。応答は説明文・Markdownを含まないJSONオブジェクト1個のみ(`{"action":"play","card":1-13,"pile":0-1}` または `{"action":"pass"}`)。出力上限64トークン・温度0で統一し、対応するProviderではネイティブ構造化出力(Ollamaの`format`、Anthropicのtool強制呼び出し)を使う。応答全体を厳密にJSONとしてparseし、最初の`{...}`を拾う正規表現フォールバックは行わない。parse失敗はそのターン`pass`として扱い`parse_errors`に計上する。1エージェントのparse失敗率が10試合以上の集計で1%を超えると、そのエージェントのランキング値は`ranking_valid: false`として無効フラグが立つ(raw値は隠さない)。詳細は `docs/arena-spec.md` を参照。

## ウォームアップ(コールドスタート除外)

対戦開始前に両エージェントへ並行して1回ウォームアップ推論を行い、結果は破棄する。ウォームアップの時間はレイテンシ計測に混入させず、`warmup_status`/`warmup_duration`として別記録する。ウォームアップに失敗した試合はランキングに算入しない(`valid: false`)。Ollamaは対戦時間中モデルをアンロードしないよう`keep_alive`を設定する。

## 公平性の注意

1台のマシンで2つのローカルLLMを同時推論させると、VRAM退避やスケジューリングで実レイテンシが歪む。厳密に測るなら `--hosts` でモデルごとに別ホスト(別GPU)を割り当てること。レイテンシがほぼ同一のモデル同士ではシステム側の微小な揺らぎが勝敗に影響するため、ペアあたり10戦以上を推奨。出力トークン数は全モデルで `num_predict`/`max_tokens=64` に統一している。

## 計測メトリクス

`results.json` にはランキング(Elo・勝敗数・parse_error_rate・ranking_valid)に加え、試合ごとにモデルごとの LLM呼び出し回数、平均レイテンシ、有効着手数、無効着手数、JSONパース失敗数、APIエラー数、ウォームアップ状態を記録する。

## ビューア

`viewer/speed-llm-arena-visualizer.jsx` は `results.json` を読み取り専用で表示するReactコンポーネント。ランキング・試合一覧・各種メトリクスを表示し、JSON不正や`schema_version`不一致、必須フィールド欠損は画面上に明示的なエラーとして表示する(バックエンドや結果ファイルの書き換えは対象外)。`tests/fixtures/` にサンプルおよびエラー確認用のfixtureがある。

## テスト

```bash
python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_engine.py` はゲームエンジン・json-v1パース・ランキング集計を、`tests/test_agents_http.py` は標準ライブラリの`http.server`によるモックサーバーでOllama/AnthropicのHTTP契約(リクエスト形状・レスポンスのパース・ウォームアップ失敗の検知)を、それぞれ実LLMサーバーなしで検証する。GitHub Actions(`.github/workflows/ci.yml`)がPython 3.10/3.11/3.12でこれらとselftestスモークテストを自動実行する。

## テスト結果

- 反応速度0.05秒 vs 0.30秒のボット対戦: 速い側が全勝(実レイテンシ勝負が機能)
- 「せーの」は各試合3〜7回発生、手詰まりからの復帰を確認
- 座席交代・シード固定・Elo集計・JSON出力の動作を確認

## 今後の拡張候補

- 観戦用リプレイ再生(`event_log`を使ったビューアのリプレイ機能。現行のビューアMVPは対象外)
- レイテンシと正確性の散布図など分析レポートの自動生成
- 「思考時間トークン換算」モードの追加でハードウェア非依存のランキング
