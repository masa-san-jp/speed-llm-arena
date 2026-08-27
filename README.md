# speed-llm-arena

トランプゲーム「スピード」をLLM同士にリアルタイムでプレイさせ、勝敗からEloランキングを算出するアリーナ。反応速度は**実レイテンシ**(推論にかかった実時間)で競わせる。主対象はOllama上のローカルLLM。

## 特徴

- ルール判定は決定論的なPythonエンジン。LLMには局面をJSONで渡し、着手をJSONで返させる
- プレイヤーごとに独立スレッドで「観測 → 思考 → 着手」をループ。思考中に相手が先に出せば競合負けとして記録される
- モデルごとに平均レイテンシ・有効手・無効手・パース失敗を計測し、「速いが雑」「遅いが正確」を分析できる
- マシンごとの永続ランキング。既定のはしご方式で新しい選手を必要な相手だけと対戦させる
- 検証用の総当たり方式と、上位3体の推移性検証にも対応

## 必要環境

Python 3.10+ と requests のみ。

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# 1. エンジン検証(LLM不要のベースライン同士、従来モード)
python speed_arena.py --mode selftest

# 2. 永続ランキングを更新 (既定は ladder、Ollama)
python speed_arena.py --mode ladder --machine-id gx10 \
    --models gpt-oss:20b gemma3:4b --games 10 \
    --quantization Q4_K_M

# 3. モデルごとに別ホストを割り当てて公平にレイテンシ計測
python speed_arena.py --mode ladder --machine-id gx10 --models gpt-oss:20b gemma3:4b \
    --hosts http://192.168.1.10:11434 http://192.168.1.11:11434

# 4. 総当たり方式で更新 (検証用)
python speed_arena.py --mode round-robin --machine-id gx10 \
    --models gpt-oss:20b gemma3:4b --games 4

# 5. 上位3体の推移性を追加検証
python speed_arena.py --mode ladder --machine-id gx10 \
    --models gpt-oss:20b gemma3:4b --verify-transitivity

# 6. Claude APIでテスト(要 ANTHROPIC_API_KEY)
python speed_arena.py --mode ladder --runtime anthropic --machine-id cloud \
    --models claude-haiku-4-5-20251001 claude-sonnet-4-6

# 7. JSONからMarkdownを再生成 (CIでも実行)
python speed_arena.py --render-rankings
```

結果は `results.json` に出力され、ランキングの正本は `rankings/<machine_id>.json` に更新される。`--machine` に `display_name`、`chip`、`memory_gb` などを指定すると、Markdownにマシン名と構成を表示できる。ランキング JSON はマシンごとにコミットし、`rankings/<machine_id>.md` は `--render-rankings` で生成する。`results.json` はコミットしない(`.gitignore` 対象)。詳細な契約は `docs/arena-spec.md` を参照する。

## マシン別ランキング

ランキングは実行したマシンごとに独立して管理し、生成された表を参照する。

- [ランキング一覧](rankings/)
- [CI 用ランキング](rankings/ci.md) — 最終更新 2026-08-27
- [MBP-M4-Max-128GB](rankings/mbp-m4-max-128gb.md) — ローカルOllama

### MBP-M4-Max-128GB

Apple M4 Max / メモリ128GB / Ollama / 2026-08-27計測

短時間で比較するため、同じ1手局面を使ったローカル確認ランキングです。順位とEloを併記しています。詳しい条件と正本データは[ランキング詳細](rankings/mbp-m4-max-128gb.md)と[JSON](rankings/mbp-m4-max-128gb.json)を参照してください。

| 順位 | モデル | 戦績 | Elo | 平均応答 |
|---:|---|---:|---:|---:|
| 1 | `gemma4:e4b` | 2勝 0敗 0分 | 1032.6 | 2507.75ms |
| 2 | `qwen3.8:27b` | 3勝 1敗 0分 | 1028.5 | 6826.96ms |
| 3 | `qwen2.5:0.5b` | 0勝 2敗 1分 | 970.8 | 277.436ms |
| 4 | `gpt-oss:20b` | 0勝 1敗 2分 | 984.0 | 955.611ms |
| 5 | `gemma4:26b` | 0勝 1敗 1分 | 984.0 | 924.92ms |

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

`results.json` にはランキング(Elo・勝敗数・parse_error_rate・ranking_valid)に加え、試合ごとにモデルごとの LLM呼び出し回数、平均レイテンシ、有効着手数、無効着手数、JSONパース失敗数、APIエラー数、ウォームアップ状態を記録する。永続ランキングは `total_requests`、`latency_requests`、`total_latency_ms` から平均値を再計算する。接続/API障害とウォームアップ失敗だけをランキング対象外とし、ゲームのタイムアウトはモデルの評価結果として勝敗に反映する。

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
