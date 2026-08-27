# speed-llm-arena 設計仕様書

このドキュメントは、ゲームルール・計測定義・公平性要件・`results.json` の JSON 契約を統合した仕様書(SSOT)である。README は入口と短い使い方だけを担当し、数値や仕様の正本はこの文書と対応する GitHub Issue に集約する。

- ウォームアップ/コールドスタート除外: Issue #1
- LLM 応答プロトコル(`json-v1`): Issue #2
- リポジトリ整備・`results.json` 契約・ビューア: Issue #3

矛盾が生じた場合は Issue 本文を先に更新し、実装・本仕様書・README の順に反映する。

## 1. ゲームルール

- 各プレイヤーはランク 1〜13(A〜K)を 2 枚ずつ、計 26 枚を持つ。
- 手札は 4 枚固定(`HAND_SIZE = 4`)。残りは山札(`stock`)としてプレイヤーが個別に保持する。
- 中央には 2 つの台札(`piles`)があり、初期化時に各プレイヤーの山札から 1 枚ずつ配置する。
- 手札のカードは、隣接するランクの台札にのみ出せる。隣接とは差が 1、または K(13)と A(1)のループを指す(`is_adjacent`)。
- カードを出したら手札が 1 枚減るので、山札があれば即座に 1 枚補充する。
- 両プレイヤーとも合法手が存在しない状態(手詰まり)が一定時間継続すると、審判が両方の台札を山札から更新する(「せーの」、`flip`)。
- 山札が尽きた状態で手詰まりになった場合は、残り枚数(手札+山札)が少ない方を勝者とする。同数なら引き分け。
- 手札+山札を先に 0 にしたプレイヤーが勝ち(`played_out`)。
- 試合には `max_duration` 秒の上限があり、超過した場合は残り枚数の少ない方を勝者、同数なら引き分けとする(タイムアウト)。

## 2. リアルタイム設計

- 各プレイヤーは専用スレッドで「観測 (`snapshot`) → 思考 (`agent.decide`) → 着手 (`try_play`)」をブロッキングで繰り返す。
- 思考中に相手が先に着手し、狙っていたカードが直前に無効化された場合は `invalid_moves` としてカウントし、即座に再観測する。
- 審判スレッドは「せーの」の誤発火を避けるため、両者に合法手が無い状態が `poll * 4`(既定 0.2 秒)継続して初めて `flip()` を実行する。
- 山札の並びは `seed` で固定され、再現可能である。

## 3. ウォームアップ(コールドスタート除外) — Issue #1

- `run_match` は `game.start()` より前に、両エージェントの `warmup()` を並行実行する。
- `warmup()` は本番と同じ Agent・接続先・system prompt・固定の合法局面(`WARMUP_SNAPSHOT`)で 1 回推論し、結果は破棄する。
- ウォームアップの時間・呼び出しは対戦の `calls` / `think_time` / `avg_latency` / 勝敗判定に含めない。`warmup_status`(`"ok"` / `"failed"`)、`warmup_duration`、`warmup_started_at`(ウォームアップ呼び出し開始時刻、UNIX epoch秒)を試合結果に別メトリクスとして記録する。`warmup_started_at` により、ウォームアップ完了後に最初のカウント対象推論が始まったことを外部から検証できる。
- いずれかのエージェントのウォームアップが失敗した場合、その試合は実対戦を行わず `valid: false`、`reason: "warmup_failed"` として記録し、ランキング(Elo・勝敗数・parse_error 集計)には算入しない。`pass` の擬似応答で隠すことはしない。
- Ollama は `keep_alive` にモデルを対戦時間中保持する値(既定は試合の最大時間 + 60 秒、または `-1` の無期限保持)を指定し、試合途中のモデルアンロードを防ぐ。
- 厳密なモデル間比較では、モデルごとに別ホスト/別 GPU を割り当てる(`--hosts`)。同一ホスト実行は VRAM 競合ありの参考値として扱う。
- `HeuristicAgent.warmup()` は副作用のない no-op だが、`Agent` と同じインターフェース(`{"status", "duration"}` を返す)を実装する。

## 4. LLM 応答プロトコル `json-v1` — Issue #2

全 Provider(Ollama / Anthropic / 将来追加分)は同じ契約で対戦する。モデルごとの例外は禁止する。

- 出力上限: 全モデル一律 64 トークン(`MAX_TOKENS = 64`)。Ollama は `num_predict=64`、Anthropic は `max_tokens=64`。
- 温度: `temperature=0.0`(`TEMPERATURE`)。
- 思考出力: Provider が無効化に対応する場合は無効化する(Ollama は `think: false`)。
- 応答は説明文・Markdown・思考過程を含まない JSON オブジェクト 1 個のみ。スキーマは以下の `ACTION_SCHEMA`:

  ```json
  {
    "type": "object",
    "properties": {
      "action": {"type": "string", "enum": ["play", "pass"]},
      "card": {"type": "integer", "minimum": 1, "maximum": 13},
      "pile": {"type": "integer", "minimum": 0, "maximum": 1}
    },
    "required": ["action"],
    "additionalProperties": false
  }
  ```

- Provider がネイティブ structured output / JSON Schema に対応する場合はそれを使う。
  - Ollama: `/api/chat` の `format` に `ACTION_SCHEMA` を渡す。
  - Anthropic: `submit_action` という 1 つの tool を `tool_choice` で強制呼び出しし、`tool_use.input` をそのまま検証する。
- パースは応答全体を厳密に JSON として解釈する(`json.loads(text.strip())`)。**最初の `{...}` だけを拾う正規表現フォールバックは使用しない。** 全体が単一の JSON オブジェクトとして解釈できない場合は parse エラーとする。
- `action` が `"play"` の場合、`card` は 1〜13 の整数、`pile` は 0 か 1 の整数でなければならない。範囲外・型不正・キー欠落は parse エラー。
- `action` が `"pass"`、または未知の `action` はそれぞれ次のとおり扱う: `"pass"` は正常、それ以外(未知の値・欠落)は parse エラー。
- キー集合はこの2形に完全一致しなければならない: `"pass"` はキーが `{"action"}` のみ、`"play"` はキーが `{"action","card","pile"}` のみ。`{"action":"pass","card":5}` のように余計なキーを含む場合(`ACTION_SCHEMA` の `additionalProperties: false` が拾いきれない組み合わせ)も parse エラーとする(`validate_action()`)。
- parse エラーはそのターン `pass` として扱い、ゲーム進行は継続するが `parse_errors` に加算し、集計から隠さない。
- 思考出力または厳格な JSON 出力を保証できないモデルは、正規表現救済やトークン上限の個別引き上げをせず、実測の parse エラー率でランキング適格性を判定する(次項)。
- `protocol_version`(`"json-v1"`)、`max_tokens`、`temperature` は `results.json` の `config` に記録する。

### 4.1 ランキング適格性

- あるエージェントについて、有効試合(`valid: true`)に限定した `parse_errors / calls` を集計する。
- そのエージェントが有効試合に 10 試合以上参加しており、かつ parse エラー率が 1% を超える場合、`ranking[].ranking_valid` を `false` にする。
- `ranking_valid: false` でも Elo・戦績・`parse_error_rate` は raw 値のまま表示し、隠蔽・改変はしない。

## 5. 公平性要件

- 全 Provider で同じ JSON スキーマ・最大出力トークン数・温度・入力局面形式(`build_user_prompt`)を使う。
- APIエラー・parse エラー・無効手(合法性違反)はそれぞれ別メトリクス(`api_errors` / `parse_errors` / `invalid_moves`)として記録し、有効手数やレイテンシへ黙って混入させない。
- 総当たり戦では先手を交互に入れ替え、座席バイアスを除去する。
- 山札のシードは `base_seed + 試合インデックス` で決定的に生成し、再現可能にする。
- 1 台のマシンで複数のローカル LLM を同時推論させると VRAM 退避やスケジューリングでレイテンシが歪むため、厳密な比較では `--hosts` でモデルごとに別ホストを割り当てる。

## 6. `results.json` 契約(schema_version 1)

`results.json` はランキング・対戦結果の唯一の出力ファイルであり、以下の構造を持つ。対戦・自己対戦で生成した `results.json` はリポジトリにコミットしない(`.gitignore` 対象)。再現用の fixture は `tests/fixtures/` に別名で管理する。

```jsonc
{
  "schema_version": 1,
  "config": {
    "mode": "selftest | ollama | anthropic",
    "models": ["..."],
    "hosts": ["..."],            // ollamaモードのみ。APIキー等の秘密情報は含めない
    "games_per_pair": 4,
    "seed": 42,
    "max_duration": 300.0,
    "protocol_version": "json-v1",
    "max_tokens": 64,
    "temperature": 0.0
  },
  "ranking": [
    {
      "name": "fast-bot",
      "elo": 1055.8,
      "win": 4,
      "loss": 0,
      "draw": 0,
      "parse_error_rate": 0.0,
      "ranking_valid": true
    }
  ],
  "matches": [
    {
      "seed": 42,
      "p0": "fast-bot",
      "p1": "slow-bot",
      "winner": "fast-bot",       // null は引き分け、または無効試合
      "reason": "played_out",     // played_out / deadlock_* / timeout_* / warmup_failed
      "duration": 7.84,
      "flips": 3,
      "valid": true,               // falseはランキング未算入(例: warmup_failed)
      "stats": [
        {
          "agent": "fast-bot",
          "calls": 44,
          "plays": 22,
          "invalid_moves": 0,
          "parse_errors": 0,
          "api_errors": 0,
          "think_time": 2.247,
          "avg_latency": 0.051,
          "warmup_status": "ok",
          "warmup_duration": 0.0,
          "warmup_started_at": 1770000000.0
        }
      ]
    }
  ]
}
```

- `schema_version` が現行値(1)と一致しない場合、消費側(ビューア等)はエラーとして明示し、黙って空表示にしない。
- `event_log`(観戦用リプレイ)はこの MVP 契約に含めない。将来追加する場合は `schema_version` を上げる。

## 10. 永続ランキング — Issue #7 / #8

ランキングは1回の実行で捨てず、マシンごとの `rankings/<machine_id>.json` に保存する。この節は対応 Issue の写しであり、ランキングの同一性・更新・破損時の動作・Markdown 生成を定義する。

### 10.1 選手の同一性

選手 ID は `モデル / 実行系 / 形式・量子化` の3つで決める。`player_id` は次の文字列を `|` で連結する。

```text
モデル|実行系|形式-量子化
qwen3.8:27b|ollama|GGUF-Q4_K_M
```

モデル名の `:` や `/` はそのまま許す。4要素のいずれかに `|` が入る場合は実行前に失敗させ、空欄も許さない。未量子化・FP16 なども `safetensors-FP16` のように形式と精度を両方指定する。

### 10.2 マシンは分け、実行系は分けない

ランキングは `machine_id` ごとに独立する。同じマシン内では実行系・形式・量子化が異なる選手も同じ表で競わせる。マシンをまたいだ対戦は行わず、比較は同じ `player_id` の点数をマシン間で並べる。

`machine_id` はファイル名になるため、`[a-z0-9][a-z0-9._-]{0,63}` のみ許可する。`.` 始まり、`..`、`/` などは実行前に拒否する。

### 10.3 はしご方式と Elo

`--mode ladder` を既定とし、順位の正本は `players[]` の `rank` 順、Elo は併記値とする。Elo で並べ替えない。新入りは `lo=1, hi=N+1` から探索し、`N<=10` では `lo`、`N>=11` では `(lo+hi)//2` の順位に挑戦する。勝ち・引き分けは `hi=相手の順位`、負けは `lo=相手の順位+1` とし、`lo>=hi` で `lo` に挿入する。`N=0` は無試合で rank 1、`N=1` は通常探索する。

既存選手の有効な再戦では、上位の勝者は順位を動かさず、下位の勝者だけが敗者の位置へ移り、間の選手を1つ押し下げる。引き分けは順位を動かさない。番狂わせでも Elo が動くのは対戦した2選手だけである。

Elo は初期値1000、K=32、期待勝率 `1/(1+10**((相手-自分)/400))`、勝ち1・引き分け0.5・負け0で、更新前の両者の値から同時に計算する。保存時は小数第1位に丸める。

`--mode round-robin` は参加選手を全ペアで対戦させ、勝ち点（勝ち1、引き分け0.5、負け0）→ Elo → `entry_no` 昇順で並べる。無効試合は勝ち点・試合数から除外し、参加しなかった選手はその下に元の相対順で置く。`entry_no` は既存最大値+1で、欠番を再利用しない。

### 10.4 検証・推移性

`--verify-transitivity` は最終ランキング上位3体の3ペアを追加対戦させる。検証試合には `verification: true` を付け、Elo・順位・戦績・レイテンシ・パースエラー率のどれにも加えない。

- 3試合の勝ち数が2/1/0なら `status: "ok"`、`transitivity_warning: false`
- 全員1勝なら `status: "cyclic"`、`transitivity_warning: true`
- 引き分けがあれば `status: "inconclusive"`, `inconclusive: true` とし、警告は変更しない
- 3体未満なら `status: "skipped"`, `reason: "players<3"` とし、警告は変更しない
- エラー・タイムアウト・パース不能は同じ組を1回だけ再試行し、それでも失敗なら `status: "aborted"`, `failed_pair`, 完了済みの0〜2件の `matches` を残す。偽の勝敗には変換しない

`transitivity_detail` は常に `checked_at` と `status` を持つ。初期値は `{"checked_at": null, "status": "skipped", "reason": "never_run"}`。`status` は `ok` / `cyclic` / `inconclusive` / `skipped` / `aborted` のいずれかである。

### 10.5 JSON の構造、母数、安全な書き込み

正本は次の構造を持つ。`avg_latency_ms` と `parse_error_rate` は母数から再計算する派生値であり、平均値だけで更新しない。

```json
{
  "schema_version": 1,
  "machine_id": "gx10",
  "machine": {"gpu": "NVIDIA GB10", "memory_gb": 121},
  "updated_at": "2026-08-27T09:40:00+09:00",
  "transitivity_warning": false,
  "transitivity_detail": {"checked_at": null, "status": "skipped", "reason": "never_run"},
  "players": []
}
```

`total_requests` は送信した全リクエスト、`latency_requests` は最後まで応答が返ったリクエスト、`parse_errors` は応答が返ったが JSON 契約に適合しなかった回数とする。タイムアウト・API エラーはレイテンシ母数に入れず、パースエラーは入れる。`latency_requests=0` の平均は `null` とする。

実行は `rankings/<machine_id>.json.lock` を `O_EXCL` で取得し、ロック本文に `pid`、`host`、`acquired_at` を JSON で書いて `fsync` する。読込・全対戦・書戻しを1つのロック内で行う。ロックを取れなければ待たずに失敗する。壊れたロックは更新時刻が30分より古い場合だけ回収し、別ホスト、生存中の PID、`EPERM` は回収しない。`ESRCH` は時間にかかわらず回収する。

ランキング JSON は読み込み時に構造を検証する。`schema_version` 欠落、rank の抜け・重複、player ID の重複・再計算値不一致、負の母数、有限でない数値、machine ID 不一致、制御文字・タブ・改行・長さ超過などは壊れたファイルとする。壊れたファイルは元を上書きせず `*.corrupt-YYYYMMDD-HHMMSS` に `O_EXCL` で退避し、同名なら `-2`, `-3` とする。存在しない JSON は空ランキングとして開始する。より新しい schema や移行不能な古い schema は読み書きせず終了する。

書き込みは同じディレクトリの予測不能な `mkstemp`（0600）へ行い、内容を `fsync`、0644 に変更して `os.replace` し、最後に親ディレクトリも `fsync` する。試合途中には書き戻さず、1回の実行を原子的な確定単位とする。

### 10.6 Markdown と CI

`python3 speed_arena.py --render-rankings` は `rankings/*.json` を読むだけで対応する `.md` を決定的に生成する。生成時刻は埋め込まない。Markdown 表のモデル名・実行系・量子化・戦績・平均応答など文字列セルはバッククォートで囲む。

README は各マシンの `rankings/<machine_id>.md` を参照し、CI は生成後に `git diff --exit-code -- rankings/` を実行する。JSON 変更時は対応する Markdown もコミットし、古い Markdown が残れば CI を失敗させる。

受け入れテストは、はしごの10体/11体境界、番狂わせの押し出し、0体/1体、Elo、無効試合の母数と再試行、JSON 破損と退避衝突、schema_version、死んだロック、原子的書き込み、Markdown 同期を `tests/` で検証する。

## 7. 計測メトリクスの定義

| フィールド | 意味 |
|---|---|
| `calls` | 対戦中(ウォームアップ除く)の `decide()` 呼び出し回数 |
| `plays` | 合法な着手として成功した回数 |
| `invalid_moves` | 着手が拒否された回数(競合・不正手を含む) |
| `parse_errors` | 応答が `json-v1` スキーマに厳密適合しなかった回数 |
| `api_errors` | HTTP/接続エラーなど Provider 呼び出し自体の失敗回数 |
| `think_time` | `decide()` の実測合計時間(秒)。ウォームアップは含まない |
| `avg_latency` | `think_time / calls`(秒) |
| `warmup_status` | `"ok"` または `"failed"` |
| `warmup_duration` | ウォームアップ推論にかかった時間(秒) |
| `warmup_started_at` | ウォームアップ呼び出しを開始した時刻(UNIX epoch秒) |

## 8. ビューア MVP — Issue #3

- `viewer/speed-llm-arena-visualizer.jsx` は `results.json` を読み取り専用で表示する React コンポーネント。
- ランキング、試合一覧、勝敗、終了理由、平均レイテンシ、`plays`、`invalid_moves`、`parse_errors`、`api_errors`、`warmup_status` を表示する。
- JSON 不正、`schema_version` 不一致、必須フィールド欠損は画面上で明示的なエラーとして表示し、黙って空画面にしない。
- バックエンド、認証、結果ファイルの書き換え、`event_log` 再生は対象外。

## 9. 非対象(Non-goals)

- モデルごとに最適化したプロンプトやトークン上限の探索。
- 思考トークン量を別ランキングにすること。
- 観戦用リプレイ再生(`event_log`)の実装。
