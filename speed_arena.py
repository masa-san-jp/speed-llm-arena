#!/usr/bin/env python3
"""speed_arena.py - スピード(トランプ)LLMアリーナ 単一ファイル版

トランプゲーム「スピード」をLLM同士にリアルタイムでプレイさせ、Eloランキングを算出する。
反応速度は実レイテンシ(推論にかかった実時間)で競わせる。

依存: Python 3.10+ と requests のみ

使用例:
  python speed_arena.py --mode selftest
  python speed_arena.py --mode ollama --models gpt-oss:20b gemma3:4b --games 10
  python speed_arena.py --mode ollama --models gpt-oss:20b gemma3:4b \
      --hosts http://192.168.1.10:11434 http://192.168.1.11:11434
  python speed_arena.py --mode anthropic --models claude-haiku-4-5-20251001 claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests

HAND_SIZE = 4
RANK_MIN, RANK_MAX = 1, 13

# 応答プロトコル契約(json-v1)。全Providerで共通。#2
PROTOCOL_VERSION = "json-v1"
MAX_TOKENS = 64
TEMPERATURE = 0.0
SCHEMA_VERSION = 1
PARSE_ERROR_RATE_THRESHOLD = 0.01
PARSE_ERROR_MIN_GAMES = 10


def is_adjacent(a: int, b: int) -> bool:
    """隣接ランク判定。差が1、またはK(13)とA(1)のループ。"""
    d = abs(a - b)
    return d == 1 or d == 12


# ============================== ゲームエンジン ==============================


@dataclass
class PlayerState:
    hand: list[int] = field(default_factory=list)   # 公開手札(最大4枚)
    stock: list[int] = field(default_factory=list)  # 山札(末尾がトップ)

    @property
    def remaining(self) -> int:
        return len(self.hand) + len(self.stock)


@dataclass
class Snapshot:
    """エージェントに渡す観測情報。相手の手札は公開情報なので含める。"""
    version: int
    piles: list[int]
    my_hand: list[int]
    my_stock_count: int
    opp_hand: list[int]
    opp_stock_count: int
    elapsed: float

    def legal_moves(self) -> list[tuple[int, int]]:
        return [
            (c, p)
            for c in set(self.my_hand)
            for p, top in enumerate(self.piles)
            if is_adjacent(c, top)
        ]


@dataclass
class PlayResult:
    ok: bool
    reason: str = ""


class SpeedGame:
    """スレッドセーフなゲーム状態。着手の合法性は適用時点の状態で判定する。"""

    def __init__(self, seed: int, max_duration: float = 300.0):
        self.rng = random.Random(seed)
        self.max_duration = max_duration
        self.lock = threading.Lock()
        self.version = 0
        self.piles: list[int] = []
        self.players = [PlayerState(), PlayerState()]
        self.winner: Optional[int] = None  # 0/1, -1=引き分け, None=進行中
        self.end_reason = ""
        self.started_at = 0.0
        self.flips = 0
        self.event_log: list[dict] = []
        self._deal()

    def _deal(self) -> None:
        deck = [r for r in range(RANK_MIN, RANK_MAX + 1) for _ in range(2)]  # 26枚
        for i in (0, 1):
            cards = deck[:]
            self.rng.shuffle(cards)
            ps = self.players[i]
            ps.hand = [cards.pop() for _ in range(HAND_SIZE)]
            ps.stock = cards
        self.piles = [self.players[0].stock.pop(), self.players[1].stock.pop()]

    def start(self) -> None:
        self.started_at = time.monotonic()

    def snapshot(self, player: int) -> Snapshot:
        with self.lock:
            me, opp = self.players[player], self.players[1 - player]
            return Snapshot(
                version=self.version,
                piles=list(self.piles),
                my_hand=list(me.hand),
                my_stock_count=len(me.stock),
                opp_hand=list(opp.hand),
                opp_stock_count=len(opp.stock),
                elapsed=time.monotonic() - self.started_at,
            )

    def try_play(self, player: int, card: int, pile: int) -> PlayResult:
        with self.lock:
            if self.winner is not None:
                return PlayResult(False, "game_over")
            if pile not in (0, 1):
                return PlayResult(False, "bad_pile")
            ps = self.players[player]
            if card not in ps.hand:
                return PlayResult(False, "card_not_in_hand")
            if not is_adjacent(card, self.piles[pile]):
                return PlayResult(False, "not_adjacent")
            ps.hand.remove(card)
            self.piles[pile] = card
            if ps.stock:
                ps.hand.append(ps.stock.pop())
            self.version += 1
            self.event_log.append({
                "t": round(time.monotonic() - self.started_at, 4),
                "ev": "play", "player": player, "card": card, "pile": pile,
            })
            if ps.remaining == 0:
                self._finish(player, "played_out")
            return PlayResult(True)

    def would_flip(self) -> bool:
        """両者に合法手が無いかを非破壊で確認する。"""
        with self.lock:
            if self.winner is not None:
                return False
            for i in (0, 1):
                for c in self.players[i].hand:
                    if any(is_adjacent(c, top) for top in self.piles):
                        return False
            return True

    def flip(self) -> None:
        """「せーの」で台札を更新する。両山札が尽きていれば残枚数で決着。"""
        with self.lock:
            if self.winner is not None:
                return
            s0, s1 = self.players[0].stock, self.players[1].stock
            if not s0 and not s1:
                r0, r1 = self.players[0].remaining, self.players[1].remaining
                if r0 == r1:
                    self._finish(-1, "deadlock_tie")
                else:
                    self._finish(0 if r0 < r1 else 1, "deadlock_fewer_cards")
                return
            if s0:
                self.piles[0] = s0.pop()
            if s1:
                self.piles[1] = s1.pop()
            self.flips += 1
            self.version += 1
            self.event_log.append({
                "t": round(time.monotonic() - self.started_at, 4),
                "ev": "flip", "piles": list(self.piles),
            })

    def check_timeout(self) -> None:
        with self.lock:
            if self.winner is None and time.monotonic() - self.started_at > self.max_duration:
                r0, r1 = self.players[0].remaining, self.players[1].remaining
                if r0 == r1:
                    self._finish(-1, "timeout_tie")
                else:
                    self._finish(0 if r0 < r1 else 1, "timeout_fewer_cards")

    def _finish(self, winner: int, reason: str) -> None:
        self.winner = winner
        self.end_reason = reason
        self.event_log.append({
            "t": round(time.monotonic() - self.started_at, 4),
            "ev": "end", "winner": winner, "reason": reason,
        })

    @property
    def is_over(self) -> bool:
        with self.lock:
            return self.winner is not None


# ============================== エージェント ==============================

SYSTEM_PROMPT = """You are playing the real-time card game Speed. Respond as FAST as possible.
Rules: you may place a card from your hand onto a center pile if its rank is adjacent
(difference of 1; King(13) and Ace(1) wrap around). First to empty hand+stock wins.
Speed matters: every millisecond you spend thinking, your opponent may act first.

Respond with ONLY a single JSON object, no explanation, no markdown:
{"action":"play","card":<rank 1-13>,"pile":<0 or 1>}  or  {"action":"pass"}"""


def build_user_prompt(s: Snapshot) -> str:
    return json.dumps({
        "center_piles": s.piles,
        "your_hand": s.my_hand,
        "your_stock_count": s.my_stock_count,
        "opponent_hand": s.opp_hand,
        "opponent_stock_count": s.opp_stock_count,
    }, separators=(",", ":"))


# json-v1: 応答はこのスキーマに厳密適合するJSONオブジェクト1個のみ。#2
# Provider側の構造化出力(Ollama format / Anthropic tool schema)へのヒントとして渡す。
# ただしこのJSON Schema自体は "pass" に card/pile が同時に付くようなケースまでは
# 弾けないため、実際の受理は validate_action() が厳密な2形("pass"のみ、または
# "play"+card+pile)への完全一致で最終判定する。
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["play", "pass"]},
        "card": {"type": "integer", "minimum": RANK_MIN, "maximum": RANK_MAX},
        "pile": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": ["action"],
    "additionalProperties": False,
}


def validate_action(obj) -> dict:
    """パース済みオブジェクトをjson-v1契約で検証する。契約の2形("pass"のみ/
    "play"+card+pile)に完全一致しない場合(余計なキーを含む場合も)はparse_errorとしてpass扱いにする。
    """
    if not isinstance(obj, dict):
        return {"action": "pass", "parse_error": True}
    action = obj.get("action")
    if action == "play":
        if set(obj.keys()) != {"action", "card", "pile"}:
            return {"action": "pass", "parse_error": True}
        try:
            card = int(obj["card"])
            pile = int(obj["pile"])
        except (KeyError, TypeError, ValueError):
            return {"action": "pass", "parse_error": True}
        if not (RANK_MIN <= card <= RANK_MAX) or pile not in (0, 1):
            return {"action": "pass", "parse_error": True}
        return {"action": "play", "card": card, "pile": pile}
    if action == "pass":
        if set(obj.keys()) != {"action"}:
            return {"action": "pass", "parse_error": True}
        return {"action": "pass"}
    return {"action": "pass", "parse_error": True}


def parse_action_text(text: str) -> dict:
    """json-v1: 応答全体を厳密にJSONとしてparseする。最初の{...}を拾う正規表現フォールバックは使わない。"""
    if not text or not text.strip():
        return {"action": "pass", "parse_error": True}
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return {"action": "pass", "parse_error": True}
    return validate_action(obj)


# ウォームアップ用の固定合法局面。本番と同じ経路で1回推論し、結果は破棄する。#1
WARMUP_SNAPSHOT = Snapshot(
    version=0, piles=[7, 8], my_hand=[6, 9, 1, 13],
    my_stock_count=22, opp_hand=[2, 3, 4, 5], opp_stock_count=22, elapsed=0.0,
)


class Agent(ABC):
    name: str
    protocol_version: str = PROTOCOL_VERSION

    @abstractmethod
    def decide(self, snapshot: Snapshot) -> dict:
        ...

    def warmup(self) -> dict:
        """本番と同じAgent/接続先/system promptで1回推論し、結果は破棄する。#1"""
        t0 = time.monotonic()
        try:
            self.decide(WARMUP_SNAPSHOT)
            status = "ok"
        except Exception:
            status = "failed"
        return {"status": status, "duration": round(time.monotonic() - t0, 3)}


class OllamaAgent(Agent):
    """ローカルLLM。model例: gpt-oss:20b, gemma3:4b。

    host をモデルごとに分ければ別GPUで公平にレイテンシ勝負できる。
    json-v1: format にJSON Schemaを渡してネイティブ構造化出力を使い、
    think=False で推論系モデルの思考出力を無効化して64トークンを回答に使わせる。#2
    keep_alive はモデルロード(コールドスタート)を試合中に発生させないため既定で無期限保持する。#1
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: float = 60.0, num_predict: int = MAX_TOKENS,
                 keep_alive: float | str = -1):
        self.name = model
        self.model = model
        self.host_label = host
        self.url = host.rstrip("/") + "/api/chat"
        self.timeout = timeout
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.session = requests.Session()

    def decide(self, snapshot: Snapshot) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": ACTION_SCHEMA,
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(snapshot)},
            ],
            "options": {"temperature": TEMPERATURE, "num_predict": self.num_predict},
        }
        try:
            r = self.session.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
        except requests.RequestException:
            return {"action": "pass", "api_error": True}
        return parse_action_text(content)


ACTION_TOOL = {
    "name": "submit_action",
    "description": "Submit your move for this turn of Speed.",
    "input_schema": ACTION_SCHEMA,
}


class AnthropicAgent(Agent):
    """要 ANTHROPIC_API_KEY 環境変数。動作確認やクラウドモデルとの比較用。

    json-v1: tool_choiceで submit_action ツールの呼び出しを強制し、
    ネイティブ構造化出力としてJSON Schema適合を保証する。#2
    """

    def __init__(self, model: str, max_tokens: int = MAX_TOKENS):
        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.session = requests.Session()

    def decide(self, snapshot: Snapshot) -> dict:
        try:
            r = self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": TEMPERATURE,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": build_user_prompt(snapshot)}],
                    "tools": [ACTION_TOOL],
                    "tool_choice": {"type": "tool", "name": "submit_action"},
                },
                timeout=60.0,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            return {"action": "pass", "api_error": True}
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_action":
                return validate_action(block.get("input", {}))
        return {"action": "pass", "parse_error": True}


class HeuristicAgent(Agent):
    """最初に見つけた合法手を即座に出す。latency で疑似反応速度を再現。"""

    def __init__(self, name: str = "heuristic", latency: float = 0.05):
        self.name = name
        self.latency = latency

    def decide(self, snapshot: Snapshot) -> dict:
        time.sleep(self.latency)
        moves = snapshot.legal_moves()
        if moves:
            card, pile = moves[0]
            return {"action": "play", "card": card, "pile": pile}
        return {"action": "pass"}

    def warmup(self) -> dict:
        """副作用のないno-op。インターフェースはAgentと共通化する。#1"""
        return {"status": "ok", "duration": 0.0}


# ============================== 対戦ランナー ==============================


@dataclass
class MatchStats:
    winner: int
    end_reason: str
    duration: float
    flips: int
    per_player: list[dict] = field(default_factory=list)
    valid: bool = True


def _player_loop(game: SpeedGame, agent: Agent, idx: int, stats: dict) -> None:
    while not game.is_over:
        snap = game.snapshot(idx)
        t0 = time.monotonic()
        try:
            action = agent.decide(snap)
        except Exception:
            action = {"action": "pass", "agent_error": True}
        latency = time.monotonic() - t0
        stats["calls"] += 1
        stats["think_time"] += latency
        if action.get("parse_error"):
            stats["parse_errors"] += 1
        if action.get("api_error") or action.get("agent_error"):
            stats["api_errors"] += 1

        if action.get("action") == "play":
            res = game.try_play(idx, action.get("card", -1), action.get("pile", -1))
            if res.ok:
                stats["plays"] += 1
            else:
                stats["invalid_moves"] += 1
        else:
            v = snap.version
            end = time.monotonic() + 0.5
            while time.monotonic() < end and not game.is_over:
                if game.snapshot(idx).version != v:
                    break
                time.sleep(0.02)


def _referee_loop(game: SpeedGame, poll: float = 0.05) -> None:
    """手詰まりが一定時間続いたら「せーの」。思考中の取りこぼしフリップを防ぐ。"""
    stuck_since = None
    while not game.is_over:
        game.check_timeout()
        if game.would_flip():
            if stuck_since is None:
                stuck_since = time.monotonic()
            elif time.monotonic() - stuck_since > poll * 4:
                game.flip()
                stuck_since = None
        else:
            stuck_since = None
        time.sleep(poll)


def _run_warmup(agents: list[Agent]) -> list[dict]:
    """試合開始前に両エージェントを並行してウォームアップする。結果は破棄する。#1

    Issue #1 の Required observability に合わせて warmup_started_at(呼び出し開始時刻、
    UNIX epoch秒)も記録し、ウォームアップ完了後に最初のカウント対象推論が始まったことを
    外部から検証可能にする。
    """
    results: list[Optional[dict]] = [None] * len(agents)

    def _do(idx: int) -> None:
        started_at = round(time.time(), 3)
        w = agents[idx].warmup()
        results[idx] = {**w, "started_at": started_at}

    threads = [threading.Thread(target=_do, args=(i,), daemon=True) for i in range(len(agents))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results  # type: ignore[return-value]


def run_match(agent_a: Agent, agent_b: Agent, seed: int,
              max_duration: float = 300.0) -> MatchStats:
    warmups = _run_warmup([agent_a, agent_b])
    stats = [
        {"agent": agent_a.name, "calls": 0, "plays": 0, "invalid_moves": 0,
         "parse_errors": 0, "api_errors": 0, "think_time": 0.0,
         "warmup_status": warmups[0]["status"], "warmup_duration": warmups[0]["duration"],
         "warmup_started_at": warmups[0]["started_at"]},
        {"agent": agent_b.name, "calls": 0, "plays": 0, "invalid_moves": 0,
         "parse_errors": 0, "api_errors": 0, "think_time": 0.0,
         "warmup_status": warmups[1]["status"], "warmup_duration": warmups[1]["duration"],
         "warmup_started_at": warmups[1]["started_at"]},
    ]
    if not all(w["status"] == "ok" for w in warmups):
        # ウォームアップ失敗試合は実対戦を行わず、無効試合としてランキングへ投入しない。
        # passの擬似応答で隠さず、warmup_statusとして明示する。
        for s in stats:
            s["avg_latency"] = 0.0
        return MatchStats(winner=-1, end_reason="warmup_failed", duration=0.0,
                           flips=0, per_player=stats, valid=False)

    game = SpeedGame(seed=seed, max_duration=max_duration)
    game.start()
    threads = [
        threading.Thread(target=_player_loop, args=(game, agent_a, 0, stats[0]), daemon=True),
        threading.Thread(target=_player_loop, args=(game, agent_b, 1, stats[1]), daemon=True),
        threading.Thread(target=_referee_loop, args=(game,), daemon=True),
    ]
    for t in threads:
        t.start()
    while not game.is_over:
        time.sleep(0.05)
    for t in threads:
        t.join(timeout=5.0)
    for s in stats:
        s["avg_latency"] = round(s["think_time"] / s["calls"], 3) if s["calls"] else 0.0
        s["think_time"] = round(s["think_time"], 3)
    return MatchStats(
        winner=game.winner if game.winner is not None else -1,
        end_reason=game.end_reason,
        duration=round(time.monotonic() - game.started_at, 2),
        flips=game.flips,
        per_player=stats,
    )


# ============================== トーナメント & Elo ==============================


def update_elo(ra: float, rb: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1.0 - score_a) - (1.0 - ea))


def build_ranking(agents: list[Agent], ratings: dict[str, float],
                   records: dict[str, dict], matches: list[dict]) -> list[dict]:
    """json-v1: parse_errors/callsが10試合以上の集計で1%を超えるエージェントは
    ranking_valid=Falseとしてランキング値を無効表示にする。raw値(elo/戦績/エラー率)は隠さない。#2
    無効試合(valid=False, 例: warmup_failed)は集計に含めない。
    """
    calls = {a.name: 0 for a in agents}
    parse_errors = {a.name: 0 for a in agents}
    games_played = {a.name: 0 for a in agents}
    for m in matches:
        if not m["valid"]:
            continue
        games_played[m["p0"]] += 1
        games_played[m["p1"]] += 1
        for s in m["stats"]:
            calls[s["agent"]] += s["calls"]
            parse_errors[s["agent"]] += s["parse_errors"]
    ranking = []
    for n, r in ratings.items():
        rate = round(parse_errors[n] / calls[n], 4) if calls[n] else 0.0
        ranking_valid = not (
            games_played[n] >= PARSE_ERROR_MIN_GAMES and rate > PARSE_ERROR_RATE_THRESHOLD
        )
        ranking.append({
            "name": n, "elo": round(r, 1), **records[n],
            "parse_error_rate": rate, "ranking_valid": ranking_valid,
        })
    ranking.sort(key=lambda x: -x["elo"])
    return ranking


def run_tournament(agents: list[Agent], config: dict, games_per_pair: int = 4,
                   base_seed: int = 42, max_duration: float = 300.0,
                   out_path: str = "results.json", verbose: bool = True) -> dict:
    ratings = {a.name: 1000.0 for a in agents}
    records = {a.name: {"win": 0, "loss": 0, "draw": 0} for a in agents}
    matches: list[dict] = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            for g in range(games_per_pair):
                # 座席バイアスを消すため先手を交互に入れ替える
                a, b = (agents[i], agents[j]) if g % 2 == 0 else (agents[j], agents[i])
                seed = base_seed + len(matches)
                ms = run_match(a, b, seed=seed, max_duration=max_duration)
                w = l = None
                if ms.valid:
                    if ms.winner == 0:
                        w, l = a.name, b.name
                    elif ms.winner == 1:
                        w, l = b.name, a.name
                    if w:
                        records[w]["win"] += 1
                        records[l]["loss"] += 1
                        sa = 1.0 if w == a.name else 0.0
                    else:
                        records[a.name]["draw"] += 1
                        records[b.name]["draw"] += 1
                        sa = 0.5
                    ratings[a.name], ratings[b.name] = update_elo(
                        ratings[a.name], ratings[b.name], sa)
                matches.append({
                    "seed": seed, "p0": a.name, "p1": b.name,
                    "winner": w, "reason": ms.end_reason,
                    "duration": ms.duration, "flips": ms.flips,
                    "stats": ms.per_player, "valid": ms.valid,
                })
                if verbose:
                    outcome = (w or "draw") if ms.valid else f"invalid:{ms.end_reason}"
                    print(f"[{len(matches):3d}] {a.name} vs {b.name} -> "
                          f"{outcome} ({ms.end_reason}, {ms.duration}s)")
    ranking = build_ranking(agents, ratings, records, matches)
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            **config,
            "games_per_pair": games_per_pair,
            "seed": base_seed,
            "max_duration": max_duration,
            "protocol_version": PROTOCOL_VERSION,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        "ranking": ranking,
        "matches": matches,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n=== RANKING ===")
        for k, row in enumerate(ranking, 1):
            flag = "" if row["ranking_valid"] else "  [INVALID: parse_error_rate>1%]"
            print(f"{k}. {row['name']:24s} Elo {row['elo']:7.1f}  "
                  f"{row['win']}W-{row['loss']}L-{row['draw']}D{flag}")
    return result


# ============================== CLI ==============================


def main() -> None:
    p = argparse.ArgumentParser(description="スピード(トランプ)LLMアリーナ")
    p.add_argument("--mode", choices=["selftest", "ollama", "anthropic"], required=True)
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--hosts", nargs="*", default=[],
                   help="Ollamaホスト。モデルと同数指定で1対1対応、1つなら共有")
    p.add_argument("--games", type=int, default=4, help="ペアごとの対戦数(偶数推奨)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    if args.mode == "selftest":
        agents: list[Agent] = [
            HeuristicAgent("fast-bot", latency=0.05),
            HeuristicAgent("slow-bot", latency=0.30),
        ]
        config = {"mode": args.mode, "models": [a.name for a in agents], "hosts": []}
    elif args.mode == "ollama":
        if len(args.models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        hosts = args.hosts or ["http://localhost:11434"]
        if len(hosts) == 1:
            hosts = hosts * len(args.models)
        if len(hosts) != len(args.models):
            p.error("--hosts はモデルと同数か1つにしてください")
        # keep_aliveは対戦の最大時間+60秒はモデルを保持し、試合中のアンロードを防ぐ。#1
        agents = [
            OllamaAgent(m, host=h, keep_alive=args.max_duration + 60.0)
            for m, h in zip(args.models, hosts)
        ]
        config = {"mode": args.mode, "models": list(args.models), "hosts": list(hosts)}
    else:
        if len(args.models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        # api_keyなどの秘密情報はconfigに含めない。
        agents = [AnthropicAgent(m) for m in args.models]
        config = {"mode": args.mode, "models": list(args.models), "hosts": []}

    run_tournament(agents, config, games_per_pair=args.games, base_seed=args.seed,
                   max_duration=args.max_duration, out_path=args.out)


if __name__ == "__main__":
    main()
