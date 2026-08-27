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
import datetime as dt
import errno
import itertools
import json
import math
import os
import random
import re
import socket
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

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
RANKINGS_DIR = "rankings"
LOCK_STALE_SECONDS = 30 * 60
MACHINE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
RANKING_DETAIL_STATUSES = {"ok", "cyclic", "inconclusive", "skipped", "aborted"}


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
    runtime: str = "unknown"
    model_format: str = "unknown"
    quantization: str = "unknown"
    runtime_version: Optional[str] = None
    size_gb: Optional[float] = None

    @abstractmethod
    def decide(self, snapshot: Snapshot) -> dict:
        ...

    def player_metadata(self) -> dict[str, Any]:
        """Return the stable identity fields used by persistent rankings."""
        return {
            "model": self.name,
            "runtime": self.runtime,
            "format": self.model_format,
            "quantization": self.quantization,
            "runtime_version": self.runtime_version,
            "size_gb": self.size_gb,
        }

    def warmup(self) -> dict:
        """本番と同じAgent/接続先/system promptで1回推論し、結果は破棄する。#1

        OllamaAgent/AnthropicAgentのdecide()はネットワークエラーを例外にせず
        {"api_error": True} として返す設計のため、例外の有無だけでは接続不通を
        検知できない。api_errorをここでも失敗として扱い、passの擬似応答で
        ウォームアップ失敗を隠さないようにする。
        """
        t0 = time.monotonic()
        try:
            result = self._warmup_decide(WARMUP_SNAPSHOT)
            status = "failed" if result.get("api_error") else "ok"
        except Exception:
            status = "failed"
        return {"status": status, "duration": round(time.monotonic() - t0, 3)}

    def _warmup_decide(self, snapshot: Snapshot) -> dict:
        """Runtimes whose first call also loads the model override this."""
        return self.decide(snapshot)


class OllamaAgent(Agent):
    """ローカルLLM。model例: gpt-oss:20b, gemma3:4b。

    host をモデルごとに分ければ別GPUで公平にレイテンシ勝負できる。
    json-v1: format にJSON Schemaを渡してネイティブ構造化出力を使い、
    think=False で推論系モデルの思考出力を無効化して64トークンを回答に使わせる。#2
    keep_alive はモデルロード(コールドスタート)を試合中に発生させないため既定で無期限保持する。#1
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: float = 60.0, warmup_timeout: float = 900.0,
                 num_predict: int = MAX_TOKENS,
                 keep_alive: float | str = -1, model_format: str = "GGUF",
                 quantization: Optional[str] = None, runtime_version: Optional[str] = None,
                 size_gb: Optional[float] = None, num_ctx: int = 4096):
        self.name = model
        self.model = model
        self.runtime = "ollama"
        self.model_format = model_format
        self._quantization = quantization
        self.runtime_version = runtime_version
        self.size_gb = size_gb
        self.host_label = host
        self.url = host.rstrip("/") + "/api/chat"
        self.show_url = host.rstrip("/") + "/api/show"
        self.timeout = timeout
        self.warmup_timeout = warmup_timeout
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.session = requests.Session()

    @property
    def quantization(self) -> str:
        # player_id は model/runtime/format/quantization で選手を一意に決める。CLI の
        # --quantization は実行単位で1つしか渡せないため、量子化の違うモデルを同じ実行に
        # 並べると片方が誤った履歴に混ざる。指定がなければ ollama 本人に1度だけ聞く。
        if self._quantization is None:
            self._quantization = self._fetch_quantization()
        return self._quantization

    def _fetch_quantization(self) -> str:
        try:
            r = self.session.post(self.show_url, json={"model": self.model},
                                  timeout=self.timeout)
            r.raise_for_status()
            level = r.json().get("details", {}).get("quantization_level")
        except (requests.RequestException, ValueError):
            return "unknown"
        return level or "unknown"

    def _warmup_decide(self, snapshot: Snapshot) -> dict:
        # The warmup call is what loads the model, and a cold 20GB load takes
        # far longer than a move is allowed to. Sharing the move timeout made
        # every first match on a cold host fail as warmup_failed, which the
        # ladder then aborted on. The move budget is unchanged.
        return self.decide(snapshot, timeout=self.warmup_timeout)

    def decide(self, snapshot: Snapshot, timeout: Optional[float] = None) -> dict:
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
            "options": {
                "temperature": TEMPERATURE, "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            r = self.session.post(self.url, json=payload,
                                  timeout=timeout if timeout is not None else self.timeout)
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

    def __init__(self, model: str, max_tokens: int = MAX_TOKENS,
                 base_url: str = "https://api.anthropic.com", model_format: str = "api",
                 quantization: str = "none", runtime_version: Optional[str] = None,
                 size_gb: Optional[float] = None):
        self.name = model
        self.model = model
        self.runtime = "anthropic"
        self.model_format = model_format
        self.quantization = quantization
        self.runtime_version = runtime_version
        self.size_gb = size_gb
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.session = requests.Session()

    def decide(self, snapshot: Snapshot) -> dict:
        try:
            r = self.session.post(
                self.url,
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
        self.runtime = "selftest"
        self.model_format = "builtin"
        self.quantization = "none"
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
    # False means the result cannot be attributed to the models (for example,
    # a failed warmup or an API/connection error).  A game timeout is still a
    # valid model result: the remaining-card rule decides its outcome.
    valid: bool = True
    # Kept separate from per_player for backwards compatibility with the
    # schema-v1 results fixtures.  These counters distinguish a response
    # latency from a timeout/API error latency.
    request_metrics: list[dict] = field(default_factory=list)


def _player_loop(game: SpeedGame, agent: Agent, idx: int, stats: dict,
                 request_metrics: Optional[dict] = None) -> None:
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
        if request_metrics is not None:
            request_metrics["total_requests"] += 1
        if action.get("parse_error"):
            stats["parse_errors"] += 1
        no_response = action.get("api_error") or action.get("agent_error")
        if no_response:
            stats["api_errors"] += 1
        elif request_metrics is not None:
            request_metrics["latency_requests"] += 1
            request_metrics["total_latency_ms"] += latency * 1000.0

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
    request_metrics = [
        {"agent": agent_a.name, "total_requests": 0, "latency_requests": 0,
         "total_latency_ms": 0.0},
        {"agent": agent_b.name, "total_requests": 0, "latency_requests": 0,
         "total_latency_ms": 0.0},
    ]
    if not all(w["status"] == "ok" for w in warmups):
        # ウォームアップ失敗試合は実対戦を行わず、無効試合としてランキングへ投入しない。
        # passの擬似応答で隠さず、warmup_statusとして明示する。
        for s in stats:
            s["avg_latency"] = 0.0
        return MatchStats(winner=-1, end_reason="warmup_failed", duration=0.0,
                           flips=0, per_player=stats, valid=False,
                           request_metrics=request_metrics)

    game = SpeedGame(seed=seed, max_duration=max_duration)
    game.start()
    threads = [
        threading.Thread(target=_player_loop,
                         args=(game, agent_a, 0, stats[0], request_metrics[0]), daemon=True),
        threading.Thread(target=_player_loop,
                         args=(game, agent_b, 1, stats[1], request_metrics[1]), daemon=True),
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
    # A timeout is part of the game protocol and therefore an evaluable model
    # result.  Only API errors make the match ineligible for ranking; a model
    # that cannot finish in time is evaluated by the timeout/remaining-card
    # result instead of being silently discarded.
    valid = not any(s["api_errors"] for s in stats)
    return MatchStats(
        winner=game.winner if game.winner is not None else -1,
        end_reason=game.end_reason,
        duration=round(time.monotonic() - game.started_at, 2),
        flips=game.flips,
        per_player=stats,
        valid=valid,
        request_metrics=[
            {**m, "total_latency_ms": round(m["total_latency_ms"], 3)}
            for m in request_metrics
        ],
    )


# ============================== トーナメント & Elo ==============================


def update_elo(ra: float, rb: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1.0 - score_a) - (1.0 - ea))


def build_ranking(agents: list[Agent], ratings: dict[str, float],
                   records: dict[str, dict], matches: list[dict]) -> list[dict]:
    """json-v1: parse_errors/callsが10試合以上の集計で1%を超えるエージェントは
    ranking_valid=Falseとしてランキング値を無効表示にする。raw値(elo/戦績/エラー率)は隠さない。#2
    エラー率は無効試合(valid=False)の呼び出しも数える。判定したいのは「約束した形式で
    返せるか」であって、その試合が勝敗として成立したかではない。試合数の下限だけは
    有効試合で数える。永続ランキング(§10)と同じ規則。
    タイムアウトはゲーム結果として集計し、残り枚数の少ない側を勝者とする。
    """
    calls = {a.name: 0 for a in agents}
    parse_errors = {a.name: 0 for a in agents}
    games_played = {a.name: 0 for a in agents}
    for m in matches:
        for s in m["stats"]:
            calls[s["agent"]] += s["calls"]
            parse_errors[s["agent"]] += s["parse_errors"]
        if not m["valid"]:
            continue
        games_played[m["p0"]] += 1
        games_played[m["p1"]] += 1
    ranking = []
    for n, r in ratings.items():
        # 判定は丸める前の比率で行う。丸めた値で見ると 1.004% が 1.00% になって通る。
        # 永続ランキング(_refresh_player_derived)も同じく生の比率で判定している。
        raw_rate = parse_errors[n] / calls[n] if calls[n] else 0.0
        rate = round(raw_rate, 4)
        ranking_valid = not (
            games_played[n] >= PARSE_ERROR_MIN_GAMES and raw_rate > PARSE_ERROR_RATE_THRESHOLD
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


# ============================== 永続ランキング ==============================


class RankingError(RuntimeError):
    """ランキングを安全に更新できないときのエラー。"""


class RankingSchemaError(RankingError):
    """読み込めないランキング schema_version のエラー。"""


class RankingBusyError(RankingError):
    """別プロセスがランキングを使用中のエラー。"""


def validate_machine_id(machine_id: str) -> str:
    if not isinstance(machine_id, str) or not MACHINE_ID_RE.fullmatch(machine_id):
        raise ValueError("machine_id は [a-z0-9][a-z0-9._-]{0,63} で指定してください")
    return machine_id


def make_player_id(model: str, runtime: str, model_format: str,
                   quantization: str) -> str:
    values = (model, runtime, model_format, quantization)
    for value in values:
        if isinstance(value, str) and (len(value) > 200 or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
            raise ValueError("player ID の構成要素に制御文字または201文字以上の文字列があります")
    if any(not isinstance(v, str) or not v or "|" in v for v in values):
        raise ValueError("model/runtime/format/quantization は空でなく | を含められません")
    result = f"{model}|{runtime}|{model_format}-{quantization}"
    if len(result) > 200:
        raise ValueError("player_id が201文字以上です")
    return result


def _now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _default_machine_id() -> str:
    candidate = os.environ.get("SPEED_ARENA_MACHINE_ID", "")
    if not candidate:
        candidate = socket.gethostname().lower()
    candidate = re.sub(r"[^a-z0-9._-]", "-", candidate).strip("-")[:64]
    return candidate if MACHINE_ID_RE.fullmatch(candidate or "") else "local"


def _new_ranking(machine_id: str, machine: Optional[dict] = None) -> dict:
    validate_machine_id(machine_id)
    _clean_strings(machine or {}, "machine")
    return {
        "schema_version": SCHEMA_VERSION,
        "machine_id": machine_id,
        "machine": dict(machine or {}),
        "updated_at": _now_iso(),
        "transitivity_warning": False,
        "transitivity_detail": {
            "checked_at": None, "status": "skipped", "reason": "never_run",
        },
        "players": [],
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _clean_strings(value: Any, where: str = "value") -> None:
    if isinstance(value, str):
        if len(value) > 200 or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise ValueError(f"{where} に制御文字または201文字以上の文字列があります")
    elif isinstance(value, dict):
        for key, child in value.items():
            _clean_strings(key, f"{where}.key")
            _clean_strings(child, f"{where}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _clean_strings(child, f"{where}[{i}]")


def _validate_match_detail(detail: dict, player_ids: set[str]) -> None:
    if not isinstance(detail, dict):
        raise ValueError("transitivity_detail がオブジェクトではありません")
    status = detail.get("status")
    if status not in RANKING_DETAIL_STATUSES:
        raise ValueError("transitivity_detail.status が不正です")
    if detail.get("checked_at") is not None and not isinstance(detail.get("checked_at"), str):
        raise ValueError("transitivity_detail.checked_at が不正です")
    if isinstance(detail.get("checked_at"), str):
        try:
            dt.datetime.fromisoformat(detail["checked_at"])
        except ValueError as exc:
            raise ValueError("transitivity_detail.checked_at がISO8601ではありません") from exc
    matches = detail.get("matches")
    if status in {"ok", "cyclic", "inconclusive"}:
        if not isinstance(matches, list) or len(matches) != 3:
            raise ValueError("transitivity_detail.matches は3件必要です")
    elif status == "aborted":
        if not isinstance(detail.get("reason"), str):
            raise ValueError("aborted の reason が不正です")
        if not isinstance(matches, list) or len(matches) > 2:
            raise ValueError("aborted の matches は0〜2件で指定します")
        failed_pair = detail.get("failed_pair")
        if not isinstance(failed_pair, dict) or not {
            "a", "b"
        } <= set(failed_pair) or failed_pair["a"] not in player_ids or failed_pair["b"] not in player_ids:
            raise ValueError("aborted の failed_pair が不正です")
    else:
        if not isinstance(detail.get("reason"), str):
            raise ValueError("skipped の reason が不正です")
        if status == "skipped" and detail["reason"] not in {"players<3", "never_run"}:
            raise ValueError("skipped の reason が不正です")
    if status == "inconclusive" and detail.get("inconclusive") is not True:
        raise ValueError("inconclusive の detail.inconclusive がありません")
    if isinstance(matches, list):
        for match in matches:
            if not isinstance(match, dict) or not {"a", "b", "winner"} <= set(match):
                raise ValueError("transitivity_detail.matches の要素が不正です")
            if match["a"] not in player_ids or match["b"] not in player_ids:
                raise ValueError("transitivity_detail の選手 ID が不正です")
            if match["winner"] is not None and match["winner"] not in {match["a"], match["b"]}:
                raise ValueError("transitivity_detail.winner が不正です")


def validate_ranking_document(data: dict, machine_id: Optional[str] = None) -> dict:
    """ランキング JSON を検証し、壊れた入力は ValueError にする。"""
    if not isinstance(data, dict):
        raise ValueError("ランキング JSON のトップレベルがオブジェクトではありません")
    _clean_strings(data)
    if set(data) < {
        "schema_version", "machine_id", "machine", "updated_at",
        "transitivity_warning", "transitivity_detail", "players",
    }:
        raise ValueError("ランキング JSON の必須キーがありません")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RankingSchemaError(f"schema_version {data.get('schema_version')!r} は未対応です")
    actual_machine_id = data.get("machine_id")
    validate_machine_id(actual_machine_id)
    if machine_id is not None and actual_machine_id != machine_id:
        raise ValueError("machine_id がファイル名と一致しません")
    if not isinstance(data.get("machine"), dict) or not isinstance(data.get("updated_at"), str):
        raise ValueError("machine または updated_at が不正です")
    if not isinstance(data.get("transitivity_warning"), bool):
        raise ValueError("transitivity_warning が不正です")
    if not isinstance(data.get("players"), list):
        raise ValueError("players が配列ではありません")

    players = data["players"]
    player_ids: set[str] = set()
    entry_nos: set[int] = set()
    ranks: list[int] = []
    for index, player in enumerate(players, 1):
        if not isinstance(player, dict):
            raise ValueError("players の要素がオブジェクトではありません")
        required = {
            "player_id", "model", "runtime", "format", "quantization", "elo",
            "win", "loss", "draw", "matches", "rank", "total_requests",
            "latency_requests", "total_latency_ms", "parse_errors", "entry_no",
            "first_seen", "last_played",
        }
        if not required <= set(player):
            raise ValueError(f"player {index} の必須キーがありません")
        for key in ("model", "runtime", "format", "quantization"):
            value = player[key]
            if not isinstance(value, str) or not value or "|" in value:
                raise ValueError(f"player {index}.{key} が不正です")
        expected_id = make_player_id(
            player["model"], player["runtime"], player["format"], player["quantization"]
        )
        if player.get("player_id") != expected_id or player["player_id"] in player_ids:
            raise ValueError(f"player {index}.player_id が不正または重複しています")
        player_ids.add(player["player_id"])
        if not _finite_number(player.get("elo")) or not _finite_number(player.get("total_latency_ms")):
            raise ValueError(f"player {index} の数値が有限ではありません")
        if player["total_latency_ms"] < 0:
            raise ValueError(f"player {index}.total_latency_ms が負です")
        for key in ("win", "loss", "draw", "matches", "total_requests", "latency_requests", "parse_errors"):
            if not _nonnegative_int(player.get(key)):
                raise ValueError(f"player {index}.{key} が不正です")
        if player["latency_requests"] > player["total_requests"] or player["parse_errors"] > player["total_requests"]:
            raise ValueError(f"player {index} の母数が不正です")
        entry_no = player["entry_no"]
        if not isinstance(entry_no, int) or isinstance(entry_no, bool) or entry_no < 1 or entry_no in entry_nos:
            raise ValueError(f"player {index}.entry_no が不正または重複しています")
        entry_nos.add(entry_no)
        rank = player["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"player {index}.rank が不正です")
        ranks.append(rank)
        if not isinstance(player.get("first_seen"), str) or player.get("last_played") is not None and not isinstance(player.get("last_played"), str):
            raise ValueError(f"player {index} の日時が不正です")
        if "runtime_version" in player and player["runtime_version"] is not None and not isinstance(player["runtime_version"], str):
            raise ValueError(f"player {index}.runtime_version が不正です")
        if "size_gb" in player and player["size_gb"] is not None and not _finite_number(player["size_gb"]):
            raise ValueError(f"player {index}.size_gb が不正です")
        if "avg_latency_ms" in player and player["avg_latency_ms"] is not None and not _finite_number(player["avg_latency_ms"]):
            raise ValueError(f"player {index}.avg_latency_ms が不正です")
        if "parse_error_rate" in player and not _finite_number(player["parse_error_rate"]):
            raise ValueError(f"player {index}.parse_error_rate が不正です")
        if "ranking_valid" in player and not isinstance(player["ranking_valid"], bool):
            raise ValueError(f"player {index}.ranking_valid が不正です")
    if ranks != list(range(1, len(players) + 1)):
        raise ValueError("rank が1からの連番ではありません")
    _validate_match_detail(data["transitivity_detail"], player_ids)
    return data


def _backup_corrupt_ranking(path: Path, raw: bytes) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(f"{path}.corrupt-{stamp}")
    suffix = 1
    while True:
        candidate = base if suffix == 1 else Path(f"{base}-{suffix}")
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            suffix += 1
            continue
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(candidate, 0o644)
        except Exception:
            try:
                os.unlink(candidate)
            except OSError:
                pass
            raise
        os.unlink(path)
        return candidate


def load_ranking(path: str | Path, machine_id: Optional[str] = None,
                 machine: Optional[dict] = None, backup_corrupt: bool = True) -> dict:
    path = Path(path)
    expected_id = machine_id or path.stem
    validate_machine_id(expected_id)
    if not path.exists():
        return _new_ranking(expected_id, machine)
    raw = b""
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(data, dict):
            raise ValueError("トップレベルがオブジェクトではありません")
        # A newer/unknown schema is never backed up: an older executable must
        # not destroy data it does not understand.
        if data.get("schema_version") != SCHEMA_VERSION:
            if "schema_version" not in data:
                raise ValueError("schema_version がありません")
            raise RankingSchemaError(f"schema_version {data.get('schema_version')!r} は未対応です")
        validate_ranking_document(data, expected_id)
        for player in data["players"]:
            _refresh_player_derived(player)
        return data
    except RankingSchemaError:
        raise
    except Exception as exc:
        if backup_corrupt:
            try:
                backup = _backup_corrupt_ranking(path, raw)
            except Exception as backup_exc:
                raise RankingError(f"壊れたランキングの退避に失敗しました: {backup_exc}") from exc
            raise RankingError(f"壊れたランキングを {backup} に退避しました: {exc}") from exc
        raise RankingError(f"ランキング JSON が不正です: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_ranking(path: str | Path, data: dict) -> None:
    path = Path(path)
    for player in data.get("players", []):
        if isinstance(player, dict) and {"total_requests", "latency_requests", "total_latency_ms", "parse_errors", "matches"} <= set(player):
            _refresh_player_derived(player)
    validate_ranking_document(data, path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _lock_is_reclaimable(lock_path: Path) -> bool:
    try:
        raw = lock_path.read_text(encoding="utf-8")
        info = json.loads(raw)
        valid = (
            isinstance(info, dict)
            and isinstance(info.get("pid"), int) and not isinstance(info.get("pid"), bool)
            and info["pid"] >= 1
            and isinstance(info.get("host"), str) and bool(info["host"])
            and isinstance(info.get("acquired_at"), str)
        )
        if valid:
            # fromisoformat is deliberately used only for lock liveness; the
            # serialized value remains the original ISO-8601 string.
            dt.datetime.fromisoformat(info["acquired_at"])
        if not valid:
            raise ValueError("invalid lock payload")
    except Exception:
        try:
            old = time.time() - lock_path.stat().st_mtime > LOCK_STALE_SECONDS
        except OSError:
            return False
        if old:
            lock_path.unlink()
            return True
        raise RankingBusyError("ロックの中身が不正で、30分以内のため回収できません")

    if info["host"] != socket.gethostname():
        raise RankingBusyError("別ホストのランキングロックを回収できません")
    try:
        os.kill(info["pid"], 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            lock_path.unlink()
            return True
        if exc.errno == errno.EPERM:
            raise RankingBusyError("ロック所有プロセスは存在します(EPERM)") from exc
        raise RankingBusyError(f"ロック所有プロセスの確認に失敗しました: {exc}") from exc
    raise RankingBusyError("ランキングロックは使用中です")


@contextmanager
def ranking_lock(path: str | Path) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{path}.lock")
    fd: Optional[int] = None
    created = False
    acquired = False
    try:
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if not _lock_is_reclaimable(lock_path):
                raise RankingBusyError("ランキングロックを取得できません")
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        payload = json.dumps({
            "pid": os.getpid(), "host": socket.gethostname(), "acquired_at": _now_iso(),
        }, ensure_ascii=False, separators=(",", ":"))
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None
        acquired = True
        yield
    finally:
        if fd is not None:
            os.close(fd)
        if created and not acquired:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        if acquired:
            try:
                lock_path.unlink()
                _fsync_directory(path.parent)
            except FileNotFoundError:
                pass


def _agent_metadata(agent: Agent) -> tuple[str, dict]:
    if hasattr(agent, "player_metadata"):
        metadata = dict(agent.player_metadata())
    else:
        metadata = {
            "model": getattr(agent, "name", ""),
            "runtime": getattr(agent, "runtime", "unknown"),
            "format": getattr(agent, "model_format", "unknown"),
            "quantization": getattr(agent, "quantization", "unknown"),
            "runtime_version": getattr(agent, "runtime_version", None),
            "size_gb": getattr(agent, "size_gb", None),
        }
    _clean_strings(metadata, "player metadata")
    if metadata.get("runtime_version") is not None and not isinstance(metadata.get("runtime_version"), str):
        raise ValueError("runtime_version は文字列または null で指定してください")
    if metadata.get("size_gb") is not None and not _finite_number(metadata.get("size_gb")):
        raise ValueError("size_gb は有限の数値または null で指定してください")
    for key in ("model", "runtime", "format", "quantization"):
        if key not in metadata:
            raise ValueError(f"選手メタデータに {key} がありません")
    player_id = make_player_id(
        metadata["model"], metadata["runtime"], metadata["format"], metadata["quantization"]
    )
    return player_id, metadata


def _new_player(metadata: dict, player_id: str, entry_no: int, rank: int, now: str) -> dict:
    return {
        "player_id": player_id,
        "model": metadata["model"],
        "runtime": metadata["runtime"],
        "runtime_version": metadata.get("runtime_version"),
        "format": metadata["format"],
        "quantization": metadata["quantization"],
        "size_gb": metadata.get("size_gb"),
        "elo": 1000.0,
        "win": 0, "loss": 0, "draw": 0, "matches": 0,
        "rank": rank,
        "total_requests": 0, "latency_requests": 0, "total_latency_ms": 0.0,
        "parse_errors": 0,
        "avg_latency_ms": None, "parse_error_rate": 0.0, "ranking_valid": True,
        "entry_no": entry_no, "first_seen": now, "last_played": None,
    }


def _renumber(players: list[dict]) -> None:
    for rank, player in enumerate(players, 1):
        player["rank"] = rank


def apply_ladder_result(players: list[dict], player_a: str, player_b: str,
                        winner: Optional[str]) -> None:
    """有効試合の順位をはしご規則で更新する。Elo は別処理で更新する。"""
    if winner is None or winner not in {player_a, player_b} or player_a == player_b:
        return
    by_id = {p["player_id"]: i for i, p in enumerate(players)}
    if player_a not in by_id or player_b not in by_id:
        raise RankingError("はしご対戦の選手がランキングにありません")
    winner_index = by_id[winner]
    loser = player_b if winner == player_a else player_a
    loser_index = by_id[loser]
    if winner_index > loser_index:
        player = players.pop(winner_index)
        players.insert(loser_index, player)
        _renumber(players)


def _metrics_for_match(match: dict) -> list[dict]:
    metrics = match.get("request_metrics")
    if isinstance(metrics, list) and len(metrics) >= 2:
        return metrics
    fallback = []
    for stats in match.get("stats", []):
        calls = stats.get("calls", 0)
        api_errors = stats.get("api_errors", 0)
        latency_requests = max(0, calls - api_errors)
        average = stats.get("avg_latency")
        if _finite_number(average):
            total_latency_ms = float(average) * latency_requests * 1000.0
        else:
            total_latency_ms = (
                float(stats.get("think_time", 0.0)) * latency_requests / calls * 1000.0
                if calls else 0.0
            )
        fallback.append({
            "total_requests": calls,
            "latency_requests": latency_requests,
            "total_latency_ms": total_latency_ms,
        })
    return fallback


def _apply_attempts(state: dict, records: list[dict], start: int,
                    ladder: bool = False, move_rank: bool = True) -> bool:
    """Apply request-level metrics for every attempt and Elo/rank for valid ones."""
    valid = False
    for record in records[start:]:
        valid = _apply_persistent_match(state, record, ladder=ladder, move_rank=move_rank) or valid
    return valid


def _refresh_player_derived(player: dict) -> None:
    total = player["total_requests"]
    latency_requests = player["latency_requests"]
    player["avg_latency_ms"] = round(player["total_latency_ms"] / latency_requests, 3) if latency_requests else None
    raw_parse_error_rate = player["parse_errors"] / total if total else 0.0
    player["parse_error_rate"] = round(raw_parse_error_rate, 4)
    player["ranking_valid"] = not (
        player["matches"] >= PARSE_ERROR_MIN_GAMES
        and raw_parse_error_rate > PARSE_ERROR_RATE_THRESHOLD
    )


def _apply_persistent_match(state: dict, match: dict, ladder: bool = False,
                            move_rank: bool = True) -> bool:
    by_id = {p["player_id"]: p for p in state["players"]}
    p0_id, p1_id = match["p0"], match["p1"]
    if p0_id not in by_id or p1_id not in by_id:
        raise RankingError("対戦結果の選手がランキングにありません")
    # Verification matches are deliberately observational. They must not
    # alter Elo, ranks, request counters, or last_played.
    if match.get("verification"):
        return False
    metrics = _metrics_for_match(match)
    for player_id, metric in zip((p0_id, p1_id), metrics):
        player = by_id[player_id]
        total = metric.get("total_requests", 0)
        latency_requests = metric.get("latency_requests", 0)
        latency_ms = metric.get("total_latency_ms", 0.0)
        player["total_requests"] += int(total)
        player["latency_requests"] += int(latency_requests)
        player["total_latency_ms"] += float(latency_ms)
        # Stats are authoritative for parse errors, while preserving the old
        # results.json shape when request_metrics is absent.
    for index, stats in enumerate(match.get("stats", [])):
        if index >= 2:
            break
        player = by_id[(p0_id, p1_id)[index]]
        player["parse_errors"] += int(stats.get("parse_errors", 0))
        player["last_played"] = match.get("played_at") or player["last_played"]
    if not match.get("verification"):
        for player_id in (p0_id, p1_id):
            by_id[player_id]["last_played"] = match.get("played_at") or by_id[player_id]["last_played"]
    for player in by_id.values():
        _refresh_player_derived(player)
    if not match.get("valid"):
        return False
    p0, p1 = by_id[p0_id], by_id[p1_id]
    winner = match.get("winner")
    p0["matches"] += 1
    p1["matches"] += 1
    if winner == p0_id:
        p0["win"] += 1
        p1["loss"] += 1
        score_a = 1.0
    elif winner == p1_id:
        p1["win"] += 1
        p0["loss"] += 1
        score_a = 0.0
    else:
        p0["draw"] += 1
        p1["draw"] += 1
        score_a = 0.5
    p0["elo"], p1["elo"] = update_elo(float(p0["elo"]), float(p1["elo"]), score_a)
    if ladder and move_rank:
        apply_ladder_result(state["players"], p0_id, p1_id, winner)
    for player in (p0, p1):
        _refresh_player_derived(player)
    return True


def _match_record(agent_a: Agent, agent_b: Agent, player_a: str, player_b: str,
                  seed: int, result: MatchStats, verification: bool = False) -> dict:
    winner = None
    if result.valid:
        if result.winner == 0:
            winner = player_a
        elif result.winner == 1:
            winner = player_b
    record = {
        "seed": seed, "p0": player_a, "p1": player_b, "winner": winner,
        "reason": result.end_reason, "duration": result.duration,
        "flips": result.flips, "stats": result.per_player, "valid": result.valid,
        "request_metrics": result.request_metrics,
        "played_at": _now_iso(),
    }
    if not result.valid:
        record["invalid"] = result.end_reason
    if verification:
        record["verification"] = True
    return record


def _run_persistent_pair(agent_a: Agent, agent_b: Agent, player_a: str, player_b: str,
                         seed: int, max_duration: float, records: list[dict],
                         verification: bool = False) -> Optional[dict]:
    first = run_match(agent_a, agent_b, seed=seed, max_duration=max_duration)
    records.append(_match_record(agent_a, agent_b, player_a, player_b, seed, first, verification))
    if first.valid:
        return records[-1]
    retry_seed = seed + 1
    retry = run_match(agent_a, agent_b, seed=retry_seed, max_duration=max_duration)
    records.append(_match_record(agent_a, agent_b, player_a, player_b, retry_seed, retry, verification))
    return records[-1] if retry.valid else None


def _ladder_insertion_position(players: list[dict], compare: Any,
                               max_duration: float, seed: int,
                               records: list[dict], new_agent: Agent,
                               new_id: str, state: dict) -> tuple[int, int]:
    """Return (rank, next_seed). The caller cannot recompute the seed from the
    record count: a retried probe writes two records but consumes one slot."""
    # The new player is temporarily appended at the bottom so its counters
    # can be updated while probing. It is not part of the search range.
    original_n = len(players) - 1
    if original_n == 0:
        return 1, seed
    lo, hi = 1, original_n + 1
    midpoint = original_n > 10
    while lo < hi:
        position = (lo + hi) // 2 if midpoint else lo
        opponent = players[position - 1]
        try:
            opponent_agent = compare(opponent["player_id"])
        except (KeyError, LookupError) as exc:
            raise RankingError(
                f"はしご探索の相手 {opponent['player_id']} が今回の実行にありません"
            ) from exc
        attempt_start = len(records)
        result = _run_persistent_pair(
            new_agent, opponent_agent, new_id, opponent["player_id"], seed,
            max_duration, records,
        )
        if result is None:
            raise RankingError("はしご探索で無効試合が再試行後も続いたため中止しました")
        _apply_attempts(state, records, attempt_start, ladder=False, move_rank=False)
        if result["winner"] == new_id:
            hi = opponent["rank"]
        else:
            # A draw is not a win for the new player. Insert it after the
            # opponent so an equal result does not promote either player.
            lo = opponent["rank"] + 1
        seed += 2
    return lo, seed


def _register_new_player(state: dict, agent: Agent, player_id: str,
                         metadata: dict, rank: Optional[int] = None) -> dict:
    entry_no = max((p["entry_no"] for p in state["players"]), default=0) + 1
    player = _new_player(metadata, player_id, entry_no, len(state["players"]) + 1, _now_iso())
    state["players"].append(player)
    if rank is not None:
        state["players"].pop()
        state["players"].insert(rank - 1, player)
        _renumber(state["players"])
    return player


def _ranking_result(state: dict, matches: list[dict], config: dict) -> dict:
    ranking = []
    for player in state["players"]:
        _refresh_player_derived(player)
        ranking.append({
            "name": player["player_id"], "elo": round(player["elo"], 1),
            "win": player["win"], "loss": player["loss"], "draw": player["draw"],
            "parse_error_rate": player["parse_error_rate"],
            "ranking_valid": player["ranking_valid"],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "ranking": ranking,
        "matches": matches,
    }


def _sort_round_robin(state: dict, participant_ids: set[str], points: dict[str, float],
                      active_ids: set[str]) -> None:
    players = state["players"]
    active = [p for p in players if p["player_id"] in active_ids]
    inactive_participants = [p for p in players if p["player_id"] in participant_ids and p["player_id"] not in active_ids]
    inactive = [p for p in players if p["player_id"] not in participant_ids]
    active.sort(key=lambda p: (-points.get(p["player_id"], 0.0), -float(p["elo"]), p["entry_no"]))
    # A participant with no valid match is not allowed to displace the
    # established table. It stays in its existing relative position below the
    # players that did obtain a valid result.
    state["players"] = active + inactive_participants + inactive
    _renumber(state["players"])


def _update_transitivity(state: dict, agents_by_id: dict[str, Agent], records: list[dict],
                         max_duration: float, seed: int) -> int:
    players = state["players"]
    now = _now_iso()
    if len(players) < 3:
        state["transitivity_detail"] = {
            "checked_at": now, "status": "skipped", "reason": "players<3",
        }
        return seed
    top = players[:3]
    completed: list[dict] = []
    for offset, (a, b) in enumerate(itertools.combinations(top, 2)):
        if a["player_id"] not in agents_by_id or b["player_id"] not in agents_by_id:
            state["transitivity_detail"] = {
                "checked_at": now, "status": "aborted",
                "reason": "agent_not_provided",
                "failed_pair": {"a": a["player_id"], "b": b["player_id"]},
                "matches": completed,
            }
            return seed + offset * 2
        result = _run_persistent_pair(
            agents_by_id[a["player_id"]], agents_by_id[b["player_id"]],
            a["player_id"], b["player_id"], seed + offset * 2,
            max_duration, records, verification=True,
        )
        if result is None:
            state["transitivity_detail"] = {
                "checked_at": now, "status": "aborted",
                "reason": records[-1].get("reason", "invalid"),
                "failed_pair": {"a": a["player_id"], "b": b["player_id"]},
                "matches": completed,
            }
            return seed + 6
        completed.append({"a": result["p0"], "b": result["p1"], "winner": result["winner"]})
    if any(m["winner"] is None for m in completed):
        state["transitivity_detail"] = {
            "checked_at": now, "status": "inconclusive", "inconclusive": True,
            "matches": completed,
        }
        return seed + 6
    wins = {p["player_id"]: 0 for p in top}
    for match in completed:
        wins[match["winner"]] += 1
    if sorted(wins.values()) == [0, 1, 2]:
        state["transitivity_warning"] = False
        status = "ok"
    else:
        state["transitivity_warning"] = True
        status = "cyclic"
    state["transitivity_detail"] = {
        "checked_at": now, "status": status, "matches": completed,
    }
    return seed + 6


def run_persistent_tournament(agents: list[Agent], config: dict,
                              strategy: str = "ladder", games_per_pair: int = 4,
                              base_seed: int = 42, max_duration: float = 300.0,
                              out_path: str = "results.json", verbose: bool = True,
                              rankings_dir: str | Path = RANKINGS_DIR,
                              machine_id: Optional[str] = None,
                              machine: Optional[dict] = None,
                              verify_transitivity: bool = False) -> dict:
    if strategy not in {"ladder", "round-robin"}:
        raise ValueError("strategy は ladder または round-robin です")
    machine_id = validate_machine_id(machine_id or config.get("machine_id") or _default_machine_id())
    rankings_dir = Path(rankings_dir)
    path = rankings_dir / f"{machine_id}.json"
    agent_info = [_agent_metadata(agent) for agent in agents]
    ids = [item[0] for item in agent_info]
    if len(ids) != len(set(ids)):
        raise ValueError("同じ player_id の選手を同じ実行に複数指定できません")
    agents_by_id = dict(zip(ids, agents))
    config = {
        **config, "strategy": strategy, "machine_id": machine_id,
        "games_per_pair": games_per_pair, "seed": base_seed,
        "max_duration": max_duration, "protocol_version": PROTOCOL_VERSION,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
    }
    records: list[dict] = []
    with ranking_lock(path):
        state = load_ranking(path, machine_id, machine)
        if machine is not None and not state["machine"]:
            state["machine"] = dict(machine)
        existing_ids = {p["player_id"] for p in state["players"]}
        initial_existing_ids = set(existing_ids)
        if not existing_ids and machine is not None:
            state["machine"] = dict(machine)
        seed = base_seed

        if strategy == "ladder":
            # New players find their position one comparator at a time. The
            # rank is held fixed while probing; only the final insertion shifts
            # existing positions.
            for (player_id, metadata), agent in zip(agent_info, agents):
                if player_id in existing_ids:
                    continue
                player = _register_new_player(state, agent, player_id, metadata)
                rank, seed = _ladder_insertion_position(
                    state["players"], lambda pid: agents_by_id[pid],
                    max_duration, seed, records, agent, player_id, state,
                )
                state["players"].remove(player)
                state["players"].insert(rank - 1, player)
                _renumber(state["players"])
                existing_ids.add(player_id)
            # Existing players may be refreshed without forcing every other
            # machine's player into a match. Upsets move only the winner.
            initial_existing = [pid for pid in ids if pid in initial_existing_ids]
            for a_id, b_id in itertools.combinations(initial_existing, 2):
                for game_no in range(games_per_pair):
                    if game_no % 2:
                        p0_id, p1_id = b_id, a_id
                    else:
                        p0_id, p1_id = a_id, b_id
                    a, b = agents_by_id[p0_id], agents_by_id[p1_id]
                    attempt_start = len(records)
                    result = _run_persistent_pair(a, b, p0_id, p1_id, seed, max_duration, records)
                    seed += 2
                    _apply_attempts(state, records, attempt_start, ladder=True)
                    if result is None:
                        continue
        else:
            for player_id, metadata in agent_info:
                if player_id not in existing_ids:
                    _register_new_player(state, agents_by_id[player_id], player_id, metadata)
                    existing_ids.add(player_id)
            points = {pid: 0.0 for pid in ids}
            active_ids: set[str] = set()
            for a_id, b_id in itertools.combinations(ids, 2):
                for game_no in range(games_per_pair):
                    if game_no % 2:
                        p0_id, p1_id = b_id, a_id
                    else:
                        p0_id, p1_id = a_id, b_id
                    a, b = agents_by_id[p0_id], agents_by_id[p1_id]
                    attempt_start = len(records)
                    result = _run_persistent_pair(a, b, p0_id, p1_id, seed, max_duration, records)
                    seed += 2
                    _apply_attempts(state, records, attempt_start, ladder=False)
                    if result is None:
                        continue
                    if result.get("valid"):
                        active_ids.update((p0_id, p1_id))
                        if result["winner"] == p0_id:
                            points[p0_id] += 1.0
                        elif result["winner"] == p1_id:
                            points[p1_id] += 1.0
                        else:
                            points[p0_id] += 0.5
                            points[p1_id] += 0.5
            _sort_round_robin(state, set(ids), points, active_ids)

        if verify_transitivity:
            seed = _update_transitivity(state, agents_by_id, records, max_duration, seed)
        state["updated_at"] = _now_iso()
        _renumber(state["players"])
        for player in state["players"]:
            player["elo"] = round(float(player["elo"]), 1)
            player["total_latency_ms"] = round(float(player["total_latency_ms"]), 3)
            _refresh_player_derived(player)
        validate_ranking_document(state, machine_id)
        atomic_write_ranking(path, state)
    result = _ranking_result(state, records, config)
    result["ranking_path"] = str(path)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n=== RANKING ===")
        for player in state["players"]:
            print(f"{player['rank']}. {player['player_id']} Elo {player['elo']:7.1f} "
                  f"{player['win']}W-{player['loss']}L-{player['draw']}D")
    return result


def _machine_summary(machine: dict, updated_at: str) -> str:
    parts = []
    if machine.get("chip"):
        parts.append(str(machine["chip"]))
    if machine.get("cpu"):
        parts.append(str(machine["cpu"]))
    if machine.get("gpu"):
        parts.append(str(machine["gpu"]))
    if machine.get("memory_gb") is not None:
        parts.append(f"メモリ {machine['memory_gb']:g}GB" if isinstance(machine["memory_gb"], (int, float)) else f"メモリ {machine['memory_gb']}GB")
    if machine.get("benchmark_variant"):
        parts.append(f"条件 {machine['benchmark_variant']}")
    parts.append(f"最終更新 {updated_at[:10]}")
    return " / ".join(parts)


def render_ranking_markdown(data: dict) -> str:
    validate_ranking_document(data, data["machine_id"])
    # Derived values are recalculated from persisted counters, so a manually
    # edited/stale derived field can never change the rendered table.
    for player in data["players"]:
        _refresh_player_derived(player)
    display_name = data.get("machine", {}).get("display_name") or data["machine_id"]
    lines = [
        f"# {display_name} ランキング", "",
        _machine_summary(data.get("machine", {}), data["updated_at"]), "",
        "| # | モデル | 実行系 | 量子化 | Elo | 戦績 | 平均応答 |",
        "|---|--------|--------|--------|-----|------|----------|",
    ]
    for player in data["players"]:
        avg = player.get("avg_latency_ms")
        if avg is None:
            avg_text = "—"
        elif float(avg).is_integer():
            avg_text = f"{int(avg)}ms"
        else:
            avg_text = f"{avg:g}ms"
        lines.append(
            f"| {player['rank']} | `{player['model']}` | `{player['runtime']}` | "
            f"`{player['quantization']}` | {float(player['elo']):.1f} | "
            f"`{player['win']}W-{player['loss']}L-{player['draw']}D` | `{avg_text}` |"
        )
    return "\n".join(lines) + "\n"


def render_rankings(rankings_dir: str | Path = RANKINGS_DIR) -> list[Path]:
    directory = Path(rankings_dir)
    if not directory.exists():
        return []
    generated = []
    for path in sorted(directory.glob("*.json")):
        machine_id = path.stem
        validate_machine_id(machine_id)
        data = load_ranking(path, machine_id, backup_corrupt=False)
        output = directory / f"{machine_id}.md"
        output.write_text(render_ranking_markdown(data), encoding="utf-8")
        generated.append(output)
    return generated


# ============================== CLI ==============================


def main() -> None:
    p = argparse.ArgumentParser(description="スピード(トランプ)LLMアリーナ")
    p.add_argument("--mode", choices=["ladder", "round-robin", "selftest", "ollama", "anthropic"],
                   default="ladder", help="ランキング方式。旧 selftest/ollama/anthropic も互換維持")
    p.add_argument("--runtime", "--provider", dest="runtime",
                   choices=["selftest", "ollama", "anthropic"], default="ollama",
                   help="ladder/round-robin で使う実行系")
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--hosts", nargs="*", default=[],
                   help="Ollamaホスト。モデルと同数指定で1対1対応、1つなら共有")
    p.add_argument("--format", dest="model_format", default=None)
    p.add_argument("--quantization", default=None)
    p.add_argument("--runtime-version", default=None)
    p.add_argument("--size-gb", type=float, default=None)
    p.add_argument("--machine-id", default=None)
    p.add_argument("--rankings-dir", default=RANKINGS_DIR)
    p.add_argument("--machine", default=None,
                   help="マシン情報の JSON (例: '{\"gpu\":\"NVIDIA GB10\",\"memory_gb\":121}')")
    p.add_argument("--verify-transitivity", action="store_true")
    p.add_argument("--render-rankings", action="store_true")
    p.add_argument("--games", type=int, default=4, help="ペアごとの対戦数(偶数推奨)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--num-ctx", type=int, default=4096,
                   help="Ollamaのコンテキスト長。ゲーム入力には4096で十分")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    if args.render_rankings:
        for generated in render_rankings(args.rankings_dir):
            print(f"generated {generated}")
        return

    machine = None
    if args.machine:
        try:
            machine = json.loads(args.machine)
        except json.JSONDecodeError as exc:
            p.error(f"--machine は JSON で指定してください: {exc}")
        if not isinstance(machine, dict):
            p.error("--machine は JSON オブジェクトで指定してください")

    if args.mode == "selftest":
        agents: list[Agent] = [
            HeuristicAgent("fast-bot", latency=0.05),
            HeuristicAgent("slow-bot", latency=0.30),
        ]
        config = {"mode": args.mode, "models": [a.name for a in agents], "hosts": []}
    elif args.mode == "ollama" or (args.mode in {"ladder", "round-robin"} and args.runtime == "ollama"):
        if len(args.models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        hosts = args.hosts or ["http://localhost:11434"]
        if len(hosts) == 1:
            hosts = hosts * len(args.models)
        if len(hosts) != len(args.models):
            p.error("--hosts はモデルと同数か1つにしてください")
        # keep_aliveは対戦の最大時間+60秒はモデルを保持し、試合中のアンロードを防ぐ。#1
        agents = [
            OllamaAgent(m, host=h, keep_alive=args.max_duration + 60.0,
                        model_format=args.model_format or "GGUF",
                        quantization=args.quantization,
                        runtime_version=args.runtime_version, size_gb=args.size_gb,
                        num_ctx=args.num_ctx)
            for m, h in zip(args.models, hosts)
        ]
        config = {"mode": args.mode, "models": list(args.models), "hosts": list(hosts)}
    elif args.mode == "anthropic" or (args.mode in {"ladder", "round-robin"} and args.runtime == "anthropic"):
        if len(args.models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        # api_keyなどの秘密情報はconfigに含めない。
        agents = [
            AnthropicAgent(m, model_format=args.model_format or "api",
                           quantization=args.quantization or "none",
                           runtime_version=args.runtime_version, size_gb=args.size_gb)
            for m in args.models
        ]
        config = {"mode": args.mode, "models": list(args.models), "hosts": []}
    else:
        models = args.models or ["fast-bot", "slow-bot"]
        if len(models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        agents = [HeuristicAgent(m, latency=0.05 + i * 0.25) for i, m in enumerate(models)]
        config = {"mode": args.mode, "models": list(models), "hosts": []}

    if args.mode in {"ladder", "round-robin"}:
        run_persistent_tournament(
            agents, {**config, "mode": args.runtime, "strategy": args.mode}, strategy=args.mode, games_per_pair=args.games,
            base_seed=args.seed, max_duration=args.max_duration, out_path=args.out,
            rankings_dir=args.rankings_dir, machine_id=args.machine_id or _default_machine_id(),
            machine=machine, verify_transitivity=args.verify_transitivity,
        )
    else:
        run_tournament(agents, config, games_per_pair=args.games, base_seed=args.seed,
                       max_duration=args.max_duration, out_path=args.out)


if __name__ == "__main__":
    main()
