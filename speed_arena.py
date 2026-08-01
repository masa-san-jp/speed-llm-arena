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
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests

HAND_SIZE = 4
RANK_MIN, RANK_MAX = 1, 13


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


_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action(text: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m:
        return {"action": "pass", "parse_error": True}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "pass", "parse_error": True}
    if obj.get("action") == "play":
        try:
            return {"action": "play", "card": int(obj["card"]), "pile": int(obj["pile"])}
        except (KeyError, TypeError, ValueError):
            return {"action": "pass", "parse_error": True}
    return {"action": "pass"}


class Agent(ABC):
    name: str

    @abstractmethod
    def decide(self, snapshot: Snapshot) -> dict:
        ...


class OllamaAgent(Agent):
    """ローカルLLM。model例: gpt-oss:20b, gemma3:4b。

    host をモデルごとに分ければ別GPUで公平にレイテンシ勝負できる。
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: float = 60.0, num_predict: int = 64):
        self.name = model
        self.model = model
        self.url = host.rstrip("/") + "/api/chat"
        self.timeout = timeout
        self.num_predict = num_predict
        self.session = requests.Session()

    def decide(self, snapshot: Snapshot) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(snapshot)},
            ],
            "options": {"temperature": 0.0, "num_predict": self.num_predict},
        }
        try:
            r = self.session.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
        except requests.RequestException:
            return {"action": "pass", "api_error": True}
        return parse_action(content)


class AnthropicAgent(Agent):
    """要 ANTHROPIC_API_KEY 環境変数。動作確認やクラウドモデルとの比較用。"""

    def __init__(self, model: str, max_tokens: int = 64):
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
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": build_user_prompt(snapshot)}],
                },
                timeout=60.0,
            )
            r.raise_for_status()
            content = "".join(
                b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text"
            )
        except requests.RequestException:
            return {"action": "pass", "api_error": True}
        return parse_action(content)


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


# ============================== 対戦ランナー ==============================


@dataclass
class MatchStats:
    winner: int
    end_reason: str
    duration: float
    flips: int
    per_player: list[dict] = field(default_factory=list)


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


def run_match(agent_a: Agent, agent_b: Agent, seed: int,
              max_duration: float = 300.0) -> MatchStats:
    game = SpeedGame(seed=seed, max_duration=max_duration)
    stats = [
        {"agent": agent_a.name, "calls": 0, "plays": 0, "invalid_moves": 0,
         "parse_errors": 0, "api_errors": 0, "think_time": 0.0},
        {"agent": agent_b.name, "calls": 0, "plays": 0, "invalid_moves": 0,
         "parse_errors": 0, "api_errors": 0, "think_time": 0.0},
    ]
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


def run_tournament(agents: list[Agent], games_per_pair: int = 4,
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
                if ms.winner == 0:
                    w, l = a.name, b.name
                elif ms.winner == 1:
                    w, l = b.name, a.name
                else:
                    w = l = None
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
                    "stats": ms.per_player,
                })
                if verbose:
                    print(f"[{len(matches):3d}] {a.name} vs {b.name} -> "
                          f"{w or 'draw'} ({ms.end_reason}, {ms.duration}s)")
    ranking = sorted(
        ({"name": n, "elo": round(r, 1), **records[n]} for n, r in ratings.items()),
        key=lambda x: -x["elo"],
    )
    result = {"ranking": ranking, "matches": matches}
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n=== RANKING ===")
        for k, row in enumerate(ranking, 1):
            print(f"{k}. {row['name']:24s} Elo {row['elo']:7.1f}  "
                  f"{row['win']}W-{row['loss']}L-{row['draw']}D")
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
    elif args.mode == "ollama":
        if len(args.models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        hosts = args.hosts or ["http://localhost:11434"]
        if len(hosts) == 1:
            hosts = hosts * len(args.models)
        if len(hosts) != len(args.models):
            p.error("--hosts はモデルと同数か1つにしてください")
        agents = [OllamaAgent(m, host=h) for m, h in zip(args.models, hosts)]
    else:
        if len(args.models) < 2:
            p.error("--models にモデルを2つ以上指定してください")
        agents = [AnthropicAgent(m) for m in args.models]

    run_tournament(agents, games_per_pair=args.games, base_seed=args.seed,
                   max_duration=args.max_duration, out_path=args.out)


if __name__ == "__main__":
    main()
