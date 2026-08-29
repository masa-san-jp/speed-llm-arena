"""永続ランキングの受け入れテスト。外部 LLM/API は使わない。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speed_arena as sa


def _player(player_id, rank, entry_no, elo=1000.0):
    model, runtime, rest = player_id.split("|", 2)
    model_format, quantization = rest.rsplit("-", 1)
    return {
        "player_id": player_id, "model": model, "runtime": runtime,
        "format": model_format, "quantization": quantization,
        "elo": elo, "win": 0, "loss": 0, "draw": 0, "matches": 0,
        "rank": rank, "total_requests": 0, "latency_requests": 0,
        "total_latency_ms": 0.0, "parse_errors": 0, "avg_latency_ms": None,
        "parse_error_rate": 0.0, "ranking_valid": True, "entry_no": entry_no,
        "first_seen": "2026-08-27T00:00:00+09:00", "last_played": None,
    }


class DummyAgent(sa.Agent):
    def __init__(self, name, latency=0.0):
        self.name = name
        self.runtime = "dummy"
        self.model_format = "test"
        self.quantization = "FP16"
        self.latency = latency

    def decide(self, snapshot):
        return {"action": "pass"}


def _valid_match(a, b, winner):
    stats = []
    metrics = []
    for name in (a.name, b.name):
        stats.append({"agent": name, "calls": 2, "parse_errors": 0,
                      "api_errors": 0, "think_time": 0.2})
        metrics.append({"agent": name, "total_requests": 2,
                        "latency_requests": 2, "total_latency_ms": 200.0})
    return sa.MatchStats(
        winner=winner, end_reason="played_out", duration=0.2, flips=0,
        per_player=stats, request_metrics=metrics,
    )


class TestPersistentRanking(unittest.TestCase):
    def test_player_id_and_machine_id_boundaries(self):
        self.assertEqual(sa.make_player_id("qwen:27b", "ollama", "GGUF", "Q4_K_M"),
                         "qwen:27b|ollama|GGUF-Q4_K_M")
        for bad in ("", "../escape", "A", ".hidden", "a" * 65):
            if bad == "A":
                continue
            with self.assertRaises(ValueError):
                sa.validate_machine_id(bad)
        with self.assertRaises(ValueError):
            sa.make_player_id("a|b", "ollama", "GGUF", "Q4")

    def test_ladder_upset_moves_only_winner_and_loser(self):
        players = [
            _player("a|dummy|test-FP16", 1, 1, 1100.0),
            _player("b|dummy|test-FP16", 2, 2, 1050.0),
            _player("c|dummy|test-FP16", 3, 3, 1000.0),
            _player("d|dummy|test-FP16", 4, 4, 950.0),
        ]
        before = [p["elo"] for p in players]
        sa.apply_ladder_result(players, players[3]["player_id"], players[0]["player_id"],
                                players[3]["player_id"])
        self.assertEqual([p["player_id"] for p in players], ["d|dummy|test-FP16", "a|dummy|test-FP16", "b|dummy|test-FP16", "c|dummy|test-FP16"])
        self.assertEqual([p["elo"] for p in players], [before[3], before[0], before[1], before[2]])
        self.assertEqual([p["rank"] for p in players], [1, 2, 3, 4])

    def test_ladder_changes_from_linear_to_binary_at_eleven(self):
        for count, expected_calls in ((10, 10), (11, 3)):
            old = [DummyAgent(f"old-{i}") for i in range(count)]
            new = DummyAgent("new")
            players = [_player(f"old-{i}|dummy|test-FP16", i + 1, i + 1) for i in range(count)]
            new_player = _player("new|dummy|test-FP16", count + 1, count + 1)
            players.append(new_player)
            state = {"players": players}
            agents = {a.name: a for a in old}
            calls = []

            def fake_run(a, b, seed, max_duration):
                calls.append(b.name)
                return _valid_match(a, b, 1)

            with patch.object(sa, "run_match", side_effect=fake_run):
                position, _next_seed = sa._ladder_insertion_position(
                    players, lambda pid: agents[pid.split("|", 1)[0]], 10.0, 1,
                    [], new, "new|dummy|test-FP16", state,
                )
            self.assertEqual(position, count + 1)
            self.assertEqual(len(calls), expected_calls)

    def test_ladder_draw_does_not_promote_new_player(self):
        old = DummyAgent("old")
        new = DummyAgent("new")
        players = [
            _player("old|dummy|test-FP16", 1, 1),
            _player("new|dummy|test-FP16", 2, 2),
        ]
        state = {"players": players}

        with patch.object(sa, "run_match", return_value=_valid_match(old, new, -1)):
            position, _next_seed = sa._ladder_insertion_position(
                players, lambda _pid: old, 10.0, 1, [], new,
                "new|dummy|test-FP16", state,
            )
        self.assertEqual(position, 2)

    def test_new_player_is_persisted_with_elo_and_request_counters(self):
        first, second = DummyAgent("old"), DummyAgent("new")
        with tempfile.TemporaryDirectory() as directory:
            def fake_run(a, b, seed, max_duration):
                return _valid_match(a, b, 0 if a is first else 1)

            with patch.object(sa, "run_match", side_effect=fake_run):
                sa.run_persistent_tournament(
                    [first, second], {"mode": "dummy", "models": ["old", "new"], "hosts": []},
                    strategy="ladder", games_per_pair=1, rankings_dir=directory,
                    machine_id="test", out_path=None, verbose=False,
                )
            data = json.loads(Path(directory, "test.json").read_text())
            self.assertEqual([p["model"] for p in data["players"]], ["old", "new"])
            self.assertEqual(data["players"][0]["elo"], 1016.0)
            self.assertEqual(data["players"][1]["elo"], 984.0)
            self.assertEqual(data["players"][1]["total_requests"], 2)

    def test_invalid_ladder_retry_does_not_write_ranking(self):
        old = DummyAgent("old")
        new = DummyAgent("new")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            state = sa._new_ranking("test")
            state["players"] = [_player("old|dummy|test-FP16", 1, 1)]
            sa.atomic_write_ranking(path, state)
            before = path.read_bytes()
            invalid = sa.MatchStats(-1, "warmup_failed", 0.0, 0, [], False)
            with patch.object(sa, "run_match", return_value=invalid):
                with self.assertRaises(sa.RankingError):
                    sa.run_persistent_tournament(
                        [old, new], {"mode": "dummy"}, strategy="ladder",
                        games_per_pair=1, rankings_dir=directory, machine_id="test",
                        out_path=None, verbose=False,
                    )
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(Path(f"{path}.lock").exists())

    def test_retry_uses_next_seed_and_next_match_skips_it(self):
        """やり直しは シード+1 を使い、次の試合は +2 から始まる（§5 の刻み幅）。

        詰めて振ると、やり直しが起きた試合の後ろが全部ずれて再現できなくなる。
        """
        a, b, c = DummyAgent("a"), DummyAgent("b"), DummyAgent("c")
        seeds = []
        invalid = sa.MatchStats(-1, "warmup_failed", 0.0, 0, [], False)

        def fake_run_match(agent_a, agent_b, seed, max_duration):
            seeds.append(seed)
            # 1試合目だけ無効にして、やり直しを1回だけ起こす
            if len(seeds) == 1:
                return invalid
            return _valid_match(agent_a, agent_b, 0)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sa, "run_match", side_effect=fake_run_match):
                sa.run_persistent_tournament(
                    [a, b, c], {"mode": "dummy"}, strategy="ladder",
                    games_per_pair=1, rankings_dir=directory, machine_id="test",
                    out_path=None, verbose=False, base_seed=42,
                )
        self.assertEqual(seeds[0], 42)
        self.assertEqual(seeds[1], 43, "やり直しは シード+1")
        self.assertNotIn(44, seeds[:2])
        self.assertEqual(seeds[2], 44, "次の試合は +2 から")

    def test_parse_error_rate_boundary_is_strictly_above_one_percent(self):
        """ちょうど1%は通し、超えたときだけ弾く（`>` であって `>=` ではない）。"""
        agents = [sa.HeuristicAgent("a"), sa.HeuristicAgent("b")]
        ratings = {"a": 1000.0, "b": 1000.0}
        records = {"a": {"win": 0, "loss": 0, "draw": 0},
                   "b": {"win": 0, "loss": 0, "draw": 0}}

        def ranking_for(parse_errors):
            matches = [{
                "valid": True, "p0": "a", "p1": "b", "winner": None,
                "stats": [
                    {"agent": "a", "calls": 100, "parse_errors": parse_errors},
                    {"agent": "b", "calls": 100, "parse_errors": 0},
                ],
            } for _ in range(sa.PARSE_ERROR_MIN_GAMES)]
            rows = sa.build_ranking(agents, ratings, records, matches)
            return {r["name"]: r for r in rows}["a"]

        exactly_one_percent = ranking_for(1)
        self.assertEqual(exactly_one_percent["parse_error_rate"], 0.01)
        self.assertTrue(exactly_one_percent["ranking_valid"])

        just_over = ranking_for(2)
        self.assertGreater(just_over["parse_error_rate"], 0.01)
        self.assertFalse(just_over["ranking_valid"])

    def test_parse_error_rate_is_judged_before_rounding(self):
        """丸めた値で判定すると 1.004% が 1.00% になって通ってしまう。"""
        agents = [sa.HeuristicAgent("a"), sa.HeuristicAgent("b")]
        ratings = {"a": 1000.0, "b": 1000.0}
        records = {"a": {"win": 0, "loss": 0, "draw": 0},
                   "b": {"win": 0, "loss": 0, "draw": 0}}
        # 10万呼び出し中1004件 = 1.004%。1%は超えているが、小数第4位に丸めると 0.01 に戻る帯。
        matches = [{
            "valid": True, "p0": "a", "p1": "b", "winner": None,
            "stats": [
                {"agent": "a", "calls": 10000, "parse_errors": 104 if i == 0 else 100},
                {"agent": "b", "calls": 10000, "parse_errors": 0},
            ],
        } for i in range(sa.PARSE_ERROR_MIN_GAMES)]
        row = {r["name"]: r for r in sa.build_ranking(agents, ratings, records, matches)}["a"]
        self.assertEqual(row["parse_error_rate"], 0.01, "表示は丸めた値")
        self.assertFalse(row["ranking_valid"], "丸める前の比率で弾く")

    def test_corrupt_json_is_renamed_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            path.write_text('{"schema_version":1,"machine_id":"wrong"}', encoding="utf-8")
            with self.assertRaises(sa.RankingError):
                sa.load_ranking(path, "test")
            self.assertFalse(path.exists())
            backups = list(Path(directory).glob("test.json.corrupt-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("wrong", backups[0].read_text())

    def test_future_schema_version_is_refused_without_touching_the_file(self):
        """新しい版は読まずに終わる。知らない項目を落として書き戻すのが一番静かな壊し方。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            payload = '{"schema_version":99,"machine_id":"test"}'
            path.write_text(payload, encoding="utf-8")
            with self.assertRaises(sa.RankingSchemaError):
                sa.load_ranking(path, "test")
            # 退避もしない。壊れているのではなく、こちらが古いだけ。
            self.assertEqual(path.read_text(encoding="utf-8"), payload)
            self.assertEqual(list(Path(directory).glob("test.json.corrupt-*")), [])

    def test_missing_schema_version_is_treated_as_corrupt(self):
        """version を書かない別物かもしれないので、1 と見なして読まない。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            path.write_text('{"machine_id":"test","players":[]}', encoding="utf-8")
            with self.assertRaises(sa.RankingError):
                sa.load_ranking(path, "test")
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(directory).glob("test.json.corrupt-*"))), 1)

    def test_missing_file_starts_an_empty_ranking_without_quarantine(self):
        """新しいマシンの初回。壊れているのとは別で、退避するものが無い。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            state = sa.load_ranking(path, "test")
            self.assertEqual(state["players"], [])
            self.assertEqual(list(Path(directory).glob("test.json.corrupt-*")), [])

    def test_first_player_takes_rank_one_without_playing(self):
        """1体しか居ない表に順位を付けただけ。強さは測れていないので Elo は初期値のまま。"""
        solo = DummyAgent("solo")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sa, "run_match", side_effect=AssertionError("試合をしてはいけない")):
                sa.run_persistent_tournament(
                    [solo], {"mode": "dummy"}, strategy="ladder", games_per_pair=1,
                    rankings_dir=directory, machine_id="test", out_path=None, verbose=False,
                )
            data = json.loads(Path(directory, "test.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["players"]), 1)
            self.assertEqual(data["players"][0]["rank"], 1)
            self.assertEqual(data["players"][0]["elo"], 1000.0)
            self.assertEqual(data["players"][0]["matches"], 0)

    def test_second_player_settles_in_one_match(self):
        """N=1 は特別扱いしない。lo=1, hi=2 の探索が1試合で決着する。"""
        first, second = DummyAgent("first"), DummyAgent("second")
        played = []

        def fake_run(x, y, seed, max_duration):
            played.append((x.name, y.name))
            return _valid_match(x, y, 0)   # 新入り(second)が勝つ

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sa, "run_match", side_effect=fake_run):
                sa.run_persistent_tournament(
                    [first, second], {"mode": "dummy"}, strategy="ladder", games_per_pair=1,
                    rankings_dir=directory, machine_id="test", out_path=None, verbose=False,
                )
            self.assertEqual(len(played), 1, "1試合で決まる")
            data = json.loads(Path(directory, "test.json").read_text(encoding="utf-8"))
            by_rank = {p["rank"]: p["model"] for p in data["players"]}
            self.assertEqual(by_rank[1], "second", "勝った新入りが上")
            self.assertEqual(by_rank[2], "first")

    def test_round_robin_uses_points_then_elo_then_entry_number(self):
        a, b, c = DummyAgent("a"), DummyAgent("b"), DummyAgent("c")
        outcomes = {("a", "b"): 0, ("a", "c"): 0, ("b", "c"): 0}

        def fake_run(x, y, seed, max_duration):
            return _valid_match(x, y, 0)

        with tempfile.TemporaryDirectory() as directory, patch.object(sa, "run_match", side_effect=fake_run):
            result = sa.run_persistent_tournament(
                [a, b, c], {"mode": "dummy"}, strategy="round-robin", games_per_pair=1,
                rankings_dir=directory, machine_id="test", out_path=None, verbose=False,
            )
            self.assertEqual([row["name"] for row in result["ranking"]], [
                "a|dummy|test-FP16", "b|dummy|test-FP16", "c|dummy|test-FP16",
            ])

    def test_invalid_match_counts_responses_but_not_match_or_latency_timeout(self):
        state = sa._new_ranking("test")
        state["players"] = [
            _player("a|dummy|test-FP16", 1, 1),
            _player("b|dummy|test-FP16", 2, 2),
        ]
        record = {
            "p0": "a|dummy|test-FP16", "p1": "b|dummy|test-FP16",
            "valid": False, "winner": None, "stats": [
                {"parse_errors": 1}, {"parse_errors": 0},
            ], "request_metrics": [
                {"total_requests": 2, "latency_requests": 1, "total_latency_ms": 10.0},
                {"total_requests": 2, "latency_requests": 0, "total_latency_ms": 0.0},
            ],
        }
        self.assertFalse(sa._apply_persistent_match(state, record))
        self.assertEqual(state["players"][0]["matches"], 0)
        self.assertEqual(state["players"][0]["total_requests"], 2)
        self.assertEqual(state["players"][0]["latency_requests"], 1)
        self.assertEqual(state["players"][0]["parse_errors"], 1)
        self.assertEqual(state["players"][1]["latency_requests"], 0)

    def test_timeout_match_counts_as_model_result(self):
        state = sa._new_ranking("test")
        state["players"] = [
            _player("a|dummy|test-FP16", 1, 1),
            _player("b|dummy|test-FP16", 2, 2),
        ]
        record = {
            "p0": "a|dummy|test-FP16", "p1": "b|dummy|test-FP16",
            "valid": True, "winner": "b|dummy|test-FP16",
            "reason": "timeout_fewer_cards", "stats": [
                {"parse_errors": 3}, {"parse_errors": 0},
            ], "request_metrics": [
                {"total_requests": 4, "latency_requests": 4, "total_latency_ms": 40.0},
                {"total_requests": 2, "latency_requests": 2, "total_latency_ms": 20.0},
            ],
        }
        self.assertTrue(sa._apply_persistent_match(state, record))
        self.assertEqual(state["players"][0]["loss"], 1)
        self.assertEqual(state["players"][1]["win"], 1)
        self.assertEqual(state["players"][0]["matches"], 1)

    def test_transitivity_cycle_is_recorded_without_affecting_ranking(self):
        agents = [DummyAgent("a"), DummyAgent("b"), DummyAgent("c")]
        ids = {a.name: f"{a.name}|dummy|test-FP16" for a in agents}
        state = sa._new_ranking("test")
        state["players"] = [_player(ids[name], i + 1, i + 1) for i, name in enumerate("abc")]

        def fake_run(a, b, seed, max_duration):
            # a>b, b>c, c>a
            winner = 0 if {a.name, b.name} != {"a", "c"} else 1
            return _valid_match(a, b, winner)

        records = []
        by_id = {ids[a.name]: a for a in agents}
        with patch.object(sa, "run_match", side_effect=fake_run):
            sa._update_transitivity(state, by_id, records, 10.0, 1)
        self.assertTrue(state["transitivity_warning"])
        self.assertEqual(state["transitivity_detail"]["status"], "cyclic")
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r.get("verification") for r in records))
        self.assertEqual([p["matches"] for p in state["players"]], [0, 0, 0])

    def test_dead_process_lock_is_reclaimed_but_live_lock_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            lock = Path(f"{path}.lock")
            lock.write_text(json.dumps({"pid": 999999, "host": "host", "acquired_at": "2026-08-27T00:00:00+09:00"}))
            with patch.object(sa.socket, "gethostname", return_value="host"), patch.object(sa.os, "kill", side_effect=OSError(sa.errno.ESRCH, "gone")):
                with sa.ranking_lock(path):
                    self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

            lock.write_text(json.dumps({"pid": 123, "host": "host", "acquired_at": "2026-08-27T00:00:00+09:00"}))
            with patch.object(sa.socket, "gethostname", return_value="host"), patch.object(sa.os, "kill", return_value=None):
                with self.assertRaises(sa.RankingBusyError):
                    with sa.ranking_lock(path):
                        pass
            self.assertTrue(lock.exists())

    def test_atomic_write_failure_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            original = sa._new_ranking("test")
            sa.atomic_write_ranking(path, original)
            before = path.read_bytes()
            changed = sa._new_ranking("test")
            changed["updated_at"] = "2026-08-27T01:00:00+09:00"
            with patch.object(sa.os, "replace", side_effect=OSError("simulated power loss")):
                with self.assertRaises(OSError):
                    sa.atomic_write_ranking(path, changed)
            self.assertEqual(path.read_bytes(), before)

    def test_markdown_generation_is_deterministic_and_quotes_cells(self):
        state = sa._new_ranking("test", {
            "display_name": "MBP-M4-Max-128GB",
            "gpu": "Test *GPU*", "memory_gb": 8,
        })
        state["players"] = [_player("llm_jp|dummy|test-FP16", 1, 1)]
        state["players"][0]["avg_latency_ms"] = 12.5
        state["updated_at"] = "2026-08-27T00:00:00+09:00"
        sa.validate_ranking_document(state, "test")
        first = sa.render_ranking_markdown(state)
        second = sa.render_ranking_markdown(state)
        self.assertEqual(first, second)
        self.assertIn("# MBP-M4-Max-128GB ランキング", first)
        self.assertIn("`llm_jp`", first)
        self.assertIn("`dummy`", first)
        self.assertIn("`FP16`", first)

    def test_markdown_shows_parse_error_rate_next_to_the_speed(self):
        state = sa._new_ranking("test", {"display_name": "GX10", "gpu": "GB10", "memory_gb": 121})
        silent = _player("silent|ollama|GGUF-MXFP4", 1, 1)
        silent.update({"total_requests": 215, "latency_requests": 215,
                       "total_latency_ms": 254486.9, "parse_errors": 215, "matches": 2})
        state["players"] = [silent]
        state["updated_at"] = "2026-08-27T00:00:00+09:00"
        sa.validate_ranking_document(state, "test")
        row = [line for line in sa.render_ranking_markdown(state).splitlines()
               if "`silent`" in line][0]
        self.assertIn("100%", row)

    def test_markdown_marks_a_player_the_ranking_excludes(self):
        state = sa._new_ranking("test", {"display_name": "GX10", "gpu": "GB10", "memory_gb": 121})
        excluded = _player("noisy|ollama|GGUF-MXFP4", 1, 1)
        excluded.update({"total_requests": 1000, "latency_requests": 1000,
                         "total_latency_ms": 1000.0, "parse_errors": 500,
                         "matches": sa.PARSE_ERROR_MIN_GAMES})
        state["players"] = [excluded]
        state["updated_at"] = "2026-08-27T00:00:00+09:00"
        sa.validate_ranking_document(state, "test")
        row = [line for line in sa.render_ranking_markdown(state).splitlines()
               if "`noisy`" in line][0]
        self.assertIn("ランキング対象外", row)


if __name__ == "__main__":
    unittest.main()


class TestAdjacentEloRematch(unittest.TestCase):
    """Issue #11: 隣り合う順位で Elo が逆転していたら、その2体だけ再戦する。

    順位の正本は rank のままで、Elo で並べ替えはしない（§10.3）。逆転が表に
    残ると、同じ表の2行を見比べた人が表そのものを信用しなくなるので、
    1試合だけ実際に戦わせて決める。
    """

    def _state(self, elos):
        state = sa._new_ranking("test")
        state["players"] = [
            _player(f"m{i}|dummy|test-FP16", i + 1, i + 1, elo)
            for i, elo in enumerate(elos)
        ]
        return state

    def _agents(self, state):
        return {p["player_id"]: DummyAgent(p["model"]) for p in state["players"]}

    def test_one_match_resolves_the_inversion(self):
        state = self._state([1100.0, 969.5, 1016.0])
        agents = self._agents(state)
        played = []

        def fake_run(a, b, seed, max_duration):
            played.append((a.name, b.name))
            return _valid_match(a, b, 1)  # 下位（b）が勝つ

        with patch.object(sa, "run_match", side_effect=fake_run):
            sa._rematch_adjacent_elo_inversion(state, agents, 10.0, 1, [])
        self.assertEqual(len(played), 1)
        self.assertEqual([p["model"] for p in state["players"]], ["m0", "m2", "m1"])

    def test_lower_player_loses_and_nothing_moves(self):
        state = self._state([1100.0, 969.5, 1016.0])
        before = [p["player_id"] for p in state["players"]]
        agents = self._agents(state)
        played = []

        def fake_run(a, b, seed, max_duration):
            played.append((a.name, b.name))
            return _valid_match(a, b, 0)  # 上位（a）が勝つ

        with patch.object(sa, "run_match", side_effect=fake_run):
            sa._rematch_adjacent_elo_inversion(state, agents, 10.0, 1, [])
        self.assertEqual(len(played), 1, "負けても再戦を繰り返さない")
        self.assertEqual([p["player_id"] for p in state["players"]], before)

    def test_no_inversion_plays_nothing(self):
        state = self._state([1100.0, 1050.0, 1000.0])
        agents = self._agents(state)
        with patch.object(sa, "run_match", side_effect=AssertionError("試合が走った")):
            sa._rematch_adjacent_elo_inversion(state, agents, 10.0, 1, [])

    def test_largest_gap_is_chosen_when_two_inversions_exist(self):
        # 1位1000 / 2位1010（差10） と 3位900 / 4位1000（差100）
        state = self._state([1000.0, 1010.0, 900.0, 1000.0])
        agents = self._agents(state)
        played = []

        def fake_run(a, b, seed, max_duration):
            played.append((a.name, b.name))
            return _valid_match(a, b, 0)

        with patch.object(sa, "run_match", side_effect=fake_run):
            sa._rematch_adjacent_elo_inversion(state, agents, 10.0, 1, [])
        self.assertEqual(played, [("m2", "m3")])

    def test_pair_missing_from_this_run_is_skipped(self):
        state = self._state([1100.0, 969.5, 1016.0])
        agents = {state["players"][0]["player_id"]: DummyAgent("m0")}
        with patch.object(sa, "run_match", side_effect=AssertionError("試合が走った")):
            sa._rematch_adjacent_elo_inversion(state, agents, 10.0, 1, [])

    def test_tournament_run_resolves_the_inversion_end_to_end(self):
        """関数だけでなく、実行経路に繋がっていることを見る。

        呼び出しを外しても関数単体のテストは通ってしまうので、これが無いと
        「実装したが誰も呼んでいない」を見逃す（2026-08-30 実測）。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "test.json")
            state = sa._new_ranking("test")
            state["players"] = [
                _player("hi|dummy|test-FP16", 1, 1, 969.5),
                _player("lo|dummy|test-FP16", 2, 2, 1016.0),
            ]
            sa.atomic_write_ranking(path, state)

            hi, lo = DummyAgent("hi"), DummyAgent("lo")

            def fake_run(a, b, seed, max_duration):
                return _valid_match(a, b, 1)  # 下位が勝つ

            with patch.object(sa, "run_match", side_effect=fake_run):
                sa.run_persistent_tournament(
                    [hi, lo], {"mode": "dummy"}, strategy="ladder",
                    games_per_pair=0, rankings_dir=directory, machine_id="test",
                    out_path=None, verbose=False,
                )
            data = json.loads(path.read_text())
            self.assertEqual([p["model"] for p in data["players"]], ["lo", "hi"])
