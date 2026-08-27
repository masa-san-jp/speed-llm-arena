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
                position = sa._ladder_insertion_position(
                    players, lambda pid: agents[pid.split("|", 1)[0]], 10.0, 1,
                    [], new, "new|dummy|test-FP16", state,
                )
            self.assertEqual(position, count + 1)
            self.assertEqual(len(calls), expected_calls)

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


if __name__ == "__main__":
    unittest.main()
