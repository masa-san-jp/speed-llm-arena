"""ゲームエンジン・json-v1パース・ランキング集計の純粋ロジックに対するユニットテスト。

ネットワークやスレッドタイミングに依存しない部分を対象とする。
HTTPを介したProvider統合テストは tests/test_agents_http.py を参照。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speed_arena as sa


class TestIsAdjacent(unittest.TestCase):
    def test_adjacent_ranks(self):
        self.assertTrue(sa.is_adjacent(5, 6))
        self.assertTrue(sa.is_adjacent(6, 5))

    def test_king_ace_wrap(self):
        self.assertTrue(sa.is_adjacent(13, 1))
        self.assertTrue(sa.is_adjacent(1, 13))

    def test_not_adjacent(self):
        self.assertFalse(sa.is_adjacent(1, 3))
        self.assertFalse(sa.is_adjacent(5, 5))


class TestSpeedGame(unittest.TestCase):
    def test_deal_sizes(self):
        game = sa.SpeedGame(seed=1)
        for ps in game.players:
            self.assertEqual(len(ps.hand), sa.HAND_SIZE)
            # 各プレイヤー26枚のうち、台札へ1枚出すため残りは25枚(手札+山札)
            self.assertEqual(ps.remaining, 26 - 1)

    def test_total_cards_conserved(self):
        game = sa.SpeedGame(seed=1)
        total = sum(p.remaining for p in game.players) + len(game.piles)
        self.assertEqual(total, 52)

    def test_try_play_rejects_bad_pile(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        card = game.players[0].hand[0]
        res = game.try_play(0, card, pile=5)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "bad_pile")

    def test_try_play_rejects_card_not_in_hand(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        missing = next(r for r in range(1, 14) if r not in game.players[0].hand)
        res = game.try_play(0, missing, pile=0)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "card_not_in_hand")

    def test_try_play_rejects_non_adjacent(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        game.piles = [7, 7]
        game.players[0].hand = [1, 2, 3, 4]
        res = game.try_play(0, 1, pile=0)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "not_adjacent")

    def test_try_play_success_refills_from_stock(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        game.piles = [7, 7]
        game.players[0].hand = [6, 2, 3, 4]
        game.players[0].stock = [9, 9, 9]
        res = game.try_play(0, 6, pile=0)
        self.assertTrue(res.ok)
        self.assertEqual(len(game.players[0].hand), 4)
        self.assertEqual(game.piles[0], 6)

    def test_finish_on_played_out(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        game.piles = [7, 7]
        game.players[0].hand = [6]
        game.players[0].stock = []
        res = game.try_play(0, 6, pile=0)
        self.assertTrue(res.ok)
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.end_reason, "played_out")

    def test_flip_deadlock_fewer_cards_wins(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        game.players[0].hand = [1]
        game.players[0].stock = []
        game.players[1].hand = [1, 2]
        game.players[1].stock = []
        game.flip()
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.end_reason, "deadlock_fewer_cards")

    def test_flip_deadlock_tie(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        game.players[0].hand = [1]
        game.players[0].stock = []
        game.players[1].hand = [2]
        game.players[1].stock = []
        game.flip()
        self.assertEqual(game.winner, -1)
        self.assertEqual(game.end_reason, "deadlock_tie")

    def test_flip_reshuffles_when_stock_available(self):
        game = sa.SpeedGame(seed=1)
        game.start()
        before = list(game.piles)
        game.players[0].stock = [11]
        game.players[1].stock = [12]
        game.flip()
        self.assertEqual(game.piles, [11, 12])
        self.assertNotEqual(game.piles, before)
        self.assertEqual(game.flips, 1)

    def test_check_timeout_declares_winner(self):
        game = sa.SpeedGame(seed=1, max_duration=0.0)
        game.start()
        game.players[0].hand = [1]
        game.players[0].stock = []
        game.players[1].hand = [1, 2]
        game.players[1].stock = []
        game.check_timeout()
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.end_reason, "timeout_fewer_cards")


class TestValidateAction(unittest.TestCase):
    def test_pass_exact_shape_ok(self):
        self.assertEqual(sa.validate_action({"action": "pass"}), {"action": "pass"})

    def test_pass_with_extra_key_is_parse_error(self):
        got = sa.validate_action({"action": "pass", "card": 5})
        self.assertTrue(got["parse_error"])
        self.assertEqual(got["action"], "pass")

    def test_play_exact_shape_ok(self):
        got = sa.validate_action({"action": "play", "card": 5, "pile": 1})
        self.assertEqual(got, {"action": "play", "card": 5, "pile": 1})

    def test_play_with_extra_key_is_parse_error(self):
        got = sa.validate_action({"action": "play", "card": 5, "pile": 1, "extra": 1})
        self.assertTrue(got["parse_error"])

    def test_play_out_of_range_card_is_parse_error(self):
        self.assertTrue(sa.validate_action({"action": "play", "card": 99, "pile": 0})["parse_error"])

    def test_play_bad_pile_is_parse_error(self):
        self.assertTrue(sa.validate_action({"action": "play", "card": 5, "pile": 2})["parse_error"])

    def test_play_missing_card_is_parse_error(self):
        self.assertTrue(sa.validate_action({"action": "play", "pile": 0})["parse_error"])

    def test_unknown_action_is_parse_error(self):
        self.assertTrue(sa.validate_action({"action": "resign"})["parse_error"])

    def test_non_dict_is_parse_error(self):
        for bad in (5, "pass", None, [1, 2], True):
            self.assertTrue(sa.validate_action(bad)["parse_error"])


class TestParseActionText(unittest.TestCase):
    def test_strict_single_object(self):
        self.assertEqual(sa.parse_action_text('{"action":"pass"}'), {"action": "pass"})

    def test_rejects_prose_around_json(self):
        got = sa.parse_action_text('Sure! {"action":"pass"}')
        self.assertTrue(got["parse_error"])

    def test_rejects_markdown_fence(self):
        got = sa.parse_action_text('```json\n{"action":"pass"}\n```')
        self.assertTrue(got["parse_error"])

    def test_rejects_empty(self):
        self.assertTrue(sa.parse_action_text("")["parse_error"])
        self.assertTrue(sa.parse_action_text(None)["parse_error"])

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(sa.parse_action_text('  {"action":"pass"}  \n'), {"action": "pass"})


class TestUpdateElo(unittest.TestCase):
    def test_equal_ratings_win_gains_half_k(self):
        ra, rb = sa.update_elo(1000.0, 1000.0, score_a=1.0, k=32.0)
        self.assertAlmostEqual(ra, 1016.0)
        self.assertAlmostEqual(rb, 984.0)

    def test_zero_sum(self):
        ra, rb = sa.update_elo(1200.0, 1000.0, score_a=0.0, k=32.0)
        self.assertAlmostEqual((ra - 1200.0) + (rb - 1000.0), 0.0)


def _match(p0, p1, winner, valid=True, stats_overrides=None):
    stats_overrides = stats_overrides or {}

    def _stat(name):
        base = {
            "agent": name, "calls": 10, "plays": 8, "invalid_moves": 0,
            "parse_errors": 0, "api_errors": 0, "think_time": 1.0, "avg_latency": 0.1,
            "warmup_status": "ok", "warmup_duration": 0.0, "warmup_started_at": 0.0,
        }
        base.update(stats_overrides.get(name, {}))
        return base

    return {
        "seed": 0, "p0": p0, "p1": p1, "winner": winner, "reason": "played_out",
        "duration": 1.0, "flips": 0, "valid": valid,
        "stats": [_stat(p0), _stat(p1)],
    }


class TestBuildRanking(unittest.TestCase):
    def test_ranking_valid_when_below_threshold(self):
        agents = [sa.HeuristicAgent("a"), sa.HeuristicAgent("b")]
        matches = [_match("a", "b", "a") for _ in range(10)]
        ratings = {"a": 1010.0, "b": 990.0}
        records = {"a": {"win": 10, "loss": 0, "draw": 0}, "b": {"win": 0, "loss": 10, "draw": 0}}
        ranking = sa.build_ranking(agents, ratings, records, matches)
        by_name = {r["name"]: r for r in ranking}
        self.assertTrue(by_name["a"]["ranking_valid"])
        self.assertTrue(by_name["b"]["ranking_valid"])

    def test_ranking_invalid_above_threshold_with_enough_games(self):
        agents = [sa.HeuristicAgent("a"), sa.HeuristicAgent("b")]
        # "a" の10試合すべてでparse_errorsを2/10callsにして20%のエラー率にする
        matches = [
            _match("a", "b", "a", stats_overrides={"a": {"parse_errors": 2}})
            for _ in range(10)
        ]
        ratings = {"a": 1010.0, "b": 990.0}
        records = {"a": {"win": 10, "loss": 0, "draw": 0}, "b": {"win": 0, "loss": 10, "draw": 0}}
        ranking = sa.build_ranking(agents, ratings, records, matches)
        by_name = {r["name"]: r for r in ranking}
        self.assertFalse(by_name["a"]["ranking_valid"])
        self.assertGreater(by_name["a"]["parse_error_rate"], sa.PARSE_ERROR_RATE_THRESHOLD)
        # raw値は隠されない
        self.assertEqual(by_name["a"]["win"], 10)

    def test_ranking_valid_when_under_min_games_even_if_error_rate_high(self):
        agents = [sa.HeuristicAgent("a"), sa.HeuristicAgent("b")]
        matches = [
            _match("a", "b", "a", stats_overrides={"a": {"parse_errors": 5}})
            for _ in range(sa.PARSE_ERROR_MIN_GAMES - 1)
        ]
        ratings = {"a": 1010.0, "b": 990.0}
        records = {"a": {"win": 9, "loss": 0, "draw": 0}, "b": {"win": 0, "loss": 9, "draw": 0}}
        ranking = sa.build_ranking(agents, ratings, records, matches)
        by_name = {r["name"]: r for r in ranking}
        self.assertTrue(by_name["a"]["ranking_valid"])

    def test_invalid_matches_count_calls_but_not_games(self):
        agents = [sa.HeuristicAgent("a"), sa.HeuristicAgent("b")]
        matches = [_match("a", "b", None, valid=False, stats_overrides={
            "a": {"parse_errors": 5, "calls": 5}, "b": {"parse_errors": 5, "calls": 5},
        }) for _ in range(20)]
        ratings = {"a": 1000.0, "b": 1000.0}
        records = {"a": {"win": 0, "loss": 0, "draw": 0}, "b": {"win": 0, "loss": 0, "draw": 0}}
        ranking = sa.build_ranking(agents, ratings, records, matches)
        by_name = {r["name"]: r for r in ranking}
        # エラー率は無効試合の呼び出しも数える（形式を守れるかは試合の成立と関係ない）。
        self.assertEqual(by_name["a"]["parse_error_rate"], 1.0)
        # 一方で試合数の下限は有効試合で数えるので、有効試合0ならまだ弾かない。
        self.assertTrue(by_name["a"]["ranking_valid"])


class TestRunMatchWithHeuristicAgents(unittest.TestCase):
    def test_full_match_produces_valid_schema_shaped_result(self):
        fast = sa.HeuristicAgent("fast", latency=0.01)
        slow = sa.HeuristicAgent("slow", latency=0.05)
        ms = sa.run_match(fast, slow, seed=123, max_duration=30.0)
        self.assertTrue(ms.valid)
        self.assertIn(ms.winner, (0, 1, -1))
        required_stat_fields = {
            "agent", "calls", "plays", "invalid_moves", "parse_errors", "api_errors",
            "think_time", "avg_latency", "warmup_status", "warmup_duration", "warmup_started_at",
        }
        for s in ms.per_player:
            self.assertEqual(set(s.keys()), required_stat_fields)
            self.assertEqual(s["warmup_status"], "ok")


if __name__ == "__main__":
    unittest.main()
