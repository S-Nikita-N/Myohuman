"""Regression tests for LoggerRL after the merge min/max fix.

Pins that LoggerRL.merge aggregates min_episode_reward with min (not max).
"""
import math

from myohuman.learning.logger_rl import LoggerRL


def _logger_with(episode_rewards, step_rewards=None):
    lg = LoggerRL()
    for i, ep_r in enumerate(episode_rewards):
        lg.start_episode(None)
        # one step carrying the whole episode reward
        lg.step(None, ep_r, {})
        lg.end_episode(None)
    lg.end_sampling()
    return lg


def test_merge_min_episode_reward_is_min():
    a = _logger_with([5.0, 8.0])   # min_episode_reward = 5.0
    b = _logger_with([2.0, 9.0])   # min_episode_reward = 2.0
    merged = LoggerRL.merge([a, b])
    assert merged.min_episode_reward == 2.0          # min across both, not max
    assert merged.max_episode_reward == 9.0


def test_merge_totals():
    a = _logger_with([5.0, 8.0])
    b = _logger_with([2.0, 9.0])
    merged = LoggerRL.merge([a, b])
    assert merged.num_episodes == 4
    assert merged.total_reward == 24.0
    assert merged.min_reward == 2.0
    assert merged.max_reward == 9.0


def test_single_logger_min_episode_reward():
    lg = _logger_with([3.0, 1.0, 7.0])
    assert lg.min_episode_reward == 1.0
    assert lg.max_episode_reward == 7.0
