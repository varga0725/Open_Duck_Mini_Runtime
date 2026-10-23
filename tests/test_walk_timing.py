import pytest
from mini_bdx_runtime.walk_timing import LoopTimer


class Clock:
    def __init__(self):
        self.now = 0.
        self.sleeps = []
    def read(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_deadlines_keep_50_hz_without_accumulating_work_time():
    clock = Clock()
    timer = LoopTimer(50, clock.read, clock.sleep)
    starts = []
    for _ in range(100):
        starts.append(clock.now)
        clock.now += .013
        timer.finish(starts[-1])
    assert clock.now == pytest.approx(2.)
    assert starts[-1] == pytest.approx(1.98)
    assert min(clock.sleeps) == pytest.approx(.007)


def test_overrun_and_pause_do_not_produce_catch_up_bursts():
    clock = Clock()
    timer = LoopTimer(50, clock.read, clock.sleep)
    clock.now = .037
    timer.finish(0.)
    assert clock.sleeps == []
    clock.now += .013
    timer.finish(.037)
    assert clock.sleeps == pytest.approx([.007])
    clock.now = 1.
    clock.now += .013
    timer.finish(1.)
    assert clock.sleeps[-1] == pytest.approx(.007)
