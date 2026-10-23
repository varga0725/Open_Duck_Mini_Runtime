"""Monotonic control deadlines and periodic measured timing summaries."""
from collections import deque
import time

import numpy as np


class LoopTimer:
    def __init__(self, control_hz, clock=None, sleep=None):
        self.period = 1 / control_hz
        self.clock = clock or time.perf_counter
        self.sleep = sleep or time.sleep
        self.deadline = None
        self.starts = deque(maxlen=250)
        self.work = deque(maxlen=250)
        self.count = 0

    def reset(self):
        self.deadline = None
        self.starts.clear()
        self.work.clear()
        self.count = 0

    @property
    def actual_hz(self):
        if len(self.starts) > 1 and self.starts[-1] > self.starts[0]:
            return (len(self.starts) - 1) / (self.starts[-1] - self.starts[0])
        return 1 / self.period

    def finish(self, start):
        now = self.clock()
        self.starts.append(start)
        self.work.append(now - start)
        self.count += 1
        # A pause resets the deadline. Never run catch-up ticks in a burst.
        if self.deadline is None or start - self.deadline > self.period:
            self.deadline = start + self.period
        if now < self.deadline:
            self.sleep(self.deadline - now)
            self.deadline += self.period
        else:
            self.deadline = now + self.period
        if self.count % 250 == 0 and self.starts[-1] > self.starts[0]:
            hz = (len(self.starts) - 1) / (self.starts[-1] - self.starts[0])
            print(
                f"[Walk timing] actual_hz={hz:.2f} "
                f"work_p95_ms={1000 * np.percentile(self.work, 95):.2f} "
                f"overruns={sum(w > self.period for w in self.work)}/{len(self.work)}",
                flush=True,
            )
