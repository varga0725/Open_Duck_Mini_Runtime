import numpy as np
import pytest

from test_hwi_bus import make_hwi, PanicException


class BulkIO:
    def __init__(self):
        self.calls = []

    def read_present_position(self, ids):
        self.calls.append(('position', list(ids)))
        return [i / 100 for i in ids]

    def read_present_velocity(self, ids):
        self.calls.append(('velocity', list(ids)))
        return [i / 10 for i in ids]


def test_bulk_reads_preserve_policy_order_signs_offsets_and_ignore(monkeypatch):
    io = BulkIO()
    hwi = make_hwi(monkeypatch, lambda: io)
    hwi.joints_offsets['left_knee'] = .1
    hwi.joints_signs['left_knee'] = -1
    q = hwi.get_present_positions(ignore=['neck_pitch'])
    v = hwi.get_present_velocities(ignore=['neck_pitch'])
    names = [n for n in hwi.joints if n != 'neck_pitch']
    assert q[names.index('left_knee')] == pytest.approx(-.13)
    assert v[names.index('left_knee')] == pytest.approx(-2.3)
    assert len(q) == len(v) == 13
    assert io.calls == [('position', list(hwi.joints.values())), ('velocity', list(hwi.joints.values()))]


@pytest.mark.parametrize('bad', [None, [0], [np.nan] * 14, OSError('malformed reply')])
def test_bad_bulk_reply_falls_back_to_complete_individual_sample(monkeypatch, bad):
    class FallbackIO(BulkIO):
        def read_present_position(self, ids):
            if len(ids) > 1:
                self.calls.append(('bulk_failed', list(ids)))
                if isinstance(bad, Exception):
                    raise bad
                return bad
            return super().read_present_position(ids)
    io = FallbackIO()
    hwi = make_hwi(monkeypatch, lambda: io)
    q = hwi.get_present_positions()
    np.testing.assert_allclose(q, np.array(list(hwi.joints.values())) / 100)
    assert len(io.calls) == 15


def test_failed_individual_fallback_does_not_return_partial_observation(monkeypatch):
    class Dead(BulkIO):
        def read_present_position(self, ids):
            raise OSError('timeout')
    hwi = make_hwi(monkeypatch, Dead)
    hwi._IO_RETRY_DELAY = 0
    assert hwi.get_present_positions() is None


def test_interrupt_is_not_swallowed_by_bulk_fallback(monkeypatch):
    class Interrupted(BulkIO):
        def read_present_position(self, ids):
            raise KeyboardInterrupt()
    hwi = make_hwi(monkeypatch, Interrupted)
    with pytest.raises(KeyboardInterrupt):
        hwi.get_present_positions()
