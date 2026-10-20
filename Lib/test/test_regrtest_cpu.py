"""Deterministic tests for percentage-job admission and CPU counters."""

import contextlib
import io
import threading
import unittest
from unittest.mock import Mock, patch

from test.libregrtest.cpu_load import CPUGate, CPUTimeSampler
from test.libregrtest.logger import Logger
from test.libregrtest.results import TestResults


class CPUGateTests(unittest.TestCase):
    def make_gate(self):
        sampler = Mock()
        gate = CPUGate(0.9, 3, sampler)
        return gate, sampler

    def update(self, gate, sampler, value):
        sampler.sample.return_value = value
        gate.next_sample = 0
        gate.update()

    def test_adjustment(self):
        gate, sampler = self.make_gate()
        for expected in (2, 3, 3):
            self.update(gate, sampler, 0.2)
            self.assertEqual(gate.limit, expected)
        for value in (0.89, 0.91, None):
            self.update(gate, sampler, value)
            self.assertEqual(gate.limit, 3)
        for expected in (2, 1, 1):
            self.update(gate, sampler, 1.0)
            self.assertEqual(gate.limit, expected)

    def test_sample_interval(self):
        gate, sampler = self.make_gate()
        gate.next_sample = float('inf')
        gate.update()
        sampler.sample.assert_not_called()

    def test_active_workers(self):
        gate, _ = self.make_gate()
        self.assertEqual(gate.get_active_workers(), 0)
        self.assertTrue(gate.acquire())
        self.assertEqual(gate.get_active_workers(), 1)
        gate.release()
        self.assertEqual(gate.get_active_workers(), 0)

    def test_waiter_shutdown(self):
        gate, _ = self.make_gate()
        self.assertTrue(gate.acquire())
        acquired = []
        thread = threading.Thread(target=lambda: acquired.append(gate.acquire()))
        thread.start()
        try:
            gate.stop()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(acquired, [False])
        finally:
            gate.stop()
            thread.join(5)
            gate.release()

    def test_release_wakes_waiter(self):
        gate, _ = self.make_gate()
        self.assertTrue(gate.acquire())
        acquired = threading.Event()

        def worker():
            if gate.acquire():
                acquired.set()
                gate.release()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertFalse(acquired.is_set())
            gate.release()
            self.assertTrue(acquired.wait(5))
        finally:
            gate.stop()
            thread.join(5)
        self.assertEqual(gate.active, 0)

    def test_lower_limit_does_not_interrupt_active_workers(self):
        gate, sampler = self.make_gate()
        gate.limit = 2
        self.assertTrue(gate.acquire())
        self.assertTrue(gate.acquire())
        self.update(gate, sampler, 1.0)
        self.assertEqual(gate.limit, 1)
        self.assertEqual(gate.active, 2)
        gate.release()
        gate.release()

    def test_increase_wakes_waiter(self):
        gate, sampler = self.make_gate()
        self.assertTrue(gate.acquire())
        acquired = threading.Event()

        def worker():
            if gate.acquire():
                acquired.set()
                gate.release()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.update(gate, sampler, 0.2)
            self.assertTrue(acquired.wait(5))
        finally:
            gate.stop()
            thread.join(5)
            gate.release()

    def test_sampling_failure(self):
        gate, sampler = self.make_gate()
        gate.next_sample = 0
        sampler.sample.side_effect = OSError('counters unavailable')
        with self.assertRaisesRegex(OSError, 'counters unavailable'):
            gate.update()
        self.assertEqual(gate.limit, 1)


class AdaptiveLoggerTests(unittest.TestCase):
    def test_active_workers_output(self):
        logger = Logger(TestResults(), quiet=False, pgo=False)
        with patch.object(logger, 'get_load_avg', return_value=None):
            for active in (0, 2):
                with self.subTest(active=active):
                    logger.get_active_workers = lambda: active
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        logger.log('test_example passed')
                    self.assertIn(f'active workers: {active} test_example passed',
                                  output.getvalue())
            logger.get_active_workers = None
            with contextlib.redirect_stdout(io.StringIO()) as output:
                logger.log('test_example passed')
            self.assertNotIn('active workers:', output.getvalue())


class CPUTimeSamplerTests(unittest.TestCase):
    def sampler(self, *snapshots):
        with (patch('sys.platform', 'linux'),
              patch('os.sched_getaffinity', return_value={0}, create=True),
              patch.object(CPUTimeSampler, '_read', side_effect=snapshots)):
            return CPUTimeSampler()

    def test_deltas(self):
        sampler = self.sampler({0: (100, 20)})
        with patch.object(sampler, '_read', return_value={0: (200, 45)}):
            self.assertEqual(sampler.sample(), 0.75)
            self.assertIsNone(sampler.sample())

    def test_invalid_deltas(self):
        for counters in ({0: (90, 20)}, {0: (200, 10)}, {0: (110, 40)}):
            with self.subTest(counters=counters):
                sampler = self.sampler({0: (100, 20)})
                with patch.object(sampler, '_read', return_value=counters):
                    with self.assertRaises(OSError):
                        sampler.sample()

    def test_read_affinity_and_guest_time(self):
        sampler = self.sampler({0: (0, 0)})
        data = 'cpu 999 0 0 0\ncpu0 10 2 3 20 5 1 1 0 8 2\ncpu1 100 0 0 0\n'
        with patch('builtins.open', return_value=io.StringIO(data)):
            self.assertEqual(sampler._read(), {0: (42, 25)})

    def test_missing_cpu(self):
        sampler = self.sampler({0: (0, 0)})
        with patch('builtins.open', return_value=io.StringIO('cpu 0 0 0 0\n')):
            with self.assertRaises(OSError):
                sampler._read()

    def test_unsupported(self):
        with patch('sys.platform', 'win32'):
            with self.assertRaisesRegex(OSError, 'requires Linux'):
                CPUTimeSampler()


if __name__ == '__main__':
    unittest.main()
