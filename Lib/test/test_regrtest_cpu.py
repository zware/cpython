"""Deterministic tests for percentage-job admission and CPU counters."""

import contextlib
import io
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from test.libregrtest.cpu_load import (
    CPUGate, CPUTimeSampler, MacOSCPUTimeAPI, WindowsCPUTimeAPI)
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
        with patch('sys.platform', 'freebsd'):
            with self.assertRaisesRegex(OSError, 'requires Linux, Windows, or macOS'):
                CPUTimeSampler()


class WindowsCPUTimeTests(unittest.TestCase):
    def make_api(self, *, groups=1, process_mask=15, system_mask=15):
        try:
            import ctypes
        except ImportError:
            self.skipTest('requires ctypes')

        kernel32 = Mock()
        kernel32.GetActiveProcessorGroupCount.return_value = groups
        kernel32.GetActiveProcessorCount.return_value = system_mask.bit_count()

        def affinity(process, allowed, system):
            allowed._obj.value = process_mask
            system._obj.value = system_mask
            return 1

        kernel32.GetProcessAffinityMask.side_effect = affinity
        with patch.object(ctypes, 'WinDLL', return_value=kernel32, create=True):
            api = WindowsCPUTimeAPI()
        return api, kernel32

    def test_counters(self):
        api, kernel32 = self.make_api()

        def times(idle, kernel, user):
            for pointer, value in ((idle, (1 << 32) + 20),
                                   (kernel, (1 << 32) + 70), (user, 30)):
                pointer._obj.low = value & 0xffffffff
                pointer._obj.high = value >> 32
            return 1

        kernel32.GetSystemTimes.side_effect = times
        self.assertEqual(api.read(), ((1 << 32) + 100, (1 << 32) + 20))
        self.assertEqual(kernel32.GetSystemTimes.restype, api.ctypes.c_int)
        self.assertEqual(kernel32.GetCurrentProcess.restype, api.ctypes.c_void_p)

    def test_scope(self):
        for kwargs, message in (({'groups': 2}, 'single Windows processor group'),
                                ({'process_mask': 3}, 'restricted Windows CPU affinity'),
                                ({'system_mask': 0}, 'restricted Windows CPU affinity')):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(OSError, message):
                    self.make_api(**kwargs)

    def test_scope_changes(self):
        api, kernel32 = self.make_api()
        kernel32.GetActiveProcessorGroupCount.return_value = 2
        with self.assertRaisesRegex(OSError, 'single Windows processor group'):
            api.read()
        kernel32.GetSystemTimes.assert_not_called()

    def test_mask_does_not_cover_all_processors(self):
        api, kernel32 = self.make_api()
        kernel32.GetActiveProcessorCount.return_value = 32
        with self.assertRaisesRegex(OSError, 'restricted Windows CPU affinity'):
            api.read()

    def test_api_failure(self):
        api, kernel32 = self.make_api()
        for function in (kernel32.GetSystemTimes, kernel32.GetProcessAffinityMask):
            with self.subTest(function=function):
                function.side_effect = None
                function.return_value = 0
                with (patch.object(api.ctypes, 'get_last_error', return_value=5, create=True),
                      patch.object(api.ctypes, 'WinError', return_value=OSError('access denied'),
                                   create=True)):
                    with self.assertRaisesRegex(OSError, 'access denied'):
                        api.read()
                function.return_value = 1

    def test_sampler_dispatch(self):
        with (patch('sys.platform', 'win32'),
              patch('test.libregrtest.cpu_load.WindowsCPUTimeAPI') as factory):
            factory.return_value.read.side_effect = [(100, 20), (200, 45)]
            sampler = CPUTimeSampler()
            self.assertEqual(sampler.sample(), 0.75)
            factory.assert_called_once_with()

    def test_ctypes_unavailable(self):
        with (patch('sys.platform', 'win32'),
              patch('test.libregrtest.cpu_load.WindowsCPUTimeAPI', side_effect=ImportError)):
            with self.assertRaisesRegex(OSError, 'requires ctypes'):
                CPUTimeSampler()

    @unittest.skipUnless(sys.platform == 'win32', 'requires Windows')
    def test_native(self):
        try:
            api = WindowsCPUTimeAPI()
        except ImportError:
            self.skipTest('requires ctypes')
        except OSError as exc:
            if 'processor group' in str(exc) or 'restricted Windows CPU affinity' in str(exc):
                self.skipTest(str(exc))
            raise
        total, idle = api.read()
        self.assertGreaterEqual(total, idle)
        self.assertGreaterEqual(idle, 0)


class MacOSCPUTimeTests(unittest.TestCase):
    def make_api(self):
        try:
            import ctypes
        except ImportError:
            self.skipTest('requires ctypes')
        lib = Mock()
        lib.mach_host_self.return_value = 42
        lib.mach_port_deallocate.return_value = 0
        with patch.object(ctypes, 'CDLL', return_value=lib):
            api = MacOSCPUTimeAPI()
        api._task_port = Mock(return_value=7)
        return api, lib

    def set_counters(self, lib, values, *, error=0, count=4):
        def statistics(host, flavor, ticks, size):
            self.assertEqual(host, 42)
            self.assertEqual(flavor, 3)
            ticks[:] = values
            size._obj.value = count
            return error
        lib.host_statistics.side_effect = statistics

    def test_counters(self):
        api, lib = self.make_api()
        self.set_counters(lib, [100, 200, 300, 400])
        self.assertEqual(api.read(), (0, 0))
        self.set_counters(lib, [110, 220, 330, 440])
        self.assertEqual(api.read(), (100, 30))
        lib.mach_port_deallocate.assert_called_with(7, 42)
        self.assertEqual(lib.mach_port_deallocate.call_count, 2)

    def test_wraparound(self):
        api, lib = self.make_api()
        self.set_counters(lib, [(1 << 32) - 2] * 4)
        api.read()
        self.set_counters(lib, [3] * 4)
        self.assertEqual(api.read(), (20, 5))

    def test_errors_release_port(self):
        for error, count, message in ((5, 4, 'host_statistics failed'),
                                      (0, 3, 'incomplete Mach CPU counters')):
            with self.subTest(error=error, count=count):
                api, lib = self.make_api()
                self.set_counters(lib, [0] * 4, error=error, count=count)
                with self.assertRaisesRegex(OSError, message):
                    api.read()
                lib.mach_port_deallocate.assert_called_once_with(7, 42)

    def test_null_port(self):
        api, lib = self.make_api()
        lib.mach_host_self.return_value = 0
        with self.assertRaisesRegex(OSError, 'null port'):
            api.read()
        lib.mach_port_deallocate.assert_not_called()

    def test_deallocation_failure(self):
        api, lib = self.make_api()
        self.set_counters(lib, [0] * 4)
        lib.mach_port_deallocate.return_value = 5
        with self.assertRaisesRegex(OSError, 'mach_port_deallocate failed'):
            api.read()

    def test_sampler_dispatch(self):
        with (patch('sys.platform', 'darwin'),
              patch('test.libregrtest.cpu_load.MacOSCPUTimeAPI') as factory):
            factory.return_value.read.side_effect = [(0, 0), (100, 25)]
            sampler = CPUTimeSampler()
            self.assertEqual(sampler.sample(), 0.75)
            factory.assert_called_once_with()

    def test_ctypes_unavailable(self):
        with (patch('sys.platform', 'darwin'),
              patch('test.libregrtest.cpu_load.MacOSCPUTimeAPI', side_effect=ImportError)):
            with self.assertRaisesRegex(OSError, 'requires ctypes'):
                CPUTimeSampler()

    @unittest.skipUnless(sys.platform == 'darwin', 'requires macOS')
    def test_native(self):
        try:
            api = MacOSCPUTimeAPI()
        except ImportError:
            self.skipTest('requires ctypes')
        self.assertEqual(api.read(), (0, 0))
        total, idle = api.read()
        self.assertGreaterEqual(total, idle)
        self.assertGreaterEqual(idle, 0)


if __name__ == '__main__':
    unittest.main()
