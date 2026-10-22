"""CPU utilization sampling and worker admission for percentage jobs."""

import os
import sys
import threading
import time


class CPUTimeSampler:
    def __init__(self):
        self.windows_api = None
        self.macos_api = None
        if sys.platform == 'win32':
            try:
                self.windows_api = WindowsCPUTimeAPI()
            except ImportError:
                raise OSError('Windows CPU utilization sampling requires ctypes') from None
        elif sys.platform == 'darwin':
            try:
                self.macos_api = MacOSCPUTimeAPI()
            except ImportError:
                raise OSError('macOS CPU utilization sampling requires ctypes') from None
        elif sys.platform == 'linux' and hasattr(os, 'sched_getaffinity'):
            self.cpus = os.sched_getaffinity(0)
            if not self.cpus:
                raise OSError('cannot determine usable CPUs')
        else:
            raise OSError('CPU utilization sampling requires Linux, Windows, or macOS')
        self.previous = self._read()

    def _read(self):
        if self.windows_api is not None:
            return {0: self.windows_api.read()}
        if self.macos_api is not None:
            return {0: self.macos_api.read()}
        counters = {}
        with open('/proc/stat', encoding='ascii') as stat:
            for line in stat:
                name, *fields = line.split()
                if name.startswith('cpu') and name[3:].isdigit():
                    cpu = int(name[3:])
                    if cpu in self.cpus:
                        # Guest time is already included in user/nice time.
                        values = list(map(int, fields[:8]))
                        if len(values) < 4:
                            raise ValueError('incomplete CPU counters')
                        idle = values[3] + (values[4] if len(values) > 4 else 0)
                        counters[cpu] = (sum(values), idle)
        if counters.keys() != self.cpus:
            raise OSError('CPU counters unavailable for affinity mask')
        return counters

    def sample(self):
        current = self._read()
        total = idle = 0
        for cpu, (ticks, idle_ticks) in current.items():
            old_ticks, old_idle = self.previous[cpu]
            delta = ticks - old_ticks
            idle_delta = idle_ticks - old_idle
            if delta < 0 or idle_delta < 0 or idle_delta > delta:
                raise OSError('CPU counters moved backwards')
            total += delta
            idle += idle_delta
        self.previous = current
        return (total - idle) / total if total else None


class WindowsCPUTimeAPI:
    """System counters for a single, unrestricted Windows processor group."""

    def __init__(self):
        import ctypes

        self.ctypes = ctypes

        class FILETIME(ctypes.Structure):
            _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]

        self.FILETIME = FILETIME
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel32 = kernel32
        kernel32.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME)] * 3
        kernel32.GetSystemTimes.restype = ctypes.c_int
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetActiveProcessorGroupCount.argtypes = []
        kernel32.GetActiveProcessorGroupCount.restype = ctypes.c_ushort
        kernel32.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
        kernel32.GetActiveProcessorCount.restype = ctypes.c_uint32
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t)]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        self._check_scope()

    def _check_scope(self):
        ctypes = self.ctypes
        # GetSystemTimes only measures the calling thread's processor group.
        if self.kernel32.GetActiveProcessorGroupCount() != 1:
            raise OSError('percentage jobs require a single Windows processor group')
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not self.kernel32.GetProcessAffinityMask(
                self.kernel32.GetCurrentProcess(), ctypes.byref(process_mask),
                ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        if (not system_mask.value or process_mask.value != system_mask.value
                or system_mask.value.bit_count() != self.kernel32.GetActiveProcessorCount(0)):
            raise OSError('percentage jobs do not support restricted Windows CPU affinity')

    def read(self):
        self._check_scope()
        idle = self.FILETIME()
        kernel = self.FILETIME()
        user = self.FILETIME()
        ctypes = self.ctypes
        if not self.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())

        def ticks(value):
            return (value.high << 32) | value.low

        # Windows kernel time already includes idle time.
        return ticks(kernel) + ticks(user), ticks(idle)


class MacOSCPUTimeAPI:
    """System CPU ticks from Mach HOST_CPU_LOAD_INFO."""

    def __init__(self):
        import ctypes

        self.ctypes = ctypes
        self.lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        self.lib.mach_host_self.argtypes = []
        self.lib.mach_host_self.restype = ctypes.c_uint32
        self.lib.host_statistics.argtypes = [
            ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32)]
        self.lib.host_statistics.restype = ctypes.c_int
        self.lib.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.lib.mach_port_deallocate.restype = ctypes.c_int
        self.previous = None
        self.total = self.idle = 0

    def _task_port(self):
        return self.ctypes.c_uint32.in_dll(self.lib, 'mach_task_self_').value

    def read(self):
        ctypes = self.ctypes
        task = self._task_port()
        host = self.lib.mach_host_self()
        if not host:
            raise OSError('mach_host_self returned a null port')
        try:
            ticks = (ctypes.c_uint32 * 4)()
            count = ctypes.c_uint32(4)
            # HOST_CPU_LOAD_INFO = 3; states: user, system, idle, nice.
            error = self.lib.host_statistics(host, 3, ticks, ctypes.byref(count))
            if error:
                raise OSError(f'host_statistics failed: Mach error {error}')
            if count.value != 4:
                raise OSError('incomplete Mach CPU counters')
        finally:
            error = self.lib.mach_port_deallocate(task, host)
            if error:
                raise OSError(f'mach_port_deallocate failed: Mach error {error}')

        current = tuple(ticks)
        if self.previous is not None:
            # Each Mach counter is a wrapping 32-bit unsigned integer.
            deltas = [(new - old) % (1 << 32)
                      for new, old in zip(current, self.previous)]
            self.total += sum(deltas)
            self.idle += deltas[2]
        self.previous = current
        return self.total, self.idle


class CPUGate:
    def __init__(self, target, maximum, sampler):
        self.target = target
        self.maximum = maximum
        self.sampler = sampler
        self.limit = 1
        self.active = 0
        self.stopped = False
        self.condition = threading.Condition()
        self.next_sample = time.monotonic() + 1.0

    def acquire(self):
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or self.active < self.limit)
            if self.stopped:
                return False
            self.active += 1
            return True

    def release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def get_active_workers(self):
        with self.condition:
            return self.active

    def update(self):
        now = time.monotonic()
        if now < self.next_sample:
            return
        self.next_sample = now + 1.0
        utilization = self.sampler.sample()
        if utilization is None:
            return
        with self.condition:
            if utilization < self.target - 0.03:
                self.limit = min(self.maximum, self.limit + 1)
            elif utilization > self.target + 0.03:
                self.limit = max(1, self.limit - 1)
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
