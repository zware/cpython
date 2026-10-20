"""CPU utilization sampling and worker admission for percentage jobs."""

import os
import sys
import threading
import time


class CPUTimeSampler:
    def __init__(self):
        if sys.platform != 'linux' or not hasattr(os, 'sched_getaffinity'):
            raise OSError('CPU utilization sampling requires Linux')
        self.cpus = os.sched_getaffinity(0)
        if not self.cpus:
            raise OSError('cannot determine usable CPUs')
        self.previous = self._read()

    def _read(self):
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
