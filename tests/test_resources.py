"""Tests for auto_worker_count and cgroup memory detection.

The detection functions read /sys/fs/cgroup/memory.max, /proc/meminfo, and
os.cpu_count() — all mocked here so tests are deterministic on any host.
"""
from unittest.mock import patch

from core.config import auto_worker_count, _cgroup_memory_limit_bytes


def _mock_open_factory(filemap: dict[str, str]):
    """Return a mock open() that serves the given path→content mapping."""
    from io import StringIO

    def _mock_open(path, *args, **kwargs):
        if path in filemap:
            return StringIO(filemap[path])
        raise FileNotFoundError(path)

    return _mock_open


# ---------- _cgroup_memory_limit_bytes ----------

def test_cgroup_v2_memory_limit():
    """Reads /sys/fs/cgroup/memory.max when present."""
    files = {"/sys/fs/cgroup/memory.max": "8589934592\n"}  # 8 GB
    with patch("builtins.open", side_effect=_mock_open_factory(files)):
        assert _cgroup_memory_limit_bytes() == 8589934592


def test_cgroup_v2_unlimited_falls_back_to_meminfo():
    """memory.max == 'max' means no limit → fall back to /proc/meminfo."""
    files = {
        "/sys/fs/cgroup/memory.max": "max\n",
        "/proc/meminfo": "MemTotal:       16384 kB\nMemAvailable:   12000 kB\n",
    }
    with patch("builtins.open", side_effect=_mock_open_factory(files)):
        # 12000 kB * 1024 = 12288000 bytes
        assert _cgroup_memory_limit_bytes() == 12288000


def test_cgroup_missing_returns_none():
    """No cgroup, no /proc/meminfo → None."""
    def _raise(*a, **kw):
        raise FileNotFoundError

    with patch("builtins.open", side_effect=_raise):
        assert _cgroup_memory_limit_bytes() is None


# ---------- auto_worker_count ----------

def test_memory_bound():
    """16 GB / 2 GB per worker = 8, but capped by CPU count."""
    files = {"/sys/fs/cgroup/memory.max": str(16 * 1024**3)}
    with patch("builtins.open", side_effect=_mock_open_factory(files)), \
         patch("core.config._cpu_count", return_value=8):
        # min(8 mem, 7 cpu, 8 cap) = 7
        assert auto_worker_count(2.0, 8) == 7


def test_cpu_bound():
    """Plenty of RAM but few CPUs → CPU limits."""
    files = {"/sys/fs/cgroup/memory.max": str(64 * 1024**3)}
    with patch("builtins.open", side_effect=_mock_open_factory(files)), \
         patch("core.config._cpu_count", return_value=4):
        # min(32 mem, 3 cpu, 8 cap) = 3
        assert auto_worker_count(2.0, 8) == 3


def test_max_workers_cap():
    """Cap applies even when both mem and CPU allow more."""
    files = {"/sys/fs/cgroup/memory.max": str(64 * 1024**3)}
    with patch("builtins.open", side_effect=_mock_open_factory(files)), \
         patch("core.config._cpu_count", return_value=16):
        # min(32 mem, 15 cpu, 4 cap) = 4
        assert auto_worker_count(2.0, 4) == 4


def test_floor_at_one():
    """Tiny memory budget → at least 1 worker."""
    files = {"/sys/fs/cgroup/memory.max": str(1 * 1024**3)}  # 1 GB
    with patch("builtins.open", side_effect=_mock_open_factory(files)), \
         patch("core.config._cpu_count", return_value=2):
        # min(0 mem, 1 cpu, 8 cap) = 0 → floored to 1
        assert auto_worker_count(2.0, 8) == 1


def test_no_memory_detectable_uses_cpu():
    """When memory can't be detected, CPU alone decides (capped)."""
    def _raise(*a, **kw):
        raise FileNotFoundError

    with patch("builtins.open", side_effect=_raise), \
         patch("core.config._cpu_count", return_value=4):
        # mem_based = max_workers (8), cpu = 3, cap = 8 → min(8,3,8) = 3
        assert auto_worker_count(2.0, 8) == 3


def test_zero_worker_memory_budget_falls_back_to_cpu_instead_of_raising():
    """worker_memory_gb: 0 is a plausible typo for 'unlimited', not a divide."""
    files = {"/sys/fs/cgroup/memory.max": str(16 * 1024**3)}
    with patch("builtins.open", side_effect=_mock_open_factory(files)), \
         patch("core.config._cpu_count", return_value=8):
        # mem_based falls back to max_workers (8), cpu = 7, cap = 8 → 7
        assert auto_worker_count(0, 8) == 7