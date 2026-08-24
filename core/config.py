import logging
import os
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def _cgroup_memory_limit_bytes() -> int | None:
    """Memory limit from cgroup v2, falling back to /proc/meminfo.

    Docker always sets cgroup v2 limits, so this is the primary source.
    /proc/meminfo MemAvailable is the fallback when no limit is set
    (e.g. no --memory flag → shows host RAM).
    """
    # cgroup v2: /sys/fs/cgroup/memory.max
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            val = fh.read().strip()
        if val != "max":
            return int(val)
    except (OSError, ValueError):
        pass

    # Fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    # "MemAvailable:  12345678 kB"
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass

    return None


def _cpu_count() -> int:
    return os.cpu_count() or 1


def auto_worker_count(worker_memory_gb: float, max_workers: int) -> int:
    """Pick a worker count that fits available RAM and CPU.

    Memory is the hard constraint (each worker's Docling converter eats
    ~1-2 GB; OOM kills the process). CPU-based count is cpus - 1, leaving
    one core for the Streamlit UI. The result is the min of both, capped
    at max_workers, floored at 1.
    """
    mem_bytes = _cgroup_memory_limit_bytes()
    if mem_bytes is not None:
        mem_gb = mem_bytes / (1024 ** 3)
        mem_based = int(mem_gb // worker_memory_gb)
    else:
        mem_based = max_workers  # no limit detectable — let CPU decide

    cpu_based = max(_cpu_count() - 1, 1)

    count = max(min(mem_based, cpu_based, max_workers), 1)

    log.info(
        "auto_worker_count: mem_based=%s cpu_based=%s max=%s → %d workers "
        "(%.1f GB available, %.1f GB/worker budget)",
        mem_based if mem_bytes is not None else "n/a",
        cpu_based, max_workers, count,
        (mem_bytes / (1024 ** 3)) if mem_bytes is not None else float("inf"),
        worker_memory_gb,
    )
    return count


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        with open(path) as fh:
            self._raw = yaml.safe_load(fh)

        self.ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.qdrant_url = os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )

    @property
    def llm_model(self) -> str:
        return self._raw["models"]["llm"]

    @property
    def embedding_model(self) -> str:
        return self._raw["models"]["embedding"]

    @property
    def embedding_dim(self) -> int:
        return self._raw["models"]["embedding_dim"]

    @property
    def reranker_model(self) -> str:
        return self._raw["models"]["reranker"]

    @property
    def llm_think(self) -> bool | None:
        """Whether to ask a thinking model to reason before answering.

        None (the default) omits the field, which is what non-thinking
        models need — Ollama rejects "think" for a model that cannot do it.
        """
        return self._raw["models"].get("think")

    @property
    def llm_seed(self) -> int | None:
        """Sampling seed, or None to let the model sample freely.

        Only matters because query generation runs at a non-zero
        temperature; pinning this is what keeps two eval runs comparable.
        """
        return self._raw["models"].get("seed")

    @property
    def collection(self) -> str:
        return self._raw["storage"]["collection"]

    @property
    def data_dir(self) -> Path:
        return Path(self._raw["storage"]["data_dir"])

    @property
    def candidates(self) -> int:
        return self._raw["retrieval"]["candidates"]

    @property
    def top_k(self) -> int:
        return self._raw["retrieval"]["top_k"]

    @property
    def score_floor(self) -> float:
        return self._raw["retrieval"]["score_floor"]

    @property
    def vector_floor(self) -> float:
        return self._raw["retrieval"]["vector_floor"]

    @property
    def chunking(self) -> dict:
        return self._raw["chunking"]

    @property
    def agentic(self) -> dict:
        return self._raw.get("agentic", {})

    @property
    def worker_count(self) -> int:
        ing = self._raw.get("ingestion", {})
        workers = ing.get("workers", 1)
        if isinstance(workers, str) and workers.lower() == "auto":
            mem_per = ing.get("worker_memory_gb", 2.0)
            cap = ing.get("max_workers", 8)
            return auto_worker_count(mem_per, cap)
        return int(workers)
