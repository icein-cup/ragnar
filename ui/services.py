"""Service construction and shared UI helpers.

Kept apart from the panels and the page script so `build_services` (the
single cached wiring point) and the small formatting helpers can be reused
without importing Streamlit page logic.
"""
import httpx
import streamlit as st

from core.config import Config
from ingestion.parser import DoclingParser
from ingestion.chunkers.registry import build_chunker
from ingestion.pipeline import Pipeline
from ingestion.storage import Storage
from ingestion.registry_db import Registry
from ingestion.worker import IngestWorker
from history.chat_store import ChatStore
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.search import Search
from retrieval.agentic import AgenticSearch
from retrieval.reranker import BGEReranker
from generation.llm import OllamaLLM
from generation.answerer import Answerer


@st.cache_resource
def build_services():
    # Cache busted for Answerer update
    cfg = Config()
    storage = Storage(cfg.data_dir)
    registry = Registry(cfg.data_dir / "registry.db")
    chats = ChatStore(cfg.data_dir / "chats.db")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    store.ensure_collection()
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)

    pipeline = Pipeline(
        DoclingParser(), build_chunker(cfg.chunking, embedder=embedder),
        embedder, store)
    worker = IngestWorker(storage, registry, pipeline)
    worker.start()   # resets stale PROCESSING rows on startup

    base_search = Search(
        embedder, store, reranker=BGEReranker(cfg.reranker_model),
        candidates=cfg.candidates, top_k=cfg.top_k,
        score_floor=cfg.score_floor, vector_floor=cfg.vector_floor,
    )
    # config.yaml's `agentic:` keys are the AgenticSearch parameter names, so
    # the defaults live in its signature alone. An unknown key here is a
    # startup TypeError rather than a silently ignored setting.
    agentic_search = AgenticSearch(base_search, llm, **cfg.agentic)

    return {
        "cfg": cfg, "storage": storage, "registry": registry, "chats": chats,
        "store": store, "pipeline": pipeline, "embedder": embedder,
        "search": base_search,
        "agentic_search": agentic_search,
        "answerer": Answerer(llm), "worker": worker,
    }


@st.cache_data(ttl=30)
def list_chat_models(ollama_url: str, exclude: str) -> list[str]:
    """Chat-capable models pulled in Ollama, excluding the embedding model.

    Filters out cloud-proxy stubs (name ends with ':cloud') and models with
    no local data (size == 0) — those either require external subscriptions
    or are not actually available to run locally.
    """
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        # Exclude the embedding model and any tagged variant of it
        # ("bge-m3", "bge-m3:latest", ...) — only chat models belong here.
        base = exclude.split(":")[0]
        return sorted(
            m["name"] for m in resp.json().get("models", [])
            if not m["name"].startswith(base)
            and not m["name"].endswith(":cloud")
            and m.get("size", 0) > 0
        )
    except Exception:
        return []


def list_loaded_models(ollama_url: str) -> list[str]:
    """Names of models currently loaded in Ollama's memory (via /api/ps).

    Not cached — callers need a live view of what is actually running.
    """
    try:
        resp = httpx.get(f"{ollama_url}/api/ps", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def warm_model(ollama_url: str, model: str) -> bool:
    """Pre-load a model into Ollama's memory without generating any text.

    Sends an empty prompt to /api/generate with keep_alive=10m so the model
    stays resident for at least ten minutes after loading. Returns True on
    success, False if Ollama rejected the request.
    """
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "10m"},
            timeout=300,
        )
        return resp.status_code < 400
    except Exception:
        return False


def format_eta(seconds: float | None) -> str:
    """A rough, human-friendly ' · ~N min remaining' suffix, or '' if unknown.

    Kept deliberately coarse — the estimate is approximate, so second-level
    precision would imply accuracy it doesn't have.
    """
    if seconds is None:
        return ""
    s = int(round(seconds))
    if s < 45:
        return " · finishing up"
    mins = max(round(s / 60), 1)
    if mins < 60:
        return f" · ~{mins} min remaining"
    hrs, rem = divmod(mins, 60)
    return (f" · ~{hrs} hr {rem} min remaining" if rem
            else f" · ~{hrs} hr remaining")
