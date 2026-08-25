import httpx
import pytest
from retrieval.embedder import OllamaEmbedder


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class StubErrorResponse:
    """A response whose raise_for_status() raises, like a real 4xx/5xx."""

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        raise httpx.HTTPStatusError("error", request=None, response=self)


class StubClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, json, timeout=None):
        self.calls.append(json)
        return StubResponse(self._payload)


class ScriptedClient:
    """Returns one queued response per call, for testing retry sequences."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json, timeout=None):
        self.calls.append(json)
        return self._responses.pop(0)


def test_embedder_sends_all_texts_in_one_batch():
    client = StubClient({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    vectors = embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls[0]["input"] == ["a", "b"]
    assert len(client.calls) == 1


def test_embedder_sends_keep_alive_by_default():
    # Without keep_alive, bge-m3 is evicted on Ollama's 5-minute default TTL
    # between chat turns and reloads cold on the next query.
    client = StubClient({"embeddings": [[0.1, 0.2]]})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    embedder.embed(["a"])

    assert client.calls[0]["keep_alive"] == "10m"


def test_embedder_returns_empty_for_no_input():
    client = StubClient({"embeddings": []})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)
    assert embedder.embed([]) == []


def test_embedder_retries_a_tokenize_eof_400_and_succeeds():
    client = ScriptedClient([
        StubErrorResponse(400, '{"error":"Post \\"http://x/tokenize\\": EOF"}'),
        StubResponse({"embeddings": [[0.1, 0.2]]}),
    ])
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    vectors = embedder.embed(["a"])

    assert vectors == [[0.1, 0.2]]
    assert len(client.calls) == 2


def test_embedder_gives_up_after_three_tokenize_eof_failures():
    responses = [StubErrorResponse(400, "...tokenize...EOF")] * 3
    client = ScriptedClient(responses)
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed(["a"])

    assert len(client.calls) == 3


def test_embedder_does_not_retry_a_genuine_bad_input_400():
    """A 400 unrelated to the tokenize/EOF hiccup is a real client error —
    retrying it would just waste ~2s on a deterministic failure."""
    client = ScriptedClient([StubErrorResponse(400, '{"error":"bad request"}')])
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed(["a"])

    assert len(client.calls) == 1


def test_embedder_does_not_retry_a_500():
    client = ScriptedClient([StubErrorResponse(500, "internal error")])
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed(["a"])

    assert len(client.calls) == 1


@pytest.mark.integration
def test_embedder_against_real_ollama_returns_1024_dims():
    import os
    embedder = OllamaEmbedder(os.environ["OLLAMA_BASE_URL"], "bge-m3")
    vectors = embedder.embed(["kontrakt serwisowy", "service contract"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
