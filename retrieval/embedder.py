import time

import httpx


class OllamaEmbedder:
    """Embeds text via Ollama's /api/embed endpoint.

    The client is injectable so unit tests can stub HTTP without a server.
    """

    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 120.0, keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Keeps bge-m3 resident between turns instead of relying on
        # Ollama's 5-minute default TTL — see OllamaLLM.keep_alive.
        self.keep_alive = keep_alive
        self._client = client or httpx.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Ollama's embed runner occasionally EOFs on its own internal
        # tokenize call and reports it as a 400 whose body names "tokenize"
        # ("Post \"...tokenize\": EOF") — retrying the identical request
        # unmodified succeeds. Scoped to that exact signature so a genuine
        # bad-input 400 (oversized input, etc.) fails immediately instead of
        # wasting ~2s retrying a deterministic failure.
        last_error = None
        for attempt in range(3):
            if attempt:
                time.sleep(1.0)
            response = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts,
                      "keep_alive": self.keep_alive},
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
                return response.json()["embeddings"]
            except httpx.HTTPStatusError as e:
                if response.status_code == 400 and "tokenize" in response.text.lower():
                    last_error = e
                    continue
                raise
        raise last_error
