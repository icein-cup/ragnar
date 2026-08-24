import json

import httpx


class OllamaLLM:
    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 300.0, temperature: float = 0.0,
                 keep_alive: str = "10m", think: bool | None = None,
                 seed: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        # Sent on every request so Ollama keeps the model resident between
        # turns instead of evicting it on its default 5-minute TTL — a
        # multi-turn chat would otherwise cold-reload on almost every reply.
        self.keep_alive = keep_alive
        # None = omit the field entirely, the default because it cannot
        # surprise a model we have not tried. False switches off a thinking
        # model's reasoning pass, which this pipeline discards anyway —
        # every structured reply it parses (Sufficient:, Complete:,
        # SUPPORTED) comes from message.content, never message.thinking.
        self.think = think
        # Pinning a seed makes a sampled run reproducible. Needed because
        # query generation runs at a non-zero temperature (see
        # AgenticSearch), which would otherwise make two eval runs over
        # identical inputs disagree.
        self.seed = seed
        self._client = client or httpx.Client()

    def _payload(self, system: str, user: str, stream: bool,
                 model: str | None, temperature: float | None, *,
                 history: list[dict] | None = None) -> dict:
        # model/temperature default to the instance values but can be
        # overridden per call, so a shared LLM instance stays safe when
        # different requests want different settings.
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            # History messages are already stripped of app-only keys
            # (citations) by the caller — pass them through as-is.
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        options: dict = {
            "temperature": self.temperature if temperature is None
            else temperature,
            # Ollama defaults num_ctx to 4096, which silently truncates the
            # system prompt when excerpts + history + system prompt exceed it.
            # qwen2.5:7b supports 32768; setting it ensures the full system
            # prompt (anti-refusal, multi-hop synthesis, NO_ANSWER contract)
            # is always visible to the model.
            "num_ctx": 32768,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        body = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        # "think" is a TOP-LEVEL field on /api/chat, not an option. Ollama
        # drops unrecognised keys inside "options" without complaining, so
        # putting it there disables nothing and fails silently.
        if self.think is not None:
            body["think"] = self.think
        return body

    def generate(self, system: str, user: str, *, model: str | None = None,
                 temperature: float | None = None,
                 history: list[dict] | None = None) -> str:
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=self._payload(system, user, False, model, temperature,
                               history=history),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def unload(self, model: str | None = None) -> bool:
        """Drop the model from Ollama's memory immediately.

        keep_alive is "10m" on every other request, which is right for an
        interactive chat and wrong for a one-shot harness: a finished
        benchmark otherwise holds its weights for ten more minutes. That
        matters because Ollama only evicts under memory pressure, so a 31 GB
        model and a 6 GB model coexist happily on a 48 GB host and starve
        whatever runs next.

        Best-effort: a failure here costs memory, never correctness, so it
        never raises.
        """
        try:
            resp = self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": model or self.model, "keep_alive": 0},
                timeout=60,
            )
            return resp.status_code < 400
        except Exception:
            return False

    def stream(self, system: str, user: str, *, model: str | None = None,
               temperature: float | None = None,
               history: list[dict] | None = None):
        """Yields token deltas. Used by the UI."""
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._payload(system, user, True, model, temperature,
                               history=history),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("done"):
                    break
                # A thinking model emits reasoning deltas whose message has
                # a "thinking" key and no "content" — indexing would raise
                # mid-answer. Empty deltas are skipped rather than yielded so
                # nothing downstream has to filter them.
                delta = payload.get("message", {}).get("content")
                if delta:
                    yield delta
