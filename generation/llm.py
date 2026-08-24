import json

import httpx


class OllamaLLM:
    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 300.0, temperature: float = 0.0,
                 keep_alive: str = "10m", think: bool | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        # Sent on every request so Ollama keeps the model resident between
        # turns instead of evicting it on its default 5-minute TTL — a
        # multi-turn chat would otherwise cold-reload on almost every reply.
        self.keep_alive = keep_alive
        # None = omit the option (non-thinking models ignore it anyway). Set
        # False to disable a thinking model's reasoning pass so its content
        # comes back clean instead of prefixed with chain-of-thought.
        self.think = think
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
        }
        if self.think is not None:
            options["think"] = self.think
        return {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": options,
        }

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
                yield payload["message"]["content"]
