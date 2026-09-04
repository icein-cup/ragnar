import pytest
from generation.llm import OllamaLLM


class StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class StubClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, json, timeout=None):
        self.calls.append(json)
        return StubResponse(self._payload)


def test_temperature_defaults_to_zero():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "m", client=client)

    llm.generate("sys", "user")

    assert client.calls[0]["options"]["temperature"] == 0.0


def test_temperature_is_settable_and_sent_in_request():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "m", client=client)

    llm.temperature = 0.7
    llm.generate("sys", "user")

    assert client.calls[0]["options"]["temperature"] == 0.7


def test_keep_alive_is_sent_by_default():
    # Without keep_alive, Ollama evicts the model on its default 5-minute
    # TTL between chat turns, forcing a cold reload on the next request.
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "m", client=client)

    llm.generate("sys", "user")

    assert client.calls[0]["keep_alive"] == "10m"


def test_keep_alive_is_settable_on_the_instance():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "m", client=client, keep_alive="30m")

    llm.generate("sys", "user")

    assert client.calls[0]["keep_alive"] == "30m"


def test_per_call_model_and_temperature_override_instance_defaults():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "default-model", client=client)

    llm.generate("sys", "user", model="other-model", temperature=0.9)

    assert client.calls[0]["model"] == "other-model"
    assert client.calls[0]["options"]["temperature"] == 0.9
    # Instance defaults are untouched — nothing was mutated.
    assert llm.model == "default-model"
    assert llm.temperature == 0.0


@pytest.mark.integration
def test_llm_answers_from_context_only():
    import os
    llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")

    reply = llm.generate(
        "Answer only from the excerpt. If absent, say you don't know.",
        "Excerpt: The contract number is SC-4471.\n\n"
        "Question: What is the contract number?",
    )

    assert "SC-4471" in reply


@pytest.mark.integration
def test_llm_answers_in_the_language_of_the_question():
    import os
    from generation.prompts import SYSTEM_PROMPT
    llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")

    reply = llm.generate(
        SYSTEM_PROMPT,
        "Excerpts:\n\n[a.pdf, p. 1]\nThe contract number is SC-4471.\n\n"
        "Question: Jaki jest numer kontraktu?",
    )

    assert "SC-4471" in reply
    # crude Polish-output check: at least one Polish-specific character
    # or common Polish word
    assert any(t in reply.lower() for t in
               ["numer", "kontrakt", "wynosi", "to "])


# "think" is a top-level /api/chat field, not an option. Ollama silently drops
# unknown keys inside "options", so putting it there disables nothing — these
# pin the placement, which is the part that fails quietly when wrong.

def test_think_is_omitted_entirely_by_default():
    body = OllamaLLM("http://x", "m")._payload("s", "u", False, None, None)
    assert "think" not in body
    assert "think" not in body["options"]


def test_think_false_is_sent_at_the_top_level_not_in_options():
    body = OllamaLLM("http://x", "m", think=False)._payload(
        "s", "u", False, None, None)
    assert body["think"] is False
    assert "think" not in body["options"]


def test_seed_is_omitted_by_default_and_sent_as_an_option_when_set():
    assert "seed" not in OllamaLLM("http://x", "m")._payload(
        "s", "u", False, None, None)["options"]
    body = OllamaLLM("http://x", "m", seed=42)._payload("s", "u", False, None, None)
    assert body["options"]["seed"] == 42


def test_stream_skips_thinking_deltas_that_carry_no_content():
    """A thinking model emits deltas whose message has "thinking" and no
    "content". Indexing those raised KeyError in the middle of an answer."""
    import json as _json

    lines = [
        _json.dumps({"message": {"thinking": "hmm", "content": ""}}),
        _json.dumps({"message": {"thinking": "still hmm"}}),
        _json.dumps({"message": {"content": "Real "}}),
        _json.dumps({"message": {"content": "answer."}}),
        _json.dumps({"done": True, "message": {"content": ""}}),
    ]

    class _Stream:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def raise_for_status(self):
            pass
        def iter_lines(self):
            return iter(lines)

    class _Client:
        def stream(self, *a, **kw):
            return _Stream()

    llm = OllamaLLM("http://x", "m", client=_Client())
    assert "".join(llm.stream("s", "u")) == "Real answer."


# keep_alive is "10m" on every request, which suits an interactive chat and
# not a one-shot harness. Ollama evicts only under memory pressure, so an idle
# 31 GB model and an idle 6 GB model sit together on a 48 GB host and starve
# whatever loads next.

def test_unload_asks_ollama_to_drop_the_model_immediately():
    sent = {}

    class _Client:
        def post(self, url, json, timeout=None):
            sent["url"] = url
            sent["json"] = json
            return StubResponse({})

    llm = OllamaLLM("http://x", "big-model", client=_Client())
    assert llm.unload() is True
    assert sent["url"].endswith("/api/generate")
    assert sent["json"] == {"model": "big-model", "keep_alive": 0}


def test_unload_can_target_a_model_other_than_the_instance_default():
    sent = {}

    class _Client:
        def post(self, url, json, timeout=None):
            sent.update(json)
            return StubResponse({})

    OllamaLLM("http://x", "small", client=_Client()).unload("other-model")
    assert sent["model"] == "other-model"


def test_unload_never_raises_when_ollama_is_unreachable():
    class _Client:
        def post(self, *a, **kw):
            raise ConnectionError("ollama is down")

    # Failing to free memory must never take down a finished run.
    assert OllamaLLM("http://x", "m", client=_Client()).unload() is False
