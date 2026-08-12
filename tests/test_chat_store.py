import time

import pytest

from history.chat_store import ChatStore, chat_title


@pytest.fixture
def store(tmp_path):
    return ChatStore(tmp_path / "chats.db")


def _msgs(question="What is the contract number?", answer="SC-4471"):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "citations": ["a.pdf, p. 4"]},
    ]


def test_save_then_read_back_round_trips_messages(store):
    store.save("c1", "a title", _msgs())
    got = store.all()[0]
    assert got.title == "a title"
    assert got.messages == _msgs()


def test_empty_store_lists_nothing(store):
    assert store.all() == []


def test_all_orders_by_most_recently_updated_first(store):
    store.save("c1", "first", _msgs())
    time.sleep(0.01)
    store.save("c2", "second", _msgs())
    assert [c.chat_id for c in store.all()] == ["c2", "c1"]


def test_saving_same_id_updates_in_place_and_preserves_created_at(store):
    store.save("c1", "old", _msgs(answer="v1"))
    created = store.all()[0].created_at
    time.sleep(0.01)
    store.save("c1", "new", _msgs(answer="v2"))

    assert len(store.all()) == 1          # updated, not duplicated
    got = store.all()[0]
    assert got.title == "new"
    assert got.messages[1]["content"] == "v2"
    assert got.created_at == created       # created_at is stable
    assert got.updated_at > created        # updated_at moved


def test_delete_removes_only_the_named_chat(store):
    store.save("c1", "one", _msgs())
    store.save("c2", "two", _msgs())
    store.delete("c1")
    assert {c.chat_id for c in store.all()} == {"c2"}


def test_unicode_content_survives_json_round_trip(store):
    store.save("c1", "polski", _msgs(question="Jaka jest łączna suma?"))
    assert store.all()[0].messages[0]["content"] == "Jaka jest łączna suma?"


# --- chat_title --------------------------------------------------------------

def test_title_comes_from_first_user_message():
    assert chat_title(_msgs(question="How many sites?")) == "How many sites?"


def test_title_collapses_whitespace():
    msgs = [{"role": "user", "content": "  what   is\n  this?  "}]
    assert chat_title(msgs) == "what is this?"


def test_long_title_is_truncated_with_ellipsis():
    long_q = "word " * 20
    title = chat_title([{"role": "user", "content": long_q}])
    assert title.endswith("…")
    assert len(title) <= 41  # 40 chars + ellipsis


def test_title_placeholder_when_no_user_message():
    assert chat_title([]) == "New chat"
    assert chat_title([{"role": "assistant", "content": "hi"}]) == "New chat"
