from eval.load_hybridqa import answer_is_unique, clean_trace


def passage_node(title):
    return ["cell text", [0, 0], f"/wiki/{title}", "passage"]


TABLE_NODE = ["cell text", [1, 0], None, "table"]


def test_clean_trace_accepts_nodes_agreeing_on_one_passage():
    assert clean_trace({"answer-node": [passage_node("A"), passage_node("A")]})


def test_clean_trace_rejects_answer_that_also_sits_in_a_cell():
    assert not clean_trace({"answer-node": [passage_node("A"), TABLE_NODE]})


def test_clean_trace_rejects_disagreeing_passages():
    assert not clean_trace({"answer-node": [passage_node("A"),
                                            passage_node("B")]})


def test_clean_trace_rejects_empty_trace():
    assert not clean_trace({"answer-node": []})


def test_answer_is_unique_accepts_a_lone_holder():
    passages = {"/wiki/Medellin": "second-largest city, after Bogota",
                "/wiki/Cali": "third-largest city"}
    assert answer_is_unique("Bogota", passages, "Medellin")


def test_answer_is_unique_rejects_a_sibling_passage_with_the_answer():
    passages = {"/wiki/Medellin": "second-largest city, after Bogota",
                "/wiki/Bogota": "Bogota is the capital"}
    assert not answer_is_unique("Bogota", passages, "Medellin")
