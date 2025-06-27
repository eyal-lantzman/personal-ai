from agent.openai_compatible_models import remove_thinking

def test_remove_thinking():
    assert remove_thinking("<think></think>\n\nnot thinking") == "not thinking"
    assert remove_thinking("not thinking") == "not thinking"
    assert remove_thinking("<think></think>") == "<think></think>"
    assert remove_thinking("<think>") == "<think>"
    assert remove_thinking("</think>") == "</think>"
    assert remove_thinking("</think>\n\n") == "</think>\n\n"
    assert remove_thinking("<think></think>\n\n") == ""
