from langgraph.errors import GraphRecursionError

from argus.patterns.recursion_limit import build_infinite_loop_demo


def test_recursion_limit_fires_on_genuinely_infinite_graph():
    app = build_infinite_loop_demo()
    try:
        app.invoke({"count": 0}, config={"recursion_limit": 10})
        assert False, "expected GraphRecursionError, graph completed instead"
    except GraphRecursionError:
        pass  # expected
