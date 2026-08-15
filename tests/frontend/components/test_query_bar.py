from climate_agent.frontend.components.query_bar import render_query_bar


def test_render_query_bar_returns_expected_types():
    result = render_query_bar()

    assert isinstance(result, tuple)
    assert len(result) == 2

    query, ask = result
    assert isinstance(query, str)
    assert isinstance(ask, bool)
