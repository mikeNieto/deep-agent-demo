from app.agents.tools import get_current_datetime


def test_datetime_tool_returns_string() -> None:
    result = get_current_datetime.invoke({})
    assert isinstance(result, str)
    assert "T" in result
