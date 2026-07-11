from app.agent import get_current_datetime


def test_datetime_tool_returns_iso_format() -> None:
    result = get_current_datetime.invoke({})
    assert isinstance(result, str)
    assert "T" in result
    assert "-" in result


def test_bitcoin_price_tool_is_async() -> None:
    import asyncio
    from app.agent import get_current_bitcoin_price
    result = asyncio.run(get_current_bitcoin_price.ainvoke({}))
    assert isinstance(result, str)
    assert result.startswith("$")
