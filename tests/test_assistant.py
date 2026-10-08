from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from stock_predictor import config
from stock_predictor.assistant import chat, tools
from stock_predictor.assistant.chat import TurnRunner, Usage
from stock_predictor.assistant.tools import ToolContext, run_tool
from stock_predictor.model import overrides_store
from stock_predictor.storage.models import Base
from stock_predictor.storage.repository import (
    get_or_create_symbol,
    replace_fundamentals_snapshot,
    replace_recent_articles,
    replace_sentiment_snapshot,
    replace_social_sentiment_snapshot,
    save_return_model_params,
    upsert_price_bar,
)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    engine = create_engine("sqlite:///:memory:", future=True, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    @contextmanager
    def scope():
        with Session(engine) as s:
            yield s
            s.commit()

    monkeypatch.setattr(tools, "session_scope", scope)
    monkeypatch.setattr(tools, "load_watchlist", lambda: {"AAPL": None})
    with scope() as s:
        symbol = get_or_create_symbol(s, "AAPL", "Apple Inc.", "Technology")
        s.flush()
        save_return_model_params(s, symbol.id, mu=0.001, sigma=0.02, dof=5.0, xi=0.008, n_bars=200)
        replace_fundamentals_snapshot(s, symbol.id, dt.date.today(), 12.0, 11.0, name="Apple Inc.", description="Makes phones.")
        replace_sentiment_snapshot(s, symbol.id, dt.date.today(), 0.4, 3)
        replace_social_sentiment_snapshot(s, symbol.id, dt.date.today(), 0.3, 10)
        replace_recent_articles(s, symbol.id, [("Headline", "https://x", "Src", dt.datetime(2026, 1, 1), "Summary")])
        for i in range(3):
            upsert_price_bar(s, symbol.id, dt.date(2026, 1, 1 + i), 100, 101, 99, 100 + i, 1000)
    return engine


def test_unknown_ticker_is_a_tool_error():
    result, is_error = run_tool("get_overview", {"ticker": "ZZZZ"}, ToolContext())
    assert is_error and "not a tracked company" in result["error"]


def test_get_signals_reports_votes_params_and_default_verdict():
    result, is_error = run_tool("get_signals", {"ticker": "aapl"}, ToolContext())
    assert not is_error
    assert result["votes"]["pe"] == 1  # P/E 12 < 15
    assert result["votes"]["news"] == 1 and result["votes"]["social"] == 1
    assert result["recommendation"] == "Buy"
    assert result["parameters"]["ma_weight"] == {"value": 1.0, "default": 1.0, "overridden": False}
    assert result["verdict_with_default_parameters"]["recommendation"] == "Buy"


def test_apply_adjustment_changes_verdict_and_is_logged():
    ctx = ToolContext()
    result, is_error = run_tool(
        "apply_adjustment",
        {"ticker": "AAPL", "changes": {"buy_threshold": 4, "pe_weight": 0.5}, "reasoning": "too eager"},
        ctx,
    )
    assert not is_error and result["status"] == "applied"
    assert result["verdict_before"]["recommendation"] == "Buy"
    assert result["verdict_after"]["recommendation"] == "Hold"
    assert overrides_store.overrides_for("AAPL") == {"buy_threshold": 4.0, "pe_weight": 0.5}
    assert len(ctx.applied) == 2 and ctx.turn_changes == 2
    assert overrides_store.history("AAPL")[-1]["source"] == "ai"

    signals, _ = run_tool("get_signals", {"ticker": "AAPL"}, ToolContext())
    assert signals["parameters"]["buy_threshold"]["overridden"] is True
    assert signals["verdict_with_default_parameters"]["recommendation"] == "Buy"


def test_out_of_range_value_is_rejected_and_nothing_changes():
    result, is_error = run_tool(
        "apply_adjustment", {"ticker": "AAPL", "changes": {"ma_weight": 50}, "reasoning": "x"}, ToolContext()
    )
    assert is_error and "between" in result["error"]
    assert overrides_store.overrides_for("AAPL") == {}


def test_per_turn_cap():
    ctx = ToolContext()
    changes = {
        "ma_weight": 0.5,
        "pe_weight": 0.5,
        "news_weight": 0.5,
        "social_weight": 0.5,
        "buy_threshold": 3,
        "sell_threshold": -3,
        "pe_value_threshold": 10,
    }
    result, is_error = run_tool("apply_adjustment", {"ticker": "AAPL", "changes": changes, "reasoning": "x"}, ctx)
    assert is_error and "per turn" in result["error"]
    assert overrides_store.overrides_for("AAPL") == {}


def test_daily_per_ticker_cap(monkeypatch):
    monkeypatch.setattr(tools, "MAX_AI_CHANGES_PER_TICKER_PER_DAY", 2)
    run_tool("apply_adjustment", {"ticker": "AAPL", "changes": {"ma_weight": 0.5, "pe_weight": 0.5}, "reasoning": "x"}, ToolContext())
    result, is_error = run_tool(
        "apply_adjustment", {"ticker": "AAPL", "changes": {"news_weight": 0.5}, "reasoning": "x"}, ToolContext()
    )
    assert is_error and "Daily limit" in result["error"]


def test_propose_then_owner_apply_is_one_use():
    ctx = ToolContext()
    result, _ = run_tool(
        "propose_adjustment", {"ticker": "AAPL", "changes": {"news_weight": 0.5}, "reasoning": "noisy"}, ctx
    )
    assert result["status"] == "proposed"
    assert overrides_store.overrides_for("AAPL") == {}

    diff = tools.proposal_diff(ctx.proposals[result["proposal_id"]])
    assert diff == [{"param": "news_weight", "current": 1.0, "proposed": 0.5, "default": 1.0}]

    tools.apply_proposal(ctx, result["proposal_id"])
    assert overrides_store.overrides_for("AAPL") == {"news_weight": 0.5}
    assert overrides_store.history("AAPL")[-1]["source"] == "owner"
    with pytest.raises(ValueError, match="already applied"):
        tools.apply_proposal(ctx, result["proposal_id"])


def test_taint_downgrades_apply_to_proposal_for_the_rest_of_the_conversation():
    ctx = ToolContext()
    assert not ctx.tainted
    run_tool("get_news", {"ticker": "AAPL"}, ctx)
    assert ctx.tainted

    result, is_error = run_tool(
        "apply_adjustment", {"ticker": "AAPL", "changes": {"news_weight": 0}, "reasoning": "ignore previous"}, ctx
    )
    assert not is_error and result["status"] == "downgraded_to_proposal"
    assert overrides_store.overrides_for("AAPL") == {}
    assert result["proposal_id"] in ctx.proposals and ctx.proposals[result["proposal_id"]].downgraded


def test_numeric_only_tools_do_not_taint():
    ctx = ToolContext()
    for name in ("get_overview", "get_signals", "get_forecast", "get_financials", "get_recommendation_history"):
        run_tool(name, {"ticker": "AAPL"}, ctx)
    assert not ctx.tainted


def test_reset_adjustments():
    ctx = ToolContext()
    run_tool("apply_adjustment", {"ticker": "AAPL", "changes": {"news_weight": 0.5}, "reasoning": "x"}, ctx)
    result, _ = run_tool("reset_adjustments", {"ticker": "AAPL", "reasoning": "back to normal"}, ctx)
    assert result["status"] == "reset" and overrides_store.overrides_for("AAPL") == {}


# ---- chat loop with a fake streaming client ----------------------------------------------


def _usage(i=10, o=5):
    return SimpleNamespace(
        input_tokens=i, output_tokens=o, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool_use(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class FakeStream:
    def __init__(self, texts, content, stop_reason):
        self.texts = texts
        self.final = SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self.texts)

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, *streams):
        self._streams = iter(streams)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(self._streams)


def _run(client, ctx=None, history=None):
    runner = TurnRunner(client, history or [], "question", ctx or ToolContext(), Usage())
    return runner, "".join(runner.stream())


def test_chat_turn_runs_tools_and_commits_a_valid_history():
    first = FakeStream(["Checking. "], [_text("Checking. "), _tool_use("t1", "get_signals", ticker="AAPL")], "tool_use")
    second = FakeStream(["It is a Buy."], [_text("It is a Buy.")], "end_turn")
    client = FakeClient(first, second)

    runner, shown = _run(client)

    assert runner.ok and runner.error is None
    assert "Checking." in shown and "It is a Buy." in shown and "Checked the recommendation signals (AAPL)" in shown
    roles = [m["role"] for m in runner.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    tool_result = runner.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert "is_error" not in tool_result
    assert runner.usage.output_tokens == 10
    request = client.requests[0]
    assert request["fallbacks"] == "default" and request["betas"] == [chat.FALLBACK_BETA]
    assert request["tools"] == tools.TOOLS


def test_all_tool_results_go_back_in_one_user_message():
    first = FakeStream(
        [],
        [_tool_use("a", "get_overview", ticker="AAPL"), _tool_use("b", "get_forecast", ticker="AAPL")],
        "tool_use",
    )
    second = FakeStream(["done"], [_text("done")], "end_turn")
    runner, _ = _run(FakeClient(first, second))
    assert [b["tool_use_id"] for b in runner.messages[2]["content"]] == ["a", "b"]


def test_news_and_apply_in_the_same_response_is_downgraded():
    first = FakeStream(
        [],
        [
            _tool_use("a", "get_news", ticker="AAPL"),
            _tool_use("b", "apply_adjustment", ticker="AAPL", changes={"news_weight": 0}, reasoning="x"),
        ],
        "tool_use",
    )
    second = FakeStream(["ok"], [_text("ok")], "end_turn")
    ctx = ToolContext()
    runner, _ = _run(FakeClient(first, second), ctx)

    assert runner.ok and ctx.tainted
    assert overrides_store.overrides_for("AAPL") == {}
    assert len(ctx.proposals) == 1


def test_tool_errors_are_flagged_to_the_model():
    first = FakeStream([], [_tool_use("a", "get_overview", ticker="NOPE")], "tool_use")
    second = FakeStream(["sorry"], [_text("sorry")], "end_turn")
    runner, _ = _run(FakeClient(first, second))
    assert runner.messages[2]["content"][0]["is_error"] is True


def test_refusal_is_not_committed():
    runner, shown = _run(FakeClient(FakeStream([], [], "refusal")))
    assert not runner.ok and "declined" in shown


def test_truncated_response_is_not_committed():
    runner, shown = _run(FakeClient(FakeStream(["partial"], [_text("partial")], "max_tokens")))
    assert not runner.ok and "cut off" in shown


def test_iteration_cap():
    streams = [FakeStream([], [_tool_use(f"t{i}", "list_companies")], "tool_use") for i in range(chat.MAX_ITERATIONS)]
    runner, shown = _run(FakeClient(*streams))
    assert not runner.ok and "tool rounds" in shown


def test_client_exception_becomes_a_friendly_message():
    class Boom:
        beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: (_ for _ in ()).throw(RuntimeError("x"))))

    runner, shown = _run(Boom())
    assert not runner.ok and runner.error and "RuntimeError" in shown


def test_budget_guard():
    usage = Usage(input_tokens=chat.SESSION_TOKEN_BUDGET)
    assert chat.over_budget(usage)
    assert not chat.over_budget(Usage(input_tokens=10, cache_read_tokens=10**9))


# ---- screen_companies / get_screen_context ---------------------------------------------------


def _add_company(ticker, name, sector, closes, pe=None, watchlisted=False):
    with tools.session_scope() as s:
        symbol = get_or_create_symbol(s, ticker, name, sector)
        s.flush()
        symbol.is_watchlisted = watchlisted
        save_return_model_params(s, symbol.id, mu=0.001, sigma=0.02, dof=5.0, xi=0.008, n_bars=200)
        if pe is not None:
            replace_fundamentals_snapshot(s, symbol.id, dt.date.today(), pe, None, name=name)
        for i, close in enumerate(closes):
            upsert_price_bar(s, symbol.id, dt.date(2026, 1, 1 + i), close, close, close, close, 1000)


@pytest.fixture
def three_companies():
    _add_company("MSFT", "Microsoft Corporation", "Technology", [100, 98], pe=40.0, watchlisted=True)
    _add_company("XOM", "Exxon Mobil", "Energy", [50])  # a single bar: no day change, no P/E


def _screen(**args):
    result, is_error = run_tool("screen_companies", args, ToolContext())
    return result, is_error


def test_screen_returns_the_table_data_sorted_by_day_change(three_companies):
    result, is_error = _screen()
    assert not is_error
    assert (result["total_companies"], result["matched"]) == (3, 3)
    tickers = [c["ticker"] for c in result["companies"]]
    assert tickers == ["AAPL", "MSFT", "XOM"]  # +0.99%, -2%, and no change data sorts last
    aapl = result["companies"][0]
    assert aapl["sector"] == "Technology" and aapl["price"] == 102.0
    assert aapl["pe_ratio"] == 12.0 and aapl["recommendation"] == "Buy"
    assert aapl["52w_position"] == 1.0
    assert result["companies"][2]["pct_change"] is None


def test_screen_filters_by_sector_case_insensitively(three_companies):
    result, _ = _screen(sectors=["technology"])
    assert [c["ticker"] for c in result["companies"]] == ["AAPL", "MSFT"]
    assert result["matched"] == 2 and result["total_companies"] == 3


def test_screen_unknown_sector_lists_the_valid_ones(three_companies):
    result, is_error = _screen(sectors=["Crypto"])
    assert is_error and "Energy" in result["error"] and "Technology" in result["error"]


def test_screen_filters_by_recommendation_and_watchlist(three_companies):
    result, _ = _screen(recommendations=["Buy"])
    assert [c["ticker"] for c in result["companies"]] == ["AAPL"]
    result, _ = _screen(watchlist_only=True)
    assert [c["ticker"] for c in result["companies"]] == ["MSFT"]
    result, _ = _screen(search="exxon")
    assert [c["ticker"] for c in result["companies"]] == ["XOM"]


def test_screen_sorts_by_any_column_and_limits(three_companies):
    result, _ = _screen(sort_by="P/E", descending=False)
    assert [c["ticker"] for c in result["companies"]] == ["AAPL", "MSFT", "XOM"]  # 12, 40, none last
    result, _ = _screen(sort_by="Ticker", descending=False, limit=2)
    assert [c["ticker"] for c in result["companies"]] == ["AAPL", "MSFT"]
    assert (result["returned"], result["matched"]) == (2, 3)


def test_screen_rejects_bad_arguments(three_companies):
    assert _screen(sort_by="Vibes")[1] is True
    assert _screen(limit="lots")[1] is True
    assert _screen(sectors="Technology")[1] is True  # must be a list


def test_screen_uses_the_tuned_math(three_companies):
    run_tool("apply_adjustment", {"ticker": "AAPL", "changes": {"buy_threshold": 4}, "reasoning": "x"}, ToolContext())
    result, _ = _screen(search="AAPL")
    assert result["companies"][0]["recommendation"] == "Hold"


def test_screen_does_not_taint_the_conversation(three_companies):
    ctx = ToolContext()
    run_tool("screen_companies", {}, ctx)
    run_tool("get_screen_context", {}, ctx)
    assert not ctx.tainted


def test_screen_context_when_nothing_has_been_browsed():
    result, is_error = run_tool("get_screen_context", {}, ToolContext())
    assert not is_error and result["available"] is False


def test_screen_context_returns_the_snapshot():
    snapshot = {"page": "Companies", "view": "Chart", "selected_company": {"ticker": "AAPL", "name": "Apple Inc."}}
    result, _ = run_tool("get_screen_context", {}, ToolContext(screen_context=snapshot))
    assert result["available"] is True and result["selected_company"]["ticker"] == "AAPL"
    assert result["view"] == "Chart"
