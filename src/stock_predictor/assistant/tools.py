"""The Assistant's tools: read-only lookups over the app's own data, plus
three write tools that change a company's recommendation-math overrides
(model/overrides_store.py).

Safety rules enforced HERE, in code, not in the prompt:
  * Tickers are validated against config/watchlist.yaml.
  * Every value goes through model/tuning.py's bounds + cross-field checks.
  * Once any third-party text (news, social posts, company profiles,
    competitor names) has been returned in this conversation -- `ctx.tainted`
    -- `apply_adjustment` is downgraded to a proposal the owner must click
    Apply on. That text could contain a prompt injection, and the taint
    deliberately lasts for the whole conversation because the text stays in
    the model's context on later turns.
  * Per-turn and per-company-per-day caps on how many parameter changes the
    AI may apply, so repeated turns can't walk a parameter to its bound.
Tool results are plain JSON-serializable dicts.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field

from stock_predictor.config import load_watchlist
from stock_predictor.dashboard.components import watchlist_dataframe
from stock_predictor.dashboard.symbol_charts import (
    METRICS,
    POSITION_COLUMN,
    RECOMMENDATIONS,
    available_sectors,
    filter_symbols,
    with_derived_metrics,
)
from stock_predictor.model import overrides_store
from stock_predictor.model.technical_indicators import compute_moving_averages
from stock_predictor.model.tuning import (
    PARAM_SPECS,
    TunableParams,
    coerce_value,
    combined_errors,
    resolve_params,
)
from stock_predictor.prediction.service import predict_symbol
from stock_predictor.storage.db import session_scope
from stock_predictor.storage.repository import (
    all_symbols,
    competitor_snapshots,
    latest_fundamentals,
    latest_sentiment,
    latest_social_sentiment,
    price_bars_for_symbol,
    quarterly_financials,
    recent_articles,
    recent_social_posts,
    recommendation_history,
    symbol_by_ticker,
)

MAX_SCREEN_ROWS = 60
DEFAULT_SCREEN_ROWS = 15
MAX_CHANGES_PER_TURN = 6
MAX_AI_CHANGES_PER_TICKER_PER_DAY = 12
FIFTY_TWO_WEEK_DAYS = 365
_TEXT_LIMIT = 300

# Tools whose results contain text written by third parties.
TAINTING_TOOLS = frozenset({"get_news", "get_social_posts", "get_company_profile", "get_competitors"})
WRITE_TOOLS = frozenset({"propose_adjustment", "apply_adjustment", "reset_adjustments"})


class ToolError(Exception):
    """A problem worth telling the model about (bad ticker, out-of-range value...)."""


@dataclass
class Proposal:
    id: str
    ticker: str
    changes: dict[str, float]
    reason: str
    created: str
    downgraded: bool = False  # True if the AI tried to apply directly but was downgraded


@dataclass
class ToolContext:
    """Per-conversation state (kept in st.session_state by the UI)."""

    tainted: bool = False
    proposals: dict[str, Proposal] = field(default_factory=dict)
    applied: list[dict] = field(default_factory=list)  # entries applied, awaiting git sync
    turn_changes: int = 0
    # What the owner last had open on the Companies/Watchlist pages (filters,
    # view, selected company); set by the Assistant page before each turn.
    screen_context: dict | None = None

    def start_turn(self) -> None:
        self.turn_changes = 0


_TICKER = {"type": "string", "description": "Ticker symbol of a tracked company, e.g. AAPL."}
_CHANGES = {
    "type": "object",
    "description": "Map of parameter name to new numeric value. See the parameter reference in the system prompt.",
    "additionalProperties": {"type": "number"},
    "minProperties": 1,
}
_REASON = {
    "type": "string",
    "description": "Short, concrete justification grounded in data you looked up (shown to the owner).",
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


TOOLS: list[dict] = [
    _tool(
        "list_companies",
        "List tracked companies (ticker, name, sector, whether on the owner's Watchlist). Optional text filter.",
        {"query": {"type": "string", "description": "Case-insensitive substring of ticker or name."}},
        [],
    ),
    _tool(
        "screen_companies",
        "Screen and rank ALL tracked companies in one call, using exactly the data the Companies table and charts show "
        "(price, day change, volume, 52-week range and position, P/E, news and social sentiment, MA signal, "
        "Buy/Hold/Sell). Filter by sector, recommendation, search text or Watchlist, then sort by any column. Prefer "
        "this over calling get_signals company by company.",
        {
            "sectors": {"type": "array", "items": {"type": "string"}, "description": "Only these sectors (e.g. Technology)."},
            "recommendations": {
                "type": "array",
                "items": {"type": "string", "enum": list(RECOMMENDATIONS)},
                "description": "Only these recommendations.",
            },
            "search": {"type": "string", "description": "Substring of ticker or company name."},
            "watchlist_only": {"type": "boolean", "description": "Only companies on the owner's Watchlist."},
            "sort_by": {
                "type": "string",
                "enum": [*METRICS, "Ticker"],
                "description": "Column to sort by. Default '% Change'.",
            },
            "descending": {"type": "boolean", "description": "Highest first (default true)."},
            "limit": {
                "type": "integer",
                "description": f"Rows to return, 1 to {MAX_SCREEN_ROWS} (default {DEFAULT_SCREEN_ROWS}).",
            },
        },
        [],
    ),
    _tool(
        "get_screen_context",
        "What the owner currently has open on the Companies/Watchlist pages: which page and view (Table or Chart), the "
        "search/sector/recommendation filters, the chart settings, the companies visible, and the selected company. "
        "Call this whenever the owner says 'this company', 'the selected one', 'what I filtered', 'the chart', or "
        "otherwise refers to what's on their screen.",
        {},
        [],
    ),
    _tool(
        "get_overview",
        "Latest price, daily change, volume, 52-week range, P/E, market cap and sector for one company, with as-of dates.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_signals",
        "The recommendation inputs for one company: the four votes, weighted total, verdict and thresholds, the raw "
        "inputs behind each vote (moving averages, P/E, sentiment scores), the effective math parameters (marking any "
        "overridden), and what the verdict would be with default parameters.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_forecast",
        "The secondary statistical forecast (Student-t model): expected return, 80% range and P(up/flat/down) over the horizon.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_news",
        "Recent news headlines for one company plus the aggregate news sentiment score. Headlines are third-party text.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_social_posts",
        "A few recent StockTwits posts for one company plus the aggregate social sentiment score. Posts are third-party text.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_company_profile",
        "The company description, industry, exchange and website. Third-party text.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_competitors",
        "The company's hand-curated competitors with price, change, market cap and P/E.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_financials",
        "Recent quarterly revenue, net income, EPS and net profit margin for one company.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "get_recommendation_history",
        "The last logged Buy/Hold/Sell recommendations for one company, and its tuning change history.",
        {"ticker": _TICKER},
        ["ticker"],
    ),
    _tool(
        "propose_adjustment",
        "Suggest parameter changes for one company. Nothing changes until the owner clicks Apply. Use for large or "
        "speculative moves, or when unsure.",
        {"ticker": _TICKER, "changes": _CHANGES, "reasoning": _REASON},
        ["ticker", "changes", "reasoning"],
    ),
    _tool(
        "apply_adjustment",
        "Apply parameter changes to one company right now (bounded, logged and undoable by the owner). Use for modest, "
        "evidence-backed fine-tuning, or when the owner asks you to. May be downgraded to a proposal.",
        {"ticker": _TICKER, "changes": _CHANGES, "reasoning": _REASON},
        ["ticker", "changes", "reasoning"],
    ),
    _tool(
        "reset_adjustments",
        "Restore default parameters for one company (all overrides, or only the named parameters).",
        {
            "ticker": _TICKER,
            "params": {"type": "array", "items": {"type": "string"}, "description": "Parameters to reset; omit for all."},
            "reasoning": _REASON,
        },
        ["ticker", "reasoning"],
    ),
]


def _clip(text: object, limit: int = _TEXT_LIMIT) -> str:
    return " ".join(str(text or "").split())[:limit]


def _iso(value: object) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _ticker(args: dict) -> str:
    raw = args.get("ticker")
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("ticker is required.")
    ticker = raw.strip().upper()
    if ticker not in load_watchlist():
        raise ToolError(f"'{ticker}' is not a tracked company. Use list_companies to find valid tickers.")
    return ticker


def _symbol(session, ticker: str):
    symbol = symbol_by_ticker(session, ticker)
    if symbol is None:
        raise ToolError(f"No data has been loaded for {ticker} yet.")
    return symbol


def _market_snapshot(price_df) -> dict:
    if price_df.empty:
        return {}
    last = price_df.iloc[-1]
    price = float(last["close"])
    out: dict = {"as_of": _iso(last["date"]), "price": round(price, 2), "volume": int(last["volume"])}
    if len(price_df) >= 2:
        previous = float(price_df.iloc[-2]["close"])
        change = price - previous
        out["change"] = round(change, 2)
        out["pct_change"] = round(change / previous, 4) if previous else None
    window = price_df[price_df["date"] >= last["date"] - dt.timedelta(days=FIFTY_TWO_WEEK_DAYS)]
    out["52w_low"] = round(float(window["low"].min()), 2)
    out["52w_high"] = round(float(window["high"].max()), 2)
    return out


def _verdict(session, symbol, tuning: TunableParams | None = None) -> dict | None:
    prediction = predict_symbol(session, symbol.id, tuning)
    if prediction is None or prediction.composite is None:
        return None
    c = prediction.composite
    return {"recommendation": c.recommendation, "total": c.total}


def _list_companies(args: dict) -> dict:
    query = str(args.get("query") or "").strip().lower()
    with session_scope() as session:
        rows = [
            {"ticker": s.ticker, "name": s.name, "sector": s.sector, "on_watchlist": bool(s.is_watchlisted)}
            for s in all_symbols(session)
            if not query or query in s.ticker.lower() or query in s.name.lower()
        ]
    return {"count": len(rows), "companies": rows[:60]}


def _num(value: object, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else round(number, digits)


def _canonical(values: object, allowed: list[str], what: str) -> list[str]:
    """Case-insensitive match of requested names against the real ones."""
    if values is None:
        return []
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ToolError(f"{what} must be a list of strings.")
    lookup = {a.lower(): a for a in allowed}
    unknown = [v for v in values if v.strip().lower() not in lookup]
    if unknown:
        raise ToolError(f"Unknown {what}: {', '.join(unknown)}. Valid values: {', '.join(allowed)}.")
    return [lookup[v.strip().lower()] for v in values]


def _screen_companies(args: dict) -> dict:
    sort_by = args.get("sort_by") or "% Change"
    if sort_by != "Ticker" and sort_by not in METRICS:
        raise ToolError(f"sort_by must be one of: {', '.join([*METRICS, 'Ticker'])}.")
    raw_limit = args.get("limit", DEFAULT_SCREEN_ROWS)
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        raise ToolError("limit must be an integer.")
    limit = max(1, min(raw_limit, MAX_SCREEN_ROWS))

    with session_scope() as session:
        df = with_derived_metrics(watchlist_dataframe(session))
    if df.empty:
        return {"total_companies": 0, "matched": 0, "companies": []}
    sectors = _canonical(args.get("sectors"), available_sectors(df), "sectors")
    recommendations = _canonical(args.get("recommendations"), list(RECOMMENDATIONS), "recommendations")

    if args.get("watchlist_only"):
        df = df[df["Watchlist"]]
    matched = filter_symbols(df, str(args.get("search") or ""), sectors, recommendations)
    ascending = not bool(args.get("descending", True))
    matched = matched.sort_values(sort_by, ascending=ascending, na_position="last")

    rows = [
        {
            "ticker": r["Ticker"],
            "name": r["Name"],
            "sector": r["Sector"] if isinstance(r["Sector"], str) else None,
            "price": _num(r["Price"], 2),
            "change": _num(r["Change"], 2),
            "pct_change": _num(r["% Change"]),
            "volume": _num(r["Volume"], 0),
            "52w_low": _num(r["52W Low"], 2),
            "52w_high": _num(r["52W High"], 2),
            "52w_position": _num(r[POSITION_COLUMN], 3),
            "pe_ratio": _num(r["P/E"], 2),
            "news_sentiment": _num(r["News Sentiment"], 3),
            "social_sentiment": _num(r["Social Sentiment"], 3),
            "ma_signal": r["MA Signal"] if isinstance(r["MA Signal"], str) else None,
            "recommendation": r["Recommendation"] if isinstance(r["Recommendation"], str) else None,
            "on_watchlist": bool(r["Watchlist"]),
        }
        for _, r in matched.head(limit).iterrows()
    ]
    return {
        "total_companies": len(df),
        "matched": len(matched),
        "returned": len(rows),
        "sorted_by": sort_by,
        "descending": not ascending,
        "note": "null means no data yet. pct_change and 52w_position are fractions (0.012 = 1.2%).",
        "companies": rows,
    }


def _get_screen_context(args: dict, ctx: ToolContext) -> dict:
    if not ctx.screen_context:
        return {
            "available": False,
            "message": (
                "The owner hasn't opened the Companies or Watchlist page in this session, "
                "so there is no screen state to read."
            ),
        }
    return {"available": True, **ctx.screen_context}


def _get_overview(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        fundamentals = latest_fundamentals(session, symbol.id)
        return {
            "ticker": ticker,
            "name": symbol.name,
            "sector": symbol.sector,
            "market": _market_snapshot(price_bars_for_symbol(session, symbol.id)),
            "pe_ratio": fundamentals.pe_ratio if fundamentals else None,
            "forward_pe": fundamentals.forward_pe if fundamentals else None,
            "market_cap": fundamentals.market_cap if fundamentals else None,
            "fundamentals_as_of": _iso(fundamentals.date) if fundamentals else None,
        }


def _get_signals(args: dict) -> dict:
    ticker = _ticker(args)
    tuning = overrides_store.params_for(ticker)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        prediction = predict_symbol(session, symbol.id, tuning)
        if prediction is None or prediction.composite is None:
            return {"ticker": ticker, "error": "No trained model yet for this company, so no recommendation."}
        c = prediction.composite
        sma_short, sma_long = compute_moving_averages(price_bars_for_symbol(session, symbol.id))
        fundamentals = latest_fundamentals(session, symbol.id)
        news = latest_sentiment(session, symbol.id)
        social = latest_social_sentiment(session, symbol.id)
        default_verdict = _verdict(session, symbol, TunableParams())
        return {
            "ticker": ticker,
            "recommendation": c.recommendation,
            "weighted_total": c.total,
            "buy_threshold": c.buy_threshold,
            "sell_threshold": c.sell_threshold,
            "votes": {
                "ma": c.ma_vote,
                "pe": c.pe_vote,
                "news": c.news_vote,
                "social": c.social_vote,
            },
            "raw_inputs": {
                "sma_50": round(sma_short, 2) if sma_short is not None else None,
                "sma_200": round(sma_long, 2) if sma_long is not None else None,
                "pe_ratio": fundamentals.pe_ratio if fundamentals else None,
                "news_sentiment": news.overall_sentiment_score if news else None,
                "news_as_of": _iso(news.date) if news else None,
                "social_sentiment": social.overall_sentiment_score if social else None,
                "social_as_of": _iso(social.date) if social else None,
            },
            "parameters": {
                name: {
                    "value": getattr(tuning, name),
                    "default": spec.default,
                    "overridden": getattr(tuning, name) != spec.default,
                }
                for name, spec in PARAM_SPECS.items()
            },
            "verdict_with_default_parameters": default_verdict,
        }


def _get_forecast(args: dict) -> dict:
    ticker = _ticker(args)
    tuning = overrides_store.params_for(ticker)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        p = predict_symbol(session, symbol.id, tuning)
        if p is None or p.breakdown is None:
            return {"ticker": ticker, "error": "No trained model yet for this company."}
        b = p.breakdown
        return {
            "ticker": ticker,
            "horizon_trading_days": b.horizon_days,
            "expected_return": round(p.expected_return, 5),
            "range_80pct": [round(p.return_range_80pct[0], 5), round(p.return_range_80pct[1], 5)],
            "p_up": round(p.p_up, 4),
            "p_flat": round(p.p_flat, 4),
            "p_down": round(p.p_down, 4),
            "model_fitted_at": b.fitted_at[:10],
            "note": "Student-t model fit only on past prices plus a small capped news-sentiment nudge; not backtested.",
        }


def _get_news(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        sentiment = latest_sentiment(session, symbol.id)
        return {
            "ticker": ticker,
            "news_sentiment": sentiment.overall_sentiment_score if sentiment else None,
            "article_count": sentiment.article_count if sentiment else 0,
            "as_of": _iso(sentiment.date) if sentiment else None,
            "articles": [
                {
                    "title": _clip(a.title, 200),
                    "source": _clip(a.source, 60),
                    "published_at": _iso(a.published_at),
                    "summary": _clip(a.summary),
                }
                for a in recent_articles(session, symbol.id)
            ],
        }


def _get_social_posts(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        social = latest_social_sentiment(session, symbol.id)
        return {
            "ticker": ticker,
            "social_sentiment": social.overall_sentiment_score if social else None,
            "post_count": social.message_count if social else 0,
            "as_of": _iso(social.date) if social else None,
            "posts": [
                {
                    "text": _clip(p.body, 280),
                    "self_tagged": p.tagged_sentiment,
                    "posted_at": _iso(p.posted_at),
                }
                for p in recent_social_posts(session, symbol.id)
            ],
        }


def _get_company_profile(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        f = latest_fundamentals(session, symbol.id)
        if f is None:
            return {"ticker": ticker, "error": "No company profile loaded yet."}
        return {
            "ticker": ticker,
            "name": f.name or symbol.name,
            "description": _clip(f.description, 800),
            "industry": f.industry,
            "exchange": f.exchange,
            "country": f.country,
            "official_site": f.official_site,
        }


def _get_competitors(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        return {
            "ticker": ticker,
            "competitors": [
                {
                    "ticker": c.competitor_ticker,
                    "name": _clip(c.competitor_name or c.competitor_ticker, 80),
                    "price": c.price,
                    "pct_change": c.pct_change,
                    "market_cap": c.market_cap,
                    "pe_ratio": c.pe_ratio,
                }
                for c in competitor_snapshots(session, symbol.id)
            ],
        }


def _get_financials(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        return {
            "ticker": ticker,
            "quarters": [
                {
                    "fiscal_date_ending": _iso(q.fiscal_date_ending),
                    "total_revenue": q.total_revenue,
                    "net_income": q.net_income,
                    "eps": q.eps,
                    "net_profit_margin": (
                        round(q.net_income / q.total_revenue, 4) if q.total_revenue and q.net_income is not None else None
                    ),
                }
                for q in quarterly_financials(session, symbol.id)
            ],
        }


def _get_recommendation_history(args: dict) -> dict:
    ticker = _ticker(args)
    with session_scope() as session:
        symbol = _symbol(session, ticker)
        logged = [
            {"recommendation": r.recommendation, "recorded_at": _iso(r.recorded_at)}
            for r in recommendation_history(session, symbol.id)
        ]
    return {
        "ticker": ticker,
        "recent_recommendations": logged,
        "tuning_changes": [
            {k: e.get(k) for k in ("ts", "param", "old", "new", "reason", "source")}
            for e in overrides_store.history(ticker)[-15:]
        ],
    }


def _validated_changes(ticker: str, raw: object) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ToolError("changes must be a non-empty object of {parameter: number}.")
    try:
        clean = {name: coerce_value(name, value) for name, value in raw.items()}
        errors = combined_errors(TunableParams(**{**overrides_store.overrides_for(ticker), **clean}))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if errors:
        raise ToolError(" ".join(errors))
    return clean


def _new_proposal(ctx: ToolContext, ticker: str, changes: dict[str, float], reason: str, downgraded: bool) -> Proposal:
    proposal = Proposal(
        id=uuid.uuid4().hex[:8],
        ticker=ticker,
        changes=changes,
        reason=_clip(reason),
        created=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        downgraded=downgraded,
    )
    ctx.proposals[proposal.id] = proposal
    return proposal


def _ai_changes_today(ticker: str) -> int:
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    return sum(1 for e in overrides_store.history(ticker) if e.get("source") == "ai" and str(e.get("ts", "")).startswith(today))


def _propose_adjustment(args: dict, ctx: ToolContext) -> dict:
    ticker = _ticker(args)
    changes = _validated_changes(ticker, args.get("changes"))
    proposal = _new_proposal(ctx, ticker, changes, args.get("reasoning", ""), downgraded=False)
    return {
        "status": "proposed",
        "proposal_id": proposal.id,
        "message": "Shown to the owner with an Apply button. Nothing has changed yet.",
    }


def _apply_adjustment(args: dict, ctx: ToolContext) -> dict:
    ticker = _ticker(args)
    changes = _validated_changes(ticker, args.get("changes"))
    reason = args.get("reasoning", "")

    if ctx.tainted:
        proposal = _new_proposal(ctx, ticker, changes, reason, downgraded=True)
        return {
            "status": "downgraded_to_proposal",
            "proposal_id": proposal.id,
            "message": (
                "This conversation includes third-party text (news, posts, profiles or competitor names), so direct "
                "changes are disabled. Your change was shown to the owner as a proposal with an Apply button. "
                "Tell the owner it needs their click."
            ),
        }
    if ctx.turn_changes + len(changes) > MAX_CHANGES_PER_TURN:
        raise ToolError(f"At most {MAX_CHANGES_PER_TURN} parameter changes can be applied per turn. Propose the rest instead.")
    if _ai_changes_today(ticker) + len(changes) > MAX_AI_CHANGES_PER_TICKER_PER_DAY:
        raise ToolError(
            f"Daily limit of {MAX_AI_CHANGES_PER_TICKER_PER_DAY} AI parameter changes for {ticker} reached. "
            "Use propose_adjustment so the owner can decide."
        )

    with session_scope() as session:
        before = _verdict(session, _symbol(session, ticker), overrides_store.params_for(ticker))
    try:
        entries = overrides_store.apply_changes(ticker, changes, _clip(reason), "ai")
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    ctx.turn_changes += len(entries)
    ctx.applied.extend(e.as_dict() for e in entries)
    with session_scope() as session:
        after = _verdict(session, _symbol(session, ticker), overrides_store.params_for(ticker))
    return {
        "status": "applied" if entries else "no_change",
        "changes": [{"param": e.param, "old": e.old, "new": e.new} for e in entries],
        "verdict_before": before,
        "verdict_after": after,
        "message": "Saved. The owner can undo this from the AI adjustments panel.",
    }


def _reset_adjustments(args: dict, ctx: ToolContext) -> dict:
    ticker = _ticker(args)
    params = args.get("params")
    if params is not None and (not isinstance(params, list) or not all(isinstance(p, str) for p in params)):
        raise ToolError("params must be a list of parameter names.")
    try:
        entries = overrides_store.reset(ticker, params, _clip(args.get("reasoning", "")), "ai")
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    ctx.applied.extend(e.as_dict() for e in entries)
    return {"status": "reset" if entries else "nothing_to_reset", "reset": [e.param for e in entries]}


_READ_HANDLERS = {
    "list_companies": _list_companies,
    "screen_companies": _screen_companies,
    "get_overview": _get_overview,
    "get_signals": _get_signals,
    "get_forecast": _get_forecast,
    "get_news": _get_news,
    "get_social_posts": _get_social_posts,
    "get_company_profile": _get_company_profile,
    "get_competitors": _get_competitors,
    "get_financials": _get_financials,
    "get_recommendation_history": _get_recommendation_history,
}
_CONTEXT_HANDLERS = {"get_screen_context": _get_screen_context}
_WRITE_HANDLERS = {
    "propose_adjustment": _propose_adjustment,
    "apply_adjustment": _apply_adjustment,
    "reset_adjustments": _reset_adjustments,
}


def run_tool(name: str, args: object, ctx: ToolContext) -> tuple[dict, bool]:
    """(result, is_error). Never raises."""
    if not isinstance(args, dict):
        return {"error": "Tool input must be an object."}, True
    try:
        if name in _READ_HANDLERS:
            result = _READ_HANDLERS[name](args)
            if name in TAINTING_TOOLS:
                ctx.tainted = True
            return result, False
        if name in _CONTEXT_HANDLERS:
            return _CONTEXT_HANDLERS[name](args, ctx), False
        if name in _WRITE_HANDLERS:
            return _WRITE_HANDLERS[name](args, ctx), False
        return {"error": f"Unknown tool '{name}'."}, True
    except ToolError as exc:
        return {"error": str(exc)}, True
    except Exception as exc:  # a broken lookup must not kill the chat turn
        return {"error": f"{type(exc).__name__} while running {name}."}, True


def apply_proposal(ctx: ToolContext, proposal_id: str) -> list[dict]:
    """Owner clicked Apply: one-use, bounds still enforced, logged as 'owner'
    (the owner's click, not the model's judgment, is what authorizes it).
    Returns the applied entries. Raises ValueError with a displayable message.
    """
    proposal = ctx.proposals.pop(proposal_id, None)
    if proposal is None:
        raise ValueError("That proposal was already applied or dismissed.")
    entries = overrides_store.apply_changes(
        proposal.ticker, proposal.changes, f"Owner-approved AI proposal: {proposal.reason}", "owner"
    )
    ctx.applied.extend(e.as_dict() for e in entries)
    return [e.as_dict() for e in entries]


def dismiss_proposal(ctx: ToolContext, proposal_id: str) -> None:
    ctx.proposals.pop(proposal_id, None)


def proposal_diff(proposal: Proposal) -> list[dict]:
    """Rows for the Apply card, built from the stored proposal and the
    current overrides -- never from model prose."""
    current = resolve_params(overrides_store.overrides_for(proposal.ticker))
    return [
        {"param": name, "current": getattr(current, name), "proposed": value, "default": PARAM_SPECS[name].default}
        for name, value in proposal.changes.items()
    ]
