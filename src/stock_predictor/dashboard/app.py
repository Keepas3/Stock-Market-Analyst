"""Streamlit multipage entrypoint. Run with:

    uv run streamlit run src/stock_predictor/dashboard/app.py

Page content lives in dashboard/views/*.py; dashboard/navigation.py builds
the st.Page objects. Direct analog of soccer-predictor's dashboard/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stock_predictor.dashboard import navigation  # noqa: E402
from stock_predictor.storage.db import init_db  # noqa: E402

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
init_db()

pg = st.navigation(
    [
        navigation.main_page(),
        navigation.watchlist_page(),
        navigation.symbol_detail_page(),
    ]
)
pg.run()
