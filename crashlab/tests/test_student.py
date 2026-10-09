from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parents[2] / "app.py")


def fixture_prices():
    rng = np.random.default_rng(77)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0002, .012, 1000))),
                       index=pd.bdate_range("2020", periods=1000))
    prices.attrs["source"] = "SYNTHETIC TEST FIXTURE — NOT MARKET HISTORY"
    return prices


def loaded_app():
    app = AppTest.from_file(APP)
    app.session_state["student_prices"] = fixture_prices()
    app.session_state["student_market"] = "S&P 500"
    app.session_state["student_origin"] = "download"
    return app.run(timeout=60)


def button(app, label):
    return next(item for item in app.button if item.label == label)


def choice(app, label):
    return next(item for item in app.selectbox if item.label == label)


def test_student_starts_without_data_or_technical_controls():
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert button(app, "Run my simulation").disabled
    assert len(app.slider) == 0
    assert len(app.number_input) == 0
    assert len(app.selectbox) == 3
    assert not app.tabs


def test_student_download_failure_does_not_create_fake_history(monkeypatch):
    import crashlab.data as data
    st.cache_data.clear()
    def unavailable(*args, **kwargs):
        raise ValueError("Provider unavailable")
    monkeypatch.setattr(data, "fetch_market", unavailable)
    app = AppTest.from_file(APP).run(timeout=20)
    button(app, "Load market prices").click()
    app.run(timeout=20)
    assert not app.exception
    assert app.error
    assert button(app, "Run my simulation").disabled
    assert "student_prices" not in app.session_state


def test_student_simulation_counts_and_stale_settings():
    app = loaded_app()
    assert not app.exception
    button(app, "Run my simulation").click()
    app.run(timeout=60)
    assert not app.exception
    results = app.session_state["student_results"]
    assert len(results) >= 3
    assert app.session_state["student_config"]["horizon"] == 63
    assert app.session_state["student_config"]["paths"] == 10000
    assert [tab.label for tab in app.tabs] == ["Your results", "Price graphs", "Class 12 maths"]
    table = app.dataframe[0].value
    for row, model in zip(table.to_dict("records"), results):
        count = int(row["Paths with this fall / 10,000"].replace(",", ""))
        assert count == round(results[model]["probability"] * 10000)
    # A changed event must hide the earlier probabilities until recomputed.
    choice(app, "What size of fall should we count?").select(30)
    app.run(timeout=20)
    assert not app.exception
    assert any("choices or data changed" in message.value for message in app.warning)
    assert not app.tabs
    button(app, "Run my simulation").click()
    app.run(timeout=60)
    assert not app.exception
    assert app.session_state["student_config"]["threshold"] == .3


def test_student_market_change_requires_its_own_prices():
    app = loaded_app()
    choice(app, "Which market would you like to explore?").select("NIFTY 50")
    app.run(timeout=20)
    assert not app.exception
    assert button(app, "Run my simulation").disabled
    assert not app.tabs


def test_student_switches_to_advanced_view_and_back():
    app = AppTest.from_file(APP).run(timeout=20)
    app.radio[0].set_value("Advanced view").run(timeout=20)
    assert not app.exception
    assert any(item.label == "Forecast trading days" for item in app.slider)
    app.radio[0].set_value("Student view").run(timeout=20)
    assert not app.exception
    assert not app.slider
    assert not app.number_input
