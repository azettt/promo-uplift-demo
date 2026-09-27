from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"

pytest.importorskip("streamlit")


def test_app_runs_and_matches_the_project_headline():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    # default scenario: budget 32,000, men's e-mail capped at 16,000, equal costs
    program, greedy = at.metric[0], at.metric[1]
    assert program.value == "2,515"   # the project's notebook 05: 2,514.8
    assert greedy.value == "1,793"    # the project's notebook 05: 1,793.3


def test_app_describes_a_customer():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert any("Buyer type: men's only" in s.value for s in at.subheader)
    at.sidebar.checkbox[1].check().run()
    assert not at.exception
    assert any("Buyer type: both" in s.value for s in at.subheader)
