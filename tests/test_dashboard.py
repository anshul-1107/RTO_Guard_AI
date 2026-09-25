from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.mark.skipif(not Path("ml/artifacts/model.txt").exists(), reason="run `make train` first")
def test_dashboard_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs][0].endswith("Overview")
