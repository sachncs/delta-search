"""Static smoke tests for the GitHub Pages site."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SITE = ROOT / "site"


def test_site_assets_exist() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assets = re.findall(r'(?:src|href)="(assets/[^"#]+)"', html)
    assert assets
    for asset in assets:
        assert (SITE / asset).is_file(), asset


def test_mobile_navigation_is_hidden_by_default() -> None:
    css = (SITE / "css/components.css").read_text(encoding="utf-8")
    assert ".nav__panel {" in css
    assert "display: none;" in css[css.index(".nav__panel {") :]
    assert "body.nav-open .nav__panel" in css


def test_code_tabs_have_complete_aria_relationships() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'id="tab-py"' in html and 'aria-controls="panel-py"' in html
    assert 'id="tab-cli"' in html and 'aria-controls="panel-cli"' in html
    assert 'id="panel-py" role="tabpanel"' in html
    assert 'id="panel-cli" role="tabpanel"' in html
    assert 'id="copy-status"' in html


def test_documentation_is_standalone_and_internal() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'href="docs/"' in html
    assert "github.com/sachncs/delta-search/tree/master/docs" not in html
    for page in ("index.html", "getting-started.html", "concepts.html", "reference.html", "faq.html"):
        doc = SITE / "docs" / page
        assert doc.is_file(), page
        assert "css/docs.css" in doc.read_text(encoding="utf-8")
