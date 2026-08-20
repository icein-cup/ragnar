from ui.viking import viking_running_html


def test_viking_html_contains_svg():
    html = viking_running_html()
    assert "<svg" in html
    assert "</svg>" in html


def test_viking_html_has_running_animation():
    html = viking_running_html()
    assert "vk-leg-back" in html
    assert "vk-leg-front" in html
    assert "@keyframes" in html


def test_viking_html_has_helmet_and_shield():
    html = viking_running_html()
    assert "Helmet" in html or "helmet" in html
    assert "Shield" in html or "shield" in html
    assert "horn" in html.lower()