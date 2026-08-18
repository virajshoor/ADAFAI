"""assert-based self-check: `python tests/test_unicode_forensics.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adafai.unicode_forensics import analyze_unicode

CLEAN = "This is a normal sentence typed on a normal keyboard. It's fine."


def test_clean_text_is_clean():
    r = analyze_unicode(CLEAN)
    assert r["invisible_characters"] == {}
    assert r["homoglyphs"] == {}
    assert r["score"] == 0.0
    assert r["flags"] == []


def test_detects_zero_width_characters():
    r = analyze_unicode("Hello​world​again​, fine.")
    assert "ZERO WIDTH SPACE" in r["invisible_characters"]
    assert r["invisible_characters"]["ZERO WIDTH SPACE"]["count"] == 3
    assert r["score"] > 0.4
    assert any("invisible" in f for f in r["flags"])


def test_detects_homoglyph_evasion():
    # Cyrillic А, О, е substituted for Latin lookalikes
    r = analyze_unicode("Thе vаlue оf the mоdel is high.")
    assert r["homoglyphs"]
    assert any("homoglyph" in f for f in r["flags"])
    assert r["score"] >= 0.3


def test_typography_is_weak_signal_not_strong():
    """Curly quotes are autocorrect artifacts, so they must stay a hint."""
    curly = analyze_unicode("“This” is “polished” prose “throughout” the piece. " * 3)
    invisible = analyze_unicode("Hello​world​again​, fine.")
    assert curly["score"] < invisible["score"], "typography must rank below invisible chars"
    assert curly["score"] <= 0.15


def test_empty_text():
    r = analyze_unicode("")
    assert r["score"] == 0.0
    assert r["word_count"] == 0


if __name__ == "__main__":
    for fn in [
        test_clean_text_is_clean,
        test_detects_zero_width_characters,
        test_detects_homoglyph_evasion,
        test_typography_is_weak_signal_not_strong,
        test_empty_text,
    ]:
        fn()
        print(f"  ok  {fn.__name__}")
    print("ok")
