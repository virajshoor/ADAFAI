"""ADAFAI - Automated Detection Algorithm For Artificial Intelligence."""
from adafai.detector import analyze
from adafai.discourse import analyze_discourse
from adafai.spans import analyze_spans
from adafai.stylometry import analyze_stylometry
from adafai.unicode_forensics import analyze_unicode

__version__ = "0.3.0"
__all__ = [
    "analyze", "analyze_discourse", "analyze_spans", "analyze_stylometry",
    "analyze_unicode",
]
