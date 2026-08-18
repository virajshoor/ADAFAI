"""ADAFAI - Automated Detection Algorithm For Artificial Intelligence."""
from adafai.detector import analyze
from adafai.stylometry import analyze_stylometry
from adafai.unicode_forensics import analyze_unicode

__version__ = "0.2.0"
__all__ = ["analyze", "analyze_stylometry", "analyze_unicode"]
