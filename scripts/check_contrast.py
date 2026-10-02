"""
WCAG AA Color Contrast Checker.

Validates contrast ratios for semantic tokens across dark and light themes against WCAG 2.1 AA requirements:
  - Normal text / body: >= 4.5:1
  - Large text / graphical components / badges: >= 3.0:1
"""

import sys
from typing import Dict, Tuple

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    return int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def channel_lum(val: int) -> float:
        c = val / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel_lum(r) + 0.7152 * channel_lum(g) + 0.0722 * channel_lum(b)


def contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = relative_luminance(hex_to_rgb(hex1))
    l2 = relative_luminance(hex_to_rgb(hex2))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


THEMES = {
    "Dark Theme (Background: #101318, Panel: #181D24)": {
        "bg": "#101318",
        "panel": "#181D24",
        "pairs": [
            ("Primary Text (--text)", "#F1F3F5", "bg", 4.5),
            ("Muted Text (--muted)", "#94A3B8", "bg", 4.5),
            ("Voice Tone (--voice)", "#FBBF24", "panel", 3.0),
            ("Text Sentiment (--text-accent)", "#38BDF8", "panel", 3.0),
            ("Positive Sentiment (--positive)", "#4ADE80", "panel", 3.0),
            ("Negative / Alert (--negative)", "#F87171", "panel", 3.0),
            ("Warning Signal (--warning)", "#FBBF24", "panel", 3.0),
            ("Fused Result (--fused)", "#E2E8F0", "panel", 4.5),
        ],
    },
    "Light Theme (Background: #F8FAFC, Panel: #FFFFFF)": {
        "bg": "#F8FAFC",
        "panel": "#FFFFFF",
        "pairs": [
            ("Primary Text (--text)", "#0F172A", "bg", 4.5),
            ("Muted Text (--muted)", "#475569", "bg", 4.5),
            ("Voice Tone (--voice)", "#B45309", "panel", 3.0),
            ("Text Sentiment (--text-accent)", "#0284C7", "panel", 3.0),
            ("Positive Sentiment (--positive)", "#15803D", "panel", 3.0),
            ("Negative / Alert (--negative)", "#DC2626", "panel", 3.0),
            ("Warning Signal (--warning)", "#B45309", "panel", 3.0),
            ("Fused Result (--fused)", "#334155", "panel", 4.5),
        ],
    },
}


def run_contrast_audit() -> bool:
    print("=" * 68)
    print("WCAG 2.1 AA COLOR CONTRAST AUDIT REPORT")
    print("=" * 68)

    all_passed = True

    for theme_name, theme_data in THEMES.items():
        print(f"\n{theme_name}:")
        print(f"{'Token / Role':<34} | {'Color':<8} | {'Against':<7} | {'Ratio':<6} | {'Target':<6} | Status")
        print("-" * 75)

        for label, fg_hex, bg_key, min_target in theme_data["pairs"]:
            bg_hex = theme_data[bg_key]
            ratio = contrast_ratio(fg_hex, bg_hex)
            passed = ratio >= min_target
            if not passed:
                all_passed = False
            status = "PASS [OK]" if passed else "FAIL [X]"
            print(f"{label:<34} | {fg_hex:<8} | {bg_hex:<7} | {ratio:>5.2f}:1 | >={min_target:<4.1f}:1 | {status}")

    print("\n" + "=" * 68)
    if all_passed:
        print("ALL TOKEN PAIRS COMPLIANT WITH WCAG AA GUIDELINES.")
    else:
        print("FAILURES DETECTED IN COLOR CONTRAST RATIOS.")
    print("=" * 68)
    return all_passed


if __name__ == "__main__":
    success = run_contrast_audit()
    sys.exit(0 if success else 1)
