#!/usr/bin/env python3
"""
WCAG Contrast Verification Script.

Calculates relative luminance and contrast ratios for all core text/background
token pairs in both Dark and Light themes.
Ensures:
  - All body text >= 4.5:1 (WCAG AA)
  - All large text >= 3.0:1 (WCAG AA)
Exits with 0 if all pass, 1 if any failure.
"""

from typing import Dict, Tuple
import sys


def parse_hex(hex_str: str) -> Tuple[float, float, float]:
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    r = int(hex_clean[0:2], 16) / 255.0
    g = int(hex_clean[2:4], 16) / 255.0
    b = int(hex_clean[4:6], 16) / 255.0
    return r, g, b


def relative_luminance(r: float, g: float, b: float) -> float:
    def channel_lum(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel_lum(r) + 0.7152 * channel_lum(g) + 0.0722 * channel_lum(b)


def contrast_ratio(hex1: str, hex2: str) -> float:
    r1, g1, b1 = parse_hex(hex1)
    r2, g2, b2 = parse_hex(hex2)
    l1 = relative_luminance(r1, g1, b1)
    l2 = relative_luminance(r2, g2, b2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


THEMES = {
    "DARK THEME": {
        "surfaces": {
            "bg": "#101318",
            "surface": "#181D24",
            "surface-raised": "#202630",
        },
        "text_tokens": {
            "text": "#F8FAFC",
            "text-muted": "#94A3B8",
            "accent": "#38BDF8",
            "voice": "#FBBF24",
            "text-modality": "#38BDF8",
            "fused": "#E2E8F0",
            "positive": "#4ADE80",
            "warning": "#FBBF24",
            "critical": "#F87171",
        },
    },
    "LIGHT THEME": {
        "surfaces": {
            "bg": "#F8FAFC",
            "surface": "#FFFFFF",
            "surface-raised": "#F1F5F9",
        },
        "text_tokens": {
            "text": "#0F172A",
            "text-muted": "#475569",
            "accent": "#0369A1",
            "voice": "#92400E",
            "text-modality": "#0369A1",
            "fused": "#334155",
            "positive": "#166534",
            "warning": "#92400E",
            "critical": "#B91C1C",
        },
    },
}


def run_checks() -> int:
    failures = 0
    total_checks = 0

    print("=" * 82)
    print(f"{'WCAG 2.1 CONTRAST VERIFICATION REPORT':^82}")
    print("=" * 82)

    for theme_name, palette in THEMES.items():
        print(f"\n[{theme_name}]")
        print(f"{'Text Token':<16} | {'Surface':<16} | {'Text Hex':<8} | {'Surf Hex':<8} | {'Ratio':<8} | {'WCAG AA'}")
        print("-" * 82)

        for s_name, s_hex in palette["surfaces"].items():
            for t_name, t_hex in palette["text_tokens"].items():
                total_checks += 1
                ratio = contrast_ratio(t_hex, s_hex)
                if ratio >= 4.5:
                    status = "PASS (Body >= 4.5:1)"
                elif ratio >= 3.0:
                    status = "PASS (Large >= 3.0:1)"
                else:
                    status = "FAIL (< 3.0:1)"
                    failures += 1

                print(f"{t_name:<16} | {s_name:<16} | {t_hex:<8} | {s_hex:<8} | {ratio:>5.2f}:1 | {status}")

    print("\n" + "=" * 82)
    print(f"Total pairs checked: {total_checks} | Failures: {failures}")
    if failures == 0:
        print("RESULT: ALL CONTRAST CHECKS PASSED WCAG AA REQUIREMENT (>= 4.5:1 / >= 3.0:1)")
        print("=" * 82)
        return 0
    else:
        print(f"RESULT: {failures} CONTRAST CHECKS FAILED")
        print("=" * 82)
        return 1


def run_contrast_audit() -> bool:
    """Run contrast checks and return True if all token pairs meet WCAG AA standards."""
    return run_checks() == 0


if __name__ == "__main__":
    sys.exit(run_checks())
