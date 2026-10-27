"""Generate csvplot/resources/icon.svg: a bipolar echo pulse rotated 90 degrees.

The trace runs top to bottom; the signal swings left/right. A strong bipolar
pulse (derivative of a Gaussian) is followed by a weaker, inverted echo.
Run: python tools/make_icon.py
"""

from pathlib import Path

import numpy as np

SIZE = 256
BG, PULSE, ECHO, GLOW = "#4F0336", "#CC088A", "#D9C0FA", "#F2EEE5"


def bipolar(u):
    # -u * exp(-u^2/2), scaled so the peaks are at +-1
    return -u * np.exp(-u * u / 2) / np.exp(-0.5)


def main():
    t = np.linspace(0, 1, 600)
    top, bottom = 0.12, 0.88
    y = top + (bottom - top) * t
    x = bipolar((t - 0.3) / 0.05) - 0.45 * bipolar((t - 0.74) / 0.045)
    cx, amp = SIZE / 2, SIZE * 0.29
    px, py = cx + amp * x, SIZE * y
    split = int(np.searchsorted(t, 0.55))

    def path(a, b):
        pts = " ".join(f"{X:.1f},{Y:.1f}" for X, Y in zip(px[a:b], py[a:b]))
        return f"M{pts.replace(' ', ' L', 1)}"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6A0848"/>
      <stop offset="1" stop-color="{BG}"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="{SIZE - 16}" height="{SIZE - 16}" rx="56" fill="url(#bg)"/>
  <line x1="{cx}" y1="{SIZE * 0.1:.0f}" x2="{cx}" y2="{SIZE * 0.9:.0f}" stroke="{GLOW}" stroke-opacity="0.18"
        stroke-width="4" stroke-dasharray="2 10" stroke-linecap="round"/>
  <path d="{path(split - 1, len(t))}" fill="none" stroke="{ECHO}" stroke-width="10"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="{path(0, split)}" fill="none" stroke="{PULSE}" stroke-width="12"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''
    out = Path(__file__).resolve().parent.parent / "csvplot" / "resources" / "icon.svg"
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
