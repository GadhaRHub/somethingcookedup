"""Generate csvplot/resources/icon.svg: two bipolar echo pulses.

The small echo runs horizontally across the top. The big echo is rotated 90°:
it runs vertically downward from the small echo's midpoint, where that pulse
crosses zero. Each pulse is a derivative of a Gaussian.
Run: python tools/make_icon.py
"""

from pathlib import Path

import numpy as np

SIZE = 256
BG, BIG, SMALL, GLOW = "#4F0336", "#CC088A", "#D9C0FA", "#F2EEE5"


def bipolar(u):
    # -u * exp(-u^2/2), scaled so the peaks are at +-1
    return -u * np.exp(-u * u / 2) / np.exp(-0.5)


def polyline(xs, ys):
    pts = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)]
    return "M" + " L".join(pts)


def main():
    t = np.linspace(0, 1, 600)
    cx, top = SIZE * 0.5, SIZE * 0.31

    # Small echo: horizontal, zero crossing at the centre.
    hx = SIZE * (0.29 + 0.42 * t)
    hy = top + SIZE * 0.13 * bipolar((t - 0.5) / 0.085)

    # Big echo: vertical, starting at the small echo's midpoint.
    vy = top + (SIZE * 0.88 - top) * t
    vx = cx + SIZE * 0.28 * bipolar((t - 0.52) / 0.085)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6A0848"/>
      <stop offset="1" stop-color="{BG}"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="{SIZE - 16}" height="{SIZE - 16}" rx="56" fill="url(#bg)"/>
  <path d="{polyline(vx, vy)}" fill="none" stroke="{BIG}" stroke-width="13"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="{polyline(hx, hy)}" fill="none" stroke="{SMALL}" stroke-width="10"
        stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{cx}" cy="{top}" r="7" fill="{GLOW}"/>
</svg>
'''
    out = Path(__file__).resolve().parent.parent / "csvplot" / "resources" / "icon.svg"
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
