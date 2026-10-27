"""Generate csvplot/resources/icon.svg: two bipolar echo pulses.

The small echo runs horizontally across the top. The big echo is rotated 90°:
it runs vertically downward from the small echo's midpoint, where that pulse
crosses zero. Each pulse is a derivative of a Gaussian.
Run: python tools/make_icon.py
"""

from pathlib import Path

import numpy as np

SIZE = 256
BG, BIG, SMALL = "#4F0336", "#CC088A", "#D9C0FA"
BLEND = "#D964C2"  # where the two echoes meet
SMALL_W, BIG_W = 11, 14


def bipolar(u):
    # -u * exp(-u^2/2), scaled so the peaks are at +-1
    return -u * np.exp(-u * u / 2) / np.exp(-0.5)


def polyline(xs, ys):
    pts = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)]
    return "M" + " L".join(pts)


def ribbon(xs, ys, widths):
    """Closed outline of a line whose width varies along its length."""
    dx, dy = np.gradient(xs), np.gradient(ys)
    norm = np.hypot(dx, dy)
    nx, ny = -dy / norm, dx / norm
    half = widths / 2
    left = zip(xs + nx * half, ys + ny * half)
    right = zip((xs - nx * half)[::-1], (ys - ny * half)[::-1])
    pts = [f"{x:.1f},{y:.1f}" for x, y in [*left, *right]]
    return "M" + " L".join(pts) + " Z"


def smoothstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def main():
    t = np.linspace(0, 1, 600)
    cx, top = SIZE * 0.5, SIZE * 0.31

    # Small echo: horizontal, zero crossing at the centre.
    hx = SIZE * (0.29 + 0.42 * t)
    hy = top + SIZE * 0.13 * bipolar((t - 0.5) / 0.085)

    # Big echo: vertical, starting at the small echo's midpoint.
    vy = top + (SIZE * 0.88 - top) * t
    vx = cx + SIZE * 0.28 * bipolar((t - 0.52) / 0.085)
    # Same width as the small echo where they meet, widening further down.
    vw = SMALL_W + (BIG_W - SMALL_W) * smoothstep(0.02, 0.3, t)
    blend_r = SIZE * 0.06  # the colour blend only spans this close to the junction

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6A0848"/>
      <stop offset="1" stop-color="{BG}"/>
    </linearGradient>
    <!-- Both strokes share the BLEND colour at the junction, so they merge. -->
    <radialGradient id="small" gradientUnits="userSpaceOnUse" cx="{cx}" cy="{top}" r="{blend_r:.1f}">
      <stop offset="0.15" stop-color="{BLEND}"/>
      <stop offset="1" stop-color="{SMALL}"/>
    </radialGradient>
    <linearGradient id="big" gradientUnits="userSpaceOnUse" x1="0" y1="{top}" x2="0" y2="{top + blend_r * 1.6:.1f}">
      <stop offset="0.2" stop-color="{BLEND}"/>
      <stop offset="1" stop-color="{BIG}"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="{SIZE - 16}" height="{SIZE - 16}" rx="56" fill="url(#bg)"/>
  <path d="{polyline(hx, hy)}" fill="none" stroke="url(#small)" stroke-width="{SMALL_W}"
        stroke-linecap="round" stroke-linejoin="round"/>
  <!-- The big echo grows out of the small one: same width and colour at the junction. -->
  <circle cx="{vx[0]:.1f}" cy="{vy[0]:.1f}" r="{SMALL_W / 2}" fill="{BLEND}"/>
  <path d="{ribbon(vx, vy, vw)}" fill="url(#big)"/>
  <circle cx="{vx[-1]:.1f}" cy="{vy[-1]:.1f}" r="{BIG_W / 2}" fill="{BIG}"/>
</svg>
'''
    out = Path(__file__).resolve().parent.parent / "csvplot" / "resources" / "icon.svg"
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
