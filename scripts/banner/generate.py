#!/usr/bin/env python3
"""Generate the animated Synthwave GitHub profile banner for Alex Baltazar Castellon.

Run from the repository root:
    uv run --with numpy,scipy,pillow python3 scripts/banner/generate.py
"""

from __future__ import annotations

import html
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"

W, H = 1180, 610
TRANSITION_SECONDS = 1.4
LOGO_HOLD_SECONDS = 3.6
TRAVELLER_COUNT = 900
HOLD_PARTICLE_COUNT = 2_200
SEED = 42
LOGO_SUFFIXES = {".png", ".webp"}
PREFERRED_LOGO_ORDER = ("docker", "nextdotjs", "nestjs", "laravel")

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Alex Baltazar Castellon"),
    (1, "role", "Full Stack Developer"),
    (1, "origin", "Bolivia 🇧🇴"),
    (1, "focus", "Next.js · NestJS · Laravel · Django"),
    (1, "status", "Building · Shipping · Always Learning"),
    (1, "toolchain", "Docker · GitHub Actions · Sentry"),
    (0, "stack", ""),
    (1, "frontend", "Next.js · React · TypeScript · Tailwind"),
    (1, "backend", "NestJS · Laravel · Django · Python"),
    (1, "mobile", "Flutter · Dart"),
    (1, "databases", "PostgreSQL · MySQL · MongoDB"),
    (1, "devops", "Docker · GitHub Actions · Linux"),
    (1, "learning", "DevOps · Cloud · AWS · GCP"),
    (0, "contact", ""),
    (1, "linkedin", "/in/alex-baltazar-castellon"),
    (1, "github", "Alexbyxd"),
    (1, "timezone", "UTC-4 · Bolivia"),
]

THEMES = {
    "dark": {
        "bg":       "#0d001a",
        "panel":    "#120526",
        "panel2":   "#180833",
        "line":     "#3a0ca3",
        "muted":    "#8f7aa8",
        "text":     "#f0e6f8",
        "portrait": "#f72585",   # synthwave hot pink
        "chrome":   "#4cc9f0",   # synthwave cyan
        "accent":   "#f72585",
        "shadow":   "#070014",
    },
    "light": {
        "bg":       "#fdf0f8",
        "panel":    "#ffffff",
        "panel2":   "#fae8f5",
        "line":     "#e0b0db",
        "muted":    "#9b7b96",
        "text":     "#2d1a2a",
        "portrait": "#f72585",
        "chrome":   "#7209b7",
        "accent":   "#f72585",
        "shadow":   "#d4a0cb",
    },
}


def normalize_logo(image: Image.Image) -> Image.Image:
    icon = image.convert("RGBA")
    icon.thumbnail((320, 320), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    offset = ((400 - icon.width) // 2, (400 - icon.height) // 2)
    canvas.alpha_composite(icon, offset)
    if canvas.getchannel("A").getbbox() is None:
        raise ValueError("icon has no visible pixels")
    return canvas


def load_logo_files() -> dict[str, Image.Image]:
    loaded: dict[str, Image.Image] = {}
    for path in sorted(LOGOS.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LOGO_SUFFIXES:
            continue
        name = path.stem.lower()
        with Image.open(path) as image:
            loaded[name] = normalize_logo(image)
    return loaded


def make_logos() -> dict[str, Image.Image]:
    LOGOS.mkdir(parents=True, exist_ok=True)
    logos = load_logo_files()
    if not logos:
        raise SystemExit(f"No PNG or WebP icons found in {LOGOS.relative_to(ROOT)}")
    ordered_names = [name for name in PREFERRED_LOGO_ORDER if name in logos]
    ordered_names.extend(name for name in sorted(logos) if name not in ordered_names)
    return {name: logos[name] for name in ordered_names}


def logo_silhouette_points(image: Image.Image) -> np.ndarray:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 120)
    # Center within 390x414 visual frame: origin (49, 124)
    # Frame center is x = 49 + 195 = 244, y = 124 + 207 = 331
    # 400x400 scaled by 0.75 -> 300x300, offset left by 49 + 45 = 94, top = 124 + 57 = 181
    return np.column_stack((94 + xs * 0.75, 181 + ys * 0.75)).astype(np.float32)


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    points = logo_silhouette_points(image)
    if not len(points):
        raise ValueError("icon has no visible pixels")
    chosen = rng.choice(len(points), count, replace=len(points) < count)
    return points[chosen]


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def time_num(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def render_svg(
    theme_name: str,
    logo_points: dict[str, np.ndarray],
    logo_hold_particles: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    names = list(logo_points.keys())
    first_name = names[0]
    n = TRAVELLER_COUNT

    # Base starts with first logo
    source = logo_points[first_name][:n]
    targets: list[tuple[str, np.ndarray]] = []
    current = source
    for name in names[1:]:
        current = transport(current, logo_points[name][:n])
        targets.append((name, current))
    # Loop back to first logo
    current = transport(current, source)
    targets.append((first_name, current))

    # Timeline calculation
    times = [0.0]
    frames = [source]
    for _, target in targets:
        # hold current
        times.append(times[-1] + LOGO_HOLD_SECONDS)
        frames.append(frames[-1])
        # transition to target
        times.append(times[-1] + TRANSITION_SECONDS)
        frames.append(target)

    # Final hold on loop point
    times.append(times[-1] + LOGO_HOLD_SECONDS)
    frames.append(frames[-1])

    loop_seconds = times[-1]
    key_times = ";".join(time_num(v / loop_seconds) for v in times)

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        f'<title id="title">Alex Baltazar Castellon — Full Stack Developer</title>',
        '<desc id="desc">Animated Synthwave terminal profile with particle logo morphing.</desc>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".45"/></filter>',
        '<filter id="glowPink" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["portrait"]}" '
        'flood-opacity=".55"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<filter id="glowCyan" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="2.5" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".55"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        '<linearGradient id="scanbeam" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0%" stop-color="{t["chrome"]}" stop-opacity="0"/>'
        f'<stop offset="50%" stop-color="{t["portrait"]}" stop-opacity="0.25"/>'
        f'<stop offset="100%" stop-color="{t["chrome"]}" stop-opacity="0.6"/>'
        '</linearGradient>',
        "</defs>",

        # Base Frame
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" stroke-width="1.5" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',

        # Window Controls
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>',
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>',
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">vim profile.yml</text>',

        # Left Visual Frame (Particle Box)
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}" stroke-width="1.2"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">PARTICLE MORPH · RETRO</text>',

        # Frame Crosshairs
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',

        # Scanline inside visual frame
        '<g clip-path="url(#visualClip)">',
        f'<rect x="49" y="124" width="390" height="24" fill="url(#scanbeam)">',
        f'<animateTransform attributeName="transform" type="translate" values="0,0; 0,400; 0,0" dur="6s" repeatCount="indefinite" ease="linear"/>',
        '</rect>',
        '</g>',

        # Particles Loop Layer
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
    ]

    # Background ambient stars/dust
    parts.append('<g opacity="0.3">')
    for sx, sy in ((70, 160), (120, 200), (380, 180), (320, 480), (90, 450), (410, 360)):
        parts.append(f'<circle cx="{sx}" cy="{sy}" r="1" fill="{t["chrome"]}"/>')
    parts.append('</g>')

    # Generate traveller morphing paths
    # We group travellers into chunks for compact SVG
    point_history = [f for f in frames]
    # To keep SVG size reasonable, bundle particles
    step = 1
    for i in range(0, n, step):
        d_init = f"M{num(point_history[0][i, 0])} {num(point_history[0][i, 1])}h1"
        vals = ";".join(
            f"{num(f[i, 0] - point_history[0][i, 0])} {num(f[i, 1] - point_history[0][i, 1])}"
            for f in point_history
        )
        parts.append(
            f'<path d="{d_init}" fill="none" stroke="{t["portrait"]}" stroke-width="1.2">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'dur="{time_num(loop_seconds)}s" repeatCount="indefinite" '
            f'values="{vals}" keyTimes="{key_times}" calcMode="spline" '
            f'keySplines="{";".join(["0.42 0 0.58 1"] * (len(times) - 1))}"/>'
            f'</path>'
        )

    parts.append('</g>')

    # Visual Frame Telemetry
    parts.extend([
        f'<text x="58" y="551" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
        f'PARTICLES {n} · DOCKER ⇄ NEXT.JS ⇄ NESTJS ⇄ LARAVEL</text>',

        # Right Panel (Vim Editor)
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}" stroke-width="1.2"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing=".5">profile.yml</text>',
        f'<text x="580" y="111" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[YAML]</text>',
        f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{t["portrait"]}" opacity=".18" '
        f'stroke="{t["portrait"]}"/>',
        f'<text x="1062" y="111" text-anchor="middle" fill="{t["portrait"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700">@Alexbyxd</text>',
    ])

    # Vim YAML lines
    row_y = 148.0
    for idx, (indent, key, value) in enumerate(YAML_ROWS, 1):
        line_num = f"{idx:2d}"
        if indent == 0:
            content = f'<tspan fill="{t["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>'
            text_x = 525.0
        else:
            content = (
                f'<tspan fill="{t["portrait"]}">{html.escape(key)}: </tspan>'
                f'<tspan fill="{t["text"]}">{html.escape(value)}</tspan>'
            )
            text_x = 542.0

        parts.extend([
            f'<text x="506" y="{num(row_y)}" text-anchor="end" fill="{t["muted"]}" opacity=".45" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13">'
            f"{line_num}</text>",
            f'<text x="{num(text_x)}" y="{num(row_y)}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13">'
            f"{content}</text>",
        ])
        row_y += 21.5

    # Blinking cursor on line 17
    parts.append(
        f'<rect x="800" y="480" width="8" height="15" fill="{t["chrome"]}">'
        '<animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/>'
        '</rect>'
    )

    # Vim status bar
    parts.extend([
        f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
        f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}" rx="0 0 5 5"/>',
        f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["portrait"]}"/>',
        f'<text x="521" y="547" text-anchor="middle" fill="{t["bg"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11" font-weight="700">NORMAL</text>',
        f'<text x="569" y="547" fill="{t["text"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="600">profile.yml</text>',
        f'<text x="740" y="547" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[utf-8]</text>',
        f'<text x="1134" y="547" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">17L, 512B  100%  17:1</text>',
        "</svg>",
    ])
    return "".join(parts)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    logos = make_logos()
    print(f"Loaded logos: {list(logos.keys())}")

    rng = np.random.default_rng(SEED)
    logo_points = {
        name: sample_logo_points(image, rng, TRAVELLER_COUNT)
        for name, image in logos.items()
    }
    hold_particles = {
        name: sample_logo_points(image, rng, HOLD_PARTICLE_COUNT)
        for name, image in logos.items()
    }

    for theme in ("dark", "light"):
        svg = render_svg(theme, logo_points, hold_particles, rng)
        out = ASSETS / f"banner-{theme}.svg"
        out.write_text(svg, encoding="utf-8")
        size_kb = out.stat().st_size / 1024
        print(f"Wrote {out} ({size_kb:.1f} KiB)")


if __name__ == "__main__":
    main()
