"""Generate assets/header-{dark,light}.svg.

The spectrum is simulated from a two-bar lofi loop at 80 BPM: kick ("boots") on 1 and 3,
snare ("cats") on 2 and 4, swung 8th-note hi-hats, a bass + chord bed and a vinyl noise floor.
Bins run low -> high frequency left to right, with instant attack, smooth release and peak-hold caps.

Run from the repo root: python scripts/gen_header.py
"""
import colorsys

import numpy as np

BPM = 80
BEAT = 60 / BPM
LOOP = 8 * BEAT  # two bars of 4/4
SWING = 0.58  # offbeat 8th lands at 58% of the beat

N_BINS, X0, BAR_W, GAP, BASE, MAX_H = 26, 740, 10, 6, 196, 110
SIM_DT = 1 / 240
KEY_DT = 1 / 40

rng = np.random.default_rng(44)
freqs = 30 * (16000 / 30) ** (np.arange(N_BINS) / (N_BINS - 1))
octs = np.log2(freqs)


def band(center_hz, width_oct):
    return np.exp(-0.5 * ((octs - np.log2(center_hz)) / width_oct) ** 2)


def env(t, t0, tau):
    # wraps around the loop so hits near the end ring into the start
    dt = (t - t0) % LOOP
    return np.exp(-dt / tau)


# (beat position, velocity)
kicks = [(0, 1.0), (2, 0.9), (4, 1.0), (6, 0.9), (6.5, 0.6)]
snares = [(1, 1.0), (3, 0.95), (5, 1.0), (7, 1.0)]
hats = []
for b in range(8):
    hats.append((b, 0.75 + rng.uniform(-0.08, 0.08)))
    hats.append((b + SWING, 0.5 + rng.uniform(-0.08, 0.08)))
open_hat = (7 + SWING, 0.8)

# bass root and chord voicing per bar (Dm9 -> G13, the lofi classic)
bed = [
    (73.4, [349, 440, 523, 659]),
    (98.0, [349, 494, 659, 740]),
]

kick_body, kick_click = band(55, 0.45), 0.25 * band(3500, 0.6)
snare_body, snare_noise = band(220, 0.5), 0.35 * band(2500, 1.0)
hat_shape, open_shape = band(10000, 0.45), band(9000, 0.6)


def raw_energy(t):
    e = np.zeros(N_BINS)
    for pos, vel in kicks:
        a = env(t, pos * BEAT, 0.22)
        drop = band(55 + 45 * env(t, pos * BEAT, 0.04), 0.45)  # pitch drop on the attack
        e += vel * (a * drop + env(t, pos * BEAT, 0.03) * kick_click)
    for pos, vel in snares:
        e += vel * (env(t, pos * BEAT, 0.11) * snare_body + env(t, pos * BEAT, 0.16) * snare_noise)
    for pos, vel in hats:
        e += vel * 0.55 * env(t, pos * BEAT, 0.045) * hat_shape
    e += open_hat[1] * 0.6 * env(t, open_hat[0] * BEAT, 0.28) * open_shape

    bar = int((t % LOOP) // (4 * BEAT))
    root, chord = bed[bar]
    e += 0.32 * band(root, 0.18) + 0.12 * band(root * 2, 0.15)
    for f in chord:
        e += 0.09 * band(f, 0.14) + 0.03 * band(f * 2, 0.14)
    e += 0.02 * band(8000, 1.5) * (1 + rng.uniform(-0.6, 0.6, N_BINS))  # vinyl hiss
    return e


def simulate():
    times = np.arange(0, 2 * LOOP, SIM_DT)  # run twice so release/peaks wrap cleanly
    level = np.zeros(N_BINS)
    peak = np.zeros(N_BINS)
    hold = np.zeros(N_BINS)
    rel = np.exp(-SIM_DT / 0.13)
    lv, pk = [], []
    for t in times:
        db = 20 * np.log10(raw_energy(t) + 1e-4)
        target = np.clip((db + 30) / 30, 0.04, 1.0)
        level = np.maximum(target, level * rel + 0.04 * (1 - rel))
        hit = level >= peak
        peak = np.where(hit, level, peak)
        hold = np.where(hit, 0.35, hold - SIM_DT)
        peak = np.where(hold <= 0, np.maximum(level, peak - 0.9 * SIM_DT), peak)
        lv.append(level.copy())
        pk.append(peak.copy())
    keep = times >= LOOP
    return times[keep] - LOOP, np.array(lv)[keep], np.array(pk)[keep]


def keyframes(times, values, fmt, tol=0.012):
    step = max(1, round(KEY_DT / SIM_DT))
    t, v = times[::step], values[::step]
    t, v = np.append(t, LOOP), np.append(v, v[0])
    keep = [0]
    for i in range(1, len(t) - 1):
        j = keep[-1]
        # drop a frame if linear interpolation between neighbours already matches it
        interp = v[j] + (v[i + 1] - v[j]) * (t[i] - t[j]) / (t[i + 1] - t[j])
        if abs(interp - v[i]) > tol or abs(v[i] - v[i - 1]) > 0.15:
            keep.append(i)
    keep.append(len(t) - 1)
    return " ".join(f"{100 * t[i] / LOOP:.2f}%{{transform:{fmt(v[i])}}}" for i in keep)


def hx(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


THEMES = {
    "dark": dict(fg=hx(320, 15, 94), accent=hx(318, 45, 72), muted=hx(310, 6, 68), line=hx(300, 10, 22)),
    "light": dict(fg=hx(300, 10, 12), accent=hx(316, 42, 36), muted=hx(300, 5, 38), line=hx(320, 12, 85)),
}


def render(c, bar_css, bars, caps):
    progress = 8 * LOOP
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 280" width="1200" height="280" role="img" aria-label="Cristian Preda: backend, LLMs, applied ML">
  <style>
    .mono {{ font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace; }}
    .b, .p {{ transform-box: fill-box; animation-duration: {LOOP:.3f}s; animation-iteration-count: infinite; animation-timing-function: linear; }}
    .b {{ fill: {c['accent']}; transform-origin: bottom; }}
    .p {{ fill: {c['fg']}; opacity: 0.7; }}
{bar_css}
    .fill {{ transform-box: fill-box; transform-origin: left; animation: play {progress:.0f}s linear infinite; }}
    .head {{ animation: head {progress:.0f}s linear infinite; }}
    @keyframes play {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
    @keyframes head {{ from {{ transform: translateX(0); }} to {{ transform: translateX(420px); }} }}
    .cursor {{ animation: blink 1.1s steps(1) infinite; }}
    @keyframes blink {{ 50% {{ opacity: 0; }} }}
  </style>

  <text x="64" y="62" class="mono" font-size="15" fill="{c['muted']}">~/criseda $ whoami</text>
  <text x="62" y="128" class="mono" font-size="56" font-weight="700" fill="{c['fg']}">Cristian Preda<tspan class="cursor" fill="{c['accent']}">_</tspan></text>
  <text x="64" y="170" class="mono" font-size="19" fill="{c['accent']}">Backend · LLMs · Applied ML</text>
  <text x="64" y="198" class="mono" font-size="15" fill="{c['muted']}">CS Graduate · University of Manchester · Ex-Bitdefender</text>

  <g>
{bars}
{caps}
  </g>
  <line x1="736" y1="{BASE + 1}" x2="1150" y2="{BASE + 1}" stroke="{c['line']}" stroke-width="2"/>

  <text x="64" y="244" class="mono" font-size="13" fill="{c['muted']}">▶ Now playing: <tspan fill="{c['fg']}">lofi_station/track_044.wav</tspan> · {BPM} BPM</text>
  <rect x="736" y="236" width="420" height="3" rx="1.5" fill="{c['line']}"/>
  <rect class="fill" x="736" y="236" width="420" height="3" rx="1.5" fill="{c['accent']}"/>
  <circle class="head" cx="736" cy="237.5" r="6" fill="{c['fg']}"/>
</svg>
'''


def main():
    times, levels, peaks = simulate()
    css, bars, caps = [], [], []
    for i in range(N_BINS):
        x = X0 + i * (BAR_W + GAP)
        css.append(f"    .b{i} {{ animation-name: b{i}; }} @keyframes b{i} {{ "
                   + keyframes(times, levels[:, i], lambda v: f"scaleY({v:.3f})") + " }")
        css.append(f"    .p{i} {{ animation-name: p{i}; }} @keyframes p{i} {{ "
                   + keyframes(times, peaks[:, i], lambda v: f"translateY({-v * MAX_H:.1f}px)") + " }")
        bars.append(f'    <rect class="b b{i}" x="{x}" y="{BASE - MAX_H}" width="{BAR_W}" height="{MAX_H}" rx="2"/>')
        caps.append(f'    <rect class="p p{i}" x="{x}" y="{BASE - 4}" width="{BAR_W}" height="2" rx="1"/>')
    for name, c in THEMES.items():
        svg = render(c, "\n".join(css), "\n".join(bars), "\n".join(caps))
        with open(f"assets/header-{name}.svg", "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
    # quick sanity print: level per bin at a kick, a snare and an offbeat hat
    for label, beat in [("kick", 0.02), ("snare", 1.02), ("hat", SWING + 0.02), ("gap", 1.45)]:
        row = levels[int(beat * BEAT / SIM_DT)]
        print(f"{label:6s}", "".join(" .:-=+*#%@"[min(9, int(v * 10))] for v in row))


if __name__ == "__main__":
    main()
