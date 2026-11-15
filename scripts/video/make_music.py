# Original soundtrack for the video (30 s, 48 kHz stereo): ambient chords, radar "ping", arpeggio, soft beat.
# Everything is synthesized here, so no third-party music is used.
# Usage: python make_music.py <output.wav>
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile
SR, DUR = 48000, 30.0
N = int(SR * DUR)
t_all = np.arange(N) / SR
rng = np.random.default_rng(7)
BPM = 96; BEAT = 60 / BPM; BAR = 4 * BEAT  # 2.5 s
mtof = lambda m: 440.0 * 2 ** ((m - 69) / 12)
def lp(x, fc, order=2): return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x, axis=0)
def hp(x, fc, order=2): return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x, axis=0)
def place(buf, sig, t0):
    i0 = int(t0 * SR); i1 = min(i0 + len(sig), len(buf))
    if i1 > i0: buf[i0:i1] += sig[: i1 - i0]
# Chords (MIDI): Dm9, Bbmaj7, Fadd9, Csus2; ends C → F (a bright, hopeful close)
CH = {"Dm": [50, 57, 65, 69, 76], "Bb": [46, 53, 62, 69, 72], "F": [53, 57, 60, 65, 67], "C": [48, 55, 62, 67, 74]}
ROOT = {"Dm": 38, "Bb": 34, "F": 41, "C": 36}
PROG = ["Dm", "Bb", "F", "C", "Dm", "Bb", "F", "C", "Dm", "Bb", "C", "F"]
# --- Pad: detuned, band-limited sawtooth
def saw(f, n, phase):
    tt = np.arange(n) / SR; out = np.zeros(n)
    for h in range(1, int(4200 / f) + 1):
        out += np.sin(2 * np.pi * f * h * tt + phase * h) / h
    return out
pad = np.zeros((N, 2))
for k, name in enumerate(PROG):
    t0 = k * BAR; last = k == len(PROG) - 1
    seg = (BAR + 1.6) if not last else (DUR - t0)
    n = int(seg * SR); tt = np.arange(n) / SR
    env = np.minimum(tt / 0.9, 1.0)
    if not last: env *= np.clip((seg - tt) / 1.6, 0, 1)
    else: env *= np.clip((DUR - t0 - tt) / 2.2, 0, 1)
    for j, m in enumerate(CH[name]):
        for cents, pan in ((-8, 0.25), (0, 0.5), (8, 0.75)):
            f = mtof(m) * 2 ** (cents / 1200)
            s = saw(f, n, rng.uniform(0, 2 * np.pi)) * env * (0.9 if j else 1.0)
            pad[:, 0][int(t0 * SR):int(t0 * SR) + n] += s[: N - int(t0 * SR)][: n] * (1 - pan) if int(t0 * SR) + n <= N else s[: N - int(t0 * SR)] * (1 - pan)
            pad[:, 1][int(t0 * SR):int(t0 * SR) + n] += s[: N - int(t0 * SR)][: n] * pan if int(t0 * SR) + n <= N else s[: N - int(t0 * SR)] * pan
# Filter opens over time (crossfade between a dark and a bright version)
dark, bright = lp(pad, 700, 4), lp(pad, 2600, 4)
build = np.clip((t_all - 2.0) / 14.0, 0, 1)[:, None]
pad = dark * (1 - build) + bright * build
# --- Bass: soft sine on the chord root
bass = np.zeros(N)
for k, name in enumerate(PROG):
    n = int((BAR + 0.3) * SR); tt = np.arange(n) / SR
    env = np.minimum(tt / 0.25, 1) * np.clip((BAR + 0.3 - tt) / 0.5, 0, 1)
    if k == len(PROG) - 1: env = np.minimum(tt / 0.25, 1) * np.clip((DUR - k * BAR - tt) / 2.2, 0, 1)
    f = mtof(ROOT[name]); s = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * env
    place(bass, s, k * BAR)
bass *= np.clip((t_all - 3.4) / 1.5, 0, 1)  # bass enters with the Sentinel-1 scene
# --- Arpeggio: bell-like FM tones, eighth notes
arp = np.zeros((N, 2))
pattern = [0, 2, 3, 4, 3, 2, 1, 2]
for k, name in enumerate(PROG):
    notes = sorted(CH[name])[1:]
    for i in range(8):
        t0 = k * BAR + i * BEAT / 2
        if t0 < 3.4 or t0 > 27.0: continue
        m = notes[pattern[i] % len(notes)] + 12
        f = mtof(m); n = int(1.4 * SR); tt = np.arange(n) / SR
        idx = 1.6 * np.exp(-tt / 0.15)
        s = np.sin(2 * np.pi * f * tt + idx * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt / 0.32) * np.minimum(tt / 0.004, 1)
        vel = (0.85 if i % 2 == 0 else 0.6) * (0.9 + 0.2 * rng.random())
        pan = 0.35 if i % 2 == 0 else 0.65
        i0 = int(t0 * SR); i1 = min(i0 + n, N)
        arp[i0:i1, 0] += s[: i1 - i0] * vel * (1 - pan); arp[i0:i1, 1] += s[: i1 - i0] * vel * pan
arp *= np.clip((t_all - 3.4) / 2.0, 0, 1)[:, None]
# Ping-pong delay (dotted eighth)
def pingpong(x, d, fb, n_rep=6):
    out = x.copy(); dn = int(d * SR); cur = x.copy()
    for r in range(1, n_rep + 1):
        cur = cur * fb; sh = np.zeros_like(x); sh[dn * r:] = (cur[:, ::-1] if r % 2 else cur)[: N - dn * r]
        out += lp(sh, 3500)
    return out
arp = pingpong(arp, BEAT * 0.75, 0.38)
# --- Radar "ping": at the very start and at the start of the NISAR scene
ping = np.zeros((N, 2))
for t0, m, a in ((0.15, 91, 1.0), (10.25, 91, 0.7)):
    n = int(2.5 * SR); tt = np.arange(n) / SR; f = mtof(m) * (1 + 0.01 * np.exp(-tt / 0.05))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.55) * np.minimum(tt / 0.003, 1) * a
    place(ping[:, 0], s, t0); place(ping[:, 1], s, t0)
ping = pingpong(ping, BEAT * 1.5, 0.45, 5)
# --- Beat: soft kick (10–27 s) and light hi-hat (15–27 s)
drums = np.zeros(N)
for b in range(int(DUR / BEAT)):
    t0 = b * BEAT
    if 10.0 <= t0 < 27.0 and b % 2 == 0:
        n = int(0.45 * SR); tt = np.arange(n) / SR
        f = 45 + 70 * np.exp(-tt / 0.035)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.16)
        place(drums, s * (0.55 if b % 4 == 0 else 0.4), t0)
hats = np.zeros(N)
for b in range(int(DUR / (BEAT / 2))):
    t0 = b * BEAT / 2
    if 15.0 <= t0 < 26.8 and b % 2 == 1:
        n = int(0.06 * SR); tt = np.arange(n) / SR
        place(hats, rng.standard_normal(n) * np.exp(-tt / 0.012), t0)
hats = hp(hats, 7000, 4)
kick_fade = np.clip((t_all - 10.0) / 3.0, 0, 1)
drums *= kick_fade
# --- Into the end card: noise riser (25.4–27.2 s) and a deep hit at 27.5 s
riser = hp(rng.standard_normal(N), 2500, 2) * np.clip((t_all - 25.4) / 1.8, 0, 1) ** 2.2 * (t_all < 27.25)
boom = np.zeros(N); n = int(3.0 * SR); tt = np.arange(n) / SR
place(boom, np.sin(2 * np.pi * np.cumsum(38 + 40 * np.exp(-tt / 0.08)) / SR) * np.exp(-tt / 0.9), 27.5)
# --- Reverb: synthetic impulse response
irn = int(3.2 * SR); itt = np.arange(irn) / SR
ir = rng.standard_normal((irn, 2)) * np.exp(-itt / 0.85)[:, None]; ir = lp(ir, 5000); ir /= np.sqrt((ir ** 2).sum(0))
def verb(x, wet):
    if x.ndim == 1: x = np.stack([x, x], 1)
    w = np.stack([fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)
    return x * (1 - wet) + w * wet * 1.4
st = lambda x, g=1.0: np.stack([x * g, x * g], 1)
mix = (verb(pad, 0.45) * 0.050 + st(bass, 0.20) + verb(arp, 0.35) * 0.11 + verb(ping, 0.5) * 0.16
       + st(drums, 0.42) + verb(hats, 0.25) * 0.035 + verb(riser, 0.4) * 0.05 + verb(boom, 0.3) * 0.5)
mix = hp(mix, 30, 2)
# Master: RMS ≈ −18 dBFS, soft limiter, fade in/out
mix *= 10 ** (-18 / 20) / np.sqrt((mix ** 2).mean())
mix = np.tanh(mix * 1.4) / 1.4
fade = np.minimum(t_all / 0.08, 1) * np.clip((DUR - t_all) / 1.2, 0, 1)
mix *= fade[:, None]
pk = np.abs(mix).max(); mix *= min(1.0, 10 ** (-1 / 20) / pk)
wavfile.write(sys.argv[1], SR, (mix * 32767).astype(np.int16))
rms = 20 * np.log10(np.sqrt((mix ** 2).mean())); print(f"ok peak {20*np.log10(np.abs(mix).max()):.1f} dBFS, rms {rms:.1f} dBFS")
