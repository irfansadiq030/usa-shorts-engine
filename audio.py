import numpy as np, wave
from scipy import ndimage
def ndimage_max(a):
    m = ndimage.maximum_filter1d(a, int(44100*.25))
    m = ndimage.uniform_filter1d(m, int(44100*.3))
    return np.clip(m/ (m.max()+1e-9)*1.4, 0, 1)
from scipy import signal
SR = 44100
rng = np.random.default_rng(3)

def env(n, a, d):
    t = np.arange(n)/SR
    e = np.minimum(t/max(a, 1e-4), 1)*np.exp(-np.maximum(t-a, 0)/max(d, 1e-4))
    return e

def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc/(SR/2), .99), 'low'); return signal.lfilter(b, a, x)
def hp(x, fc, order=2):
    b, a = signal.butter(order, min(fc/(SR/2), .99), 'high'); return signal.lfilter(b, a, x)
def bp(x, f1, f2):
    b, a = signal.butter(2, [f1/(SR/2), min(f2/(SR/2), .99)], 'band'); return signal.lfilter(b, a, x)

def whoosh(dur=.55, up=True):
    n = int(SR*dur); x = rng.standard_normal(n)
    out = np.zeros(n); chunks = 24
    for i in range(chunks):
        a, b = i*n//chunks, (i+1)*n//chunks
        f = 300+3200*((i/chunks) if up else (1-i/chunks))**1.5
        seg = bp(x[max(0, a-400):b], f*.6, f*1.6)[-(b-a):]
        out[a:b] = seg
    t = np.linspace(0, 1, n); e = np.sin(np.pi*t**(0.7 if up else 1.3))**2
    return out*e*1.2

def impact(dur=1.6):
    n = int(SR*dur); t = np.arange(n)/SR
    f = 38+70*np.exp(-t*9); ph = 2*np.pi*np.cumsum(f)/SR
    sub = np.sin(ph)*np.exp(-t*2.6)
    nz = lp(rng.standard_normal(n), 900)*np.exp(-t*7)*.9
    crack = hp(rng.standard_normal(n), 2500)*np.exp(-t*40)*.5
    return np.tanh((sub*1.1+nz+crack)*1.4)*.9

def pop(f0=520):
    n = int(SR*.12); t = np.arange(n)/SR
    f = f0+900*np.exp(-t*60); ph = 2*np.pi*np.cumsum(f)/SR
    return np.sin(ph)*np.exp(-t*38)*.6

def tick():
    n = int(SR*.03); t = np.arange(n)/SR
    return (np.sin(2*np.pi*2400*t)*.5+hp(rng.standard_normal(n), 3000)*.4)*np.exp(-t*220)*.5

def ding():
    n = int(SR*1.4); t = np.arange(n)/SR
    s = sum(np.sin(2*np.pi*f*t)*a*np.exp(-t*d) for f, a, d in [(1318, .5, 3), (1760, .35, 4), (2637, .2, 6), (3520, .1, 8)])
    return s*.6

def riser(dur=1.8):
    n = int(SR*dur); t = np.arange(n)/SR
    f = 180*np.exp(np.log(6)*t/dur); ph = 2*np.pi*np.cumsum(f)/SR
    nz = whoosh(dur, True)*.6
    s = (np.sin(ph)+.4*np.sin(2*ph))*(t/dur)**2*.35
    return s+nz*(t/dur)

def stamp():
    n = int(SR*.5); t = np.arange(n)/SR
    th = np.sin(2*np.pi*(90+120*np.exp(-t*30))*t)*np.exp(-t*12)
    sn = lp(rng.standard_normal(n), 3000)*np.exp(-t*30)*.6
    return np.tanh((th+sn)*1.6)*.8

def swoosh_short(): return whoosh(.32, True)*.9
def cash():
    a = ding(); n = int(SR*.25); t = np.arange(n)/SR
    rattle = hp(rng.standard_normal(n), 4000)*np.exp(-t*25)*.3
    a[:n] += rattle; return a

def music(dur, bpm=92, root=55.0, seed=0, dark=True):
    n = int(SR*dur); t = np.arange(n)/SR
    out = np.zeros(n)
    # drone pad: root, fifth, minor third, octave (detuned saws -> lowpass)
    chord = [1, 1.5, 1.189, 2, 3] if dark else [1, 1.5, 1.26, 2, 3]
    pad = np.zeros(n)
    for i, r in enumerate(chord):
        for det in (-0.3, 0.3):
            f = root*r+det
            pad += signal.sawtooth(2*np.pi*f*t+i)*(.35 if r < 3 else .12)
    pad = lp(pad, 700)
    swell = .55+.45*np.sin(2*np.pi*t/8-np.pi/2)
    out += pad*swell*.12
    # pulse
    beat = 60/bpm
    kick_n = int(SR*.35); kt = np.arange(kick_n)/SR
    kick = np.sin(2*np.pi*np.cumsum(45+90*np.exp(-kt*25))/SR)*np.exp(-kt*9)
    hat_n = int(SR*.05); hat = hp(rng.standard_normal(hat_n), 7000)*np.exp(-np.arange(hat_n)/SR*90)
    k = 0.
    i = 0
    while k < dur:
        s = int(k*SR)
        if s+kick_n < n: out[s:s+kick_n] += kick*.35
        hs = int((k+beat/2)*SR)
        if hs+hat_n < n: out[hs:hs+hat_n] += hat*.12
        # arp pluck every 2 beats
        if i % 2 == 0:
            notes = [4, 3, 6, 4.75]
            f = root*notes[(i//2) % 4]
            pn = int(SR*.6); pt = np.arange(pn)/SR
            pl = signal.square(2*np.pi*f*pt)*np.exp(-pt*7)
            pl = lp(pl, 1800)*.06
            if s+pn < n: out[s:s+pn] += pl
        k += beat; i += 1
    # fades
    fi = int(SR*.4); out[:fi] *= np.linspace(0, 1, fi)
    fo = int(SR*1.5); out[-fo:] *= np.linspace(1, 0, fo)
    return out

LIB = {'whoosh': lambda: whoosh(.55, True), 'whoosh_out': lambda: whoosh(.6, False), 'impact': impact, 'pop': pop,
       'pop_hi': lambda: pop(800), 'tick': tick, 'ding': ding, 'riser': riser, 'stamp': stamp, 'swoosh': swoosh_short, 'cash': cash}
VOL = {'whoosh': .55, 'whoosh_out': .5, 'impact': .95, 'pop': .45, 'pop_hi': .4, 'tick': .35, 'ding': .4, 'riser': .5, 'stamp': .8, 'swoosh': .45, 'cash': .45}

def mix(dur, events, out_path, bpm=92, root=55.0, vo=None):
    n = int(SR*dur)+SR
    m = music(dur, bpm, root)
    mus = np.zeros(n); mus[:len(m)] = m
    fx = np.zeros(n)
    for (t, name) in events:
        s = LIB[name]()*VOL[name]; i = int(t*SR)
        if i < 0: s = s[-i:]; i = 0
        e = min(n, i+len(s)); fx[i:e] += s[:e-i]
    # duck music under impacts
    duck = np.ones(n)
    for (t, name) in events:
        if name in ('impact', 'stamp'):
            i = int(t*SR); L = int(SR*1.0)
            seg = 1-.6*np.exp(-np.arange(L)/SR*3)
            e = min(n, i+L); duck[i:e] = np.minimum(duck[i:e], seg[:e-i])
    if vo is not None:
        v = np.zeros(n); v[:min(n, len(vo))] = vo[:n]
        envv = ndimage_max(np.abs(v))
        duck = np.minimum(duck, 1-.62*envv)
        fx = fx*.75
        y = mus*duck*.9+fx+v*1.55
    else:
        y = mus*duck*.9+fx
    y = y[:int(SR*dur)]
    y = y/max(1e-6, np.max(np.abs(y)))*.89
    st = np.stack([y, y], 1)
    st = (st*32767).astype(np.int16)
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
