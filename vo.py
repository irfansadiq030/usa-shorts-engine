import json, sys, numpy as np
from kokoro_onnx import Kokoro
from scipy import signal
k = Kokoro('/tmp/tts/kokoro.onnx', '/tmp/tts/voices.bin')
VOICE = 'am_michael'
def gen(text, speed):
    s, sr = k.create(text, voice=VOICE, speed=speed, lang='en-us')
    s = np.asarray(s, np.float32)
    nz = np.where(np.abs(s) > 0.01)[0]
    s = s[max(0, nz[0]-240):nz[-1]+1200]
    return signal.resample_poly(s, 147, 80)  # 24k -> 44.1k

def build(scenes, out_json, out_npy, base_speed=1.12, gap=.12, lead=.25, tail=.45):
    SR = 44100; durs = []; ovr = {}; clips = []
    t0 = 0.
    for dur, phrases in scenes:
        sp = base_speed
        for _ in range(4):
            aud = [gen(s, sp) for _, s in phrases]
            tot = lead+sum(len(a)/SR for a in aud)+gap*(len(aud)-1)+tail
            if tot <= dur or sp >= 1.32: break
            sp = min(1.32, sp*tot/dur+.02)
        d = max(dur, tot); durs.append(round(d*30)/30)
        t = lead
        for (cap, _), a in zip(phrases, aud):
            L = len(a)/SR; ovr[cap] = [t, t+L+ (gap if cap != phrases[-1][0] else tail*.8)]
            clips.append((t0+t, a)); t += L+gap
        print(f'scene dur {dur} -> {d:.2f} speed {sp:.2f}', flush=True)
        t0 += durs[-1]
    v = np.zeros(int(SR*(t0+1)), np.float32)
    for st, a in clips:
        i = int(st*SR); v[i:i+len(a)] += a
    v = v/np.max(np.abs(v))*.5
    np.save(out_npy, v)
    json.dump({'durs': durs, 'ovr': ovr}, open(out_json, 'w'), indent=1)
