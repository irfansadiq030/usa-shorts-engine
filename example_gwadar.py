import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from engine import *
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Polygon, box
from scipy import ndimage
import audio

OUT = '/home/claude/vid/out_gwadar'; os.makedirs(OUT, exist_ok=True)

# ================= region map =================
M = GeoMap(54, 78, 18, 40.5, 160)
C = load_countries()
names = ['Pakistan', 'India', 'China', 'Oman', 'Iran', 'Afghanistan', 'United Arab Emirates', 'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Kyrgyzstan', 'Nepal', 'Saudi Arabia', 'Qatar', 'Yemen']
bb = box(M.lon0-1, M.lat0-1, M.lon1+1, M.lat1+1)
def ring_px(r): return [M.xy(*p) for p in r]

def build_region():
    land = Image.new('L', (M.w, M.h), 0); d = ImageDraw.Draw(land)
    for n in names:
        if n in C:
            g = C[n].intersection(bb)
            if g.is_empty: continue
            for r in geom_rings(g): d.polygon(ring_px(r), fill=255)
    land = np.asarray(land.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)/255
    h, w = land.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lon = M.lon0+xx/M.ppd; lat = M.lat1-yy/M.ppd
    dist = ndimage.distance_transform_edt(land > .5)
    inland = np.clip(dist/90, 0, 1)
    rid = fbm(h, w, 160, 5, 11, ridged=True)
    base = fbm(h, w, 300, 4, 12)
    north = np.clip((lat-33.2)/2.2, 0, 1)*np.clip((lon-68.5)/2, 0, 1)
    indus = np.exp(-((lon-(68.4+(lat-24)*0.55))**2)/1.6)*np.clip((33.5-lat)/2, 0, 1)*(lat > 23.6)
    punjab = np.clip((lon-71)/1.5, 0, 1)*np.clip((33-lat)/1.5, 0, 1)*np.clip((lat-28)/1, 0, 1)
    plains = np.clip(indus+punjab, 0, 1)
    tarim = np.clip((lat-37.6)/1, 0, 1)*np.clip((lon-75.6)/.8, 0, 1)
    elev = (0.12+0.42*rid*base*1.6)*inland
    elev = elev*(1-plains*.85)*(1-tarim*.8) + north*rid*0.75*(1-tarim)
    elev = np.clip(elev, 0, 1)*land
    veg = plains*.9 + north*.3
    img = terrain_rgb(land, elev, seed=5, veg=veg)
    return Image.fromarray(img), land

if os.path.exists(OUT+'/region.png'):
    REG = Image.open(OUT+'/region.png').convert('RGB')
else:
    REG, _ = build_region(); REG.save(OUT+'/region.png')
RC = Canvas(REG)

# ================= gwadar close-up canvas =================
GW, GH = 2800, 4800
def build_close():
    r = np.random.default_rng(4)
    land = Image.new('L', (GW, GH), 0); d = ImageDraw.Draw(land)
    # mainland top with wavy coast
    xs = np.linspace(-50, GW+50, 60)
    coast = 1700+ 90*np.sin(xs/380)+ 50*np.sin(xs/130+1)
    poly = [(-50, -50), (GW+50, -50)]+[(x, y) for x, y in zip(xs[::-1], coast[::-1])]
    d.polygon(poly, fill=255)
    # isthmus (concave bays)
    ys = np.linspace(1600, 2760, 40)
    left = [1080+ 95*math.sin((y-1600)/1160*math.pi)*1.0 + (y-1600)*0.08 for y in ys]
    right = [1520- 120*math.sin((y-1600)/1160*math.pi) - (y-1600)*0.12 for y in ys]
    # widen at top
    left = [l-260*max(0, 1-(y-1600)/420)**2 for l, y in zip(left, ys)]
    right = [rr+300*max(0, 1-(y-1600)/420)**2 for rr, y in zip(right, ys)]
    d.polygon([(l, y) for l, y in zip(left, ys)]+[(rr, y) for rr, y in zip(right[::-1], ys[::-1])], fill=255)
    # hammerhead
    hx = np.linspace(260, 2420, 70)
    top = [2740- 60*math.sin((x-260)/2160*math.pi) + 25*math.sin(x/90) for x in hx]
    bot = [3080+ 70*math.sin((x-260)/2160*math.pi) + 30*math.sin(x/70+2) for x in hx]
    ptsH = [(x, y) for x, y in zip(hx, top)]+[(x, y) for x, y in zip(hx[::-1], bot[::-1])]
    # rounded ends
    d.polygon(ptsH, fill=255)
    d.ellipse([200, 2760, 360, 3060], fill=255); d.ellipse([2330, 2740, 2510, 3060], fill=255)
    landf = np.asarray(land.filter(ImageFilter.GaussianBlur(3))).astype(np.float32)/255
    # elevation: hammerhead plateau high, isthmus flat
    hm = Image.new('L', (GW, GH), 0); ImageDraw.Draw(hm).polygon(ptsH, fill=255)
    hm = np.asarray(hm.filter(ImageFilter.GaussianBlur(30))).astype(np.float32)/255
    rid = fbm(GH, GW, 220, 5, 21, ridged=True)
    nz = fbm(GH, GW, 500, 3, 22)
    yy = np.arange(GH)[:, None].astype(np.float32)
    mainland_hills = np.clip((1300-yy)/900, 0, 1)
    elev = (0.03+0.06*nz)*landf + hm*(0.42+0.3*rid) + mainland_hills*(0.25+0.35*rid)*landf
    elev = np.clip(elev, 0, 1)
    img = terrain_rgb(landf, elev, seed=9, sea_deep=(8, 52, 74), sea_shallow=(52, 150, 150))
    im = Image.fromarray(img)
    dd = ImageDraw.Draw(im, 'RGBA')
    # town: grid of small blocks on isthmus
    for i in range(1400):
        y = r.uniform(1750, 2700);
        k = int((y-1600)/1160*39); k = min(max(k, 0), 39)
        l, rr_ = left[k]+30, right[k]-30
        if rr_-l < 20: continue
        x = r.uniform(l, rr_); s = r.uniform(8, 22)
        c = int(r.uniform(170, 225)); dd.rectangle([x, y, x+s, y+s*r.uniform(.6, 1.3)], fill=(c, c-6, c-14, 200))
    # port on east side
    px, py = right[22]+10, ys[22]
    dd.rectangle([px-40, py-120, px+260, py+80], fill=(200, 200, 196, 255))
    dd.rectangle([px+260, py-90, px+520, py-50], fill=(205, 205, 200, 255))
    dd.line([(px+40, py-130), (px+700, py-260), (px+760, py-180)], fill=(190, 190, 186, 255), width=26)
    for j in range(5): dd.rectangle([px+300+j*40, py-80, px+330+j*40, py-60], fill=(180, 60, 50, 255))
    return im, dict(port=(px+300, py-100), town=((left[25]+right[25])/2, ys[25]), head=(1340, 2910), isth=((left[18]+right[18])/2, ys[18]), wb=(560, 2300), eb=(2150, 2250))

if os.path.exists(OUT+'/close.png') and os.path.exists(OUT+'/close.npy'):
    CLOSE = Image.open(OUT+'/close.png').convert('RGB'); P = np.load(OUT+'/close.npy', allow_pickle=True).item()
else:
    CLOSE, P = build_close(); CLOSE.save(OUT+'/close.png'); np.save(OUT+'/close.npy', P)
CC = Canvas(CLOSE)

# ================ geography helpers ================
GWD = M.xy(62.32, 25.12); MUS = M.xy(58.59, 23.61)
pak = C['Pakistan']; oman = C['Oman'].intersection(bb)
enclave = Polygon([(61.6, 25.0), (61.6, 25.6), (62.8, 25.6), (62.8, 25.1)]).intersection(pak.buffer(0.05))
ENC = [ring_px(r) for r in geom_rings(enclave)]
OMAN = [ring_px(r) for r in geom_rings(oman)]
PAK = [ring_px(r) for r in geom_rings(pak)]

def scr(cam, rings): return [[cam.to_screen(*p) for p in r] for r in rings]

EVENTS = []
def ev(t, name): EVENTS.append((t, name))

def finish(img, layer, t=None, fl=0):
    img = vignette(img)
    img = img.convert('RGBA'); img.alpha_composite(layer.final())
    img = img.convert('RGB')
    if fl: img = flash(img, fl)
    return img

# ================= scenes =================
T0 = {}
def s1(t):  # HOOK
    cam = cam_lerp((P['town'][0], 2600, 3600), (P['town'][0], 2450, 2300), ease_out(t/4.5))
    img = CC.view(cam.cx, cam.cy, cam.vw)
    sp = cam.to_screen(*P['town'])
    img = spotlight(img, sp, 330, ease_io(prog(t, .2, .9)))
    L = Layer()
    tag(L, sp, 'Gwadar', prog(t, .1, .3), 30)
    big_text(L, (W/2, 330), 'PAKISTAN', .25, 120, t=t)
    big_text(L, (W/2, 470), 'BOUGHT THIS CITY', .55, 92, (255, 214, 0), t=t)
    captions(L, t, [(0.1, 2.3, "Pakistan didn't conquer this city..."), (2.3, 4.5, 'it BOUGHT it.')])
    return finish(img, L, fl=1-prog(t, 0, .15))

def s2(t):  # Oman owned it
    cam = cam_lerp((GWD[0], GWD[1], 700), (60.3*0+ (GWD[0]+MUS[0])/2, (GWD[1]+MUS[1])/2-60, 1450), ease_expo(prog(t, 0, 1.1)))
    img = RC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    a = prog(t, 1.2, 1.8)
    if a > 0: draw_region_fast(L, scr(cam, OMAN), (220, 40, 40), 120*a, (255, 80, 80, int(255*a)), 4)
    gp = cam.to_screen(*GWD); mp = cam.to_screen(*MUS)
    tag(L, cam.to_screen(*M.xy(57.0, 21.2)), 'OMAN', prog(t, 1.4, 1.7), 34)
    tag(L, cam.to_screen(*M.xy(63.2, 26.9)), 'PAKISTAN', prog(t, 1.4, 1.7), 34)
    path = polyline_pts([mp, (lerp(mp[0], gp[0], .5)+40, lerp(mp[1], gp[1], .5)+60), gp])
    draw_dashed(L, path, (255, 255, 255, 255), 9, offset=-t*80, frac=ease_io(prog(t, 2.2, 3.4)))
    dot(L, gp, 13, (230, 30, 30, 255), (t*1.2) % 1)
    tag(L, gp, 'Gwadar', 1, 28, dy=-52)
    big_text(L, (W/2, 330), "OMAN'S TERRITORY", 3.3, 74, (255, 70, 70), t=t)
    captions(L, t, [(0.2, 3.3, 'For 175 years, Gwadar was NOT part of Pakistan.'), (3.3, 6.5, 'It belonged to the Sultan of Oman.')])
    return finish(img, L, fl=.6*(1-prog(t, 0, .2)))

def s3(t):  # enclave + year counter
    cam = cam_lerp((GWD[0]-40, GWD[1], 1450), (GWD[0]-20, GWD[1]+40, 560), ease_io(prog(t, 0, 1.6)))
    img = RC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    a = prog(t, .8, 1.3)
    pulse = .5+.5*math.sin(t*6)
    if a > 0: draw_region_fast(L, scr(cam, ENC), (220, 40, 40), (110+60*pulse)*a, (255, 255, 255, int(255*a)), 5)
    gp = cam.to_screen(*GWD)
    tag(L, (gp[0], gp[1]-150), 'Gwadar enclave', prog(t, 1.0, 1.3), 30)
    yr = int(lerp(1783, 1958, ease_io(prog(t, 2.2, 4.6))))
    if t > 1.8:
        big_text(L, (W/2, 330), str(yr), 1.8, 170, (255, 255, 255), t=t)
        big_text(L, (W/2, 500), '175 YEARS OF OMANI RULE', 4.7, 62, (255, 214, 0), t=t)
    captions(L, t, [(0.2, 2.2, 'This coastal strip was its own enclave...'), (2.2, 6.0, 'Omani from 1783 all the way to 1958.')])
    return finish(img, L)

_bg4 = None
def s4(t):  # the price
    global _bg4
    if _bg4 is None:
        _bg4 = dim_blur(RC.view(GWD[0], GWD[1], 900), .55, 10)
    k = 1+.04*t
    img = _bg4.resize((int(W*k), int(H*k)), Image.BILINEAR).crop((int((W*k-W)/2), int((H*k-H)/2), int((W*k-W)/2)+W, int((H*k-H)/2)+H))
    L = Layer()
    big_text(L, (W/2, 420), 'THE PRICE', .1, 70, (200, 200, 200), 'ExtraBold', t=t)
    v = int(lerp(0, 3_000_000, ease_out(prog(t, .5, 2.4))))
    if t > .5: big_text(L, (W/2, 600), f'£{v:,}', .5, 128, (90, 230, 120), t=t)
    for i, (x, y, tt) in enumerate([(240, 860, 1.0), (840, 880, 1.3), (540, 960, 1.6)]):
        badge(L, (x, y), '£', t, tt, 54, (40, 160, 70), symsize=70)
    big_text(L, (W/2, 1130), '8 SEPTEMBER 1958', 3.0, 78, (255, 255, 255), rot=4, t=t)
    big_text(L, (W/2, 1240), 'DEAL SIGNED', 3.4, 60, (255, 214, 0), rot=4, t=t)
    captions(L, t, [(0.2, 2.9, 'Then Pakistan paid around £3 million...'), (2.9, 6.5, 'and by December 1958, Gwadar was Pakistani.')], y=1460)
    return finish(img, L, fl=.5*(1-prog(t, 0, .2)))

def s5(t):  # hammerhead geography
    cx = lerp(1340, 1360, t/7.5)
    cam = cam_lerp((1340, 2700, 1500), (1360, 2560, 2750), ease_io(prog(t, 0, 2.2)))
    img = CC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    big_text(L, (W/2, 300), 'WHY THIS SHAPE?', .2, 96, t=t)
    hp = cam.to_screen(*P['head']); ip = cam.to_screen(*P['isth']); wb = cam.to_screen(*P['wb']); eb = cam.to_screen(*P['eb'])
    # hammerhead outline arrow
    big_text(L, (hp[0], hp[1]), 'ROCKY HAMMERHEAD', 2.2, 52, (255, 214, 0), t=t)
    tag(L, (ip[0]-250, ip[1]-60), 'Sand strip (tombolo)', prog(t, 3.2, 3.5), 30)
    draw_dashed(L, polyline_pts([(ip[0]-230, ip[1]-25), (ip[0]-120, ip[1]+10), (ip[0]-25, ip[1]+10)]), (255, 255, 255, 255), 7, 20, 14, -t*40, ease_io(prog(t, 3.3, 3.9)))
    tag(L, wb, 'West Bay', prog(t, 4.4, 4.7), 30)
    tag(L, eb, 'East Bay', prog(t, 4.7, 5.0), 30)
    captions(L, t, [(0.2, 2.2, 'Why does Gwadar look like this?'), (2.2, 4.4, 'A rocky hammerhead, joined to land by sand...'), (4.4, 7.5, 'creating two natural bays on either side.')])
    return finish(img, L)

def s6(t):  # CPEC gateway + outro
    route_ll = [(62.32, 25.12), (63.4, 26.3), (65.4, 28.4), (67.0, 30.18), (70.9, 31.83), (73.05, 33.68), (73.6, 35.0), (74.3, 35.92), (75.42, 36.85), (75.6, 38.2), (75.99, 39.47)]
    route = [M.xy(*p) for p in route_ll]
    mid = ((route[0][0]+route[-1][0])/2+80, (route[0][1]+route[-1][1])/2)
    cam = cam_lerp((GWD[0], GWD[1], 600), (mid[0], mid[1], 1700), ease_io(prog(t, .9, 2.6)))
    img = RC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    a = prog(t, 2.2, 2.7)
    if a > 0: draw_region_fast(L, scr(cam, PAK), (40, 170, 80), 70*a, (120, 255, 150, int(220*a)), 3)
    sp = polyline_pts([cam.to_screen(*p) for p in route], 8)
    draw_dashed(L, sp, (255, 214, 0, 255), 10, offset=-t*90, frac=ease_io(prog(t, 2.4, 4.6)))
    gp = cam.to_screen(*GWD)
    badge(L, (gp[0]+10, gp[1]-90), '⚓', t, .3, 48, (30, 110, 200), symsize=58)
    tag(L, cam.to_screen(*M.xy(75.99, 39.47)), 'Kashgar, CHINA', prog(t, 4.4, 4.7), 28, dy=-48)
    big_text(L, (W/2, 300), 'GATEWAY OF CPEC', 2.8, 90, (255, 214, 0), t=t)
    big_text(L, (W/2, 1200), 'FOLLOW FOR MORE', 5.4, 76, t=t)
    captions(L, t, [(0.2, 2.6, 'Today it is a deep-sea port...'), (2.6, 5.3, 'and the gateway of CPEC to China.'), (5.3, 7.0, 'Follow for more!')], y=1480)
    return finish(img, L)

SCENES = [(s1, 4.5), (s2, 6.5), (s3, 6.0), (s4, 6.5), (s5, 7.5), (s6, 7.0)]
# sound design (global times)
import json, engine
_vo = json.load(open('/home/claude/vid/out_gwadar/vo.json'))
SCENES = [(fn, d) for (fn, _), d in zip(SCENES, _vo['durs'])]
engine.CAP_OVR.update({k: tuple(v) for k, v in _vo['ovr'].items()})
VO = np.load('/home/claude/vid/out_gwadar/vo.npy')
st = np.cumsum([0]+[d for _, d in SCENES])
def G(i, t): return st[i]+t
ev(0.0, 'whoosh'); ev(.25, 'impact'); ev(.55, 'pop'); ev(.6, 'swoosh')
ev(G(1, 0)-.25, 'whoosh_out'); ev(G(1, 1.2), 'pop'); ev(G(1, 1.45), 'pop_hi'); ev(G(1, 2.2), 'swoosh'); ev(G(1, 3.3), 'stamp')
ev(G(2, 0)-.2, 'whoosh'); ev(G(2, .8), 'pop'); ev(G(2, 1.8), 'impact')
for k in range(24): ev(G(2, 2.2+k*.1), 'tick')
ev(G(2, 4.7), 'pop_hi')
ev(G(3, 0)-.25, 'whoosh'); [ev(G(3, .5+k*.08), 'tick') for k in range(23)]; ev(G(3, 2.4), 'cash')
ev(G(3, 1.0), 'pop'); ev(G(3, 1.3), 'pop'); ev(G(3, 1.6), 'pop'); ev(G(3, 3.0), 'stamp'); ev(G(3, 3.4), 'pop_hi')
ev(G(4, 0)-.25, 'whoosh_out'); ev(G(4, .2), 'impact'); ev(G(4, 2.2), 'pop'); ev(G(4, 3.3), 'swoosh'); ev(G(4, 4.4), 'pop'); ev(G(4, 4.7), 'pop')
ev(G(5, 0)-.25, 'whoosh'); ev(G(5, .3), 'pop_hi'); ev(G(5, .9), 'whoosh_out'); ev(G(5, 2.4), 'riser'); ev(G(5, 2.8), 'impact'); ev(G(5, 5.4), 'pop'); ev(G(5, 4.4), 'ding')

if __name__ == '__main__':
    only = sys.argv[1:]
    if only and only[0] == 'still':
        i, t = int(only[1]), float(only[2])
        SCENES[i][0](t).save(f'{OUT}/still_{i}_{t}.jpg', quality=90); sys.exit()
    files = []
    for i, (fn, d) in enumerate(SCENES):
        p = f'{OUT}/scene{i}.mp4'; files.append(p)
        if not os.path.exists(p): render_scene(p, d, fn); print('scene', i, 'done', flush=True)
    with open(OUT+'/list.txt', 'w') as f:
        for p in files: f.write(f"file '{p}'\n")
    total = float(st[-1])
    audio.mix(total, EVENTS, OUT+'/audio.wav', bpm=92, root=55.0, vo=VO)
    os.system(f"ffmpeg -y -loglevel error -f concat -safe 0 -i {OUT}/list.txt -c copy {OUT}/silent.mp4 && ffmpeg -y -loglevel error -i {OUT}/silent.mp4 -i {OUT}/audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest /home/claude/vid/Sample1_Gwadar_VO.mp4")
    print('done')
