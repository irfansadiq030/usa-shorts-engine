import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from engine import *
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import box
from scipy import ndimage
import audio

OUT = '/home/claude/vid/out_kkh'; os.makedirs(OUT, exist_ok=True)

M = GeoMap(66, 80.5, 28.5, 44, 270)
C = load_countries()
names = ['Pakistan', 'India', 'China', 'Afghanistan', 'Tajikistan', 'Kyrgyzstan', 'Uzbekistan', 'Turkmenistan', 'Nepal', 'Iran']
bb = box(M.lon0-1, M.lat0-1, M.lon1+1, M.lat1+1)
def ring_px(r): return [M.xy(*p) for p in r]

def build_region():
    land = np.ones((M.h, M.w), np.float32)
    h, w = land.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lon = M.lon0+xx/M.ppd; lat = M.lat1-yy/M.ppd
    rid = np.clip(fbm(h, w, 220, 6, 31, ridged=True), 0, 1)
    base = fbm(h, w, 420, 4, 32)
    mtn = np.clip((lat-33.6)/1.6, 0, 1)*np.clip((lon-69)/2, 0, 1)
    mtn = np.maximum(mtn, np.clip((lat-34)/2, 0, 1)*.7)
    plains = np.clip((33.9-lat)/1.0, 0, 1)*np.clip((lon-70.2)/1.2, 0, 1)
    tarim = np.clip((lat-38.4)/.8, 0, 1)*np.clip((lon-76.2)/.8, 0, 1)
    # river valleys (Indus + Hunza) carve
    elev = 0.18+0.3*rid*base*1.5
    elev = elev*(1-plains*.85) + mtn*(rid**1.3)*0.95*(1-tarim)
    elev = elev*(1-tarim*.85)
    elev = np.clip(elev, 0, 1)
    veg = plains*.9
    img = terrain_rgb(land, elev, seed=7, veg=veg)
    return Image.fromarray(img)

if os.path.exists(OUT+'/region.png'): REG = Image.open(OUT+'/region.png').convert('RGB')
else: REG = build_region(); REG.save(OUT+'/region.png')
RC = Canvas(REG)

# mountain close-up canvas
MW, MH = 2400, 4800
def river_x(y): return 1200+360*np.sin(y/720)+140*np.sin(y/240+1)
def build_mtn():
    h, w = MH, MW
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = river_x(yy)
    d = np.abs(xx-rx)
    rid = fbm(h, w, 380, 6, 41, ridged=True)
    rid2 = fbm(h, w, 120, 4, 42, ridged=True)
    valley = np.clip((d-60)/700, 0, 1)**0.8
    elev = (0.25+0.55*rid+0.2*rid2)*valley
    elev = np.clip(elev*1.05, 0, 1)
    land = (d > 34).astype(np.float32)
    land = ndimage.gaussian_filter(land, 2)
    img = terrain_rgb(land, elev, seed=13, sea_deep=(70, 120, 120), sea_shallow=(110, 150, 140))
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, 'RGBA')
    ys = np.arange(0, h, 6)
    road = [(river_x(y)+72+10*math.sin(y/90), y) for y in ys]
    dr.line(road, fill=(60, 58, 56, 255), width=18); dr.line(road, fill=(150, 146, 140, 255), width=12)
    return im

if os.path.exists(OUT+'/mtn.png'): MTN = Image.open(OUT+'/mtn.png').convert('RGB')
else: MTN = build_mtn(); MTN.save(OUT+'/mtn.png')
MC = Canvas(MTN)
ROAD = [(river_x(y)+72+10*math.sin(y/90), float(y)) for y in np.arange(0, MH, 6)]

def road_from(y0, y1):
    return [p for p in ROAD if y1 <= p[1] <= y0][::-1]

PLACES = {'Hasan Abdal': (72.69, 33.82), 'Abbottabad': (73.21, 34.15), 'Thakot': (72.93, 34.86), 'Dasu': (73.29, 35.29), 'Chilas': (74.1, 35.42),
          'Raikot': (74.6, 35.5), 'Gilgit': (74.3, 35.92), 'Karimabad': (74.66, 36.32), 'Sost': (74.82, 36.69), 'Khunjerab': (75.42, 36.85),
          'Tashkurgan': (75.23, 37.77), 'Kashgar': (75.99, 39.47)}
RT = ['Hasan Abdal', 'Abbottabad', 'Thakot', 'Dasu', 'Chilas', 'Raikot', 'Gilgit', 'Karimabad', 'Sost', 'Khunjerab', 'Tashkurgan', 'Kashgar']
ROUTE = [M.xy(*PLACES[k]) for k in RT]
pak = C['Pakistan']; china = C['China'].intersection(bb)
PAK = [ring_px(r) for r in geom_rings(pak)]; CHN = [ring_px(r) for r in geom_rings(china)]
def scr(cam, rings): return [[cam.to_screen(*p) for p in r] for r in rings]

EVENTS = []
def ev(t, n): EVENTS.append((t, n))
def finish(img, layer, fl=0):
    img = vignette(img).convert('RGBA'); img.alpha_composite(layer.final()); img = img.convert('RGB')
    return flash(img, fl) if fl else img

def s1(t):  # hook: fly up the valley, road lights up
    y = lerp(3700, 2700, ease_io(t/4.5))
    vw = lerp(1500, 1150, ease_out(t/4.5)); vh = vw*H/W
    cam = Cam(clamp(float(river_x(y-200)), vw/2, MW-vw/2), clamp(y-200, vh/2, MH-vh/2), vw)
    img = MC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    seg = road_from(4700, 1500)
    sp = [cam.to_screen(*p) for p in seg]
    draw_solid_path(L, sp, (255, 60, 40, 255), 9, frac=ease_io(prog(t, .2, 4.3)), glow=(255, 60, 40, 70))
    big_text(L, (W/2, 330), 'THIS ROAD COST', .25, 104, t=t)
    big_text(L, (W/2, 480), '~1,000 LIVES', .65, 140, (255, 70, 50), t=t)
    captions(L, t, [(0.1, 2.4, 'Nearly 1,000 workers died...'), (2.4, 4.5, 'building this one road.')])
    return finish(img, L, fl=1-prog(t, 0, .15))

def s2(t):  # route map
    g = M.xy(74.3, 35.92)
    mid = ((ROUTE[0][0]+ROUTE[-1][0])/2-30, (ROUTE[0][1]+ROUTE[-1][1])/2)
    cam = cam_lerp((g[0], g[1], 600), (mid[0]+40, ROUTE[-1][1]+650, 1950), ease_expo(prog(t, 0, 1.2)))
    img = RC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    a = prog(t, 1.0, 1.5)
    if a > 0:
        draw_region_fast(L, scr(cam, PAK), (40, 170, 80), 70*a, (120, 255, 150, int(230*a)), 3)
        draw_region_fast(L, scr(cam, CHN), (220, 40, 40), 70*a, (255, 110, 110, int(230*a)), 3)
    tag(L, cam.to_screen(*M.xy(71.9, 33.1)), 'PAKISTAN', prog(t, 1.2, 1.5), 34)
    tag(L, cam.to_screen(*M.xy(78.4, 38.9)), 'CHINA', prog(t, 1.2, 1.5), 34)
    sp = polyline_pts([cam.to_screen(*p) for p in ROUTE], 8)
    fr = ease_io(prog(t, 1.5, 4.2))
    draw_dashed(L, sp, (255, 214, 0, 255), 10, offset=-t*90, frac=fr)
    for k, tt in [('Hasan Abdal', 1.5), ('Gilgit', 2.7), ('Khunjerab', 3.5), ('Kashgar', 4.1)]:
        if t > tt:
            p = cam.to_screen(*M.xy(*PLACES[k])); dot(L, p, 10, (255, 214, 0, 255))
            tag(L, p, k, prog(t, tt, tt+.2), 28, dy=-46)
    big_text(L, (W/2, 330), 'KARAKORAM HIGHWAY', .9, 80, t=t)
    big_text(L, (W/2, 470), '1,300 KM', 4.3, 120, (255, 214, 0), t=t)
    captions(L, t, [(0.2, 2.5, 'The Karakoram Highway...'), (2.5, 6.5, '1,300 km from Pakistan to Kashgar, China.')], y=1580)
    return finish(img, L, fl=.6*(1-prog(t, 0, .2)))

PROF = [(0, 500), (60, 560), (160, 700), (260, 950), (330, 1250), (420, 1350), (480, 1500), (580, 2400), (690, 2800), (740, 3600), (790, 4693), (840, 4000), (900, 3100), (1050, 2000), (1300, 1290)]
_bg3 = None
def s3(t):  # elevation profile
    global _bg3
    if _bg3 is None: _bg3 = dim_blur(MC.view(1200, 2500, 2400), .6, 6)
    k = 1+.03*t
    img = _bg3.resize((int(W*k), int(H*k))).crop((int((W*k-W)/2), int((H*k-H)/2), int((W*k-W)/2)+W, int((H*k-H)/2)+H))
    L = Layer(); d = L.d
    x0, x1, y0, y1 = 110, 970, 1260, 640
    def P(km, m): return (lerp(x0, x1, km/1300), lerp(y0, y1, m/5000))
    a = prog(t, .1, .5)
    d.line([S((x0, y0)), S((x1, y0))], fill=(255, 255, 255, int(200*a)), width=3*SS)
    for m in (1000, 2000, 3000, 4000, 5000):
        yy = P(0, m)[1]
        d.line([S((x0, yy)), S((x1, yy))], fill=(255, 255, 255, int(40*a)), width=SS*2)
        d.text(S((x0-8, yy-30)), f'{m:,} m', font=font(22*SS, 'Bold'), fill=(255, 255, 255, int(150*a)))
    fr = ease_io(prog(t, .6, 3.4))
    pts = polyline_pts([P(*q) for q in PROF], 10)
    up, _ = path_upto(pts, fr)
    if len(up) > 1:
        tmp = Image.new('RGBA', L.im.size, (0, 0, 0, 0)); td = ImageDraw.Draw(tmp)
        td.polygon([S(p) for p in up]+[S((up[-1][0], y0)), S((x0, y0))], fill=(255, 214, 0, 70)); L.im.alpha_composite(tmp)
        draw_solid_path(L, up, (255, 214, 0, 255), 7)
        hx, hy = up[-1]
        cur_m = int(lerp(0, 5000, (y0-hy)/(y0-y1)))
        if t < 3.4: big_text(L, (hx, hy-70), f'{cur_m:,} m', .6, 44, t=t)
    pk = P(790, 4693)
    if t > 2.4:
        dot(L, pk, 12, (255, 60, 40, 255), (t*1.3) % 1)
        big_text(L, (pk[0], pk[1]-120), '4,693 m', 2.4, 92, (255, 255, 255), t=t)
        tag(L, (pk[0], pk[1]-50), 'Khunjerab Pass', prog(t, 2.6, 2.9), 28)
    big_text(L, (W/2, 330), 'HOW HIGH?', .1, 100, t=t)
    big_text(L, (W/2, 1360), 'HIGHEST PAVED BORDER', 4.0, 64, (255, 214, 0), t=t)
    big_text(L, (W/2, 1440), 'CROSSING ON EARTH', 4.2, 64, (255, 214, 0), t=t)
    captions(L, t, [(0.2, 3.6, 'It climbs to 4,693 metres at Khunjerab Pass...'), (3.6, 7.0, 'the highest paved border crossing on Earth.')], y=1580)
    d.text(S((x1-160, y0+14)), 'profile approx.', font=font(20*SS, 'Medium'), fill=(255, 255, 255, int(130*a)))
    return finish(img, L, fl=.5*(1-prog(t, 0, .2)))

def s4(t):  # Mintaka vs Khunjerab
    sost = M.xy(*PLACES['Sost']); kh = M.xy(*PLACES['Khunjerab']); mi = M.xy(74.85, 37.03); tk = M.xy(*PLACES['Tashkurgan'])
    c = ((sost[0]+kh[0])/2, (sost[1]+tk[1])/2+40)
    cam = cam_lerp((c[0], c[1]+120, 1500), (c[0], c[1], 560), ease_expo(prog(t, 0, 1.2)))
    img = RC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    S0, K, Mi, T = [cam.to_screen(*p) for p in (sost, kh, mi, tk)]
    draw_dashed(L, polyline_pts([S0, (lerp(S0[0], Mi[0], .5)-30, lerp(S0[1], Mi[1], .5)), Mi, (lerp(Mi[0], T[0], .5)-40, lerp(Mi[1], T[1], .5)), T]), (255, 70, 60, 255), 10, offset=-t*80, frac=ease_io(prog(t, 1.0, 2.4)))
    tag(L, Mi, 'Mintaka Pass', prog(t, 1.4, 1.7), 30, dy=-115)
    cross_out(L, Mi, t, 3.2, 150)
    big_text(L, (Mi[0]-10, Mi[1]+110), 'EXPOSED TO', 3.4, 40, (255, 90, 80), t=t)
    big_text(L, (Mi[0]-10, Mi[1]+160), 'AIR STRIKES', 3.5, 40, (255, 90, 80), t=t)
    if t > 4.0:
        draw_dashed(L, polyline_pts([S0, (lerp(S0[0], K[0], .5)+20, lerp(S0[1], K[1], .5)+30), K, (lerp(K[0], T[0], .5)+40, lerp(K[1], T[1], .5)), T]), (80, 230, 110, 255), 11, offset=-t*80, frac=ease_io(prog(t, 4.0, 5.4)))
    tag(L, K, 'Khunjerab Pass', prog(t, 4.4, 4.7), 30, dy=70)
    badge(L, (K[0]+90, K[1]-40), '✓', t, 5.4, 46, (40, 170, 70))
    dot(L, S0, 9, (255, 255, 255, 255)); tag(L, S0, 'Sost', 1, 26, dy=46)
    big_text(L, (W/2, 330), 'WHICH PASS?', .3, 104, t=t)
    captions(L, t, [(0.3, 3.2, 'Pakistan first favoured the Mintaka Pass...'), (3.2, 7.0, 'but China chose Khunjerab — safer from air strikes.')], y=1460)
    return finish(img, L, fl=.5*(1-prog(t, 0, .2)))

def person(L, cx, cy, s, col, a=1.):
    d = L.d; x, y = cx*SS, cy*SS; s *= SS
    d.ellipse([x-s*.18, y-s*.48, x+s*.18, y-s*.12], fill=col+(int(255*a),))
    d.rounded_rectangle([x-s*.3, y-s*.06, x+s*.3, y+s*.48], radius=int(s*.16), fill=col+(int(255*a),))

def s5(t):  # workers grid
    global _bg3
    if _bg3 is None: _bg3 = dim_blur(MC.view(1200, 2500, 2400), .6, 6)
    img = dim_blur(_bg3, .35)
    L = Layer()
    cols, cell = 10, 86
    gx0 = W/2-(cols-1)*cell/2; gy0 = 690
    shown = int(lerp(0, 101, ease_io(prog(t, .6, 4.2))))
    for i in range(101):
        if i >= shown: break
        r_, c_ = divmod(i, cols)
        col = (70, 210, 100) if i < 81 else (240, 70, 60)
        k = back_out(clamp((shown-i)/4), 2)
        person(L, gx0+c_*cell, gy0+r_*cell, 74*k, col)
    np_ = int(lerp(0, 810, ease_io(prog(t, .6, 3.6)))); nc = int(lerp(0, 200, ease_io(prog(t, 3.4, 4.3))))
    big_text(L, (290, 470), f'~{np_}', .5, 110, (70, 210, 100), t=t)
    big_text(L, (290, 570), 'PAKISTANI', .5, 40, (255, 255, 255), 'ExtraBold', t=t)
    if t > 3.4:
        big_text(L, (790, 470), f'~{nc}', 3.4, 110, (240, 70, 60), t=t)
        big_text(L, (790, 570), 'CHINESE', 3.4, 40, (255, 255, 255), 'ExtraBold', t=t)
    big_text(L, (W/2, 260), '1 ICON = 10 WORKERS', .2, 46, (200, 200, 200), 'ExtraBold', t=t)
    captions(L, t, [(0.2, 3.4, 'About 810 Pakistani workers lost their lives...'), (3.4, 7.0, 'and about 200 Chinese — mostly in landslides and falls.')], y=1680)
    return finish(img, L, fl=.4*(1-prog(t, 0, .2)))

def s6(t):  # outro
    y = lerp(2900, 2300, ease_io(t/6.5))
    vw = lerp(1900, 1700, t/6.5); cam = Cam(clamp(float(river_x(y)), vw/2, MW-vw/2), y, vw)
    img = MC.view(cam.cx, cam.cy, cam.vw)
    L = Layer()
    seg = road_from(4700, 0)
    draw_solid_path(L, [cam.to_screen(*p) for p in seg], (255, 214, 0, 255), 7, glow=(255, 214, 0, 60))
    big_text(L, (W/2, 330), "THE WORLD'S", .3, 84, t=t)
    big_text(L, (W/2, 470), '8TH WONDER', .6, 140, (255, 214, 0), t=t)
    big_text(L, (W/2, 1230), 'FOLLOW FOR MORE', 4.4, 76, t=t)
    captions(L, t, [(0.2, 4.3, "That's why they call it the 8th Wonder of the World."), (4.3, 6.5, 'Follow for more!')], y=1440)
    return finish(img, L, fl=.6*(1-prog(t, 0, .2)))

SCENES = [(s1, 4.5), (s2, 6.5), (s3, 7.0), (s4, 7.0), (s5, 7.0), (s6, 6.5)]
import json, engine
_vo = json.load(open('/home/claude/vid/out_kkh/vo.json'))
SCENES = [(fn, d) for (fn, _), d in zip(SCENES, _vo['durs'])]
engine.CAP_OVR.update({k: tuple(v) for k, v in _vo['ovr'].items()})
VO = np.load('/home/claude/vid/out_kkh/vo.npy')
st = np.cumsum([0]+[d for _, d in SCENES])
def G(i, t): return st[i]+t
ev(0, 'riser'); ev(.25, 'impact'); ev(.65, 'pop'); ev(.7, 'swoosh')
ev(G(1, 0)-.25, 'whoosh_out'); ev(G(1, .9), 'pop'); ev(G(1, 1.0), 'swoosh'); ev(G(1, 1.5), 'pop_hi'); ev(G(1, 2.7), 'pop'); ev(G(1, 3.5), 'pop'); ev(G(1, 4.1), 'pop'); ev(G(1, 4.3), 'impact')
ev(G(2, 0)-.25, 'whoosh'); ev(G(2, .1), 'pop'); ev(G(2, .6), 'riser'); [ev(G(2, .7+k*.12), 'tick') for k in range(22)]; ev(G(2, 2.4), 'impact'); ev(G(2, 4.0), 'pop_hi'); ev(G(2, 4.2), 'pop_hi')
ev(G(3, 0)-.25, 'whoosh_out'); ev(G(3, .3), 'pop'); ev(G(3, 1.0), 'swoosh'); ev(G(3, 1.4), 'pop'); ev(G(3, 3.2), 'stamp'); ev(G(3, 3.4), 'pop'); ev(G(3, 4.0), 'swoosh'); ev(G(3, 4.4), 'pop'); ev(G(3, 5.4), 'ding')
ev(G(4, 0)-.25, 'whoosh'); ev(G(4, .2), 'pop'); [ev(G(4, .6+k*.17), 'tick') for k in range(22)]; ev(G(4, 3.4), 'impact')
ev(G(5, 0)-.25, 'whoosh'); ev(G(5, .3), 'pop'); ev(G(5, .6), 'impact'); ev(G(5, 4.4), 'ding')

if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'still':
        i, t = int(a[1]), float(a[2]); SCENES[i][0](t).save(f'{OUT}/still_{i}_{t}.jpg', quality=90); sys.exit()
    files = []
    for i, (fn, d) in enumerate(SCENES):
        p = f'{OUT}/scene{i}.mp4'; files.append(p)
        if not os.path.exists(p): render_scene(p, d, fn); print('scene', i, 'done', flush=True)
    with open(OUT+'/list.txt', 'w') as f:
        for p in files: f.write(f"file '{p}'\n")
    audio.mix(float(st[-1]), EVENTS, OUT+'/audio.wav', bpm=96, root=49.0, vo=VO)
    os.system(f"ffmpeg -y -loglevel error -f concat -safe 0 -i {OUT}/list.txt -c copy {OUT}/silent.mp4 && ffmpeg -y -loglevel error -i {OUT}/silent.mp4 -i {OUT}/audio.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest /home/claude/vid/Sample2_Karakoram_Highway_VO.mp4")
    print('done')
