import numpy as np, math, os, subprocess, functools
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

W, H, FPS = 1080, 1920, 30
SS = 2  # overlay supersampling
FONT_DIR = '/usr/share/fonts/truetype/google-fonts/ofl/'
RNG = np.random.default_rng(7)

# ---------------- fonts ----------------
@functools.lru_cache(None)
def font(size, weight='Black'):
    f = ImageFont.truetype(FONT_DIR + 'montserrat/Montserrat[wght].ttf', size)
    try: f.set_variation_by_name(weight)
    except Exception: pass
    return f

@functools.lru_cache(None)
def sym_font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)

# ---------------- easing ----------------
def clamp(x, a=0., b=1.): return max(a, min(b, x))
def ease_io(t): t = clamp(t); return t*t*(3-2*t)
def ease_out(t): t = clamp(t); return 1-(1-t)**3
def ease_in(t): t = clamp(t); return t**3
def ease_expo(t):
    t = clamp(t)
    return 0.5*(2**(20*t-10)) if t < .5 else 1-0.5*(2**(-20*t+10)) if t < 1 else 1
def back_out(t, s=2.2):
    t = clamp(t); t -= 1; return t*t*((s+1)*t+s)+1
def prog(t, a, b): return clamp((t-a)/(b-a)) if b > a else float(t >= a)
def lerp(a, b, t): return a+(b-a)*t

# ---------------- noise / terrain ----------------
def value_noise(h, w, cell, seed):
    r = np.random.default_rng(seed)
    gh, gw = int(h/cell)+3, int(w/cell)+3
    g = r.random((gh, gw)).astype(np.float32)
    z = ndimage.zoom(g, (h/(gh-3)*1.0, w/(gw-3)*1.0), order=3)
    return z[:h, :w]

def fbm(h, w, base, octaves, seed, ridged=False):
    out = np.zeros((h, w), np.float32); amp = 1.; tot = 0.
    cell = base
    for o in range(octaves):
        n = value_noise(h, w, max(cell, 2), seed+o*17)
        if ridged: n = 1-np.abs(n*2-1)
        out += n*amp; tot += amp; amp *= .5; cell /= 2
    return out/tot

def terrain_rgb(land, elev, seed=1, sea_deep=(10, 38, 58), sea_shallow=(38, 112, 122), veg=None):
    """land: float mask 0..1, elev: 0..1 (land). returns uint8 HxWx3 satellite-ish image"""
    h, w = land.shape
    tex = fbm(h, w, 60, 5, seed+100)
    tex2 = fbm(h, w, 400, 3, seed+200)
    # land palette by elevation
    e = np.clip(elev, 0, 1)
    desert = np.array([186, 162, 120], np.float32)
    dry = np.array([150, 128, 92], np.float32)
    rock = np.array([108, 92, 76], np.float32)
    dark = np.array([72, 64, 56], np.float32)
    snow = np.array([238, 240, 245], np.float32)
    green = np.array([86, 104, 58], np.float32)
    c = desert[None, None]*(1-np.clip(e/.35, 0, 1))[..., None] + dry[None, None]*np.clip(e/.35, 0, 1)[..., None]
    m2 = np.clip((e-.35)/.3, 0, 1)[..., None]
    c = c*(1-m2)+rock*m2
    m3 = np.clip((e-.62)/.15, 0, 1)[..., None]
    c = c*(1-m3)+dark*m3
    if veg is not None:
        vm = (np.clip(veg, 0, 1)*np.clip((tex2-.45)*3, 0, 1))[..., None]
        c = c*(1-vm*.8)+green*vm*.8
    sm = np.clip((e-.8)/.08, 0, 1)*np.clip((tex-.35)*3, 0, 1)
    c = c*(1-sm[..., None])+snow*sm[..., None]
    c *= (0.82+0.36*tex)[..., None]*(0.9+0.2*tex2)[..., None]
    # hillshade
    gy, gx = np.gradient(elev*260)
    shade = np.clip(1+(-gx*0.7-gy*0.7)*1.0, .45, 1.5)
    c *= shade[..., None]
    # sea
    dist = ndimage.gaussian_filter(land, 18)
    shallow = np.clip(dist*2.2, 0, 1)
    sd, ss_ = np.array(sea_deep, np.float32), np.array(sea_shallow, np.float32)
    sea = sd*(1-shallow[..., None])+ss_*shallow[..., None]
    sea *= (0.94+0.12*tex2)[..., None]
    lm = np.clip(land, 0, 1)[..., None]
    # beach rim
    rim = np.clip(ndimage.gaussian_filter(land, 2)-land*0.0, 0, 1)
    out = sea*(1-lm)+c*lm
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------- camera / map ----------------
class Canvas:
    def __init__(self, img):
        self.levels = [img]
        im = img
        while min(im.size) > 600:
            im = im.resize((im.width//2, im.height//2), Image.BOX); self.levels.append(im)
    def view(self, cx, cy, vw, size=(W, H)):
        s = vw/size[0]
        L = 0
        while L+1 < len(self.levels) and s/2**(L+1) >= 0.8: L += 1
        k = 2**L; ss = s/k
        x0 = (cx - vw/2)/k; y0 = (cy - vw*size[1]/size[0]/2)/k
        return self.levels[L].transform(size, Image.AFFINE, (ss, 0, x0, 0, ss, y0), resample=Image.BILINEAR, fillcolor=(10, 38, 58))

class Cam:
    def __init__(self, cx, cy, vw): self.cx, self.cy, self.vw = cx, cy, vw
    def to_screen(self, x, y):
        s = W/self.vw
        return ((x-self.cx)*s+W/2, (y-self.cy)*s+H/2)
    @property
    def scale(self): return W/self.vw

def cam_lerp(a, b, t):
    # zoom in log space for natural feel
    vw = math.exp(lerp(math.log(a[2]), math.log(b[2]), t))
    return Cam(lerp(a[0], b[0], t), lerp(a[1], b[1], t), vw)

class GeoMap:
    def __init__(self, lon0, lon1, lat0, lat1, ppd):
        self.lon0, self.lon1, self.lat0, self.lat1, self.ppd = lon0, lon1, lat0, lat1, ppd
        self.w = int((lon1-lon0)*ppd); self.h = int((lat1-lat0)*ppd)
    def xy(self, lon, lat): return ((lon-self.lon0)*self.ppd, (self.lat1-lat)*self.ppd)

def load_countries():
    import shapefile
    from shapely.geometry import shape
    r = shapefile.Reader('/tmp/gp/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp')
    f = [x[0] for x in r.fields[1:]]
    out = {}
    for sr in r.iterShapeRecords():
        d = dict(zip(f, sr.record)); out[d['name']] = shape(sr.shape.__geo_interface__)
    return out

def geom_rings(g):
    polys = [g] if g.geom_type == 'Polygon' else list(g.geoms)
    return [list(p.exterior.coords) for p in polys]

# ---------------- overlay drawing (on SS-scaled RGBA layer) ----------------
class Layer:
    def __init__(self):
        self.im = Image.new('RGBA', (W*SS, H*SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
    def final(self):
        return self.im.resize((W, H), Image.BOX)

def S(p): return (p[0]*SS, p[1]*SS)

def polyline_pts(pts, n=12):
    """catmull-rom smooth"""
    if len(pts) < 3: return pts
    P = [pts[0]]+list(pts)+[pts[-1]]; out = []
    for i in range(1, len(P)-2):
        p0, p1, p2, p3 = map(np.array, P[i-1:i+3])
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(tuple(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3)))
    out.append(pts[-1]); return out

def path_upto(pts, frac):
    seg = [math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    L = sum(seg); tgt = L*clamp(frac); acc = 0; out = [pts[0]]
    for i, s in enumerate(seg):
        if acc+s >= tgt:
            t = (tgt-acc)/s if s else 0
            out.append((lerp(pts[i][0], pts[i+1][0], t), lerp(pts[i][1], pts[i+1][1], t))); return out, L
        out.append(pts[i+1]); acc += s
    return out, L

def draw_dashed(layer, pts, color, width=10, dash=34, gap=22, offset=0, frac=1., arrow=True, outline=True):
    if frac <= 0 or len(pts) < 2: return
    pts, _ = path_upto(pts, frac)
    pts = [S(p) for p in pts]; wv = width*SS; d = layer.d
    # resample
    res = [pts[0]]; acc = []
    dist = 0; samples = [(pts[0], 0.)]
    for i in range(len(pts)-1):
        a, b = pts[i], pts[i+1]; L = math.dist(a, b); n = max(1, int(L/(3*SS)))
        for k in range(1, n+1):
            t = k/n; samples.append(((lerp(a[0], b[0], t), lerp(a[1], b[1], t)), dist+L*t))
        dist += L
    per = (dash+gap)*SS
    segs = []; cur = []
    for p, s in samples:
        on = ((s - offset*SS) % per) < dash*SS
        if on: cur.append(p)
        elif cur: segs.append(cur); cur = []
    if cur: segs.append(cur)
    for sg in segs:
        if len(sg) > 1:
            if outline: d.line(sg, fill=(0, 0, 0, 120), width=int(wv+5*SS), joint='curve')
    for sg in segs:
        if len(sg) > 1: d.line(sg, fill=color, width=int(wv), joint='curve')
    if arrow and len(pts) >= 2:
        tip = pts[-1]; j = len(samples)-1
        while j > 0 and math.dist(samples[j][0], tip) < 20*SS: j -= 1
        base = samples[j][0]; ang = math.atan2(tip[1]-base[1], tip[0]-base[0])
        sz = width*SS*2.6
        pa = [(tip[0]+math.cos(ang)*sz*.6, tip[1]+math.sin(ang)*sz*.6),
              (tip[0]+math.cos(ang+2.5)*sz, tip[1]+math.sin(ang+2.5)*sz),
              (tip[0]+math.cos(ang-2.5)*sz, tip[1]+math.sin(ang-2.5)*sz)]
        d.polygon(pa, fill=color, outline=(0, 0, 0, 140))

def draw_solid_path(layer, pts, color, width=8, frac=1., glow=None):
    if frac <= 0: return
    p, _ = path_upto(pts, frac); p = [S(q) for q in p]
    if glow: layer.d.line(p, fill=glow, width=int((width+10)*SS), joint='curve')
    layer.d.line(p, fill=(0, 0, 0, 150), width=int((width+4)*SS), joint='curve')
    layer.d.line(p, fill=color, width=int(width*SS), joint='curve')

def draw_region(layer, rings_screen, fill, alpha, stroke=None, sw=4):
    for r in rings_screen:
        r2 = [S(p) for p in r]
        if len(r2) < 3: continue
        tmp = Image.new('RGBA', layer.im.size, (0, 0, 0, 0)); td = ImageDraw.Draw(tmp)
        td.polygon(r2, fill=fill+(int(alpha),))
        layer.im.alpha_composite(tmp)
        if stroke: layer.d.line(r2+[r2[0]], fill=stroke, width=int(sw*SS), joint='curve')

def draw_region_fast(layer, rings_screen, fill, alpha, stroke=None, sw=4):
    tmp = Image.new('RGBA', layer.im.size, (0, 0, 0, 0)); td = ImageDraw.Draw(tmp)
    for r in rings_screen:
        r2 = [S(p) for p in r]
        if len(r2) >= 3: td.polygon(r2, fill=fill+(int(alpha),))
    layer.im.alpha_composite(tmp)
    if stroke:
        for r in rings_screen:
            r2 = [S(p) for p in r]
            if len(r2) >= 3: layer.d.line(r2+[r2[0]], fill=stroke, width=int(sw*SS), joint='curve')

def tag(layer, p, text, a=1., size=26, dy=0):
    """google-earth style small black label"""
    if a <= 0: return
    f = font(size*SS, 'Bold')
    bb = layer.d.textbbox((0, 0), text, font=f)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    x, y = p[0]*SS, (p[1]+dy)*SS
    pad = 10*SS
    layer.d.rounded_rectangle([x-tw/2-pad, y-th/2-pad*.8, x+tw/2+pad, y+th/2+pad*.8], radius=6*SS, fill=(15, 15, 18, int(215*a)))
    layer.d.text((x-tw/2, y-th/2-bb[1]), text, font=f, fill=(255, 255, 255, int(255*a)))

def dot(layer, p, r, color, ring_t=None):
    x, y = S(p); r *= SS
    if ring_t is not None:
        rr = r*(1+ring_t*3); al = int(200*(1-ring_t))
        layer.d.ellipse([x-rr, y-rr, x+rr, y+rr], outline=color[:3]+(al,), width=int(4*SS))
    layer.d.ellipse([x-r-3*SS, y-r-3*SS, x+r+3*SS, y+r+3*SS], fill=(255, 255, 255, 255))
    layer.d.ellipse([x-r, y-r, x+r, y+r], fill=color)

@functools.lru_cache(512)
def text_img(text, size, color=(255, 255, 255), weight='Black', stroke=0, shadow=True):
    f = font(size*SS, weight)
    d0 = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
    bb = d0.textbbox((0, 0), text, font=f, stroke_width=stroke*SS)
    pad = 30*SS
    w, h = bb[2]-bb[0]+pad*2, bb[3]-bb[1]+pad*2
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    if shadow:
        sh = Image.new('RGBA', (w, h), (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        sd.text((pad-bb[0]+4*SS, pad-bb[1]+8*SS), text, font=f, fill=(0, 0, 0, 200), stroke_width=stroke*SS+4*SS, stroke_fill=(0, 0, 0, 200))
        sh = sh.filter(ImageFilter.GaussianBlur(8*SS)); im.alpha_composite(sh)
    d = ImageDraw.Draw(im)
    d.text((pad-bb[0], pad-bb[1]), text, font=f, fill=color+(255,), stroke_width=stroke*SS, stroke_fill=(0, 0, 0, 255))
    return im

def big_text(layer, p, text, t_in, size=96, color=(255, 255, 255), weight='Black', stroke=5, rot=0, out=None, t=None, anchor='c'):
    """pop-in text. t = local time, t_in = appear time"""
    if t < t_in: return
    k = back_out(prog(t, t_in, t_in+.35))
    a = 1.
    if out is not None and t > out: a = 1-prog(t, out, out+.25); k *= (1+.15*(1-a))
    if a <= 0 or k <= 0.01: return
    im = text_img(text, size, color, weight, stroke)
    if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    w, h = int(im.width*k), int(im.height*k)
    if w < 2 or h < 2: return
    im2 = im.resize((w, h), Image.BILINEAR)
    if a < 1:
        al = im2.getchannel('A').point(lambda v: int(v*a)); im2.putalpha(al)
    x, y = S(p)
    if anchor == 'l': x += w/2 - 30*SS*k
    layer.im.alpha_composite(im2, (int(x-w/2), int(y-h/2)))

def badge(layer, p, sym, t, t_in, r=56, bg=(220, 40, 40), fg=(255, 255, 255), symsize=None, ring=(255, 255, 255)):
    if t < t_in: return
    k = back_out(prog(t, t_in, t_in+.3), 3)
    x, y = S(p); R = r*SS*k
    if R < 1: return
    layer.d.ellipse([x-R-6*SS, y-R-6*SS+8*SS, x+R+6*SS, y+R+6*SS+8*SS], fill=(0, 0, 0, 110))
    layer.d.ellipse([x-R-6*SS*k, y-R-6*SS*k, x+R+6*SS*k, y+R+6*SS*k], fill=ring+(255,))
    layer.d.ellipse([x-R, y-R, x+R, y+R], fill=bg+(255,))
    f = sym_font(int((symsize or r*1.1)*SS*k)+1)
    bb = layer.d.textbbox((0, 0), sym, font=f)
    layer.d.text((x-(bb[2]+bb[0])/2, y-(bb[3]+bb[1])/2), sym, font=f, fill=fg+(255,))

def cross_out(layer, p, t, t_in, size=120, color=(235, 30, 30)):
    if t < t_in: return
    k = back_out(prog(t, t_in, t_in+.25), 2.5)
    x, y = S(p); s = size*SS*k
    for a, b in [((-1, -1), (1, 1)), ((1, -1), (-1, 1))]:
        layer.d.line([(x+a[0]*s/2, y+a[1]*s/2), (x+b[0]*s/2, y+b[1]*s/2)], fill=(0, 0, 0, 160), width=int(30*SS*k))
        layer.d.line([(x+a[0]*s/2, y+a[1]*s/2), (x+b[0]*s/2, y+b[1]*s/2)], fill=color+(255,), width=int(22*SS*k))

# ---------------- captions ----------------
CAP_OVR = {}
def captions(layer, t, phrases, y=1440):
    """phrases: list of (start, end, text). words reveal one by one; current word yellow."""
    fac = 0.75
    if CAP_OVR:
        phrases = [(CAP_OVR[x][0], CAP_OVR[x][1], x) if x in CAP_OVR else (a, b, x) for a, b, x in phrases]; fac = 0.95
    for (a, b, txt) in phrases:
        if a <= t < b:
            words = txt.split()
            # line wrap max ~ 22 chars
            lines = []; cur = []
            for w in words:
                if len(' '.join(cur+[w])) > 20 and cur: lines.append(cur); cur = [w]
                else: cur.append(w)
            if cur: lines.append(cur)
            n = len(words); dur = (b-a)*fac
            idx = int(clamp((t-a)/max(dur, .01))*n*0.999)+0 if dur > 0 else n
            f = font(64*SS, 'Black')
            k = 0
            for li, ln in enumerate(lines):
                widths = [layer.d.textlength(w+' ', font=f) for w in ln]
                tot = sum(widths); x = W*SS/2-tot/2; yy = (y+li*84)*SS
                for w, wd in zip(ln, widths):
                    if k <= idx:
                        col = (255, 222, 40) if k == idx else (255, 255, 255)
                        pop = 1.0
                        layer.d.text((x, yy), w, font=f, fill=col+(255,), stroke_width=7*SS, stroke_fill=(0, 0, 0, 255))
                    x += wd; k += 1
            return

# ---------------- post ----------------
_vig = None
def vignette(img, strength=.45):
    global _vig
    if _vig is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx-W/2)/(W*.7))**2+((yy-H/2)/(H*.62))**2)
        _vig = np.clip(1-strength*np.clip(r-.45, 0, 1)**1.4*1.6, 0, 1)[..., None].astype(np.float32)
    a = np.asarray(img).astype(np.float32)*_vig
    return Image.fromarray(a.astype(np.uint8))

def spotlight(img, p, radius, amount):
    if amount <= 0: return img
    a = np.asarray(img).astype(np.float32)
    yy, xx = np.ogrid[0:H, 0:W]
    d = np.sqrt((xx-p[0])**2+(yy-p[1])**2)
    m = np.clip((d-radius)/(radius*.35), 0, 1)
    f = 1-amount*.78*m
    return Image.fromarray((a*f[..., None]).astype(np.uint8))

def flash(img, a):
    if a <= 0: return img
    return Image.blend(img, Image.new('RGB', img.size, (255, 255, 255)), clamp(a))

def dim_blur(img, amt, blur=0):
    if blur: img = img.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(img, Image.new('RGB', img.size, (0, 0, 0)), amt)

# ---------------- encode ----------------
def render_scene(path, dur, frame_fn):
    n = int(round(dur*FPS))
    tmp = path[:-4]+'.part.mp4'
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
    for i in range(n):
        im = frame_fn(i/FPS)
        p.stdin.write(im.convert('RGB').tobytes())
    p.stdin.close(); assert p.wait() == 0
    os.replace(tmp, path)
