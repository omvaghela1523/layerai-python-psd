from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io, os, struct
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

# ── Helpers ───────────────────────────────────────────────────────
p = lambda fmt, *a: struct.pack(fmt, *a)

def pstring(s, pad=4):
    """Pascal string padded to multiple of `pad`"""
    b = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(b)]) + b
    r = len(raw) % pad
    if r: raw += b'\x00' * (pad - r)
    return raw

def make_vignette(w, h):
    img = Image.new('RGBA', (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    for i in range(30):
        t = i / 30
        a = int(150*(1-t))
        x0,y0 = int(w*t*0.4), int(h*t*0.4)
        d.rectangle([x0,y0,w-x0,h-y0], outline=(0,0,0,a))
    return img.filter(ImageFilter.GaussianBlur(15))

# ── PSD builder ───────────────────────────────────────────────────
def build_psd(layer_list, W, H):
    """
    Spec-correct PSD 1.0
    All lengths are big-endian uint32.
    Channel data: compression=0 (raw), full plane per channel.
    """

    # ── 1. Header (26 bytes) ──────────────────────────────────────
    s1  = b'8BPS'
    s1 += p('>H', 1)            # version
    s1 += b'\x00'*6             # reserved
    s1 += p('>H', 4)            # channels  (RGBA)
    s1 += p('>I', H)
    s1 += p('>I', W)
    s1 += p('>H', 8)            # depth
    s1 += p('>H', 3)            # color mode: RGB

    # ── 2. Color Mode Data ────────────────────────────────────────
    s2 = p('>I', 0)

    # ── 3. Image Resources ────────────────────────────────────────
    s3 = p('>I', 0)

    # ── 4. Layer & Mask Info ──────────────────────────────────────
    def channel_bytes(arr2d):
        """raw channel: uint16 compression=0 + raw pixels"""
        return p('>H', 0) + arr2d.astype(np.uint8).tobytes()

    records = b''
    chandata = b''

    for lname, limg, blend4, opac in layer_list:
        img = limg.convert('RGBA').resize((W, H), Image.LANCZOS)
        a = np.array(img, dtype=np.uint8)

        # channel sizes
        chs = [(-1,3),(0,0),(1,1),(2,2)]
        parts = []
        for cid, ci in chs:
            cb = channel_bytes(a[:,:,ci])
            parts.append((cid, cb))

        # layer record
        rec  = p('>IIII', 0, 0, H, W)   # top left bottom right
        rec += p('>H', 4)                # num channels
        for cid, cb in parts:
            rec += p('>hI', cid, len(cb))

        bm = blend4.encode('ascii').ljust(4)[:4]
        rec += b'8BIM' + bm
        rec += p('>B', opac)    # opacity
        rec += p('>B', 0)       # clipping
        rec += p('>B', 0)       # flags  (visible)
        rec += p('>B', 0)       # filler

        extra  = p('>I', 0)            # layer mask (none)
        extra += p('>I', 0)            # blending ranges (none)
        extra += pstring(lname, 4)     # name

        rec += p('>I', len(extra)) + extra
        records += rec
        for _, cb in parts:
            chandata += cb

    # layer info block: signed layer count + records + channel data
    li  = p('>h', len(layer_list))
    li += records
    li += chandata
    if len(li) % 2:
        li += b'\x00'

    # global layer mask info: 4 bytes of length=0
    glmi = p('>I', 0)

    # layer & mask section body = layerInfoLength + layerInfo + glmi
    body  = p('>I', len(li)) + li
    body += glmi

    s4 = p('>I', len(body)) + body

    # ── 5. Image Data (merged composite, raw) ─────────────────────
    merged = Image.new('RGBA', (W, H), (0,0,0,255))
    for _, limg, _, _ in reversed(layer_list):
        limg2 = limg.convert('RGBA').resize((W,H), Image.LANCZOS)
        merged = Image.alpha_composite(merged, limg2)

    rgb = np.array(merged.convert('RGB'), dtype=np.uint8)
    s5  = p('>H', 0)           # compression: raw
    s5 += rgb[:,:,0].tobytes() # R plane
    s5 += rgb[:,:,1].tobytes() # G plane
    s5 += rgb[:,:,2].tobytes() # B plane

    return s1 + s2 + s3 + s4 + s5

# ── Routes ────────────────────────────────────────────────────────
@app.route('/'); 
@app.route('/health')
def health():
    return jsonify({"status":"ok","service":"LayerAI PSD","version":"8.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"No image uploaded"}), 400

        raw   = request.files['image'].read()
        orig  = Image.open(io.BytesIO(raw)).convert('RGBA')
        W, H  = orig.size

        MAX = 1000
        if W > MAX or H > MAX:
            r = min(MAX/W, MAX/H)
            W, H = int(W*r), int(H*r)
            orig = orig.resize((W,H), Image.LANCZOS)

        layers = []
        layers.append(('Background',          orig.copy(),                              'norm', 255))

        if REMOVE_BG_API_KEY:
            try:
                rsp = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file':('i.jpg', raw, 'image/jpeg')},
                    data={'size':'auto'},
                    headers={'X-Api-Key': REMOVE_BG_API_KEY},
                    timeout=20)
                if rsp.status_code == 200:
                    subj = Image.open(io.BytesIO(rsp.content)).convert('RGBA').resize((W,H),Image.LANCZOS)
                    layers.append(('Subject — Masked', subj, 'norm', 255))
            except Exception as e:
                print('removebg:', e)

        layers.append(('Curves 1',            Image.new('RGBA',(W,H),(255,200,150,30)), 'over', 60))
        layers.append(('Brightness/Contrast 1',Image.new('RGBA',(W,H),(200,200,200,20)),'norm', 40))
        layers.append(('Hue/Saturation 1',    Image.new('RGBA',(W,H),(100,150,200,18)), 'scrn', 35))
        layers.append(('Color Balance 1',     Image.new('RGBA',(W,H),(0,120,130,15)),   'norm', 30))
        layers.append(('Vignette',            make_vignette(W,H),                       'mul ', 180))

        psd  = build_psd(layers, W, H)
        buf  = io.BytesIO(psd); buf.seek(0)

        return send_file(buf, mimetype='application/octet-stream',
                         as_attachment=True, download_name='layerai-export.psd')

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))
