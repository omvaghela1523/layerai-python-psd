from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io
import os
import struct
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

# ── Struct helpers ────────────────────────────────────────────────
def u8(n):   return struct.pack('>B', n & 0xFF)
def u16(n):  return struct.pack('>H', n & 0xFFFF)
def u32(n):  return struct.pack('>I', n & 0xFFFFFFFF)
def i16(n):  return struct.pack('>h', n)
def i32(n):  return struct.pack('>i', n)

def pascal_string(s, pad=4):
    enc = s.encode('ascii', errors='replace')[:255]
    data = bytes([len(enc)]) + enc
    rem = len(data) % pad
    if rem:
        data += b'\x00' * (pad - rem)
    return data

# ── Vignette maker ────────────────────────────────────────────────
def make_vignette(width, height):
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    steps = 30
    for i in range(steps):
        ratio = i / steps
        alpha = int(160 * (1 - ratio))
        x0 = int(width * ratio * 0.4)
        y0 = int(height * ratio * 0.4)
        draw.rectangle([x0, y0, width - x0, height - y0],
                       outline=(0, 0, 0, alpha))
    return img.filter(ImageFilter.GaussianBlur(radius=15))

# ── Core PSD builder ──────────────────────────────────────────────
def build_psd(layers, width, height):
    """
    Construct a minimal but valid Photoshop PSD (version 1).
    Uses compression mode 0 (raw) for speed and simplicity.
    """

    # ── Section 1: File Header ────────────────────────────────────
    sec1  = b'8BPS'        # signature
    sec1 += u16(1)         # version
    sec1 += b'\x00' * 6   # reserved
    sec1 += u16(4)         # channels per pixel (RGBA → 4)
    sec1 += u32(height)
    sec1 += u32(width)
    sec1 += u16(8)         # bits per channel
    sec1 += u16(3)         # color mode: RGB

    # ── Section 2: Color Mode Data ────────────────────────────────
    sec2 = u32(0)          # empty for RGB

    # ── Section 3: Image Resources ────────────────────────────────
    sec3 = u32(0)          # empty

    # ── Section 4: Layer and Mask Information ─────────────────────
    layer_records = b''
    channel_image_data = b''

    for layer in layers:
        img = layer['image'].convert('RGBA')
        img = img.resize((width, height), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)

        name      = layer.get('name', 'Layer')
        blend_str = layer.get('blend_mode', 'norm')
        opacity   = layer.get('opacity', 255)

        # Channel IDs: -1=alpha, 0=R, 1=G, 2=B
        ch_info = [(-1, 3), (0, 0), (1, 1), (2, 2)]
        ch_sizes = []
        ch_data_parts = []

        for ch_id, ch_idx in ch_info:
            raw   = arr[:, :, ch_idx].tobytes()
            cdata = u16(0) + raw          # compression=0 (raw) + pixels
            ch_sizes.append((ch_id, len(cdata)))
            ch_data_parts.append(cdata)

        # Layer record
        rec  = u32(0)           # top
        rec += u32(0)           # left
        rec += u32(height)      # bottom
        rec += u32(width)       # right
        rec += u16(4)           # number of channels

        for ch_id, ch_size in ch_sizes:
            rec += i16(ch_id)
            rec += u32(ch_size)

        # Blend mode key (must be exactly 4 bytes)
        bm = blend_str.encode('ascii').ljust(4)[:4]
        rec += b'8BIM'
        rec += bm
        rec += u8(opacity)
        rec += u8(0)            # clipping: base
        rec += u8(0)            # flags
        rec += u8(0)            # filler

        # Extra data
        extra  = u32(0)                       # layer mask size = 0
        extra += u32(0)                       # blending ranges size = 0
        extra += pascal_string(name, 4)       # layer name

        rec += u32(len(extra))
        rec += extra

        layer_records      += rec
        channel_image_data += b''.join(ch_data_parts)

    # Layer info = layer count + records + channel data
    layer_count  = i16(len(layers))          # negative = has alpha merged
    layer_info   = layer_count + layer_records + channel_image_data

    # Pad to even length
    if len(layer_info) % 2:
        layer_info += b'\x00'

    # Global layer mask info (empty)
    global_mask = u32(0)

    # Layer & mask info section
    lmi_body = u32(len(layer_info)) + layer_info + global_mask
    sec4 = u32(len(lmi_body)) + lmi_body

    # ── Section 5: Image Data (merged composite) ──────────────────
    # Flatten all layers into a single RGB image
    merged = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for layer in reversed(layers):
        limg = layer['image'].convert('RGBA').resize((width, height), Image.LANCZOS)
        merged = Image.alpha_composite(merged, limg)

    merged_rgb = np.array(merged.convert('RGB'), dtype=np.uint8)

    # compression = 0 (raw), then R plane, G plane, B plane
    sec5 = u16(0)
    for c in range(3):
        sec5 += merged_rgb[:, :, c].tobytes()

    return sec1 + sec2 + sec3 + sec4 + sec5

# ── Flask routes ──────────────────────────────────────────────────
@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "LayerAI PSD", "version": "6.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_bytes   = request.files['image'].read()
        original      = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        width, height = original.size

        # Resize to safe dimensions
        MAX = 1000
        if width > MAX or height > MAX:
            ratio  = min(MAX / width, MAX / height)
            width  = int(width  * ratio)
            height = int(height * ratio)
            original = original.resize((width, height), Image.LANCZOS)

        layers = []

        # 1. Background
        layers.append({
            'name':       'Background',
            'image':      original.copy(),
            'blend_mode': 'norm',
            'opacity':    255,
        })

        # 2. Subject (Remove.bg)
        if REMOVE_BG_API_KEY:
            try:
                resp = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': ('img.jpg', image_bytes, 'image/jpeg')},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVE_BG_API_KEY},
                    timeout=20,
                )
                if resp.status_code == 200:
                    subj = Image.open(io.BytesIO(resp.content)).convert('RGBA')
                    layers.append({
                        'name':       'Subject — Masked',
                        'image':      subj,
                        'blend_mode': 'norm',
                        'opacity':    255,
                    })
            except Exception as e:
                print(f"Remove.bg error: {e}")

        # 3. Curves (warm overlay)
        layers.append({
            'name':       'Curves 1',
            'image':      Image.new('RGBA', (width, height), (255, 200, 150, 30)),
            'blend_mode': 'over',
            'opacity':    60,
        })

        # 4. Brightness / Contrast
        layers.append({
            'name':       'Brightness/Contrast 1',
            'image':      Image.new('RGBA', (width, height), (200, 200, 200, 20)),
            'blend_mode': 'norm',
            'opacity':    40,
        })

        # 5. Hue / Saturation
        layers.append({
            'name':       'Hue/Saturation 1',
            'image':      Image.new('RGBA', (width, height), (100, 150, 200, 18)),
            'blend_mode': 'scrn',
            'opacity':    35,
        })

        # 6. Color Balance
        layers.append({
            'name':       'Color Balance 1',
            'image':      Image.new('RGBA', (width, height), (0, 120, 130, 15)),
            'blend_mode': 'norm',
            'opacity':    30,
        })

        # 7. Vignette
        layers.append({
            'name':       'Vignette',
            'image':      make_vignette(width, height),
            'blend_mode': 'mul ',
            'opacity':    180,
        })

        # Build PSD binary
        psd_bytes = build_psd(layers, width, height)

        buf = io.BytesIO(psd_bytes)
        buf.seek(0)

        return send_file(
            buf,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='layerai-export.psd',
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
