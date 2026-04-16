from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io
import os
import struct
import zipfile
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

def u32(n): return struct.pack('>I', n)
def u16(n): return struct.pack('>H', n)
def u8(n):  return struct.pack('>B', n)
def i16(n): return struct.pack('>h', n)
def i32(n): return struct.pack('>i', n)

def pascal_string(s, pad=4):
    enc = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(enc)]) + enc
    rem = len(raw) % pad
    if rem: raw += b'\x00' * (pad - rem)
    return raw

def make_vignette(width, height):
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    steps = 40
    for i in range(steps):
        ratio = i / steps
        alpha = int(180 * (1 - ratio))
        x0 = int(width * ratio * 0.4)
        y0 = int(height * ratio * 0.4)
        draw.rectangle([x0, y0, width - x0, height - y0], outline=(0, 0, 0, alpha))
    return img.filter(ImageFilter.GaussianBlur(radius=max(width, height) // 20))

def create_psd(layers, width, height):
    """Build PSD using raw (uncompressed) channel data — fast and memory efficient"""

    # Header
    hdr  = b'8BPS' + u16(1) + b'\x00'*6
    hdr += u16(4) + u32(height) + u32(width) + u16(8) + u16(3)

    # Color mode + resources
    color_mode = u32(0)
    resources  = u32(0)

    # Build layer records + channel data
    all_records = b''
    all_chandata = b''

    for layer in layers:
        img = layer['image'].convert('RGBA').resize((width, height), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        name      = layer['name']
        blend     = layer.get('blend_mode', 'norm').encode('ascii').ljust(4)[:4]
        opacity   = layer.get('opacity', 255)

        # Channel data: -1(alpha), 0(R), 1(G), 2(B)
        chan_ids  = [-1, 0, 1, 2]
        chan_idxs = [3, 0, 1, 2]
        chan_bytes_list = []

        for ch_id, ch_idx in zip(chan_ids, chan_idxs):
            raw = arr[:, :, ch_idx].tobytes()
            # Raw uncompressed: compression=0, then raw bytes
            ch_data = u16(0) + raw
            chan_bytes_list.append((ch_id, ch_data))

        # Layer record
        rec  = u32(0) + u32(0) + u32(height) + u32(width)  # bbox
        rec += u16(4)  # num channels
        for ch_id, ch_data in chan_bytes_list:
            rec += i16(ch_id) + u32(len(ch_data))
        rec += b'8BIM' + blend
        rec += u8(opacity) + u8(0) + u8(0) + u8(0)

        extra = u32(0) + u32(0) + pascal_string(name, 4)
        rec  += u32(len(extra)) + extra

        all_records  += rec
        for _, ch_data in chan_bytes_list:
            all_chandata += ch_data

    # Layer info
    layer_info  = i16(len(layers)) + all_records + all_chandata
    if len(layer_info) % 2:
        layer_info += b'\x00'

    lmi = u32(len(layer_info)) + layer_info + u32(0)
    layer_mask_info = u32(len(lmi)) + lmi

    # Merged composite (raw)
    merged = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for layer in reversed(layers):
        limg = layer['image'].convert('RGBA').resize((width, height), Image.LANCZOS)
        merged = Image.alpha_composite(merged, limg)

    merged_arr = np.array(merged.convert('RGB'), dtype=np.uint8)
    image_data = u16(0)  # Raw uncompressed
    for c in range(3):
        image_data += merged_arr[:, :, c].tobytes()

    return hdr + color_mode + resources + layer_mask_info + image_data

@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "LayerAI Python PSD", "version": "5.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_bytes = request.files['image'].read()
        original    = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        width, height = original.size

        # Limit size to save memory
        MAX = 1200
        if width > MAX or height > MAX:
            ratio  = min(MAX / width, MAX / height)
            width  = int(width * ratio)
            height = int(height * ratio)
            original = original.resize((width, height), Image.LANCZOS)

        layers = []

        # Background
        layers.append({
            'name': 'Background',
            'image': original.copy(),
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # Subject — Remove.bg
        if REMOVE_BG_API_KEY:
            try:
                resp = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': ('img.jpg', image_bytes, 'image/jpeg')},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVE_BG_API_KEY},
                    timeout=25
                )
                if resp.status_code == 200:
                    subj = Image.open(io.BytesIO(resp.content)).convert('RGBA')
                    layers.append({
                        'name': 'Subject — Masked',
                        'image': subj,
                        'blend_mode': 'norm',
                        'opacity': 255,
                    })
            except Exception as e:
                print(f"Remove.bg error: {e}")

        # Curves — warm tones
        layers.append({
            'name': 'Curves 1',
            'image': Image.new('RGBA', (width, height), (255, 210, 160, 35)),
            'blend_mode': 'over',
            'opacity': 70,
        })

        # Brightness/Contrast
        layers.append({
            'name': 'Brightness/Contrast 1',
            'image': Image.new('RGBA', (width, height), (200, 200, 200, 25)),
            'blend_mode': 'norm',
            'opacity': 50,
        })

        # Hue/Saturation
        layers.append({
            'name': 'Hue/Saturation 1',
            'image': Image.new('RGBA', (width, height), (100, 160, 210, 20)),
            'blend_mode': 'scrn',
            'opacity': 40,
        })

        # Color Balance
        layers.append({
            'name': 'Color Balance 1',
            'image': Image.new('RGBA', (width, height), (0, 120, 130, 18)),
            'blend_mode': 'norm',
            'opacity': 35,
        })

        # Vignette
        layers.append({
            'name': 'Vignette',
            'image': make_vignette(width, height),
            'blend_mode': 'mul ',
            'opacity': 200,
        })

        # Generate PSD
        psd_bytes = create_psd(layers, width, height)
        buf = io.BytesIO(psd_bytes)
        buf.seek(0)

        return send_file(
            buf,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='layerai-export.psd'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
