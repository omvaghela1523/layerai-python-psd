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

# ─── PackBits RLE Compression ─────────────────────────────────────
def packbits_encode(data):
    result = bytearray()
    i = 0
    while i < len(data):
        if i + 1 < len(data) and data[i] == data[i + 1]:
            run_val = data[i]
            run_len = 1
            while i + run_len < len(data) and data[i + run_len] == run_val and run_len < 128:
                run_len += 1
            result.append((-(run_len - 1)) & 0xFF)
            result.append(run_val)
            i += run_len
        else:
            start = i
            literals = [data[i]]
            i += 1
            while i < len(data) and len(literals) < 128:
                if i + 1 < len(data) and data[i] == data[i + 1]:
                    break
                literals.append(data[i])
                i += 1
            result.append(len(literals) - 1)
            result.extend(literals)
    return bytes(result)

# ─── PSD Builder ──────────────────────────────────────────────────
def u32(n): return struct.pack('>I', n)
def u16(n): return struct.pack('>H', n)
def u8(n):  return struct.pack('>B', n)
def i16(n): return struct.pack('>h', n)

def pascal_string(s, pad=4):
    enc = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(enc)]) + enc
    remainder = len(raw) % pad
    if remainder:
        raw += b'\x00' * (pad - remainder)
    return raw

def encode_channel(arr2d):
    """Encode one channel (2D numpy array) with PackBits, return (bytecounts, compressed_rows)"""
    rows = []
    counts = []
    for row in arr2d:
        compressed = packbits_encode(row.tobytes())
        rows.append(compressed)
        counts.append(len(compressed))
    return counts, rows

def build_layer_record(name, top, left, bottom, right, blend_mode, opacity, channels_data):
    """Build a single layer record for PSD"""
    rec = b''
    rec += u32(top) + u32(left) + u32(bottom) + u32(right)

    num_ch = len(channels_data)
    rec += u16(num_ch)

    # Channel lengths placeholder
    for ch_id, ch_bytes in channels_data:
        rec += i16(ch_id)
        rec += u32(len(ch_bytes))

    # Blend mode signature + mode + opacity + clipping + flags + filler
    bm = blend_mode.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm
    rec += u8(opacity) + u8(0) + u8(0) + u8(0)

    # Extra data
    extra = b''
    extra += u32(0)  # Layer mask
    extra += u32(0)  # Blending ranges
    extra += pascal_string(name, 4)

    rec += u32(len(extra)) + extra
    return rec

def image_to_channels(img, width, height):
    """Convert RGBA PIL image to PSD channel byte strings"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img = img.resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    results = {}
    # -1 = alpha, 0 = R, 1 = G, 2 = B
    for ch_id, ch_idx in [(-1, 3), (0, 0), (1, 1), (2, 2)]:
        counts, rows = encode_channel(arr[:, :, ch_idx])
        ch_bytes = u16(1)  # PackBits
        for c in counts:
            ch_bytes += u16(c)
        ch_bytes += b''.join(rows)
        results[ch_id] = ch_bytes

    return [(-1, results[-1]), (0, results[0]), (1, results[1]), (2, results[2])]

def create_psd(layers, width, height):
    """Create a valid PSD file"""

    # ── PSD Header ────────────────────────────────────────────────
    header  = b'8BPS'
    header += u16(1)          # Version 1
    header += b'\x00' * 6    # Reserved
    header += u16(3)          # 3 channels (RGB, no alpha on merged)
    header += u32(height)
    header += u32(width)
    header += u16(8)          # 8 bits per channel
    header += u16(3)          # RGB color mode

    # ── Color Mode Data ───────────────────────────────────────────
    color_mode_data = u32(0)

    # ── Image Resources ───────────────────────────────────────────
    img_resources = u32(0)

    # ── Layer & Mask Info ─────────────────────────────────────────
    all_records = b''
    all_channel_data = b''

    for layer in layers:
        img       = layer['image']
        name      = layer['name']
        blend     = layer.get('blend_mode', 'norm')
        opacity   = layer.get('opacity', 255)

        channels_data = image_to_channels(img, width, height)
        rec = build_layer_record(
            name, 0, 0, height, width,
            blend, opacity, channels_data
        )
        all_records += rec
        for _, ch_bytes in channels_data:
            all_channel_data += ch_bytes

    layer_count_bytes = i16(len(layers))
    layer_info = layer_count_bytes + all_records + all_channel_data

    # Pad to multiple of 2
    if len(layer_info) % 2:
        layer_info += b'\x00'

    layer_info_block = u32(len(layer_info)) + layer_info
    global_mask = u32(0)

    lmi_content = layer_info_block + global_mask
    layer_mask_info = u32(len(lmi_content)) + lmi_content

    # ── Merged Image Data ─────────────────────────────────────────
    merged = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for layer in reversed(layers):
        limg = layer['image'].convert('RGBA').resize((width, height), Image.LANCZOS)
        merged = Image.alpha_composite(merged, limg)

    merged_rgb = merged.convert('RGB')
    merged_arr = np.array(merged_rgb, dtype=np.uint8)

    image_data = u16(1)  # PackBits
    all_row_counts = []
    all_compressed = []

    for c in range(3):
        counts, rows = encode_channel(merged_arr[:, :, c])
        all_row_counts.extend(counts)
        all_compressed.extend(rows)

    for cnt in all_row_counts:
        image_data += u16(cnt)
    for row in all_compressed:
        image_data += row

    # ── Assemble PSD ──────────────────────────────────────────────
    psd  = header
    psd += color_mode_data
    psd += img_resources
    psd += layer_mask_info
    psd += image_data

    return psd

# ─── Adjustment Layer Helpers ─────────────────────────────────────
def make_vignette(width, height):
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    steps = 80
    for i in range(steps):
        ratio = i / steps
        alpha = int(200 * (1 - ratio))
        x0 = int(width * ratio * 0.45)
        y0 = int(height * ratio * 0.45)
        x1 = width - x0
        y1 = height - y0
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, alpha))
    return img.filter(ImageFilter.GaussianBlur(radius=max(width, height) // 15))

def make_color_grade(width, height, r, g, b, alpha):
    return Image.new('RGBA', (width, height), (r, g, b, alpha))

# ─── Routes ───────────────────────────────────────────────────────
@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "LayerAI Python PSD", "version": "4.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_bytes = request.files['image'].read()
        original    = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        width, height = original.size

        # Limit size
        MAX = 1800
        if width > MAX or height > MAX:
            ratio  = min(MAX / width, MAX / height)
            width  = int(width * ratio)
            height = int(height * ratio)
            original = original.resize((width, height), Image.LANCZOS)

        layers = []

        # ── Background ────────────────────────────────────────────
        layers.append({
            'name': 'Background',
            'image': original.copy(),
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # ── Subject (Remove.bg) ───────────────────────────────────
        subject_extracted = False
        if REMOVE_BG_API_KEY:
            try:
                resp = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': ('img.jpg', image_bytes, 'image/jpeg')},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVE_BG_API_KEY},
                    timeout=30
                )
                if resp.status_code == 200:
                    subj = Image.open(io.BytesIO(resp.content)).convert('RGBA')
                    layers.append({
                        'name': 'Subject — Masked',
                        'image': subj,
                        'blend_mode': 'norm',
                        'opacity': 255,
                    })
                    subject_extracted = True
            except Exception as e:
                print(f"Remove.bg error: {e}")

        # ── Curves (warm tone) ────────────────────────────────────
        layers.append({
            'name': 'Curves 1',
            'image': make_color_grade(width, height, 255, 200, 150, 40),
            'blend_mode': 'over',
            'opacity': 80,
        })

        # ── Brightness/Contrast ───────────────────────────────────
        layers.append({
            'name': 'Brightness/Contrast 1',
            'image': make_color_grade(width, height, 180, 180, 180, 30),
            'blend_mode': 'norm',
            'opacity': 60,
        })

        # ── Hue/Saturation ────────────────────────────────────────
        layers.append({
            'name': 'Hue/Saturation 1',
            'image': make_color_grade(width, height, 100, 150, 200, 25),
            'blend_mode': 'scrn',
            'opacity': 50,
        })

        # ── Color Balance ─────────────────────────────────────────
        layers.append({
            'name': 'Color Balance 1',
            'image': make_color_grade(width, height, 0, 128, 128, 20),
            'blend_mode': 'norm',
            'opacity': 40,
        })

        # ── Vignette ──────────────────────────────────────────────
        layers.append({
            'name': 'Vignette',
            'image': make_vignette(width, height),
            'blend_mode': 'mul ',
            'opacity': 200,
        })

        # ── Generate PSD ──────────────────────────────────────────
        psd_bytes  = create_psd(layers, width, height)
        buf        = io.BytesIO(psd_bytes)
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
