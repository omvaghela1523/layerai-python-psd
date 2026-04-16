from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io
import os
import struct
import zlib
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

# ─── PSD Writer ───────────────────────────────────────────────────
def write_pascal_string(s, padding=2):
    encoded = s.encode('macroman') if s else b''
    length = len(encoded)
    data = struct.pack('>B', length) + encoded
    pad = (padding - (len(data) % padding)) % padding
    return data + b'\x00' * pad

def write_uint32(n): return struct.pack('>I', n)
def write_uint16(n): return struct.pack('>H', n)
def write_int16(n):  return struct.pack('>h', n)
def write_uint8(n):  return struct.pack('>B', n)

def compress_channel(data):
    """PackBits RLE compression for PSD channels"""
    result = []
    bytecounts = []
    i = 0
    while i < len(data):
        run_start = i
        if i + 1 < len(data) and data[i] == data[i+1]:
            run_val = data[i]
            run_len = 1
            while i + run_len < len(data) and data[i + run_len] == run_val and run_len < 128:
                run_len += 1
            result.append(struct.pack('b', -(run_len - 1)))
            result.append(bytes([run_val]))
            i += run_len
        else:
            literal = [data[i]]
            i += 1
            while i < len(data) and len(literal) < 128:
                if i + 1 < len(data) and data[i] == data[i+1]:
                    break
                literal.append(data[i])
                i += 1
            result.append(struct.pack('b', len(literal) - 1))
            result.extend([bytes([b]) for b in literal])
        row_bytes = b''.join(result[run_start:])
    return b''.join(result)

def image_to_psd_channels(img, width, height):
    """Convert PIL image to PSD channel data"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    arr = np.array(img)
    channels = []
    # Alpha, R, G, B order for PSD
    channel_order = [3, 0, 1, 2]
    for c in channel_order:
        channel_data = arr[:, :, c].tobytes()
        channels.append(channel_data)
    return channels

def build_psd(layers_info, width, height):
    """Build a proper PSD file from scratch"""
    
    # ── Header ──────────────────────────────────────────────────
    header = b'8BPS'           # Signature
    header += write_uint16(1)  # Version
    header += b'\x00' * 6     # Reserved
    header += write_uint16(4)  # Channels (RGBA)
    header += write_uint32(height)
    header += write_uint32(width)
    header += write_uint16(8)  # Bits per channel
    header += write_uint16(3)  # Color mode: RGB

    # ── Color Mode Data ─────────────────────────────────────────
    color_mode = write_uint32(0)  # Empty for RGB

    # ── Image Resources ─────────────────────────────────────────
    image_resources = write_uint32(0)  # Empty

    # ── Layer and Mask Info ──────────────────────────────────────
    layer_records = b''
    channel_image_data = b''

    num_layers = len(layers_info)

    for layer in layers_info:
        img = layer['image']
        name = layer['name']
        blend_mode = layer.get('blend_mode', 'norm')
        opacity = layer.get('opacity', 255)
        layer_type = layer.get('type', 'pixel')

        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        arr = np.array(img)
        top = layer.get('top', 0)
        left = layer.get('left', 0)
        bottom = top + height
        right = left + width

        # Channel info: alpha + RGB = 4 channels
        num_channels = 4
        channel_ids = [-1, 0, 1, 2]  # Alpha, R, G, B

        # Layer record
        layer_record = b''
        layer_record += write_uint32(top)
        layer_record += write_uint32(left)
        layer_record += write_uint32(bottom)
        layer_record += write_uint32(right)
        layer_record += write_uint16(num_channels)

        # Channel data sizes (placeholder, fill later)
        channel_data_list = []
        for idx, cid in enumerate(channel_ids):
            if cid == -1:
                ch_arr = arr[:, :, 3]
            else:
                ch_arr = arr[:, :, cid]
            
            # Compress with PackBits per row
            rows = []
            row_lengths = []
            for row in ch_arr:
                compressed = compress_channel(row.tobytes())
                rows.append(compressed)
                row_lengths.append(len(compressed))
            
            # Channel data: compression type + row lengths + compressed rows
            ch_data = write_uint16(1)  # PackBits
            for rl in row_lengths:
                ch_data += write_uint16(rl)
            ch_data += b''.join(rows)
            channel_data_list.append((cid, ch_data))

        for cid, ch_data in channel_data_list:
            layer_record += write_int16(cid)
            layer_record += write_uint32(len(ch_data))

        # Blend mode
        layer_record += b'8BIM'
        blend_bytes = blend_mode.encode('ascii').ljust(4)[:4]
        layer_record += blend_bytes
        layer_record += write_uint8(opacity)
        layer_record += write_uint8(0)   # Clipping
        layer_record += write_uint8(4 if layer.get('visible', True) else 6)  # Flags
        layer_record += write_uint8(0)   # Filler

        # Extra data
        extra = b''

        # Layer mask (empty)
        extra += write_uint32(0)

        # Layer blending ranges (empty)
        extra += write_uint32(0)

        # Layer name (pascal string, padded to 4 bytes)
        name_enc = name.encode('ascii', errors='replace')[:255]
        name_len = len(name_enc)
        name_data = bytes([name_len]) + name_enc
        pad = (4 - (len(name_data) % 4)) % 4
        name_data += b'\x00' * pad
        extra += name_data

        layer_record += write_uint32(len(extra))
        layer_record += extra

        layer_records += layer_record

        # Channel image data
        for cid, ch_data in channel_data_list:
            channel_image_data += ch_data

    # Layer info section
    layer_info = write_uint16(num_layers)
    layer_info += layer_records
    layer_info += channel_image_data

    # Pad to 4 bytes
    if len(layer_info) % 4:
        layer_info += b'\x00' * (4 - len(layer_info) % 4)

    layer_and_mask = write_uint32(len(layer_info) + 4)
    layer_and_mask += write_uint32(len(layer_info))
    layer_and_mask += layer_info
    layer_and_mask += write_uint32(0)  # Global mask info

    # ── Image Data (merged composite) ────────────────────────────
    # Create merged image
    merged = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for layer in reversed(layers_info):
        if layer.get('visible', True) and layer['image']:
            merged = Image.alpha_composite(merged, layer['image'].resize((width, height)))

    merged_arr = np.array(merged.convert('RGB'))
    image_data = write_uint16(1)  # PackBits
    for c in range(3):  # RGB only for merged
        for row in merged_arr[:, :, c]:
            compressed = compress_channel(row.tobytes())
            image_data += write_uint16(len(compressed))
    for c in range(3):
        for row in merged_arr[:, :, c]:
            compressed = compress_channel(row.tobytes())
            image_data += compressed

    # Combine all sections
    psd = header
    psd += write_uint32(len(color_mode) - 4) + color_mode[4:]
    psd += write_uint32(len(image_resources) - 4) + image_resources[4:]
    psd += layer_and_mask
    psd += image_data

    # ── Proper PSD Assembly ──────────────────────────────────────
    final = b'8BPS'
    final += write_uint16(1)
    final += b'\x00' * 6
    final += write_uint16(4)
    final += write_uint32(height)
    final += write_uint32(width)
    final += write_uint16(8)
    final += write_uint16(3)
    final += write_uint32(0)  # Color mode data length
    final += write_uint32(0)  # Image resources length
    final += write_uint32(len(layer_and_mask))
    final += layer_and_mask
    final += image_data

    return final


def create_adjustment_layer(width, height, adj_type, values, opacity=200):
    """Create adjustment layer overlay image"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    if adj_type == 'curves':
        brightness = values.get('brightness', 0)
        r = max(0, min(255, 128 + brightness * 2))
        g = max(0, min(255, 128 + brightness))
        b = max(0, min(255, 128))
        img = Image.new('RGBA', (width, height), (r, g, b, opacity))

    elif adj_type == 'brightness_contrast':
        brightness = values.get('brightness', 0)
        contrast = values.get('contrast', 0)
        base = max(0, min(255, 128 + brightness))
        img = Image.new('RGBA', (width, height), (base, base, base, max(0, min(255, abs(contrast) * 2))))

    elif adj_type == 'hue_saturation':
        saturation = values.get('saturation', 0)
        hue = values.get('hue', 0)
        s_val = max(0, min(255, 128 + saturation))
        img = Image.new('RGBA', (width, height), (s_val, 128, 200, opacity // 2))

    elif adj_type == 'color_balance':
        cyan_red = values.get('cyan_red', 0)
        magenta_green = values.get('magenta_green', 0)
        yellow_blue = values.get('yellow_blue', 0)
        r = max(0, min(255, 128 + cyan_red * 2))
        g = max(0, min(255, 128 + magenta_green * 2))
        b = max(0, min(255, 128 + yellow_blue * 2))
        img = Image.new('RGBA', (width, height), (r, g, b, opacity // 3))

    elif adj_type == 'vignette':
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        steps = 60
        for i in range(steps):
            ratio = i / steps
            alpha = int(180 * (1 - ratio))
            x0 = int(width * ratio * 0.5)
            y0 = int(height * ratio * 0.5)
            x1 = width - x0
            y1 = height - y0
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, alpha))
        img = img.filter(ImageFilter.GaussianBlur(radius=width // 20))

    return img


# ─── Routes ───────────────────────────────────────────────────────
@app.route("/")
@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "LayerAI Python PSD", "version": "3.0.0"})


@app.route("/generate-psd", methods=["POST"])
def generate_psd():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        # Load original image
        original = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = original.size

        # Limit size for performance
        max_size = 2000
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            width = int(width * ratio)
            height = int(height * ratio)
            original = original.resize((width, height), Image.LANCZOS)

        layers_info = []

        # ── Layer 1: Background ──────────────────────────────────
        layers_info.append({
            'name': 'Background',
            'image': original.copy(),
            'blend_mode': 'norm',
            'opacity': 255,
            'visible': True,
            'type': 'pixel',
            'top': 0, 'left': 0
        })

        # ── Layer 2: Subject (Remove.bg) ─────────────────────────
        subject_extracted = False
        if REMOVE_BG_API_KEY:
            try:
                resp = requests.post(
                    "https://api.remove.bg/v1.0/removebg",
                    files={"image_file": ("image.jpg", image_bytes, "image/jpeg")},
                    data={"size": "auto"},
                    headers={"X-Api-Key": REMOVE_BG_API_KEY},
                    timeout=30
                )
                if resp.status_code == 200:
                    subject_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    subject_img = subject_img.resize((width, height), Image.LANCZOS)
                    layers_info.append({
                        'name': 'Subject — Masked',
                        'image': subject_img,
                        'blend_mode': 'norm',
                        'opacity': 255,
                        'visible': True,
                        'type': 'pixel',
                        'top': 0, 'left': 0
                    })
                    subject_extracted = True
            except Exception as e:
                print(f"Remove.bg error: {e}")

        # ── Layer 3: Curves Adjustment ───────────────────────────
        curves_img = create_adjustment_layer(width, height, 'curves',
            {'brightness': 15}, opacity=180)
        layers_info.append({
            'name': 'Curves 1',
            'image': curves_img,
            'blend_mode': 'over',  # Overlay
            'opacity': 180,
            'visible': True,
            'type': 'adjustment',
            'top': 0, 'left': 0
        })

        # ── Layer 4: Brightness/Contrast ─────────────────────────
        bc_img = create_adjustment_layer(width, height, 'brightness_contrast',
            {'brightness': 10, 'contrast': 20}, opacity=120)
        layers_info.append({
            'name': 'Brightness/Contrast 1',
            'image': bc_img,
            'blend_mode': 'norm',
            'opacity': 120,
            'visible': True,
            'type': 'adjustment',
            'top': 0, 'left': 0
        })

        # ── Layer 5: Hue/Saturation ──────────────────────────────
        hs_img = create_adjustment_layer(width, height, 'hue_saturation',
            {'saturation': 20, 'hue': 5}, opacity=100)
        layers_info.append({
            'name': 'Hue/Saturation 1',
            'image': hs_img,
            'blend_mode': 'norm',
            'opacity': 100,
            'visible': True,
            'type': 'adjustment',
            'top': 0, 'left': 0
        })

        # ── Layer 6: Color Balance ────────────────────────────────
        cb_img = create_adjustment_layer(width, height, 'color_balance',
            {'cyan_red': -10, 'magenta_green': 5, 'yellow_blue': 15}, opacity=80)
        layers_info.append({
            'name': 'Color Balance 1',
            'image': cb_img,
            'blend_mode': 'norm',
            'opacity': 80,
            'visible': True,
            'type': 'adjustment',
            'top': 0, 'left': 0
        })

        # ── Layer 7: Vignette ─────────────────────────────────────
        vig_img = create_adjustment_layer(width, height, 'vignette', {}, opacity=150)
        layers_info.append({
            'name': 'Vignette',
            'image': vig_img,
            'blend_mode': b'mul ',  # Multiply
            'opacity': 150,
            'visible': True,
            'type': 'pixel',
            'top': 0, 'left': 0
        })

        # ── Build PSD ─────────────────────────────────────────────
        psd_bytes = build_psd(layers_info, width, height)

        psd_buffer = io.BytesIO(psd_bytes)
        psd_buffer.seek(0)

        return send_file(
            psd_buffer,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='layerai-export.psd'
        )

    except Exception as e:
        print(f"PSD generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
