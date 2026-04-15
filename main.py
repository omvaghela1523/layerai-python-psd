from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io
import os
import struct
import zlib
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

# ============================================================
# PURE PYTHON PSD WRITER
# PSD format manually implement kiya hai
# ============================================================

def pack_be(fmt, *args):
    return struct.pack(">" + fmt, *args)

def psd_pascal_string(s, pad_to=2):
    b = s.encode("ascii")
    length = len(b)
    data = bytes([length]) + b
    while len(data) % pad_to != 0:
        data += b'\x00'
    return data

def compress_rle(data, width, height, channels):
    """PackBits RLE compression for PSD"""
    compressed_rows = []
    byte_counts = []
    
    for c in range(channels):
        for row in range(height):
            row_data = data[c, row, :]
            compressed = packbits_encode(row_data)
            compressed_rows.append(compressed)
            byte_counts.append(len(compressed))
    
    return byte_counts, compressed_rows

def packbits_encode(data):
    """PackBits compression"""
    result = bytearray()
    i = 0
    n = len(data)
    
    while i < n:
        # Find run of same bytes
        run_start = i
        run_byte = data[i]
        run_len = 1
        
        while i + run_len < n and data[i + run_len] == run_byte and run_len < 128:
            run_len += 1
        
        if run_len > 1:
            result.append(256 - run_len + 1)
            result.append(run_byte)
            i += run_len
        else:
            # Find literal run
            lit_start = i
            lit_len = 0
            while i + lit_len < n and lit_len < 128:
                # Check if next 2 bytes are same (start of run)
                if i + lit_len + 1 < n and data[i + lit_len] == data[i + lit_len + 1]:
                    break
                lit_len += 1
            if lit_len == 0:
                lit_len = 1
            result.append(lit_len - 1)
            result.extend(data[lit_start:lit_start + lit_len])
            i += lit_len
    
    return bytes(result)

def image_to_channels(img):
    """PIL image ko channel arrays mein convert karo"""
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    # PSD order: R, G, B, A
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    a = arr[:, :, 3]
    return np.stack([r, g, b, a], axis=0)  # shape: (4, H, W)

def write_layer_channel_data(img):
    """Layer ke channel data compress karke likho"""
    channels = image_to_channels(img)
    height, width = img.size[1], img.size[0]
    num_channels = 4
    
    byte_counts, compressed_rows = compress_rle(channels, width, height, num_channels)
    
    # Channel data buffer
    buf = io.BytesIO()
    
    # Compression type: 1 = RLE
    buf.write(pack_be("H", 1))
    
    # Row byte counts for all channels
    for bc in byte_counts:
        buf.write(pack_be("H", bc))
    
    # Compressed row data
    for row_data in compressed_rows:
        buf.write(row_data)
    
    return buf.getvalue()

def create_psd(layers_info, width, height):
    """
    layers_info = list of (name, PIL_image, blend_mode, opacity, visible)
    blend_mode: 'norm', 'mul ', 'scrn', 'over', 'lum '
    """
    buf = io.BytesIO()
    
    num_layers = len(layers_info)
    
    # ── SECTION 1: File Header ──────────────────────────────
    buf.write(b'8BPS')           # Signature
    buf.write(pack_be("H", 1))   # Version
    buf.write(b'\x00' * 6)       # Reserved
    buf.write(pack_be("H", 4))   # Channels (R,G,B,A)
    buf.write(pack_be("I", height))
    buf.write(pack_be("I", width))
    buf.write(pack_be("H", 8))   # Bits per channel
    buf.write(pack_be("H", 3))   # Color mode: RGB
    
    # ── SECTION 2: Color Mode Data ──────────────────────────
    buf.write(pack_be("I", 0))   # Empty
    
    # ── SECTION 3: Image Resources ──────────────────────────
    buf.write(pack_be("I", 0))   # Empty
    
    # ── SECTION 4: Layer and Mask Information ───────────────
    layer_mask_buf = io.BytesIO()
    
    # Layer Info
    layer_info_buf = io.BytesIO()
    
    # Layer count (negative = merged alpha)
    layer_info_buf.write(pack_be("h", -num_layers))
    
    # Layer records
    layer_channel_data_list = []
    
    for (name, img, blend_mode, opacity, visible) in layers_info:
        img = img.convert("RGBA").resize((width, height))
        
        top = 0
        left = 0
        bottom = height
        right = width
        
        layer_info_buf.write(pack_be("I", top))
        layer_info_buf.write(pack_be("I", left))
        layer_info_buf.write(pack_be("I", bottom))
        layer_info_buf.write(pack_be("I", right))
        
        # 4 channels: A, R, G, B
        layer_info_buf.write(pack_be("H", 4))
        channel_ids = [-1, 0, 1, 2]  # Alpha, R, G, B
        for cid in channel_ids:
            layer_info_buf.write(pack_be("h", cid))
            layer_info_buf.write(pack_be("I", 0))  # placeholder length
        
        # Blend mode
        layer_info_buf.write(b'8BIM')
        bm = blend_mode.encode('ascii')
        bm = bm + b' ' * (4 - len(bm))
        layer_info_buf.write(bm)
        layer_info_buf.write(bytes([opacity]))      # Opacity 0-255
        layer_info_buf.write(bytes([0]))            # Clipping
        flags = 0 if visible else 2
        layer_info_buf.write(bytes([flags]))        # Flags
        layer_info_buf.write(bytes([0]))            # Filler
        
        # Extra data
        extra_buf = io.BytesIO()
        
        # Layer mask (empty)
        extra_buf.write(pack_be("I", 0))
        
        # Blending ranges (empty)
        extra_buf.write(pack_be("I", 0))
        
        # Layer name (pascal string, padded to 4)
        name_bytes = name.encode('ascii')[:255]
        name_len = len(name_bytes)
        extra_buf.write(bytes([name_len]))
        extra_buf.write(name_bytes)
        # Pad to multiple of 4
        total = 1 + name_len
        while total % 4 != 0:
            extra_buf.write(b'\x00')
            total += 1
        
        extra_data = extra_buf.getvalue()
        layer_info_buf.write(pack_be("I", len(extra_data)))
        layer_info_buf.write(extra_data)
        
        # Channel data store karo
        layer_channel_data_list.append((img, channel_ids))
    
    # Channel image data
    channel_image_buf = io.BytesIO()
    for (img, channel_ids) in layer_channel_data_list:
        channels = image_to_channels(img)
        h, w = img.size[1], img.size[0]
        
        # Order: Alpha(id=-1), R(0), G(1), B(2)
        channel_order = [3, 0, 1, 2]  # RGBA array indices
        
        for ch_idx in channel_order:
            ch_data = channels[ch_idx]  # shape (H, W)
            
            # RLE compress
            byte_counts = []
            compressed_rows = []
            for row in range(h):
                compressed = packbits_encode(ch_data[row])
                compressed_rows.append(compressed)
                byte_counts.append(len(compressed))
            
            # Write compression type
            channel_image_buf.write(pack_be("H", 1))
            # Write byte counts
            for bc in byte_counts:
                channel_image_buf.write(pack_be("H", bc))
            # Write compressed data
            for rd in compressed_rows:
                channel_image_buf.write(rd)
    
    layer_info_data = layer_info_buf.getvalue()
    
    # Pad to even
    if len(layer_info_data) % 2 != 0:
        layer_info_data += b'\x00'
    
    layer_mask_buf.write(pack_be("I", len(layer_info_data) + 4))
    layer_mask_buf.write(pack_be("I", len(layer_info_data)))
    layer_mask_buf.write(layer_info_data)
    layer_mask_buf.write(channel_image_buf.getvalue())
    
    layer_mask_data = layer_mask_buf.getvalue()
    buf.write(pack_be("I", len(layer_mask_data)))
    buf.write(layer_mask_data)
    
    # ── SECTION 5: Merged Image Data ────────────────────────
    # Flatten all layers
    merged = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for (name, img, blend_mode, opacity, visible) in reversed(layers_info):
        if visible:
            layer_img = img.convert("RGBA").resize((width, height))
            merged = Image.alpha_composite(merged, layer_img)
    
    merged_rgb = merged.convert("RGB")
    merged_arr = np.array(merged_rgb, dtype=np.uint8)
    
    buf.write(pack_be("H", 1))  # RLE compression
    
    # Write byte counts for all rows (R, G, B channels)
    all_counts = []
    all_compressed = []
    for c in range(3):
        for row in range(height):
            rd = packbits_encode(merged_arr[row, :, c])
            all_compressed.append(rd)
            all_counts.append(len(rd))
    
    for bc in all_counts:
        buf.write(pack_be("H", bc))
    for rd in all_compressed:
        buf.write(rd)
    
    return buf.getvalue()


# ============================================================
# LAYER BUILDING HELPERS
# ============================================================

def make_color_grade_layer(width, height, image):
    """Color grade overlay layer"""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    
    # Warm highlights + cool shadows
    grade = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    grade_arr = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Subtle orange/warm tint in highlights
    brightness = (arr[:,:,0] * 0.299 + arr[:,:,1] * 0.587 + arr[:,:,2] * 0.114) / 255.0
    
    grade_arr[:,:,0] = (brightness * 30).astype(np.uint8)   # R
    grade_arr[:,:,1] = (brightness * 10).astype(np.uint8)   # G  
    grade_arr[:,:,2] = ((1-brightness) * 20).astype(np.uint8) # B (shadows cool)
    grade_arr[:,:,3] = 80  # opacity
    
    return Image.fromarray(grade_arr, "RGBA")

def make_vignette_layer(width, height):
    """Dark vignette around edges"""
    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vignette)
    steps = min(width, height) // 2
    for i in range(steps):
        opacity = int(180 * (1 - i / steps) ** 2)
        draw.rectangle([i, i, width-1-i, height-1-i], outline=(0, 0, 0, opacity))
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=40))
    return vignette

def make_brightness_contrast_layer(width, height, brightness=1.0, contrast=1.0):
    """Subtle brightness/contrast adjustment as overlay"""
    # Create a neutral gray layer with slight adjustment
    val = 128
    if brightness > 1.0:
        val = min(255, int(128 + (brightness - 1.0) * 128))
    elif brightness < 1.0:
        val = max(0, int(128 * brightness))
    
    layer = Image.new("RGBA", (width, height), (val, val, val, 30))
    return layer

def make_highlight_layer(width, height, image):
    """Soft light / highlight enhancement layer"""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    brightness = (arr[:,:,0] * 0.299 + arr[:,:,1] * 0.587 + arr[:,:,2] * 0.114) / 255.0
    
    highlight_arr = np.zeros((height, width, 4), dtype=np.uint8)
    # White highlights where bright
    highlight_arr[:,:,0] = (brightness * 255).astype(np.uint8)
    highlight_arr[:,:,1] = (brightness * 255).astype(np.uint8)
    highlight_arr[:,:,2] = (brightness * 255).astype(np.uint8)
    highlight_arr[:,:,3] = (brightness * 40).astype(np.uint8)
    
    return Image.fromarray(highlight_arr, "RGBA")


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "LayerAI Python PSD", "version": "2.0.0"})

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/generate-psd", methods=["POST"])
def generate_psd():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        # Original image load karo
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = original_image.size
        
        print(f"Image size: {width}x{height}")

        # ── Subject extraction via Remove.bg ────────────────
        subject_image = None
        background_image = None
        
        if REMOVE_BG_API_KEY:
            try:
                response = requests.post(
                    "https://api.remove.bg/v1.0/removebg",
                    files={"image_file": ("image.jpg", image_bytes, "image/jpeg")},
                    data={"size": "auto"},
                    headers={"X-Api-Key": REMOVE_BG_API_KEY},
                    timeout=30
                )
                if response.status_code == 200:
                    subject_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
                    subject_image = subject_image.resize((width, height))
                    print("Subject extracted via remove.bg ✓")
                    
                    # Background = original with subject area dimmed
                    background_image = original_image.copy()
                else:
                    print(f"Remove.bg failed: {response.status_code}")
            except Exception as e:
                print(f"Remove.bg error: {e}")

        # ── Build all layers ────────────────────────────────
        color_grade = make_color_grade_layer(width, height, original_image)
        vignette = make_vignette_layer(width, height)
        brightness_layer = make_brightness_contrast_layer(width, height, brightness=1.05)
        highlight_layer = make_highlight_layer(width, height, original_image)

        # ── Assemble PSD layers (bottom → top) ──────────────
        # (name, image, blend_mode, opacity_0_255, visible)
        layers = []

        # 1. Background (bottom)
        layers.append(("Background", original_image, "norm", 255, True))

        # 2. Subject masked layer (if available)
        if subject_image:
            layers.append(("Subject_Masked", subject_image, "norm", 255, True))

        # 3. Color Grade
        layers.append(("Color_Grade", color_grade, "norm", 180, True))

        # 4. Highlight layer
        layers.append(("Highlights", highlight_layer, "norm", 200, True))

        # 5. Brightness/Contrast
        layers.append(("Brightness_Contrast", brightness_layer, "norm", 100, True))

        # 6. Vignette (top)
        layers.append(("Vignette", vignette, "norm", 220, True))

        # ── Generate PSD ─────────────────────────────────────
        print("Generating PSD...")
        psd_data = create_psd(layers, width, height)
        print(f"PSD size: {len(psd_data)} bytes")

        psd_buffer = io.BytesIO(psd_data)
        psd_buffer.seek(0)

        return send_file(
            psd_buffer,
            mimetype="image/vnd.adobe.photoshop",
            as_attachment=True,
            download_name="layerai-edited.psd"
        )

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
