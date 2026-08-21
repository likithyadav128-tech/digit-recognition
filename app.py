import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Digit — AI Handwritten Digit Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --violet-400: #a78bfa;
        --violet-500: #8b5cf6;
        --indigo-400: #818cf8;
        --indigo-500: #6366f1;
        --cyan-400: #22d3ee;
        --green-400: #4ade80;
    }

    .stApp {
        background: #06060e;
        background-image:
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(99,102,241,0.1) 0%, transparent 70%),
            radial-gradient(ellipse 60% 50% at 80% 80%, rgba(139,92,246,0.06) 0%, transparent 60%);
    }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    h1, h2, h3, p, span, div, label { font-family: 'Outfit', sans-serif !important; }

    .hero-title {
        text-align: center; font-size: 2.6rem; font-weight: 900;
        letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 4px; color: #f0eef6;
    }
    .hero-gradient {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-desc { text-align: center; font-size: 1rem; color: #8b86a8; font-weight: 400; margin-bottom: 2rem; }

    .status-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 16px; border-radius: 999px;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        font-size: 0.78rem; color: #8b86a8; margin: 0 auto 1.5rem;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #4ade80; box-shadow: 0 0 8px rgba(74,222,128,0.5); display: inline-block;
    }

    .glass-card {
        background: rgba(12, 12, 30, 0.7); backdrop-filter: blur(24px);
        border: 1px solid rgba(139, 92, 246, 0.1); border-radius: 20px;
        padding: 28px; box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 1px rgba(139,92,246,0.05);
    }

    .section-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #5c577a; margin-bottom: 14px;
    }

    .prediction-box { text-align: center; padding: 20px 0; }
    .pred-digit {
        font-size: 6rem; font-weight: 900; line-height: 1;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;
    }
    .pred-confidence { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #4ade80; }
    .pred-label { font-size: 0.7rem; color: #5c577a; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

    .prob-container { margin-top: 16px; }
    .prob-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
    .prob-digit-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
        font-weight: 600; color: #5c577a; width: 16px; text-align: center; flex-shrink: 0;
    }
    .prob-track { flex: 1; height: 16px; background: rgba(255,255,255,0.03); border-radius: 4px; overflow: hidden; }
    .prob-fill {
        height: 100%; border-radius: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6); opacity: 0.4;
    }
    .prob-fill.top-pred {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        opacity: 1; box-shadow: 0 0 10px rgba(139,92,246,0.25);
    }
    .prob-pct {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        font-weight: 500; color: #5c577a; width: 42px; text-align: right; flex-shrink: 0;
    }

    canvas { border-radius: 14px !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: rgba(255,255,255,0.02); border-radius: 10px;
        padding: 4px; border: 1px solid rgba(255,255,255,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; font-family: 'Outfit', sans-serif;
        font-weight: 600; font-size: 0.85rem; color: #8b86a8; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(20, 20, 50, 0.55); color: #f0eef6; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-border"], .stTabs [data-baseweb="tab-highlight"] { display: none; }

    .stButton > button {
        font-family: 'Outfit', sans-serif; font-weight: 700;
        border-radius: 10px; padding: 10px 24px; border: none;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        color: #fff; box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); color: #8b86a8;
    }

    .stFileUploader {
        border: 2px dashed rgba(139,92,246,0.2) !important;
        border-radius: 14px !important; background: rgba(255,255,255,0.02) !important;
    }

    .history-chip {
        display: inline-flex; flex-direction: column; align-items: center; justify-content: center;
        width: 42px; height: 52px; border-radius: 10px;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        margin-right: 6px; margin-bottom: 6px;
    }
    .chip-digit {
        font-size: 1.1rem; font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .chip-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.5rem; color: #5c577a; }

    .app-footer { text-align: center; font-size: 0.7rem; color: #5c577a; padding: 24px 0 8px; }
    .empty-state { text-align: center; padding: 60px 20px; color: #5c577a; font-size: 0.9rem; }

    .preview-box {
        display: flex; align-items: center; gap: 10px; justify-content: center;
        padding: 10px; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.04);
    }
    .preview-label { font-size: 0.68rem; color: #5c577a; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model: Robust CNN with Data Augmentation
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_cnn_v2.keras")
ACC_PATH = os.path.join(MODEL_DIR, "accuracy_v2.txt")


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load model from disk if available, otherwise train with augmentation and save."""

    # Try loading saved model first (instant, ~2 seconds)
    if os.path.exists(MODEL_PATH) and os.path.exists(ACC_PATH):
        model = keras.models.load_model(MODEL_PATH)
        with open(ACC_PATH, "r") as f:
            acc = float(f.read().strip())
        return model, acc

    # First run: train with data augmentation
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Data augmentation: makes the model robust to real-world image variations
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.08),        # ±15° rotation
        keras.layers.RandomTranslation(0.08, 0.08),  # ±8% shift
        keras.layers.RandomZoom(0.08),            # ±8% zoom
    ])

    # Stronger CNN architecture
    inputs = keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)

    # Block 1
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 2
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 3
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense head
    x = keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Use learning rate schedule for better convergence
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train for more epochs with augmentation
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    _, acc = model.evaluate(x_test, y_test, verbose=0)

    # Save to disk so next startup is instant
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(ACC_PATH, "w") as f:
        f.write(str(acc))

    return model, acc


def _center_by_mass_and_fit(gray_arr: np.ndarray) -> np.ndarray:
    """
    Core MNIST-style preprocessing:
    1. Crop to bounding box of content
    2. Scale to fit inside 20x20 preserving aspect ratio
    3. Center by CENTER OF MASS in 28x28 (this is how MNIST was created)
    4. Light Gaussian blur for anti-aliased edges
    Returns a (1, 28, 28, 1) float32 tensor, or None.
    """
    arr = gray_arr.astype(np.float32)

    # Find bounding box of non-zero content
    thresh = arr.max() * 0.15  # adaptive threshold: 15% of peak brightness
    mask = arr > thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]
    ch, cw = cropped.shape
    if ch < 2 or cw < 2:
        return None

    # Scale to fit in 20x20 box, preserving aspect ratio
    scale = min(20.0 / ch, 20.0 / cw)
    new_h = max(1, int(round(ch * scale)))
    new_w = max(1, int(round(cw * scale)))

    cropped_pil = Image.fromarray(cropped.astype(np.uint8))
    resized = cropped_pil.resize((new_w, new_h), Image.LANCZOS)
    resized_arr = np.array(resized).astype(np.float32)

    # Place in 28x28 canvas, centered by CENTER OF MASS
    # (This is the key MNIST preprocessing step that most implementations miss)
    canvas = np.zeros((28, 28), dtype=np.float32)

    # Compute center of mass of the resized digit
    total_mass = resized_arr.sum()
    if total_mass == 0:
        return None

    ys, xs = np.mgrid[0:new_h, 0:new_w]
    com_y = (ys * resized_arr).sum() / total_mass
    com_x = (xs * resized_arr).sum() / total_mass

    # Shift so center of mass is at (14, 14) — the center of 28x28
    shift_x = int(round(14.0 - com_x))
    shift_y = int(round(14.0 - com_y))

    # Paste with the computed shift
    y_start = max(0, shift_y)
    x_start = max(0, shift_x)
    y_end = min(28, shift_y + new_h)
    x_end = min(28, shift_x + new_w)

    src_y_start = max(0, -shift_y)
    src_x_start = max(0, -shift_x)
    src_y_end = src_y_start + (y_end - y_start)
    src_x_end = src_x_start + (x_end - x_start)

    if y_end > y_start and x_end > x_start:
        canvas[y_start:y_end, x_start:x_end] = resized_arr[src_y_start:src_y_end, src_x_start:src_x_end]

    # Light Gaussian blur for anti-aliased edges
    canvas_pil = Image.fromarray(canvas.astype(np.uint8))
    canvas_pil = canvas_pil.filter(ImageFilter.GaussianBlur(radius=0.65))
    canvas = np.array(canvas_pil).astype(np.float32)

    # Normalize to 0-1
    if canvas.max() > 0:
        canvas = canvas / 255.0

    return canvas.reshape(1, 28, 28, 1)


def preprocess_canvas_image(image_data: np.ndarray) -> np.ndarray:
    """
    Preprocess the canvas RGBA image for prediction.
    Canvas has black background with white strokes.
    """
    # Extract grayscale from RGBA/RGB
    if image_data.shape[2] >= 3:
        gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = image_data[:, :, 0].astype(np.uint8)

    # Check if anything was actually drawn
    if gray.max() < 25:
        return None

    return _center_by_mass_and_fit(gray)


def preprocess_uploaded_image(pil_img: Image.Image) -> np.ndarray:
    """
    Robustly preprocess an uploaded image (PNG/JPG/JPEG) to 28x28 white-on-black.
    Handles: camera photos, scans, screenshots, gray/noisy backgrounds,
    thin pencil strokes, uneven lighting, any orientation.
    """
    # --- Step 0: Resize large images and convert to grayscale ---
    max_dim = 500
    if max(pil_img.size) > max_dim:
        ratio = max_dim / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    gray = pil_img.convert("L")

    # Apply slight blur to reduce camera noise
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr = np.array(gray, dtype=np.float64)
    h, w = arr.shape

    # --- Step 1: Detect background and invert if needed ---
    border_w = max(5, int(w * 0.08))
    border_h = max(5, int(h * 0.08))
    border_samples = np.concatenate([
        arr[:border_h, :].flatten(),
        arr[-border_h:, :].flatten(),
        arr[:, :border_w].flatten(),
        arr[:, -border_w:].flatten(),
    ])
    bg_brightness = np.median(border_samples)

    if bg_brightness > 90:
        arr = 255.0 - arr

    # --- Step 2: Adaptive LOCAL thresholding ---
    # Unlike global Otsu, this handles uneven lighting across the image
    # (common in camera photos with shadows)
    block_size = max(15, int(min(h, w) * 0.15)) | 1  # ensure odd
    binary = np.zeros_like(arr)

    # Pad the image for block processing
    pad = block_size // 2
    padded = np.pad(arr, pad, mode='reflect')

    for y in range(h):
        for x in range(w):
            # Local neighborhood
            local = padded[y:y + block_size, x:x + block_size]
            local_mean = local.mean()
            # Pixel is foreground if it's significantly brighter than local average
            if arr[y, x] > local_mean + 8:
                binary[y, x] = arr[y, x]

    # Fallback: if adaptive threshold produced very little, use global Otsu
    if binary.sum() < 100:
        # Global Otsu as fallback
        arr_flat = arr.flatten()
        hist, _ = np.histogram(arr_flat, bins=256, range=(0, 256))
        total = arr_flat.size
        sum_total = np.sum(np.arange(256) * hist)
        sum_bg, weight_bg, max_var, best_t = 0.0, 0, 0.0, 0
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0: continue
            weight_fg = total - weight_bg
            if weight_fg == 0: break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var > max_var:
                max_var = var
                best_t = t
        binary = arr.copy()
        binary[binary < best_t] = 0

    # --- Step 3: Morphological dilation to thicken thin strokes ---
    # This helps with thin pencil/pen strokes that might be too thin
    binary_uint8 = np.clip(binary, 0, 255).astype(np.uint8)
    dilated_img = Image.fromarray(binary_uint8)
    dilated_img = dilated_img.filter(ImageFilter.MaxFilter(size=3))
    binary_uint8 = np.array(dilated_img)

    # --- Step 4: Remove small noise blobs ---
    # Only keep the largest connected region(s)
    thresh_mask = binary_uint8 > 20
    if thresh_mask.sum() > 0:
        # Simple flood-fill connected component analysis
        from scipy import ndimage
        try:
            labeled, num_features = ndimage.label(thresh_mask)
            if num_features > 1:
                # Keep only components with significant size (>1% of largest)
                comp_sizes = ndimage.sum(thresh_mask, labeled, range(1, num_features + 1))
                max_size = max(comp_sizes)
                for i, size in enumerate(comp_sizes):
                    if size < max_size * 0.01:
                        binary_uint8[labeled == (i + 1)] = 0
        except ImportError:
            pass  # scipy not available, skip this step

    # --- Step 5: Contrast stretch ---
    max_val = binary_uint8.max()
    if max_val > 0:
        binary_uint8 = (binary_uint8.astype(np.float64) / max_val * 255.0).astype(np.uint8)

    return _center_by_mass_and_fit(binary_uint8)


def render_prob_bars(probs, predicted):
    html = '<div class="prob-container">'
    for i in range(10):
        pct = probs[i] * 100
        cls = "prob-fill top-pred" if i == predicted else "prob-fill"
        html += f'''
        <div class="prob-row">
            <span class="prob-digit-label">{i}</span>
            <div class="prob-track"><div class="{cls}" style="width: {pct:.1f}%"></div></div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>'''
    html += '</div>'
    return html


def render_history(hist):
    if not hist:
        return ""
    html = '<div style="display:flex;flex-wrap:wrap;gap:4px;">'
    for item in hist:
        html += f'''
        <div class="history-chip">
            <span class="chip-digit">{item['digit']}</span>
            <span class="chip-conf">{item['conf']:.1f}%</span>
        </div>'''
    html += '</div>'
    return html


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# ──────────────────────────────────────────────
# Train model
# ──────────────────────────────────────────────
with st.spinner("🧠 Training neural network on MNIST... (first run only, ~30s)"):
    model, test_accuracy = load_trained_model()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    Handwritten Digit<br>
    <span class="hero-gradient">Recognition</span>
</div>
<div class="hero-desc">Draw a digit or upload an image — AI identifies it instantly</div>
<div style="text-align:center">
    <div class="status-badge">
        <span class="status-dot"></span>
        Model ready · {acc:.1f}% accuracy
    </div>
</div>
""".format(acc=test_accuracy * 100), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────
col_input, col_spacer, col_result = st.columns([5, 0.5, 5])

with col_input:
    tab_draw, tab_upload = st.tabs(["✏️  Draw", "📤  Upload"])

    with tab_draw:
        st.markdown('<div class="section-label">Draw a digit on the canvas</div>', unsafe_allow_html=True)

        brush = st.slider("Brush size", 12, 40, 22, label_visibility="collapsed")

        # Use pure black background for clean preprocessing
        # Dynamic key forces a fresh canvas when Clear is clicked
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=False,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_draw = st.button("🔮 Recognize", type="primary", use_container_width=True, key="pred_draw")
        with btn_col2:
            clear_draw = st.button("🗑️ Clear", type="secondary", use_container_width=True, key="clear_draw")
            if clear_draw:
                st.session_state.canvas_key += 1
                st.session_state.last_prediction = None
                st.rerun()

    with tab_upload:
        st.markdown('<div class="section-label">Upload a digit image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop image here or click to browse",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            upload_img = Image.open(uploaded_file)
            st.image(upload_img, caption="Uploaded image", use_container_width=False, width=280)
            predict_upload = st.button("🔮 Recognize", type="primary", use_container_width=True, key="pred_upload")
        else:
            predict_upload = False


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
prediction_made = False
probs = None
predicted_digit = None
preview_img = None

# --- From canvas ---
if predict_draw and canvas_result.image_data is not None:
    tensor = preprocess_canvas_image(canvas_result.image_data)
    if tensor is not None:
        probs = model.predict(tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        prediction_made = True
        # Save the 28x28 preview
        preview_img = (tensor.reshape(28, 28) * 255).astype(np.uint8)
        st.session_state.last_prediction = {
            "probs": probs, "digit": predicted_digit, "preview": preview_img
        }
    else:
        st.toast("⚠️ Please draw a digit on the canvas first!", icon="✏️")

# --- From upload ---
if predict_upload and uploaded_file is not None:
    tensor = preprocess_uploaded_image(upload_img)
    if tensor is not None:
        probs = model.predict(tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        prediction_made = True
        preview_img = (tensor.reshape(28, 28) * 255).astype(np.uint8)
        st.session_state.last_prediction = {
            "probs": probs, "digit": predicted_digit, "preview": preview_img
        }
    else:
        st.toast("⚠️ Could not detect a digit in the image.", icon="📤")

# Use stored prediction if available (persists across reruns)
if not prediction_made and st.session_state.last_prediction is not None:
    probs = st.session_state.last_prediction["probs"]
    predicted_digit = st.session_state.last_prediction["digit"]
    preview_img = st.session_state.last_prediction["preview"]
    prediction_made = True


# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────
with col_result:
    if prediction_made and probs is not None:
        conf = float(probs[predicted_digit]) * 100

        # Add to history (only on fresh prediction)
        if predict_draw or predict_upload:
            st.session_state.history.insert(0, {"digit": predicted_digit, "conf": conf})
            if len(st.session_state.history) > 12:
                st.session_state.history.pop()

        st.markdown(f"""
        <div class="glass-card">
            <div class="prediction-box">
                <div class="pred-digit">{predicted_digit}</div>
                <div class="pred-confidence">{conf:.1f}%</div>
                <div class="pred-label">Confidence</div>
            </div>
            <div class="section-label" style="margin-top:16px">Class Probabilities</div>
            {render_prob_bars(probs, predicted_digit)}
        </div>
        """, unsafe_allow_html=True)

        # Show what the model sees (28x28 preview)
        if preview_img is not None:
            st.markdown('<div class="section-label" style="margin-top:16px">What the model sees (28×28)</div>', unsafe_allow_html=True)
            st.image(Image.fromarray(preview_img), width=112, caption="Preprocessed input")

    else:
        st.markdown("""
        <div class="glass-card">
            <div class="empty-state">
                <div style="font-size:2.5rem; margin-bottom:12px; opacity:0.3">🔮</div>
                <p>Draw or upload a digit<br>to see the prediction</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # History
    if st.session_state.history:
        st.markdown(f"""
        <div style="margin-top:16px">
            <div class="section-label">Recent Predictions</div>
            {render_history(st.session_state.history)}
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Powered by TensorFlow & Keras · CNN trained on MNIST · Built with Streamlit
</div>
""", unsafe_allow_html=True)
