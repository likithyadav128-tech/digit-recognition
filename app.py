import streamlit as st
import numpy as np
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
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --violet-400: #a78bfa;
        --violet-500: #8b5cf6;
        --indigo-400: #818cf8;
        --indigo-500: #6366f1;
        --cyan-400: #22d3ee;
        --green-400: #4ade80;
    }

    /* Global background */
    .stApp {
        background: #06060e;
        background-image:
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(99,102,241,0.1) 0%, transparent 70%),
            radial-gradient(ellipse 60% 50% at 80% 80%, rgba(139,92,246,0.06) 0%, transparent 60%);
    }

    /* Hide default elements */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* Typography */
    h1, h2, h3, p, span, div, label {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Hero title */
    .hero-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 4px;
        color: #f0eef6;
    }
    .hero-gradient {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        text-align: center;
        font-size: 1rem;
        color: #8b86a8;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 999px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        font-size: 0.78rem;
        color: #8b86a8;
        margin: 0 auto 1.5rem;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 8px rgba(74,222,128,0.5);
        display: inline-block;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(12, 12, 30, 0.7);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(139, 92, 246, 0.1);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 1px rgba(139,92,246,0.05);
    }

    /* Section labels */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5c577a;
        margin-bottom: 14px;
    }

    /* Prediction display */
    .prediction-box {
        text-align: center;
        padding: 20px 0;
    }
    .pred-digit {
        font-size: 6rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .pred-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #4ade80;
    }
    .pred-label {
        font-size: 0.7rem;
        color: #5c577a;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 2px;
    }

    /* Probability bars */
    .prob-container { margin-top: 16px; }
    .prob-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }
    .prob-digit-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #5c577a;
        width: 16px;
        text-align: center;
        flex-shrink: 0;
    }
    .prob-track {
        flex: 1;
        height: 16px;
        background: rgba(255,255,255,0.03);
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        opacity: 0.4;
        transition: width 0.5s ease;
    }
    .prob-fill.top-pred {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 100%);
        opacity: 1;
        box-shadow: 0 0 10px rgba(139,92,246,0.25);
    }
    .prob-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        color: #5c577a;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }

    /* Canvas styling */
    canvas {
        border-radius: 14px !important;
    }
    .stCanvas > div {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(139,92,246,0.08);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        color: #8b86a8;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(20, 20, 50, 0.55);
        color: #f0eef6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* Buttons */
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        color: #fff;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 28px rgba(99,102,241,0.5);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        color: #8b86a8;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(255,255,255,0.07);
        color: #f0eef6;
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(139,92,246,0.2) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.02) !important;
    }
    .stFileUploader:hover {
        border-color: rgba(139,92,246,0.4) !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #818cf8, #c084fc);
        box-shadow: 0 0 8px rgba(139,92,246,0.4);
    }

    /* History chips */
    .history-chip {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 52px;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .chip-digit {
        font-size: 1.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chip-conf {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.5rem;
        color: #5c577a;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        font-size: 0.7rem;
        color: #5c577a;
        padding: 24px 0 8px;
        letter-spacing: 0.01em;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #5c577a;
        font-size: 0.9rem;
    }

    /* Metric override */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #4ade80 !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
        color: #5c577a !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-size: 0.7rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model: Build & Train (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Build and train CNN on MNIST, cached across reruns."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Build CNN
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test), verbose=0)

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    return model, acc


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert any image to 28x28 grayscale, white-on-black, centered."""
    # Convert to grayscale
    img = img.convert("L")

    # Determine if we need to invert (white bg → black bg)
    arr = np.array(img)
    if arr.mean() > 128:
        img = ImageOps.invert(img)

    # Apply slight Gaussian blur for smoother edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Find bounding box of content
    arr = np.array(img)
    rows = np.any(arr > 15, axis=1)
    cols = np.any(arr > 15, axis=0)

    if not rows.any():
        return np.zeros((1, 28, 28, 1), dtype="float32")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to content
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Fit into 20x20 maintaining aspect ratio (like MNIST preprocessing)
    ch, cw = cropped.shape
    scale = min(20.0 / cw, 20.0 / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))

    cropped_img = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)

    # Center in 28x28
    final = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    final.paste(cropped_img, (paste_x, paste_y))

    result = np.array(final).astype("float32") / 255.0
    return result.reshape(1, 28, 28, 1)


def render_prob_bars(probs, predicted):
    """Render beautiful probability bars as HTML."""
    html = '<div class="prob-container">'
    for i in range(10):
        pct = probs[i] * 100
        cls = "prob-fill top-pred" if i == predicted else "prob-fill"
        html += f'''
        <div class="prob-row">
            <span class="prob-digit-label">{i}</span>
            <div class="prob-track">
                <div class="{cls}" style="width: {pct:.1f}%"></div>
            </div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>'''
    html += '</div>'
    return html


def render_history(hist):
    """Render history chips as HTML."""
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
# Initialize session state
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ──────────────────────────────────────────────
# Train model (with loading spinner)
# ──────────────────────────────────────────────
with st.spinner("🧠 Training neural network on MNIST data... (first run only)"):
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
# Main layout
# ──────────────────────────────────────────────
col_input, col_spacer, col_result = st.columns([5, 0.5, 5])

with col_input:
    tab_draw, tab_upload = st.tabs(["✏️  Draw", "📤  Upload"])

    with tab_draw:
        st.markdown('<div class="section-label">Draw a digit below</div>', unsafe_allow_html=True)

        brush = st.slider("Brush size", 10, 35, 20, label_visibility="collapsed")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush,
            stroke_color="#FFFFFF",
            background_color="#0a0a14",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=False,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_draw = st.button("🔮 Recognize", type="primary", use_container_width=True, key="pred_draw")
        with btn_col2:
            if st.button("🗑️ Clear", type="secondary", use_container_width=True, key="clear_draw"):
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
# Prediction logic
# ──────────────────────────────────────────────
prediction_made = False
probs = None
predicted_digit = None

# Predict from canvas
if predict_draw and canvas_result.image_data is not None:
    img_array = canvas_result.image_data[:, :, :3]  # Remove alpha
    if img_array.sum() > 0:
        pil_img = Image.fromarray(img_array.astype("uint8"))
        tensor = preprocess_image(pil_img)
        probs = model.predict(tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        prediction_made = True

# Predict from upload
if predict_upload and uploaded_file is not None:
    tensor = preprocess_image(upload_img)
    probs = model.predict(tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    prediction_made = True

# ──────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────
with col_result:
    if prediction_made and probs is not None:
        conf = float(probs[predicted_digit]) * 100

        # Add to history
        st.session_state.history.insert(0, {"digit": predicted_digit, "conf": conf})
        if len(st.session_state.history) > 12:
            st.session_state.history.pop()

        # Prediction display
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
