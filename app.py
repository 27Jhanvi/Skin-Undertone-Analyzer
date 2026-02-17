import cv2
import numpy as np
import streamlit as st

from src.color_calibration import gray_world_white_balance
from src.undertone import lab_stats_from_rgb_mask, classify_from_two_conditions


# -----------------------------
# THEME (PASTEL GIRLISH)
# -----------------------------
def apply_girlish_theme():
    st.set_page_config(
        page_title="Skin Undertone Analyzer",
        page_icon="💖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    /* ====== App Background (Pastel) ====== */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(255, 214, 235, 0.75) 0%, transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(204, 244, 255, 0.80) 0%, transparent 45%),
            radial-gradient(circle at 80% 85%, rgba(231, 214, 255, 0.75) 0%, transparent 45%),
            linear-gradient(180deg, #fff7fb 0%, #f6fbff 55%, #fbf7ff 100%);
        color: #1c1b22;
        font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial;
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }

    /* ====== Hero Card ====== */
    .hero {
        padding: 22px 26px;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(30, 30, 40, 0.10);
        box-shadow: 0 18px 55px rgba(120, 90, 140, 0.18);
        backdrop-filter: blur(10px);
        margin-bottom: 14px;
    }
    .hero h1 {
        font-size: 42px;
        margin: 0;
        line-height: 1.08;
        letter-spacing: -0.6px;
        color: #1f1d2b;
    }
    .hero p {
        margin: 8px 0 0 0;
        font-size: 15.5px;
        color: rgba(31, 29, 43, 0.78);
    }

    /* ====== Sidebar (Pastel glass) ====== */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.62);
        border-right: 1px solid rgba(30, 30, 40, 0.10);
        backdrop-filter: blur(12px);
    }
    section[data-testid="stSidebar"] * {
        color: #1f1d2b !important;
    }

    /* ====== File uploader (Pastel card) ====== */
    div[data-testid="stFileUploader"] section {
        border-radius: 18px !important;
        border: 1px dashed rgba(120, 90, 140, 0.35) !important;
        background: rgba(255, 255, 255, 0.55) !important;
    }

    /* ====== Buttons (Cute gradient) ====== */
    .stButton>button, button[kind="primary"] {
        border-radius: 999px !important;
        padding: 10px 16px !important;
        border: 1px solid rgba(120, 90, 140, 0.25) !important;
        background: linear-gradient(90deg, rgba(255, 140, 200, 0.55), rgba(155, 225, 255, 0.60)) !important;
        color: #1f1d2b !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 28px rgba(120, 90, 140, 0.18) !important;
    }

    /* ====== Alerts softer ====== */
    div[data-testid="stAlert"] {
        border-radius: 16px !important;
        background: rgba(255,255,255,0.65) !important;
        border: 1px solid rgba(30,30,40,0.10) !important;
        box-shadow: 0 14px 40px rgba(120, 90, 140, 0.14) !important;
    }

    /* ====== Captions / small text ====== */
    .stCaption, .stMarkdown, .stText, p, label {
        color: rgba(31, 29, 43, 0.82);
    }
    </style>
    """, unsafe_allow_html=True)


# IMPORTANT: theme must be applied before the UI
apply_girlish_theme()


# -----------------------------
# HEADER / HERO
# -----------------------------
st.markdown("""
<div class="hero">
  <h1>💖 Skin Undertone Analyzer</h1>
  <p>Computer Vision + LAB stats • works best with Natural + Indoor photos ✨</p>
</div>
""", unsafe_allow_html=True)

st.markdown("Upload **one** image OR upload **two images** (Natural + Indoor) for better stability.")


# -----------------------------
# HELPERS
# -----------------------------
def read_image_to_rgb(uploaded_file):
    if uploaded_file is None:
        return None

    data = uploaded_file.getvalue()
    if not data:
        return None

    file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_skin_mask(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # slightly wider than classic range
    lower = np.array([0, 25, 60])
    upper = np.array([25, 220, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def overlay_mask(image_rgb, mask):
    overlay = image_rgb.copy()
    overlay[mask == 255] = [255, 0, 120]  # pink overlay
    return cv2.addWeighted(image_rgb, 0.78, overlay, 0.22, 0)


# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("Options")
apply_wb = st.sidebar.checkbox("Apply Gray-World White Balance", value=True)
min_sat = st.sidebar.slider("Min Saturation filter (mask refinement)", 0, 80, 35, 5)
neutral_band = st.sidebar.slider("Neutral band (Lab a*/b* tolerance)", 1.0, 6.0, 2.5, 0.5)
show_debug = st.sidebar.checkbox("Show debug details", value=True)
show_masks = st.sidebar.checkbox("Show skin masks", value=True)


# -----------------------------
# UPLOAD
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    nat_file = st.file_uploader(
        "Upload Natural-light image (optional)",
        type=["jpg", "jpeg", "png"],
        key="natural"
    )
with c2:
    ind_file = st.file_uploader(
        "Upload Indoor-light image (optional)",
        type=["jpg", "jpeg", "png"],
        key="indoor"
    )

nat_rgb = read_image_to_rgb(nat_file)
ind_rgb = read_image_to_rgb(ind_file)

if nat_rgb is None and ind_rgb is None:
    st.info("Upload at least one image to start.")
    st.stop()


# -----------------------------
# PROCESS
# -----------------------------
def process_one(rgb):
    if apply_wb:
        rgb = gray_world_white_balance(rgb)

    mask = make_skin_mask(rgb)
    stats = lab_stats_from_rgb_mask(rgb, mask, min_sat=min_sat)

    vis = {
        "rgb": rgb,
        "mask_overlay": overlay_mask(rgb, mask),
    }
    return stats, vis


nat_stats, nat_vis = (None, None)
ind_stats, ind_vis = (None, None)

if nat_rgb is not None:
    nat_stats, nat_vis = process_one(nat_rgb)

if ind_rgb is not None:
    ind_stats, ind_vis = process_one(ind_rgb)


# -----------------------------
# CLASSIFY (safe even if one is None)
# -----------------------------
result = classify_from_two_conditions(nat_stats, ind_stats, neutral_band=neutral_band)

if result.label == "Unknown":
    st.error("No valid skin pixels detected. Try a clearer face/hand image (good lighting, not too bright).")
else:
    st.success(f"Undertone: {result.label}  |  Confidence: {result.confidence:.2f}")

if show_debug:
    st.caption(str(result.debug))


# -----------------------------
# PREVIEW
# -----------------------------
st.subheader("Preview")
v1, v2 = st.columns(2)

with v1:
    st.write("Natural")
    if nat_vis is None:
        st.info("Natural image not uploaded.")
    else:
        st.image(nat_vis["mask_overlay"] if show_masks else nat_vis["rgb"], use_container_width=True)

with v2:
    st.write("Indoor")
    if ind_vis is None:
        st.info("Indoor image not uploaded.")
    else:
        st.image(ind_vis["mask_overlay"] if show_masks else ind_vis["rgb"], use_container_width=True)

st.divider()
st.caption("Tip: Upload both natural + indoor images for more stable undertone prediction.")
