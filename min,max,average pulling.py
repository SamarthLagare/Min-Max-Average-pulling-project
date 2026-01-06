import streamlit as st
import cv2
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import plotly.express as px
import io

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="MatrixFlow Studio",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS STYLING (IMPRESSIVE UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* HERO SECTION STYLES */
    .hero-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 700;
        background: -webkit-linear-gradient(120deg, #ffffff, #4b7bff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #B0B0B0;
        max-width: 600px;
        margin: 0 auto;
    }

    /* FEATURE CARDS */
    .feature-card {
        background-color: #161920;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #262730;
        text-align: center;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #4b7bff;
        box-shadow: 0 4px 20px rgba(75, 123, 255, 0.1);
    }
    .card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: block;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #fff;
    }
    .card-text {
        font-size: 0.9rem;
        color: #a0a0a0;
    }

    /* DATA GRID STYLING */
    .stDataFrame { border: 1px solid #333; border-radius: 8px; }

    /* CUSTOM BUTTON */
    div.stButton > button {
        background: linear-gradient(90deg, #262730 0%, #31333F 100%);
        color: white;
        border: 1px solid #444;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #4b7bff;
        color: #4b7bff;
        box-shadow: 0 0 10px rgba(75, 123, 255, 0.2);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. LOGIC FUNCTIONS ---
def manual_pooling(mat, kernel_size, stride, method='max'):
    h, w = mat.shape
    k = kernel_size
    s = stride
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    if out_h <= 0 or out_w <= 0: return None 
    itemsize = mat.itemsize
    output_shape = (out_h, out_w, k, k)
    strides = (s * w * itemsize, s * itemsize, w * itemsize, itemsize)
    windows = as_strided(mat, shape=output_shape, strides=strides)
    if method == 'max': return np.max(windows, axis=(2, 3))
    elif method == 'min': return np.min(windows, axis=(2, 3))
    elif method == 'avg': return np.mean(windows, axis=(2, 3)).astype(int)
    return mat

def render_heatmap(matrix, title):
    show_text = True if matrix.shape[0] <= 32 else False
    fig = px.imshow(
        matrix, color_continuous_scale='gray', zmin=0, zmax=255,
        text_auto=show_text, title=f"{title} (Visual)"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300, font=dict(color="white"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_coloraxes(showscale=False)
    if show_text: fig.update_traces(texttemplate="%{z:.0f}", textfont_size=14)
    st.plotly_chart(fig, use_container_width=True)

def render_data_grid(matrix, label):
    st.markdown(f"**{label} (Data)**")
    if matrix.shape[0] > 20:
        st.warning(f"Large Grid ({matrix.shape[0]}x{matrix.shape[1]}). Previewing top-left 10x10.")
        st.dataframe(matrix[:10, :10], use_container_width=True)
        s = io.BytesIO()
        np.savetxt(s, matrix, fmt='%d', delimiter=",")
        st.download_button(f"Download {label} CSV", s.getvalue(), f"{label}.csv", "text/csv", key=f"dl_{label}")
    else:
        st.dataframe(matrix, use_container_width=True)

# --- 4. NAVIGATION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'home'

def go_home(): st.session_state.page = 'home'
def go_lab(): st.session_state.page = 'lab'

# --- 5. PAGE CONTENT ---

# ====== HOME PAGE UI ======
if st.session_state.page == 'home':
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">MATRIXFLOW STUDIO</h1>
            <p class="hero-subtitle">
                Deconstruct images into raw numbers. Visualize the mathematics behind Computer Vision and Convolutional Neural Networks (CNNs).
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature Grid
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        st.markdown("""
        <div class="feature-card">
            <span class="card-icon">💠</span>
            <div class="card-title">Grid Compression</div>
            <div class="card-text">
                Transform high-res images into manageable numerical grids ranging from 8x8 to 256x256 pixels.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="feature-card">
            <span class="card-icon">🧮</span>
            <div class="card-title">Pooling Engine</div>
            <div class="card-text">
                Apply real-time Max, Min, and Average pooling with custom kernel sizes and strides.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <span class="card-icon">📊</span>
            <div class="card-title">Dual Visualization</div>
            <div class="card-text">
                See the visual heatmap side-by-side with the raw data matrix to understand the math.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("<br><br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        if st.button("🚀 Launch Workspace", use_container_width=True):
            go_lab()
            st.rerun()

# ====== LAB PAGE UI ======
elif st.session_state.page == 'lab':
    # Sidebar only appears in Lab Mode
    with st.sidebar:
        st.header("MatrixFlow")
        if st.button("← Back to Home"):
            go_home()
            st.rerun()
        st.markdown("---")
        st.info("Upload an image to start processing.")

    st.markdown("## 🧪 Matrix Lab")
    
    # 1. INPUT
    with st.expander("📂 1. Upload & Settings", expanded=True):
        c_up, c_set = st.columns([2, 1])
        with c_up:
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        with c_set:
            st.write("**Compression Target**")
            grid_size = st.selectbox(
                "Grid Size", [8, 16, 32, 64, 128, 256], index=1,
                format_func=lambda x: f"{x} x {x} Matrix"
            )
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        compressed_matrix = cv2.resize(img_gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA)

        st.markdown("---")
        st.markdown(f"### 2. Base Matrix Analysis")
        c_visual, c_data = st.columns([1, 1])
        with c_visual: st.image(img_rgb, caption="Original Input (RGB)", use_container_width=True)
        with c_data: render_data_grid(compressed_matrix, "Compressed Matrix Values")

        st.markdown("---")
        st.markdown("### 3. Pooling Operations")
        c1, c2 = st.columns(2)
        with c1: k_size = st.slider("Kernel (Window) Size", 2, 5, 2)
        with c2: stride = st.slider("Stride (Step)", 1, 3, 2)

        pool_max = manual_pooling(compressed_matrix, k_size, stride, 'max')
        pool_min = manual_pooling(compressed_matrix, k_size, stride, 'min')
        pool_avg = manual_pooling(compressed_matrix, k_size, stride, 'avg')

        if pool_max is None:
            st.error("Kernel is too big for this image size.")
        else:
            col_max, col_min, col_avg = st.columns(3)
            with col_max:
                st.markdown("#### Max Pooling")
                st.caption("Extracts Edges")
                render_heatmap(pool_max, "Max Result")
                render_data_grid(pool_max, "Max Data")
            with col_min:
                st.markdown("#### Min Pooling")
                st.caption("Extracts Background")
                render_heatmap(pool_min, "Min Result")
                render_data_grid(pool_min, "Min Data")
            with col_avg:
                st.markdown("#### Avg Pooling")
                st.caption("Smooths Features")
                render_heatmap(pool_avg, "Avg Result")
                render_data_grid(pool_avg, "Avg Data")
    else:
        st.info("Please upload an image to begin the analysis.")
