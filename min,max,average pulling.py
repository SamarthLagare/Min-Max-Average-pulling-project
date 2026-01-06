import streamlit as st
import cv2
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import plotly.express as px
import io

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="MatrixFlow Studio v2.0",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
        background-color: #0E1117;
    }
    
    /* Make Dataframes look more like grids */
    .stDataFrame { border: 1px solid #444; border-radius: 5px; }

    /* Buttons */
    div.stButton > button {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #4b7bff;
        color: #4b7bff;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. NUMPY POOLING ENGINE ---
def manual_pooling(mat, kernel_size, stride, method='max'):
    h, w = mat.shape
    k = kernel_size
    s = stride

    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1

    if out_h <= 0 or out_w <= 0:
        return None 

    itemsize = mat.itemsize
    output_shape = (out_h, out_w, k, k)
    strides = (s * w * itemsize, s * itemsize, w * itemsize, itemsize)
    
    windows = as_strided(mat, shape=output_shape, strides=strides)

    if method == 'max':
        return np.max(windows, axis=(2, 3))
    elif method == 'min':
        return np.min(windows, axis=(2, 3))
    elif method == 'avg':
        # Return Integer for clean display
        return np.mean(windows, axis=(2, 3)).astype(int)
    return mat

def render_heatmap(matrix, title):
    """Renders the visual heatmap (Grayscale)."""
    # Show numbers inside the squares ONLY if the grid is small (16x16 or similar)
    show_text = True if matrix.shape[0] <= 32 else False
    
    fig = px.imshow(
        matrix, 
        color_continuous_scale='gray', 
        zmin=0, zmax=255,
        text_auto=show_text,
        title=f"{title} (Visual)"
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0), 
        height=300,
        font=dict(color="white")
    )
    fig.update_coloraxes(showscale=False)
    
    if show_text:
        fig.update_traces(texttemplate="%{z:.0f}", textfont_size=14)
    
    st.plotly_chart(fig, use_container_width=True)

def render_data_grid(matrix, label):
    """Renders the raw numbers grid (DataFrame)."""
    st.markdown(f"**{label} (Data)**")
    
    if matrix.shape[0] > 20:
        st.warning(f"Grid is large ({matrix.shape[0]}x{matrix.shape[1]}). Showing top-left 10x10.")
        st.dataframe(matrix[:10, :10], use_container_width=True)
        
        # CSV Download for large files
        s = io.BytesIO()
        np.savetxt(s, matrix, fmt='%d', delimiter=",")
        st.download_button(f"Download {label} CSV", s.getvalue(), f"{label}.csv", "text/csv", key=f"dl_{label}")
    else:
        # Full grid for 16x16
        st.dataframe(matrix, use_container_width=True)

# --- 4. NAVIGATION ---
if 'page' not in st.session_state: st.session_state.page = 'home'

with st.sidebar:
    st.title("Navigation")
    st.caption("v2.0 Updated Build") # LOOK FOR THIS TO CONFIRM UPDATE
    
    page = st.radio("Go to:", ["Home", "Matrix Lab"], label_visibility="collapsed")
    
    if page == "Home": st.session_state.page = 'home'
    else: st.session_state.page = 'lab'
    
    st.markdown("---")
    st.info("**MatrixFlow Studio**\n\nCNN Feature Extraction Workbench.")

# --- 5. PAGE LOGIC ---

# === HOME ===
if st.session_state.page == 'home':
    st.title("MATRIXFLOW STUDIO v2.0")
    st.markdown("### Feature Extraction & Pooling Workbench")
    st.markdown("---")
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 1. RGB Compression")
        st.write("Convert image to numbers. (Visualization: Color Image + Data Grid)")
    with c2:
        st.markdown("#### 2. Pooling Logic")
        st.write("Apply Max, Min, Avg logic. (Visualization: Heatmap + Data Grid)")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start Experiment >"):
        st.session_state.page = 'lab'
        st.rerun()

# === LAB ===
elif st.session_state.page == 'lab':
    st.markdown("## Matrix Lab")
    
    # 1. INPUT
    with st.expander("📂 1. Upload & Settings", expanded=True):
        c_up, c_set = st.columns([2, 1])
        with c_up:
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        with c_set:
            st.write("**Compression Target**")
            grid_size = st.selectbox("Grid Size", [16, 256], format_func=lambda x: f"{x} x {x} Matrix")
    
    if uploaded_file:
        # 2. PROCESS
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Resize/Compress
        compressed_matrix = cv2.resize(img_gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA)

        # 3. BASE MATRIX DISPLAY
        st.markdown("---")
        st.markdown(f"### 2. Base Matrix Analysis")
        
        c_visual, c_data = st.columns([1, 1])
        
        with c_visual:
            st.image(img_rgb, caption="Original Input (RGB)", use_container_width=True)
            # NOTE: No grayscale heatmap here, as requested.
        
        with c_data:
            # PURE DATA GRID
            render_data_grid(compressed_matrix, "Compressed Matrix Values")

        # 4. POOLING OPERATIONS
        st.markdown("---")
        st.markdown("### 3. Pooling Operations")
        
        # Controls
        c1, c2 = st.columns(2)
        with c1: k_size = st.slider("Kernel (Window) Size", 2, 5, 2)
        with c2: stride = st.slider("Stride (Step)", 1, 3, 2)

        # Compute
        pool_max = manual_pooling(compressed_matrix, k_size, stride, 'max')
        pool_min = manual_pooling(compressed_matrix, k_size, stride, 'min')
        pool_avg = manual_pooling(compressed_matrix, k_size, stride, 'avg')

        if pool_max is None:
            st.error("Kernel is too big for this image size.")
        else:
            # Display: 3 Columns. Each Column has Heatmap THEN Data Grid
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

            # Formula
            st.markdown("---")
            with st.expander("ℹ️ Calculation Details"):
                st.latex(r''' Output_{size} = \lfloor \frac{Input - Kernel}{Stride} \rfloor + 1 ''')
                st.write(f"({grid_size} - {k_size}) / {stride} + 1 = **{pool_max.shape[0]}x{pool_max.shape[0]}**")

    else:
        st.info("Please upload an image to begin.")
