import streamlit as st
import cv2
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="MatrixFlow Studio",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING (The "Clear View" Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
        background-color: #0E1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Cards/Containers */
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #444;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #4b7bff;
        color: #4b7bff;
    }

    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. NUMPY POOLING ENGINE ---
def manual_pooling(mat, kernel_size, stride, method='max'):
    """
    Applies sliding window pooling efficiently using NumPy strides.
    """
    h, w = mat.shape
    k = kernel_size
    s = stride

    # Calculate output dimensions
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1

    if out_h <= 0 or out_w <= 0:
        return np.zeros((1,1)) # Return empty if kernel > image

    # Create sliding windows view (4D array)
    itemsize = mat.itemsize
    output_shape = (out_h, out_w, k, k)
    strides = (s * w * itemsize, s * itemsize, w * itemsize, itemsize)
    
    windows = as_strided(mat, shape=output_shape, strides=strides)

    # Apply the mathematical operation
    if method == 'max':
        return np.max(windows, axis=(2, 3))
    elif method == 'min':
        return np.min(windows, axis=(2, 3))
    elif method == 'avg':
        return np.mean(windows, axis=(2, 3))
    return mat

def render_heatmap(matrix, title):
    """Uses Plotly to render a nice heatmap for the matrix."""
    fig = px.imshow(
        matrix, 
        color_continuous_scale='gray', 
        zmin=0, zmax=255,
        title=f"{title} ({matrix.shape[0]}x{matrix.shape[1]})"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. NAVIGATION ---
if 'page' not in st.session_state: st.session_state.page = 'home'

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["Home", "Matrix Lab"], label_visibility="collapsed")
    
    if page == "Home": st.session_state.page = 'home'
    else: st.session_state.page = 'lab'
    
    st.markdown("---")
    st.info("**MatrixFlow Studio**\n\nSpecialized tool for converting images into raw feature grids and applying CNN pooling concepts.")

# --- 5. PAGE LOGIC ---

# === HOME PAGE ===
if st.session_state.page == 'home':
    st.title("MATRIXFLOW STUDIO")
    st.markdown("### Feature Extraction & Pooling Workbench")
    st.markdown("---")
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 1. RGB Compression")
        st.write("Import any image and compress it into a standardized single-channel matrix (16x16 or 256x256).")
    with c2:
        st.markdown("#### 2. Sliding Window Pooling")
        st.write("Apply Max, Min, and Average pooling algorithms to visualize how Neural Networks downsample features.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Start Experiment >"):
        st.session_state.page = 'lab'
        st.rerun()

# === LAB PAGE ===
elif st.session_state.page == 'lab':
    st.markdown("## Matrix Lab")
    
    # 1. INPUT SECTION
    with st.expander("📂 1. Upload & Compression Settings", expanded=True):
        c_up, c_set = st.columns([2, 1])
        with c_up:
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        with c_set:
            st.write("**Compression Target**")
            grid_size = st.selectbox("Grid Size", [16, 256], format_func=lambda x: f"{x} x {x} Matrix")
    
    if uploaded_file:
        # 2. PRE-PROCESSING
        # Read file buffer -> OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to Single Matrix (Grayscale) as requested
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Compress (Resize)
        # We use INTER_AREA for better quality when shrinking images
        compressed_matrix = cv2.resize(img_gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA)

        # 3. VISUALIZATION OF BASE MATRIX
        st.markdown("---")
        st.markdown(f"### 2. Base Matrix ({grid_size}x{grid_size})")
        
        col_orig, col_mat = st.columns([1, 2])
        
        with col_orig:
            st.image(img_rgb, caption="Original RGB", use_container_width=True)
        
        with col_mat:
            # If 16x16, show Dataframe for clarity
            if grid_size == 16:
                t1, t2 = st.tabs(["Visual", "Data"])
                with t1: render_heatmap(compressed_matrix, "Compressed Input")
                with t2: st.dataframe(compressed_matrix, height=300)
            else:
                st.image(compressed_matrix, caption=f"Compressed Single Matrix ({grid_size}x{grid_size})", width=300)

        # 4. POOLING OPERATIONS
        st.markdown("---")
        st.markdown("### 3. Pooling Operations")
        
        # Controls
        c_ctrl1, c_ctrl2 = st.columns(2)
        with c_ctrl1:
            k_size = st.slider("Kernel Size (k)", 2, 5, 2)
        with c_ctrl2:
            stride = st.slider("Stride (s)", 1, 3, 2)

        # Calculation
        pool_max = manual_pooling(compressed_matrix, k_size, stride, 'max')
        pool_min = manual_pooling(compressed_matrix, k_size, stride, 'min')
        pool_avg = manual_pooling(compressed_matrix, k_size, stride, 'avg')

        if pool_max is None:
            st.error("Kernel size is larger than the image grid! Reduce Kernel Size.")
        else:
            # Display Results
            p1, p2, p3 = st.columns(3)
            
            with p1:
                st.markdown("#### Max Pooling")
                st.caption("Captures most prominent features (edges/textures)")
                render_heatmap(pool_max, "Max Result")
                st.write(f"Shape: {pool_max.shape}")

            with p2:
                st.markdown("#### Min Pooling")
                st.caption("Captures darkest areas/background")
                render_heatmap(pool_min, "Min Result")
                st.write(f"Shape: {pool_min.shape}")

            with p3:
                st.markdown("#### Avg Pooling")
                st.caption("Smooths the image")
                render_heatmap(pool_avg, "Avg Result")
                st.write(f"Shape: {pool_avg.shape}")

            # Formula Explanation
            with st.expander("ℹ️ How calculations work"):
                st.latex(r''' Output_{size} = \lfloor \frac{Input - Kernel}{Stride} \rfloor + 1 ''')
                st.write(f"Input: {grid_size}, Kernel: {k_size}, Stride: {stride}")
                st.write(f"Result: ({grid_size} - {k_size}) / {stride} + 1 = **{pool_max.shape[0]}**")

    else:
        st.info("Waiting for image upload...")
