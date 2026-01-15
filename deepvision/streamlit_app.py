import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ASSET_DIR = Path(__file__).parent / "web" / "assets"
DATA_DIR = ASSET_DIR / "data"
IMG_DIR = ASSET_DIR / "images"
SAMPLE_VIDEO = Path(__file__).parent / "sampleDashboard" / "video.mp4"

st.set_page_config(
    page_title="DeepVision Crowd Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def load_meta() -> Dict:
    path = DATA_DIR / "meta.json"
    if not path.exists():
        st.error("EDA assets are missing. Run `python scripts/build_eda_assets.py` first.")
        st.stop()
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_dataframes() -> Dict[str, pd.DataFrame]:
    payload_path = DATA_DIR / "dataframes.json"
    overview_csv = DATA_DIR / "crowd_dataset_overview.csv"
    if not payload_path.exists() or not overview_csv.exists():
        st.error("Required dataframe exports are missing. Re-run the EDA asset builder.")
        st.stop()
    with open(payload_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    overview_df = pd.read_csv(overview_csv)
    return {
        "head": pd.DataFrame(payload["head"]),
        "group_stats": pd.DataFrame(payload["group_stats"]),
        "summary_cards": pd.DataFrame(payload["summary_cards"]),
        "overview": overview_df,
    }


def load_image_safe(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    return Image.open(path)


def render_overview_tab(meta: Dict):
    st.markdown("### DeepVision Crowd Monitor")
    st.write(
        "Real-time deep learning platform that estimates crowd density, pinpoints "
        "overcrowded zones, and enables proactive safety interventions across transit hubs, "
        "pilgrimage sites, smart campuses, and mega events."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total labeled frames", f"{meta['total_images']:,}")
    col2.metric("Avg people / frame", f"{meta['avg_people_per_image']}")
    col3.metric("Max detected", f"{meta['max_people_detected']}")
    col4.metric("Min detected", f"{meta['min_people_detected']}")

    st.markdown("#### Project Workflow")
    workflow = [
        "Video input from CCTV or drone feeds",
        "Frame extraction and preprocessing",
        "CNN-based density estimation (CSRNet / MCNN)",
        "Counting + overcrowding detection",
        "Alerting and visual dashboards",
    ]
    st.write(" ➜ ".join(workflow))

    st.markdown("#### Architecture Highlights")
    col_a, col_b = st.columns(2)
    col_a.write(
        "- **Deep Learning:** PyTorch, CSRNet/MCNN backbones, density-map supervision\n"
        "- **Computer Vision:** OpenCV, Pillow, NumPy\n"
        "- **Visualization:** Matplotlib/Plotly overlays, Streamlit dashboard"
    )
    col_b.write(
        "- **Alerts:** SMTP/Twilio integrations for escalation\n"
        "- **Deployment:** Docker + (optional) Nginx reverse proxy\n"
        "- **Acceleration:** NVIDIA CUDA for real-time throughput"
    )


def render_eda_tab(meta: Dict, frames: Dict[str, pd.DataFrame]):
    st.markdown("### Exploratory Data Analysis")
    st.caption("Summaries derived from ShanghaiTech Part A & Part B via `scripts/build_eda_assets.py`.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Sample records")
        st.dataframe(frames["head"], use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Summary cards")
        for _, row in frames["summary_cards"].iterrows():
            st.metric(
                label=f"{row['part'].replace('_', ' ').title()} frames",
                value=int(row["count"]),
                delta=f"avg crowd {row['mean']}",
                help=f"Range {row['min']} – {row['max']} people",
            )

    st.subheader("Descriptive statistics")
    st.dataframe(frames["group_stats"], use_container_width=True, hide_index=True)

    st.subheader("Distributions & density map")
    col3, col4, col5 = st.columns(3)
    col3.image(IMG_DIR / meta["assets"]["distribution"]["hist"], caption="Crowd count histogram")
    col4.image(IMG_DIR / meta["assets"]["distribution"]["box"], caption="Crowd count boxplot")
    col5.image(IMG_DIR / meta["assets"]["density_map"], caption="Sample density map")

    st.subheader("Random samples")
    random_images: List[str] = meta["assets"]["random_samples"]
    cols = st.columns(len(random_images))
    for col, img_name in zip(cols, random_images):
        col.image(IMG_DIR / img_name, use_container_width=True)


def estimate_density_naive(pil_img: Image.Image) -> Dict[str, float]:
    """Lightweight demo estimator (placeholder for CSRNet inference)."""
    arr = np.asarray(pil_img.convert("L"))
    normalized = arr / 255.0
    density_score = float(normalized.mean())
    estimated_count = int(np.clip((normalized > density_score).sum() / 50, 0, 4000))
    return {"score": density_score, "count": estimated_count}


def render_live_tab(meta: Dict):
    st.markdown("### Live Monitor Prototype")
    st.caption(
        "Upload a frame or use the sample video to simulate density estimation. "
        "Replace the naive estimator with CSRNet inference for production deployments."
    )

    threshold = st.slider("Alert threshold (people)", min_value=100, max_value=3000, value=800, step=50)

    col_left, col_right = st.columns(2)
    with col_left:
        uploaded = st.file_uploader("Upload frame (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Uploaded frame", use_container_width=True)
            estimate = estimate_density_naive(pil_img)
            st.metric("Estimated crowd", estimate["count"], help=f"Density score: {estimate['score']:.3f}")
            if estimate["count"] >= threshold:
                st.error("⚠️ Overcrowding detected! Trigger alert.")
            else:
                st.success("Crowd level within safe limits.")
        else:
            st.info("Upload a still frame to run the crowd estimator.")

    with col_right:
        if SAMPLE_VIDEO.exists():
            st.video(str(SAMPLE_VIDEO))
            st.caption("Sample live feed (replace with CCTV capture in production).")
        else:
            st.warning("Sample video not found. Place a clip at `sampleDashboard/video.mp4`.")

    st.markdown("#### Deployment Checklist")
    st.write(
        "- Connect webcam/RTSP feeds through OpenCV and stream frames to the model.\n"
        "- Replace `estimate_density_naive` with CSRNet inference (PyTorch) and density-map rendering.\n"
        "- Wire alert routing to SMTP/Twilio and log events for audits.\n"
        "- Containerize with Docker + enable GPU pass-through for real-time throughput."
    )


def main():
    meta = load_meta()
    frames = load_dataframes()

    tab_overview, tab_eda, tab_live = st.tabs(["Overview", "EDA Explorer", "Live Monitor"])

    with tab_overview:
        render_overview_tab(meta)

    with tab_eda:
        render_eda_tab(meta, frames)

    with tab_live:
        render_live_tab(meta)


if __name__ == "__main__":
    main()

