import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib

# Use a non-interactive backend so the script can run headless
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
sns.set_theme(style="whitegrid")

# Make sure we can import the project's EDA helpers
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from Eda.basic_stats import summarize_dataset  # noqa: E402
from Eda.density_map_demo import generate_density_map  # noqa: E402

DATASET_ROOT = ROOT_DIR / "dataset" / "ShanghaiTech"
ASSET_ROOT = ROOT_DIR / "web" / "assets"
IMG_DIR = ASSET_ROOT / "images"
DATA_DIR = ASSET_ROOT / "data"


def ensure_paths():
    for path in (IMG_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def serialize_dataframes(df: pd.DataFrame):
    """Persist the dataset summary and compact JSON slices that power the EDA page."""
    csv_path = DATA_DIR / "crowd_dataset_overview.csv"
    df.to_csv(csv_path, index=False)

    head_payload = df.head(12).to_dict(orient="records")
    grouped = (
        df.groupby(["part", "mode"])["people_count"]
        .describe()
        .reset_index()
        .round(2)
    )

    summary_cards = (
        df.groupby("part")["people_count"]
        .agg(["count", "mean", "max", "min"])
        .reset_index()
        .round(2)
    )

    payload = {
        "head": head_payload,
        "group_stats": grouped.to_dict(orient="records"),
        "summary_cards": summary_cards.to_dict(orient="records"),
    }

    with open(DATA_DIR / "dataframes.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_random_samples(part: str, mode: str, n: int = 4):
    """Draw annotated random samples and save a grid for HTML consumption."""
    img_dir = DATASET_ROOT / part / mode / "images"
    gt_dir = DATASET_ROOT / part / mode / "ground-truth"
    files = sorted(list(img_dir.glob("*.jpg")))
    if not files:
        raise FileNotFoundError(f"No images found at {img_dir}")

    picks = random.sample(files, min(n, len(files)))
    cols = 2
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = np.array(axes).reshape(rows, cols)

    for ax, img_path in zip(axes.flatten(), picks):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_path = gt_dir / f"GT_{img_path.stem}.mat"
        mat = loadmat(str(gt_path))
        pts = mat["image_info"][0, 0][0, 0][0]
        ax.imshow(img)
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=10, edgecolors="red", facecolors="none")
        ax.set_title(f"{img_path.name} | crowd={len(pts)}", fontsize=8)
        ax.axis("off")

    for ax in axes.flatten()[len(picks) :]:
        ax.axis("off")

    fig.suptitle(f"Random Samples • {part} / {mode}", fontsize=14)
    fig.tight_layout()
    out_path = IMG_DIR / f"samples_{part}_{mode}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def create_distribution_plots(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df, x="people_count", hue="part", element="step", stat="density", ax=ax
    )
    ax.set_title("Crowd Count Distribution by Dataset Part")
    ax.set_xlabel("People per Image")
    ax.set_ylabel("Density")
    hist_path = IMG_DIR / "distribution_hist.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="part", y="people_count", ax=ax2)
    ax2.set_title("Crowd Count Spread (Boxplot)")
    box_path = IMG_DIR / "distribution_box.png"
    fig2.tight_layout()
    fig2.savefig(box_path, dpi=160)
    plt.close(fig2)

    return hist_path.name, box_path.name


def create_density_example():
    img_dir = DATASET_ROOT / "part_A" / "train_data" / "images"
    img_files = sorted(list(img_dir.glob("*.jpg")))
    if not img_files:
        raise FileNotFoundError(f"No train images found at {img_dir}")
    img_path = img_files[0]
    gt_path = (
        DATASET_ROOT / "part_A" / "train_data" / "ground-truth" / f"GT_{img_path.stem}.mat"
    )
    mat = loadmat(str(gt_path))
    pts = mat["image_info"][0, 0][0, 0][0]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    density = generate_density_map(img.shape[:2], pts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"Original ({len(pts)} people)")
    axes[0].axis("off")

    im = axes[1].imshow(density, cmap="turbo")
    axes[1].set_title(f"Density Map (sum≈{density.sum():.0f})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Estimated crowd density")

    fig.tight_layout()
    out_path = IMG_DIR / "density_demo.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def persist_meta(df: pd.DataFrame, assets):
    summary = {
        "total_images": int(len(df)),
        "avg_people_per_image": round(float(df.people_count.mean()), 2),
        "max_people_detected": int(df.people_count.max()),
        "min_people_detected": int(df.people_count.min()),
        "assets": assets,
    }
    with open(DATA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    ensure_paths()
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset missing at {DATASET_ROOT}")

    print("Building dataset summary (this can take a minute)...")
    df = summarize_dataset(str(DATASET_ROOT))
    serialize_dataframes(df)

    assets = {
        "random_samples": [],
        "distribution": {},
        "density_map": "",
    }

    print("Saving random sample grids...")
    for part in ("part_A", "part_B"):
        asset_name = create_random_samples(part, "train_data")
        assets["random_samples"].append(asset_name)

    print("Saving distribution plots...")
    hist, box = create_distribution_plots(df)
    assets["distribution"]["hist"] = hist
    assets["distribution"]["box"] = box

    print("Creating density demo...")
    assets["density_map"] = create_density_example()

    persist_meta(df, assets)
    print(f"EDA assets saved to {ASSET_ROOT}")


if __name__ == "__main__":
    main()

