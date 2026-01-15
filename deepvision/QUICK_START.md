# Quick Start Guide - DeepVision Crowd Monitor

## 🎯 Quick Checklist

### Initial Setup (Do Once)
1. ✅ Create directory structure:
   ```
   deepvision/web/assets/data/
   deepvision/web/assets/images/
   deepvision/scripts/
   ```

2. ✅ Create `deepvision/scripts/build_eda_assets.py`
   - 206 lines
   - Imports from `Eda.basic_stats` and `Eda.density_map_demo`
   - Generates CSV, JSON, PNG files

3. ✅ Create `deepvision/streamlit_app.py`
   - 198 lines
   - 3 tabs: Overview, EDA Explorer, Live Monitor
   - Uses `use_container_width=True` (not deprecated)

4. ✅ Update `deepvision/requirements.txt`
   - Add `streamlit` line

5. ✅ Install dependencies:
   ```bash
   cd deepvision
   .\venv\Scripts\activate
   python -m pip install streamlit seaborn pandas pillow scipy tqdm
   ```

6. ✅ Generate assets:
   ```bash
   python scripts\build_eda_assets.py
   ```

7. ✅ Run dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📋 Files Created/Modified

### New Files:
- ✅ `deepvision/streamlit_app.py`
- ✅ `deepvision/scripts/build_eda_assets.py`
- ✅ `deepvision/web/assets/data/*.csv, *.json`
- ✅ `deepvision/web/assets/images/*.png`

### Modified Files:
- ✅ `deepvision/requirements.txt` (added streamlit)

---

## 🔑 Key Code Sections

### `streamlit_app.py` - Main Sections:
- Lines 11-14: Path definitions
- Lines 24-31: `load_meta()` function
- Lines 34-49: `load_dataframes()` function
- Lines 58-93: `render_overview_tab()`
- Lines 96-127: `render_eda_tab()`
- Lines 130-136: `estimate_density_naive()`
- Lines 139-176: `render_live_tab()`

### `build_eda_assets.py` - Main Sections:
- Lines 32-34: `ensure_paths()` - Creates directories
- Lines 37-64: `serialize_dataframes()` - Creates CSV/JSON
- Lines 67-101: `create_random_samples()` - Generates sample grids
- Lines 104-125: `create_distribution_plots()` - Histogram/boxplot
- Lines 128-156: `create_density_example()` - Density map
- Lines 171-200: `main()` - Orchestrates everything

---

## ⚠️ Common Issues & Fixes

1. **Path Errors:** Ensure `streamlit_app.py` is in `deepvision/` directory
2. **Missing Assets:** Run `python scripts\build_eda_assets.py` first
3. **Deprecation Warnings:** Use `use_container_width=True` not `use_column_width=True`
4. **Import Errors:** Activate venv: `.\venv\Scripts\activate`

---

## 🧪 Test Commands

```bash
# Test syntax
python -m py_compile streamlit_app.py
python -m py_compile scripts/build_eda_assets.py

# Test imports
python -c "import streamlit_app; print('OK')"

# Test paths
python -c "from pathlib import Path; import streamlit_app; print(streamlit_app.ASSET_DIR.exists())"
```

---

## 📊 Generated Assets (Must Exist)

**Data Files:**
- `web/assets/data/meta.json`
- `web/assets/data/dataframes.json`
- `web/assets/data/crowd_dataset_overview.csv`

**Images:**
- `web/assets/images/samples_part_A_train_data.png`
- `web/assets/images/samples_part_B_train_data.png`
- `web/assets/images/distribution_hist.png`
- `web/assets/images/distribution_box.png`
- `web/assets/images/density_demo.png`

---

**Status:** ✅ Complete and tested

