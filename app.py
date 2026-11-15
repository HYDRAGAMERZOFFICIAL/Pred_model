import os
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import streamlit as st
import rasterio
from skimage.filters import threshold_otsu
import tensorflow as tf

from src.data_processor import LakeDataProcessor, resolve_data_dir, resolve_dataset_csv
from src.predictor import predict_next_season, predict_lake_for_date, resolve_model_artifacts, map_month_to_season_code
from src.forecasting import forecast_annual_water_area
from src.spectrum_to_ndwi import SpectrumNDWIProcessor
import io

st.set_page_config(
    page_title="Lake Intelligence | Geospatial Analysis",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
}
h1, h2, h3, h4 {
    font-weight: 700 !important;
}
[data-testid="stMetric"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    background-color: #ffffff;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.title("Lake Intelligence: Geospatial Analysis & Forecasting")
st.markdown(
    "Satellite-derived water extent, spectral diagnostics and temporal forecasting for Jakkur Lake. "
    "Explore historical scenes, review spatial measurements and generate deep-learning forecasts."
)
st.divider()

processor = LakeDataProcessor()
dataset_dir = processor.data_dir
tif_files = processor.tif_files
csv_path = resolve_dataset_csv(dataset_dir)

# Standardized scientific cartographic palettes
cmap_water_binary = ListedColormap(['#e2e8f0', '#0284c7'])
cmap_depth_4class = ListedColormap(['#e2e8f0', '#7dd3fc', '#0284c7', '#0c4a6e'])

with st.sidebar.expander("Project setup and model readiness", expanded=False):
    csv_ready = os.path.isfile(csv_path)
    st.write(f"Dataset folder: **`{dataset_dir}`**")
    st.write(f"NDWI scenes detected: **{len(tif_files)}**")
    st.write(f"Pixel-label CSV: **{'Found' if csv_ready else 'Missing'}**")
    ready_model, ready_report, ready_source = resolve_model_artifacts("knowledge_base")
    if os.path.isfile(ready_model) and os.path.isfile(ready_report):
        try:
            with open(ready_report, encoding="utf-8") as stream:
                readiness = json.load(stream)
            if readiness.get("deployment_eligible", False):
                st.success("A validated model is available.")
            else:
                st.warning("A trained candidate is available for experimental preview only; it has not passed validation.")
            st.caption(f"Model source: {ready_source}")
            st.caption(f"Input schema: {readiness.get('input_channels', 'legacy')} channels")
        except (OSError, ValueError):
            st.warning("A model report exists but could not be read. Review the latest training run.")
    else:
        st.warning("No trained model artifact was found. Complete audit and training before using seasonal prediction.")
        st.code(
            ".\\.venv-dml\\Scripts\\python.exe -m src.dataset_audit\n"
            ".\\.venv-dml\\Scripts\\python.exe -m src.trainer --mode full --epochs 50",
            language="powershell",
        )
    st.markdown("[Open the setup and training guide](https://github.com/HYDRAGAMERZOFFICIAL/Pred_model/blob/main/SETUP_AND_RUN.md)")

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

def map_month_to_season(m):
    m_lower = m.lower()
    if m_lower in ["january", "february", "march", "april", "jan", "feb", "mar", "apr"]:
        return "JAN_APR", "Jan - Apr (Dry / Pre-Monsoon)"
    elif m_lower in ["may", "june", "july", "august", "jun", "jul", "aug"]:
        return "MAY_AUG", "May - Aug (Monsoon)"
    else:
        return "SEP_DEC", "Sep - Dec (Post-Monsoon)"

def get_base_raster_for_selection(year_str, month_str):
    code, _ = map_month_to_season(month_str)
    yr = str(year_str)[-2:]
    matches = [f for f in tif_files if code in os.path.basename(f) and yr in os.path.basename(f)]
    if matches:
        return matches[0]
    season_matches = [f for f in tif_files if code in os.path.basename(f)]
    if season_matches:
        return season_matches[0]
    return tif_files[0] if tif_files else None

# Sidebar Controls for Map Selection
st.sidebar.header("Scene explorer")
side_lake = st.sidebar.selectbox("Lake / Site", ["Jakkur Lake"], key="side_lake")
st.sidebar.caption("Only Jakkur Lake is represented by the committed raster archive. Other sites can be enabled when their scenes are added.")

# Dynamically discover all years present in Jakkur_lake_datasets
available_years = set()
for f in tif_files:
    m = re.search(r"(?:20\d{2}|(?<=_)\d{2})(?=[_.-]|$)", os.path.basename(f))
    if m:
        yr = m.group()
        if len(yr) == 2:
            yr = f"20{yr}"
        available_years.add(yr)
years_list = sorted(list(available_years)) if available_years else ["2022", "2023", "2024", "2025"]

side_year = st.sidebar.selectbox("Year", years_list, index=len(years_list) - 1, key="side_year")
side_month = st.sidebar.selectbox("Month", months, index=0, key="side_month")

current_tif = get_base_raster_for_selection(side_year, side_month)
filename = os.path.basename(current_tif) if current_tif else "N/A"
st.sidebar.caption(f"Active raster: `{filename}`")

if current_tif and os.path.exists(current_tif):
    arr, pixel_area_m2, crs = processor.load_tif(current_tif, apply_lake_boundary=False)
    thresh, binary_water, water_pixels, total_valid = processor.compute_otsu_threshold(arr)
    depth_map = processor.compute_depth_map(arr, binary_water)
    metrics = processor.compute_water_metrics(arr, binary_water, pixel_area_m2)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Raster analysis",
        "Area analytics",
        "Forecasting",
        "Spectral processing"
    ])

    with tab1:
        st.subheader(f"Analysis for {side_lake} — {side_month} {side_year}")
        st.caption("Spatial analysis is clipped to the reviewed Jakkur Lake footprint. Surrounding vegetation, buildings and the two internal land islands are excluded.")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. Original NDWI Raster")
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            im1 = ax1.imshow(arr, cmap='viridis')
            ax1.set_title(f"NDWI: {side_lake}")
            fig1.colorbar(im1, ax=ax1, label='NDWI Value')
            st.pyplot(fig1)

            st.markdown("### 2. Grayscale Normalized Image")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            norm_arr = np.nan_to_num(arr)
            norm_arr = (norm_arr - norm_arr.min()) / (norm_arr.max() - norm_arr.min() + 1e-8)
            ax2.imshow(norm_arr, cmap='gray')
            ax2.set_title("Normalized Grayscale")
            st.pyplot(fig2)

            st.markdown("### 5. Delineated Water Extent (Binary Mask)")
            fig_bin, ax_bin = plt.subplots(figsize=(6, 4.5))
            ax_bin.imshow(binary_water, cmap=cmap_water_binary, vmin=0, vmax=1)
            patches_bin = [
                mpatches.Patch(facecolor='#e2e8f0', edgecolor='#94a3b8', label=f'Land ({total_valid - water_pixels:,} px)'),
                mpatches.Patch(facecolor='#0284c7', edgecolor='#0369a1', label=f'Water ({water_pixels:,} px)'),
            ]
            ax_bin.legend(handles=patches_bin, loc='lower right', framealpha=0.9, fontsize=8)
            ax_bin.set_title("Binary Water Extent")
            st.pyplot(fig_bin)

        with col2:
            st.markdown("### 3. Relative 3D Depth Map")
            fig3, ax3 = plt.subplots(figsize=(6, 4.5))
            im3 = ax3.imshow(depth_map, cmap='jet', vmin=0, vmax=3)
            ax3.set_title("Depth Classes (0:Land, 1:Shallow, 2:Mod, 3:Deep)")
            fig3.colorbar(im3, ax=ax3, label='Depth Class')
            st.pyplot(fig3)

            st.markdown("### 4. Spectral Image (Pseudo-Color)")
            fig4, ax4 = plt.subplots(figsize=(6, 4.5))
            im4 = ax4.imshow(arr, cmap='inferno')
            ax4.set_title("Spectral Intensity Map")
            fig4.colorbar(im4, ax=ax4, label='Intensity')
            st.pyplot(fig4)

            st.markdown("### 6. NDWI Pixel Histogram & Otsu Threshold")
            fig5, ax5 = plt.subplots(figsize=(6, 4.5))
            valid_vals = arr[~np.isnan(arr)]
            ax5.hist(valid_vals.ravel(), bins=50, color='#1f77b4', alpha=0.8, edgecolor='black')
            ax5.axvline(thresh, color='red', linestyle='--', linewidth=2.5, label=f'Otsu Threshold: {thresh:.4f}')
            ax5.set_title("NDWI Histogram with Otsu Line")
            ax5.set_xlabel("NDWI Value")
            ax5.set_ylabel("Pixel Frequency")
            ax5.legend()
            st.pyplot(fig5)

        st.markdown("### Threshold statistics")
        st.markdown(f"""
        - **Otsu Threshold Value:** `{thresh:.4f}`
        - **Total Valid Pixels:** `{total_valid:,}`
        - **Water Pixels:** `{water_pixels:,}` ({(water_pixels / total_valid) * 100:.2f}%)
        - **Land Pixels:** `{total_valid - water_pixels:,}` ({(1 - water_pixels / total_valid) * 100:.2f}%)
        - **Water Coverage:** `{(water_pixels / total_valid) * 100:.2f}%`
        - **Water Surface Area:** `{metrics['area_km2']:.4f} km²` ({metrics['area_m2']:,.0f} m²)
        """)

    with tab2:
        st.subheader("Water Surface Area & Volumetric Analytics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Water Surface Area", f"{metrics['area_km2']:.4f} km²", f"{metrics['area_m2']:,.0f} m²")
        m2.metric("Water Volume", "Not available", help="NDWI is not bathymetry. Supply an aligned measured-depth raster to calculate volume.")
        m3.metric("Spatial Resolution", f"{np.sqrt(pixel_area_m2):.2f} m/pixel", f"CRS: {crs}")

        st.markdown("### Comprehensive Summary Table")
        _, season_desc = map_month_to_season(side_month)
        summary_df = pd.DataFrame({
            "Parameter": [
                "Lake Name", "Year", "Month", "Season Category", "Raster Scene", "Otsu Threshold",
                "Total Water Pixels", "Surface Area (km²)", "Depth / Volume"
            ],
            "Value": [
                side_lake, side_year, side_month, season_desc, filename, f"{thresh:.4f}",
                f"{water_pixels:,}", f"{metrics['area_km2']:.4f} km²",
                "Requires measured depth raster"
            ]
        })
        st.table(summary_df)

    with tab3:
        st.subheader("Seasonal and Annual Forecasts")
        st.caption("Forecasts are derived from the dated NDWI archive. Seasonal maps come from the trained multi-task U-Net; annual area is a separate statistical trend projection.")

        st.markdown("### 1. Spatial Lake Prediction & Forecasting")
        st.caption("Select a year and month to generate deep-learning spatial predictions for Jakkur Lake.")

        model_path, model_report_path, model_source = resolve_model_artifacts("knowledge_base")
        model_is_eligible = False
        if os.path.isfile(model_report_path):
            with open(model_report_path, encoding="utf-8") as model_report_file:
                model_is_eligible = bool(json.load(model_report_file).get("deployment_eligible", False))
        model_available = os.path.isfile(model_path) and os.path.isfile(model_report_path)
        if model_available:
            if not model_is_eligible:
                st.warning("Experimental prediction mode is enabled. The available model has not passed the persistence-baseline validation gate.")
                st.caption("You can generate and inspect maps, but treat them as research previews—not validated forecasts or decision-grade outputs.")
            if model_source == "versioned_candidate_preview":
                st.info("No validated active model is available yet. The latest versioned training candidate is loaded for experimental preview only.")

            # Year & Month Dropdown Selectors
            col_target_yr, col_target_mo = st.columns(2)
            with col_target_yr:
                target_forecast_year = st.selectbox(
                    "Forecast Year",
                    [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
                    index=3,  # default to 2026
                    key="target_forecast_year",
                    help="Choose a future year (2026+) for multi-step projection, or a historical year (2023-2025) to backcast and validate against satellite ground truth."
                )
            with col_target_mo:
                target_forecast_month = st.selectbox(
                    "Forecast Month",
                    months,
                    index=2,  # default to March
                    key="target_forecast_month",
                    help="The month is mapped to its corresponding 4-month hydrological period (Jan-Apr, May-Aug, Sep-Dec)."
                )

            t_code, _, t_desc = map_month_to_season_code(target_forecast_month)
            if target_forecast_year > 2025:
                st.caption(f"Target Selection: **{target_forecast_month} {target_forecast_year}** ({t_code} · {t_desc}) — Future Autoregressive Projection")
            else:
                st.caption(f"Target Selection: **{target_forecast_month} {target_forecast_year}** ({t_code} · {t_desc}) — Historical Backcast & Validation")

            button_label = f"Predict Lake for {target_forecast_month} {target_forecast_year}"
            if st.button(button_label, type="primary", key="run_seasonal_forecast"):
                try:
                    with st.spinner(f"Computing spatial U-Net forecast for {target_forecast_month} {target_forecast_year}..."):
                        forecast_result = predict_lake_for_date(
                            target_year=target_forecast_year,
                            target_month=target_forecast_month,
                            data_dir=dataset_dir,
                            model_dir="knowledge_base",
                            output_dir="outputs",
                            dataset_csv=csv_path,
                            allow_unvalidated=True,
                            model_path=model_path,
                            metadata_path=model_report_path,
                        )
                    st.session_state["seasonal_forecast_result"] = forecast_result
                except Exception as exc:
                    st.error(f"Prediction failed for {target_forecast_month} {target_forecast_year}: {exc}")

            result = st.session_state.get("seasonal_forecast_result")
            if result:
                if result.get("model_validation_status") == "experimental_unvalidated":
                    st.warning("Experimental output: this model has not passed the persistence-baseline validation gate.")
                else:
                    st.success(f"Spatial prediction generated for {result.get('predicted_for', 'selected period')}.")

                if result.get("steps_ahead", 0) > 0:
                    st.info(f"Forecast Horizon: **{result['steps_ahead']} seasonal step(s) ahead** from the latest observed baseline.")
                elif result.get("ground_truth_available"):
                    st.success("Historical satellite ground-truth observation is available for this period. Validation comparison shown below.")

                st.write("Input baselines used:", " → ".join(result["input_scenes"]))

                # Metrics row
                if result.get("ground_truth_available") and result.get("observed_water_area_km2") is not None:
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Predicted Water Area", f'{result["water_area_km2"]:.4f} km²', f'{result["water_pixels"]:,} px')
                    mcol2.metric("Observed Satellite Area", f'{result["observed_water_area_km2"]:.4f} km²', f'{result["observed_water_pixels"]:,} px')
                    mcol3.metric("Spatial IoU Agreement", f'{result["iou"] * 100:.2f}%', help="Intersection over Union between predicted and observed water bodies.")
                    mcol4.metric("Pixel Accuracy", f'{result["accuracy"] * 100:.2f}%')
                else:
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Predicted Water Pixels", f'{result["water_pixels"]:,}')
                    if result["water_area_km2"] is not None:
                        mcol2.metric("Predicted Water Area", f'{result["water_area_km2"]:.5f} km²')
                    else:
                        mcol2.metric("Predicted Water Area", "CRS area unavailable")
                    mcol3.metric("Valid Scene Pixels", f'{result["valid_pixels"]:,}')
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
                
                # 1. Predicted Water Probability
                im0 = axes[0].imshow(result["water_probability"], cmap="Blues", vmin=0.0, vmax=1.0)
                prob_data = np.nan_to_num(result["water_probability"], nan=0.0)
                if np.any(prob_data >= 0.5):
                    axes[0].contour(prob_data, levels=[0.5], colors=['#dc2626'], linewidths=1.6, linestyles='--')
                    axes[0].plot([], [], color='#dc2626', linestyle='--', linewidth=1.6, label='Decision Boundary (p ≥ 0.50)')
                    axes[0].legend(loc='lower right', framealpha=0.92, fontsize=7.5)
                axes[0].set_title("1. Water Occurrence Probability", fontsize=10, fontweight='bold', pad=8)
                cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                cb0.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
                cb0.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
                cb0.ax.tick_params(labelsize=8)

                # 2. Predicted Water Extent
                pred_water_binary = (result["water_probability"] >= 0.5).astype(np.uint8)
                axes[1].imshow(pred_water_binary, cmap=cmap_water_binary, vmin=0, vmax=1)
                patches_pred_binary = [
                    mpatches.Patch(facecolor='#e2e8f0', edgecolor='#94a3b8', label=f'Land ({result["valid_pixels"] - result["water_pixels"]:,} px)'),
                    mpatches.Patch(facecolor='#0284c7', edgecolor='#0369a1', label=f'Predicted Water ({result["water_pixels"]:,} px)'),
                ]
                axes[1].legend(handles=patches_pred_binary, loc='lower right', framealpha=0.92, fontsize=7.5)
                axes[1].set_title("2. Delineated Water Extent", fontsize=10, fontweight='bold', pad=8)

                # 3. Predicted Relative Classes
                axes[2].imshow(result["relative_classes"], cmap=cmap_depth_4class, vmin=0, vmax=3)
                patches_pred_classes = [
                    mpatches.Patch(facecolor='#e2e8f0', edgecolor='#94a3b8', label='0: Land / Non-Water'),
                    mpatches.Patch(facecolor='#7dd3fc', edgecolor='#38bdf8', label='1: Shallow Water'),
                    mpatches.Patch(facecolor='#0284c7', edgecolor='#0284c7', label='2: Moderate Depth'),
                    mpatches.Patch(facecolor='#0c4a6e', edgecolor='#082f49', label='3: Deep Water'),
                ]
                axes[2].legend(handles=patches_pred_classes, loc='lower right', framealpha=0.92, fontsize=7.5)
                axes[2].set_title("3. Predicted Relative Classes", fontsize=10, fontweight='bold', pad=8)

                for axis in axes:
                    axis.set_xlabel("Pixel X", fontsize=8)
                    axis.set_ylabel("Pixel Y", fontsize=8)
                    axis.tick_params(labelsize=7.5)
                plt.tight_layout()
                st.pyplot(fig)

                buf_pred = io.BytesIO()
                fig.savefig(buf_pred, format="png", dpi=200, bbox_inches="tight")
                file_tag = f"{result.get('target_year', 'season')}_{result.get('period_code', 'forecast')}"
                st.download_button(
                    f"Download high-resolution prediction map ({file_tag}.png, 200 DPI)",
                    data=buf_pred.getvalue(),
                    file_name=f"lake_prediction_{file_tag}_report.png",
                    mime="image/png",
                )

                with st.expander("Export prediction GeoTIFFs and individual raster products", expanded=False):
                    st.caption(f"Raw raster products for {result.get('predicted_for', 'forecast')} (97×162 px) for GIS software integration (QGIS, ArcGIS).")
                    gcol1, gcol2 = st.columns(2)
                    with gcol1:
                        st.download_button(
                            "Download water probability GeoTIFF",
                            data=open(result["water_probability_path"], "rb").read(),
                            file_name=os.path.basename(result["water_probability_path"]),
                            mime="image/tiff",
                        )
                    with gcol2:
                        st.download_button(
                            "Download relative class GeoTIFF",
                            data=open(result["relative_class_path"], "rb").read(),
                            file_name=os.path.basename(result["relative_class_path"]),
                            mime="image/tiff",
                        )
                    pcol1, pcol2, pcol3 = st.columns(3)
                    with pcol1:
                        st.caption("Probability raster (97×162 px)")
                        with open(result["water_probability_png_path"], "rb") as image_file:
                            st.download_button("Download probability PNG", image_file.read(), file_name=os.path.basename(result["water_probability_png_path"]), mime="image/png")
                    with pcol2:
                        st.caption("Extent raster (97×162 px)")
                        with open(result["water_extent_png_path"], "rb") as image_file:
                            st.download_button("Download water extent PNG", image_file.read(), file_name=os.path.basename(result["water_extent_png_path"]), mime="image/png")
                    with pcol3:
                        st.caption("Class raster (97×162 px)")
                        with open(result["relative_class_png_path"], "rb") as image_file:
                            st.download_button("Download class map PNG", image_file.read(), file_name=os.path.basename(result["relative_class_png_path"]), mime="image/png")
            else:
                st.info("The model is available. Generate a forecast to create georeferenced GeoTIFF outputs.")
        else:
            st.warning("No trained model or versioned candidate was found. Run the first training workflow from the guide.")
            st.code(
                "python -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --patch-size 64 --stride 32 --epochs 50",
                language="bash",
            )
            st.caption("Training uses every supplied CSV-labelled pixel and masks unlabelled pixels from the loss; Otsu/quantile fallback is used only when a scene has no usable CSV labels.")

        report_path = model_report_path if os.path.isfile(model_report_path) else os.path.join("knowledge_base", "multitask_training_report.json")
        if os.path.isfile(report_path):
            with st.expander("Model validation metrics and confusion matrices", expanded=False):
                with open(report_path, encoding="utf-8") as report_file:
                    training_report = json.load(report_file)
                st.caption("These scores compare predictions with supplied CSV labels or documented pseudo-labels in the chronological holdout. They are not independent field-validated accuracy.")
                st.write("Deployment eligible:", bool(training_report.get("deployment_eligible", False)))
                st.write("Input channels:", training_report.get("input_channels", "legacy"))
                if training_report.get("persistence_baseline"):
                    st.write("Persistence baseline:")
                    st.json(training_report["persistence_baseline"])
                water_metrics = training_report.get("water_validation_metrics", {})
                class_metrics = training_report.get("relative_class_validation_metrics", {})
                if water_metrics:
                    st.markdown("**Water / non-water validation metrics**")
                    st.json(water_metrics)
                    st.write("Confusion matrix (rows = actual, columns = predicted)")
                    st.dataframe(pd.DataFrame(
                        training_report.get("water_validation_confusion_matrix", []),
                        index=["Actual non-water", "Actual water"],
                        columns=["Predicted non-water", "Predicted water"],
                    ), width="stretch")
                if class_metrics:
                    st.markdown("**Relative NDWI class validation metrics**")
                    class_table = pd.DataFrame([
                        {"Class": name, **values}
                        for name, values in class_metrics.items()
                        if name.isdigit()
                    ]).replace({"Class": {"0": "Non-water", "1": "Shallow", "2": "Moderate", "3": "Deep"}})
                    st.dataframe(class_table, width="stretch", hide_index=True)
                    st.write("Confusion matrix (rows = actual, columns = predicted)")
                    st.dataframe(pd.DataFrame(
                        training_report.get("relative_class_validation_confusion_matrix", []),
                        index=["Actual non-water", "Actual shallow", "Actual moderate", "Actual deep"],
                        columns=["Predicted non-water", "Predicted shallow", "Predicted moderate", "Predicted deep"],
                    ), width="stretch")

        st.divider()
        st.markdown("### 2. Annual water-area projection")
        try:
            annual_result = forecast_annual_water_area(dataset_dir)
            annual_rows = annual_result["annual_history"]
            if annual_rows:
                annual_df = pd.DataFrame(annual_rows).rename(columns={
                    "year": "Year",
                    "mean_seasonal_water_area_km2": "Mean seasonal water area (km²)",
                    "scenes": "Scenes available",
                })
                st.dataframe(annual_df, width="stretch", hide_index=True)
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(
                    annual_df["Year"],
                    annual_df["Mean seasonal water area (km²)"],
                    marker="o",
                    linewidth=2,
                    label="Historical annual mean",
                )
                if annual_result["forecast"] is not None:
                    projection = annual_result["forecast"]
                    ax.scatter(
                        [projection["year"]],
                        [projection["mean_seasonal_water_area_km2"]],
                        marker="D",
                        s=75,
                        label="Annual trend projection",
                    )
                    low, high = projection["prediction_interval_95_km2"]
                    if low is not None and high is not None:
                        ax.vlines(projection["year"], low, high, linewidth=2, label="95% prediction interval")
                    st.metric(
                        f'{projection["year"]} projected annual mean area',
                        f'{projection["mean_seasonal_water_area_km2"]:.5f} km²',
                    )
                    st.caption(
                        f'Linear trend R²: {projection["r_squared"]:.3f}. '
                        f'Projection interval: {low:.5f}–{high:.5f} km².'
                    )
                    st.warning(projection["warning"])
                else:
                    st.info(annual_result.get("reason", "Insufficient annual observations for projection."))
                ax.set_xlabel("Year")
                ax.set_ylabel("Mean seasonal water area (km²)")
                ax.set_title("Historical and projected annual water area")
                ax.grid(alpha=0.25)
                ax.legend()
                st.pyplot(fig)
                st.download_button(
                    "Download annual projection data",
                    data=annual_df.to_csv(index=False).encode("utf-8"),
                    file_name="annual_water_area_history.csv",
                    mime="text/csv",
                )
            else:
                st.info("No dated seasonal NDWI rasters were found for annual analysis.")
        except Exception as exc:
            st.error(f"Annual projection could not be calculated: {exc}")

        st.info(
            "Interpretation: seasonal U-Net outputs are predictions of algorithm-derived labels. "
            "Annual values are linear trend projections of observed annual mean seasonal area. "
            "Neither output is a physical water-depth forecast or a hydrological simulation."
        )

    with tab4:
        st.subheader("Multispectral processing")
        st.markdown("""
        Relative NDWI classes are spectral ranks, not physical depth measurements. Volume is not reported without measured bathymetry.

        Transform raw satellite spectrum GeoTIFF rasters (**Red, Green, Blue, NIR, SWIR**) into fully processed,
        georeferenced **NDWI GeoTIFF images** matching the project's multi-temporal raster standards.
        Features automatic **spatial alignment, reprojection, Otsu thresholding, 4-tier relative depth classification,
        and geospatial water-area calculation**. Physical volume requires measured bathymetry or depth data.
        """)

        sp_processor = SpectrumNDWIProcessor()

        conv_col1, conv_col2 = st.columns([1, 1])

        with conv_col1:
            st.markdown("### 1. Select Spectral Input Source")
            input_mode = st.radio(
                "Input Mode",
                [
                    "Demonstration dataset (Jakkur Lake)",
                    "Local directory of spectral bands",
                    "Upload individual spectral bands (.tif)",
                    "Multi-band composite GeoTIFF (.tif)"
                ],
                key="conv_input_mode"
            )

            band_paths = {}
            multiband_file = None
            preset_choice = 'auto'

            if input_mode == "Demonstration dataset (Jakkur Lake)":
                demo_dir = "sample_spectral_data"
                if not os.path.exists(demo_dir) or not os.path.exists(os.path.join(demo_dir, "Jakkur_2022_B03_Green.tif")):
                    from scripts.create_sample_spectra import generate_sample_bands
                    generate_sample_bands()
                band_paths = {
                    'green': os.path.join(demo_dir, "Jakkur_2022_B03_Green.tif"),
                    'nir': os.path.join(demo_dir, "Jakkur_2022_B08_NIR.tif"),
                    'red': os.path.join(demo_dir, "Jakkur_2022_B04_Red.tif"),
                    'swir1': os.path.join(demo_dir, "Jakkur_2022_B11_SWIR_20m.tif"),
                    'blue': os.path.join(demo_dir, "Jakkur_2022_B02_Blue.tif")
                }
                st.info(f"Loaded 5 spectral bands from `{demo_dir}/` (Green 10m, NIR 10m, Red 10m, SWIR 20m, Blue 10m).")

            elif input_mode == "Local directory of spectral bands":
                target_dir = st.text_input("Local Folder Path", value="sample_spectral_data", key="local_dir_input")
                if os.path.exists(target_dir):
                    detected = sp_processor.auto_detect_bands_in_directory(target_dir)
                    st.success(f"Detected {len(detected)} spectral bands: {list(detected.keys())}")
                    band_paths = detected
                else:
                    st.error("Directory not found.")

            elif input_mode == "Upload individual spectral bands (.tif)":
                up_green = st.file_uploader("Green Band (.tif) [Required for NDWI/MNDWI]", type=['tif', 'tiff'], key="up_green")
                up_nir = st.file_uploader("NIR Band (.tif) [Required for NDWI/NDVI]", type=['tif', 'tiff'], key="up_nir")
                up_red = st.file_uploader("Red Band (.tif) [Optional / Required for NDVI/NDTI]", type=['tif', 'tiff'], key="up_red")
                up_swir = st.file_uploader("SWIR Band (.tif) [Required for MNDWI]", type=['tif', 'tiff'], key="up_swir")
                up_blue = st.file_uploader("Blue Band (.tif) [Optional for True-Color RGB]", type=['tif', 'tiff'], key="up_blue")

                temp_upload_dir = os.path.join("sample_spectral_data", "_uploads")
                os.makedirs(temp_upload_dir, exist_ok=True)

                def save_uploaded(up_file, name):
                    if up_file is not None:
                        fpath = os.path.join(temp_upload_dir, f"{name}.tif")
                        with open(fpath, "wb") as f:
                            f.write(up_file.getbuffer())
                        return fpath
                    return None

                if up_green: band_paths['green'] = save_uploaded(up_green, "custom_green")
                if up_nir: band_paths['nir'] = save_uploaded(up_nir, "custom_nir")
                if up_red: band_paths['red'] = save_uploaded(up_red, "custom_red")
                if up_swir: band_paths['swir1'] = save_uploaded(up_swir, "custom_swir")
                if up_blue: band_paths['blue'] = save_uploaded(up_blue, "custom_blue")

            elif input_mode == "Multi-band composite GeoTIFF (.tif)":
                mb_choice = st.radio("Multi-Band Source", ["Use Demo 4-Band PlanetScope Composite", "Upload Multi-Band .tif"], key="mb_choice")
                if mb_choice == "Use Demo 4-Band PlanetScope Composite":
                    multiband_file = os.path.join("sample_spectral_data", "Jakkur_2022_4Band_PlanetScope_Composite.tif")
                    preset_choice = 'planetscope'
                else:
                    up_mb = st.file_uploader("Upload Multi-band GeoTIFF", type=['tif', 'tiff'], key="up_mb")
                    if up_mb:
                        multiband_file = os.path.join("sample_spectral_data", "_uploads", "custom_multiband.tif")
                        os.makedirs(os.path.dirname(multiband_file), exist_ok=True)
                        with open(multiband_file, "wb") as f:
                            f.write(up_mb.getbuffer())
                    preset_choice = st.selectbox(
                        "Band Layout Preset",
                        ["auto", *sp_processor.PRESETS.keys()],
                        index=0,
                        key="preset_choice"
                    )

        with conv_col2:
            st.markdown("### 2. Spectral Algorithm & Parameters")
            idx_map = {
                "NDWI (McFeeters: Green - NIR)": "NDWI",
                "MNDWI (Xu: Green - SWIR - Urban/Turbid Suppression)": "MNDWI",
                "NDVI (Rouse: NIR - Red - Vegetation Delineation)": "NDVI",
                "NDTI (Lacaux: Red - Green - Turbidity Index)": "NDTI",
                "WRI (Water Ratio Index: (Green+Red)/(NIR+SWIR))": "WRI"
            }
            selected_idx_label = st.selectbox("Spectral Water / Vegetation Index", list(idx_map.keys()), index=0, key="selected_idx")
            selected_index = idx_map[selected_idx_label]

            thresh_mode = st.selectbox(
                "Water Classification Threshold",
                ["Dynamic Otsu Thresholding (Recommended)", "Standard NDWI > 0.0 (Zero Cutoff)", "Custom Numeric Threshold"],
                key="thresh_mode"
            )
            if thresh_mode == "Standard NDWI > 0.0 (Zero Cutoff)":
                thresh_val_arg = 'zero'
            elif thresh_mode == "Custom Numeric Threshold":
                thresh_val_arg = st.slider("Cutoff Value", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key="cust_thresh")
            else:
                thresh_val_arg = 'otsu'

            apply_morph = st.checkbox("Apply Morphological Filter (Patch cleanup & hole filling)", value=True, key="apply_morph")
            target_out_name = st.text_input("Output Raster Filename", value=f"NDWI_PROCESSED_{side_lake.split()[0].upper()}.tif", key="target_out_name")

            convert_button = st.button("Process spectral bands", key="conv_btn")

        if convert_button or 'last_conv_results' in st.session_state:
            if convert_button:
                try:
                    with st.spinner("Aligning spectral bands, resampling grids, and computing index..."):
                        if multiband_file and os.path.exists(multiband_file):
                            res = sp_processor.process_from_multiband(
                                multiband_path=multiband_file,
                                preset=preset_choice,
                                index_type=selected_index,
                                threshold_method=thresh_val_arg,
                                lake_boundary=False,
                            )
                        elif band_paths:
                            res = sp_processor.process_from_files(
                                band_paths=band_paths,
                                index_type=selected_index,
                                reference_band='green' if 'green' in band_paths else list(band_paths.keys())[0],
                                threshold_method=thresh_val_arg,
                                lake_boundary=False,
                            )
                        else:
                            st.error("Please supply valid spectral bands.")
                            res = None

                        if res:
                            st.session_state['last_conv_results'] = res
                            st.session_state['last_conv_out_name'] = target_out_name
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

            if 'last_conv_results' in st.session_state:
                res = st.session_state['last_conv_results']
                st.success(f"Processing Complete for index: **{res['index_name']}**!")

                m = res['metrics']
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("Otsu Water Cutoff", f"{res['otsu_threshold']:.4f}")
                rm2.metric("Water Surface Area", f"{m['area_km2']:.4f} km²", f"{m['area_m2']:,.0f} m²")
                rm3.metric("Water Volume", "Not available", help="Provide measured bathymetry/depth data to calculate physical volume.")
                rm4.metric("Water Coverage", f"{m['water_coverage_pct']:.2f}%", f"{m['water_pixels']:,} / {m['total_valid_pixels']:,} px")

                # Visual Panels (Row 1: Spectral Bands & Composites)
                st.markdown("### Visual diagnostics and georeferenced products")
                vcol1, vcol2 = st.columns(2)

                aligned = res.get('aligned_bands', {})
                valid_mask = res['valid_mask']
                ndwi_arr = res['index_raster']

                with vcol1:
                    st.markdown("#### 1. False-Color CIR / True-Color Composite")
                    fig_c, ax_c = plt.subplots(figsize=(6, 4.5))
                    if 'nir' in aligned and 'red' in aligned and 'green' in aligned:
                        cir = sp_processor.create_cir_false_color(aligned['nir'], aligned['red'], aligned['green'])
                        ax_c.imshow(cir)
                        ax_c.set_title("False Color CIR (NIR-Red-Green)")
                    elif 'red' in aligned and 'green' in aligned and 'blue' in aligned:
                        rgb = sp_processor.create_rgb_composite(aligned['red'], aligned['green'], aligned['blue'])
                        ax_c.imshow(rgb)
                        ax_c.set_title("True Color RGB Composite")
                    else:
                        first_k = list(aligned.keys())[0]
                        ax_c.imshow(aligned[first_k], cmap='viridis')
                        ax_c.set_title(f"Input Band: {first_k}")
                    ax_c.axis('off')
                    st.pyplot(fig_c)

                    st.markdown("#### 2. Processed NDWI GeoTIFF Heatmap")
                    fig_ndwi, ax_ndwi = plt.subplots(figsize=(6, 4.5))
                    masked_index = np.where(valid_mask, ndwi_arr, np.nan)
                    im_ndwi = ax_ndwi.imshow(masked_index, cmap='RdYlBu')
                    ax_ndwi.set_title(f"Processed {res['index_name']} Raster")
                    fig_ndwi.colorbar(im_ndwi, ax=ax_ndwi, label=res['index_name'])
                    ax_ndwi.axis('off')
                    st.pyplot(fig_ndwi)

                with vcol2:
                    st.markdown("#### 3. NDWI Histogram & Otsu Threshold Cutoff")
                    fig_h, ax_h = plt.subplots(figsize=(6, 4.5))
                    val_data = ndwi_arr[valid_mask]
                    ax_h.hist(val_data, bins=50, color='#1f77b4', alpha=0.8, edgecolor='black', density=True)
                    ax_h.axvline(res['otsu_threshold'], color='red', linestyle='--', linewidth=2.5,
                                 label=f"Otsu Cutoff: {res['otsu_threshold']:.4f}")
                    ax_h.set_title("NDWI Spectrum Histogram with Otsu Line")
                    ax_h.set_xlabel(res['index_name'])
                    ax_h.set_ylabel("Pixel Frequency")
                    ax_h.legend()
                    st.pyplot(fig_h)

                    st.markdown("#### 4. Relative Spectral Classification (Proxy Ranks)")
                    fig_d, ax_d = plt.subplots(figsize=(6, 4.5), dpi=140)
                    ax_d.imshow(res['depth_map'], cmap=cmap_depth_4class, vmin=0, vmax=3)
                    ax_d.set_title("Relative Spectral Classes (Not Bathymetry)", fontsize=10, fontweight='bold', pad=8)
                    patches_d = [
                        mpatches.Patch(facecolor='#e2e8f0', edgecolor='#94a3b8', label='0: Land / Non-Water'),
                        mpatches.Patch(facecolor='#7dd3fc', edgecolor='#38bdf8', label='1: Shallow Water'),
                        mpatches.Patch(facecolor='#0284c7', edgecolor='#0284c7', label='2: Moderate Depth'),
                        mpatches.Patch(facecolor='#0c4a6e', edgecolor='#082f49', label='3: Deep Water'),
                    ]
                    ax_d.legend(handles=patches_d, loc='lower right', framealpha=0.92, fontsize=8)
                    ax_d.set_xlabel("Pixel X", fontsize=8)
                    ax_d.set_ylabel("Pixel Y", fontsize=8)
                    ax_d.tick_params(labelsize=8)
                    st.pyplot(fig_d)

                # Export and Integration
                st.markdown("### Export and integrate results")
                save_col1, save_col2, save_col3 = st.columns(3)

                with save_col1:
                    save_name = st.session_state.get('last_conv_out_name', 'NDWI_NEW_CONVERTED.tif')
                    if not save_name.endswith(('.tif', '.tiff')):
                        save_name += '.tif'
                    if st.button(f"Save `{save_name}` to project root", key="save_to_root"):
                        out_fpath = os.path.join(".", save_name)
                        sp_processor.export_geotiff(
                            data_arr=res['index_raster'],
                            meta=res['reference_meta'],
                            output_path=out_fpath
                        )
                        st.success(f"Saved `{out_fpath}`! It is now permanently in your project library.")
                        st.rerun()

                with save_col2:
                    # GeoTIFF in-memory download
                    mem_tif_path = os.path.join("sample_spectral_data", "_temp_download.tif")
                    sp_processor.export_geotiff(
                        data_arr=res['index_raster'],
                        meta=res['reference_meta'],
                        output_path=mem_tif_path
                    )
                    with open(mem_tif_path, "rb") as f:
                        tif_bytes = f.read()
                    st.download_button(
                        label="Download processed NDWI GeoTIFF",
                        data=tif_bytes,
                        file_name=save_name,
                        mime="image/tiff",
                        key="dl_tif"
                    )

                with save_col3:
                    # Diagnostic figure download
                    diag_fig = sp_processor.generate_diagnostic_figure(res, title=f"{side_lake} Multi-Spectral Analysis")
                    buf = io.BytesIO()
                    diag_fig.savefig(buf, format="png", dpi=200, bbox_inches='tight')
                    st.download_button(
                        label="Download 8-panel analysis report (PNG)",
                        data=buf.getvalue(),
                        file_name="NDWI_Spectral_Analysis_Report.png",
                        mime="image/png",
                        key="dl_report"
                    )

