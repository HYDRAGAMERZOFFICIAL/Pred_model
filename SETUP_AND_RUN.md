# Setup and Run Guide

This guide covers installing the Lake Water Analysis application on Windows,
starting the Streamlit dashboard, and preparing data for analysis and
prediction.

## Requirements

- Windows 10 or 11 (64-bit)
- Python 3.10, 3.11, or 3.12
- Git, if cloning the repository
- Enough disk space for Python packages, satellite rasters, and model files

TensorFlow on native Windows generally runs on the CPU. GPU acceleration may
require WSL2 or a supported DirectML setup.

## Get the project

If you have not cloned the repository yet, run these commands in PowerShell:

```powershell
git clone https://github.com/HYDRAGAMERZOFFICIAL/Pred_model.git
Set-Location Pred_model
```

If you already have the project, open PowerShell in its root directory (the
directory containing `app.py` and `requirements.txt`).

## Create an environment and install dependencies

Create a virtual environment and install the project requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.12 is not installed, replace `-3.12` with an installed supported
version, such as `-3.11`.

## Start the dashboard

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the **Local URL** printed in the terminal, usually
`http://localhost:8501`. Keep the terminal open while using the app. To stop
the server, press `Ctrl+C` in that terminal.

## Prepare input data

### Dated NDWI scenes

Place NDWI GeoTIFFs in the project root. Filenames must contain a year and one
of these seasonal labels:

- `JAN_APR`
- `MAY_AUG`
- `SEP_DEC`

For example: `NDWI_JAN_APR_26.tif`. The app uses raster geospatial metadata;
scenes should have a valid CRS and compatible grids.

The project also includes a sample NDWI dataset. To generate or run temporal
predictions, use at least three dated scenes in a regular seasonal sequence.

### Spectral bands

In the spectral processing section:

- Choose **Upload Individual Spectral Bands** when providing separate band
  files. For NDWI, provide Green and NIR; other indices require their
  corresponding bands.
- Choose **Multi-Band Composite GeoTIFF** for a raster stack and leave the
  layout on **auto** when possible. If selecting a preset, make sure its band
  order matches the uploaded raster.
- A single-band raster cannot be used as a multi-band spectral input. Use the
  individual-band workflow and provide the bands required by the selected
  index.

## Train the seasonal model

Training is optional for general raster and spectral analysis. Seasonal
prediction requires a trained model and its training report under
`knowledge_base/`. From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --patch-size 64 --stride 32 --epochs 50
```

Training can take a long time and use substantial memory. The generated model
and report are local artifacts; back them up separately and do not commit them
to Git.

## Run tests

Run the test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- **`ModuleNotFoundError`**: Install requirements using the virtual
  environment's Python command shown above.
- **Seasonal prediction says the model is missing**: Train the model first and
  confirm the model and report files exist in `knowledge_base/`.
- **Band index is outside the raster's band range**: Use automatic detection
  or select a preset matching the raster. A single-band file cannot provide
  the multiple inputs required for a spectral index.
- **The app still behaves like an older version after code updates**: Stop
  Streamlit with `Ctrl+C` and start it again so Python reloads imported
  modules.
- **TensorFlow reports that GPU support is unavailable on Windows**: This
  usually means TensorFlow will use the CPU; it is not by itself a startup
  failure.

For interpretation limits and model behavior, see
[README.md](README.md) and
[docs/DATA_AND_MODEL_GUIDELINES.md](docs/DATA_AND_MODEL_GUIDELINES.md).


## Prediction image exports

After a successful seasonal prediction, the application creates three PNG previews in
`outputs/` in addition to the georeferenced GeoTIFFs:

- `next_season_water_probability.png`: continuous probability from 0 to 1.
- `next_season_water_extent.png`: binary predicted water / non-water map.
- `next_season_relative_classes.png`: four relative NDWI classes (0–3).

The PNGs are visualization products and do not retain GeoTIFF georeferencing.
Use the corresponding `.tif` files in GIS software when spatial coordinates,
CRS, and raster metadata are required.

## Adding future training data

Keep each new NDWI scene as a dated, single-band GeoTIFF with a valid CRS and
the same pixel grid as the existing time series. Use a filename containing
the year and one supported season token, for example
`NDWI_JAN_APR_26.tif`, `NDWI_MAY_AUG_26.tif`, or
`NDWI_SEP_DEC_26.tif`. Do not overwrite an older scene.

If independently reviewed pixel labels are available, append them to the CSV
with columns `image,row,column,water,depth_class`; use 0/1 for water and
0–3 for relative classes, with `water == (depth_class > 0)`. Keep the
`image` value equal to the GeoTIFF filename. Then run ingestion followed by
training:

```powershell
.\.venv\Scripts\python.exe -m src.knowledge_base --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv
.\.venv\Scripts\python.exe -m src.trainer --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv --mode both --patch-size 64 --stride 32 --epochs 50
```

The archive is append-only and content-addressed. Re-ingesting identical
GeoTIFF bytes does not duplicate a scene; new scenes and newly supplied pixel
labels are incorporated into the next training run. Training creates versioned
candidates and promotes a candidate only when it passes the configured
validation-loss comparison. Back up `knowledge_base/` separately because it
contains the archive, model versions, active model, and reports.


## Audit the NDWI dataset before training

Run the dataset audit before starting a training job:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir . --dataset-csv NDWI_pixel_depth_dataset.csv
```

The command writes `dataset_audit_report.json` and reports raster dimensions,
CRS, grid consistency, NoData coverage, missing seasonal periods, CSV label
coverage, pseudo-label fallback scenes, and usable consecutive temporal
windows. Resolve reported errors before training. Missing seasons are reported
as warnings because they interrupt temporal training windows.

For the current repository data, add the missing
`NDWI_SEP_DEC_24.tif` scene if it is available. The temporal pipeline skips
windows that cross that missing period; it does not invent an observation.


## Train using GitHub Actions (cloud CPU)

The repository includes a manual **Train NDWI model** workflow. Open the
GitHub repository, select **Actions → Train NDWI model → Run workflow**, choose
the training mode and epoch count, and start the run. The workflow installs
the pinned project requirements, audits the committed rasters and labels,
trains the temporal U-Net, and uploads the dataset audit plus versioned model
artifacts and reports as a downloadable workflow artifact.

This workflow does not commit trained weights to Git. Download and back up the
artifact after the run. Review `training_report.json` and the label provenance
before interpreting validation metrics. If the audit reports missing seasonal
periods, the affected temporal windows are skipped; add the missing source
scene before expecting a continuous seasonal history.


### Local files versus GitHub Actions training

The repository's ignore rules intentionally keep large `.tif`/`.tiff` and
`.csv` files out of Git by default. Local training still discovers dated NDWI
rasters placed in the project root. The local command also keeps the archive
and active model in `knowledge_base/` between runs, enabling incremental
fine-tuning after new scenes are added.

The GitHub Actions training workflow can only see files committed to the
repository. If you choose cloud training for newly acquired scenes, explicitly
track and commit those input files (or use Git LFS / external object storage
for larger scenes). The workflow caches `knowledge_base/` between runs so
subsequent runs can restore the prior archive and active model before ingesting
new scenes. Download the uploaded run artifact as a separate backup.
