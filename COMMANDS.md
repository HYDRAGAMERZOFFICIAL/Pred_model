# Execution Command Reference

A complete, tested reference of all execution commands for **Windows Terminal (PowerShell)** and **macOS / Linux Terminal (Bash / Zsh)**.

---

## 1. Quick Reference Matrix

| Task | Windows (PowerShell) | macOS / Linux (Terminal) |
| :--- | :--- | :--- |
| **Activate Virtual Env** | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| **Run All Unit Tests** | `python -m unittest discover -s tests -v` | `python3 -m unittest discover -s tests -v` |
| **Audit Dataset Files** | `python -m src.dataset_audit` | `python3 -m src.dataset_audit` |
| **Ingest / Sync Archive** | `python -m src.knowledge_base` | `python3 -m src.knowledge_base` |
| **Fast Model Training** | `python -m src.trainer --mode full --patch-size 64 --stride 32 --epochs 50` | `python3 -m src.trainer --mode full --patch-size 64 --stride 32 --epochs 50` |
| **Gate-Passing Dense Training** | `python -m src.trainer --mode full --patch-size 64 --stride 16 --epochs 50` | `python3 -m src.trainer --mode full --patch-size 64 --stride 16 --epochs 50` |
| **Check Model Gate Status** | `python -c "import json; r=json.load(open('knowledge_base/multitask_training_report.json')); print('Eligible:', r.get('deployment_eligible'))"` | `python3 -c "import json; r=json.load(open('knowledge_base/multitask_training_report.json')); print('Eligible:', r.get('deployment_eligible'))"` |
| **Launch Streamlit Web UI** | `python -m streamlit run app.py` | `python3 -m streamlit run app.py` |
| **Generate Demo Spectra** | `python -m scripts.create_sample_spectra` | `python3 -m scripts.create_sample_spectra` |
| **Automated Pipeline** | `python -m src.pipeline --mode full --epochs 50` | `python3 -m src.pipeline --mode full --epochs 50` |

---

## 2. Windows Terminal (PowerShell)

### A. Environment Setup & Verification

```powershell
# 1. Verify Python installation (Python 3.10 - 3.12 recommended)
python --version

# 2. Navigate to project root
cd A:\Pred_model

# 3. Create a clean virtual environment (named .venv)
python -m venv .venv

# 4. If script execution is blocked in PowerShell, permit local scripts:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 5. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 6. Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# 7. Install project dependencies
pip install -r requirements.txt
```

> **Note**: Always use `python -m <module>` instead of calling `.exe` shortcuts directly to prevent Python path collision or broken launcher shims.

---

### B. Verification & Quality Assurance

```powershell
# Run the complete test suite (all 32 unit and integration tests)
python -m unittest discover -s tests -v

# Run dataset audit on Jakkur_lake_datasets/
python -m src.dataset_audit

# Ingest and archive rasters into the versioned knowledge base
python -m src.knowledge_base
```

---

### C. Model Training Workflows

#### Option 1: Fast Exploratory Training (Stride 32)
```powershell
python -m src.trainer --mode full --patch-size 64 --stride 32 --batch-size 32 --epochs 50
```

#### Option 2: Dense Patch Sampling (Stride 16 — Passes Persistence Baseline Gate)
```powershell
python -m src.trainer --mode full --patch-size 64 --stride 16 --batch-size 16 --epochs 50
```

#### Check Model Deployment Gate Status
```powershell
python -c "import json; r=json.load(open('knowledge_base/multitask_training_report.json')); print('Deployment eligible:', r.get('deployment_eligible', False)); print('Persistence gate:', r.get('persistence_baseline', {}))"
```

---

### D. Running the Streamlit Web Application

```powershell
# Standard launch (opens browser at http://localhost:8501)
python -m streamlit run app.py

# Launch on a custom port (if 8501 is busy)
python -m streamlit run app.py --server.port 8502

# Headless / remote server launch
python -m streamlit run app.py --server.headless true --server.port 8501
```

---

### E. Multispectral Tools & Automated Pipeline

```powershell
# Generate sample synthetic multispectral bands (B2, B3, B4, B8, B11) for testing Tab 4
python -m scripts.create_sample_spectra

# Run full automated lifecycle pipeline (audit -> ingest -> train -> forecast)
python -m src.pipeline --mode full --epochs 50
```

---

## 3. macOS & Linux Terminal (Bash / Zsh)

### A. Environment Setup & Verification

```bash
# 1. Verify Python 3 installation (Python 3.10 - 3.12 recommended)
python3 --version

# 2. Navigate to project root
cd /path/to/Pred_model

# 3. Create a clean virtual environment
python3 -m venv .venv

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Upgrade core packaging tools
pip install --upgrade pip setuptools wheel

# 6. Install project dependencies
pip install -r requirements.txt
```

> **macOS Note**: If `rasterio` or `GDAL` fails to compile, install GDAL via Homebrew first:
> ```bash
> brew install gdal
> pip install rasterio
> ```

---

### B. Verification & Quality Assurance

```bash
# Run the complete test suite (all 32 unit and integration tests)
python3 -m unittest discover -s tests -v

# Run dataset audit on Jakkur_lake_datasets/
python3 -m src.dataset_audit

# Ingest and archive rasters into the versioned knowledge base
python3 -m src.knowledge_base
```

---

### C. Model Training Workflows

#### Option 1: Fast Exploratory Training (Stride 32)
```bash
python3 -m src.trainer --mode full --patch-size 64 --stride 32 --batch-size 32 --epochs 50
```

#### Option 2: Dense Patch Sampling (Stride 16 — Passes Persistence Baseline Gate)
```bash
python3 -m src.trainer --mode full --patch-size 64 --stride 16 --batch-size 16 --epochs 50
```

#### Check Model Deployment Gate Status
```bash
python3 -c "import json; r=json.load(open('knowledge_base/multitask_training_report.json')); print('Deployment eligible:', r.get('deployment_eligible', False)); print('Persistence gate:', r.get('persistence_baseline', {}))"
```

---

### D. Running the Streamlit Web Application

```bash
# Standard launch (opens browser at http://localhost:8501)
python3 -m streamlit run app.py

# Launch on a custom port (if 8501 is busy)
python3 -m streamlit run app.py --server.port 8502

# Headless / remote server launch
python3 -m streamlit run app.py --server.headless true --server.port 8501
```

---

### E. Multispectral Tools & Automated Pipeline

```bash
# Generate sample synthetic multispectral bands for testing Tab 4
python3 -m scripts.create_sample_spectra

# Run full automated lifecycle pipeline (audit -> ingest -> train -> forecast)
python3 -m src.pipeline --mode full --epochs 50
```

---

## 4. Troubleshooting & FAQ

### Issue: `Fatal error in launcher: Unable to create process using ... streamlit.exe`
- **Cause**: The global or virtual environment executable shim has a hardcoded path pointing to a deleted or moved Python installation.
- **Resolution**: Do NOT invoke `streamlit run app.py`. Always use the active Python interpreter module call:
  ```powershell
  # Windows
  python -m streamlit run app.py
  # macOS / Linux
  python3 -m streamlit run app.py
  ```

### Issue: `Execution of scripts is disabled on this system` (PowerShell)
- **Cause**: Windows execution policy restricts running PowerShell activation scripts.
- **Resolution**: Execute this once in your current PowerShell session:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```

### Issue: Model shows "Experimental prediction mode / Not passed persistence validation"
- **Cause**: The persistence gate requires the trained U-Net to strictly beat a trivial baseline that predicts the current season unchanged. On small 97x162 scenes, `--stride 32` yields only ~45 training patches, which underfits boundaries.
- **Resolution**: Train with dense patch extraction:
  ```powershell
  python -m src.trainer --mode full --patch-size 64 --stride 16 --batch-size 16 --epochs 50
  ```
  This extracts >500 patches, improving boundary learning and satisfying the gate condition (`deployment_eligible: true`).

### Issue: "Water Volume: Not available" in the UI
- **Cause**: NDWI (Normalized Difference Water Index) is a surface spectral reflectance ratio `(Green - NIR) / (Green + NIR)`, not underwater bathymetry.
- **Resolution**: Surface water area ($km^2$) is computed with high precision from pixel counts and CRS affine resolution. True physical volume ($m^3$) requires an aligned bathymetric depth raster. If bathymetry is unavailable, volume is properly labeled "Not available" per scientific integrity standards.
