# Copilot Instructions for 2025_atp1a3a

This repository contains investigation code for the project: "Focalized forebrain dysregulation and widespread behavioural abnormalities in a larval zebrafish model of ATP1A3-related disorders."

## 🧠 Project Architecture & Context
This is a scientific data analysis codebase linking raw experimental data to a LaTeX manuscript.
- **Domain:** Neuroscience, Zebrafish Behavior, Calcium Imaging (2-Photon).
- **Core Components:**
    - `Analyze_atp1a3a.py` & `FishTrack.py`: Behavioral data analysis pipelines.
    - `Ca2+Imaging/`: Dedicated folder for 2-Photon imaging analysis pipelines (`2PAnalysis_atp1a3a.py`).
    - `Manuscript/`: LaTeX source for the final publication.
- **Shared Libraries:**
    - `HabTrackFunctions.py`: Utilities for behavioral tracking and plotting.
    - `Ca2+Imaging/Ca2ImagingFns.py`: Specialized functions for imaging data, utilizing `numba` for performance.

## 💻 Tech Stack & Dependencies
- **Language:** Python (Scripts), LaTeX (Manuscript).
- **Key Libraries:** `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `gspread` (Google Sheets), `tifffile`, `pynwb` (Neurodata Without Borders), `nrrd`, `scikit-posthocs`.
- **Performance:** `numba` (`@njit`) is used in `Ca2ImagingFns.py` for computationally intensive array operations.

## 📝 Coding Standards & Patterns
- **Interactive Execution:** Most `.py` files utilize `# %%` markers to define code cells. Treat these files as hybrid scripts/notebooks intended for interactive execution in VS Code or Spyder.
- **Data Handling:**
    - Images are handled as `tiff` or `nrrd`.
    - Dataframes are heavily used for storing behavioral metrics.
    - Files are often sorted using `natsort` to ensure correct experimental order.
- **Path Management:** 
    - ⚠️ **Caution:** The codebase often references hardcoded local/network paths (e.g., `Q:\atp1a3a_Data\BigRigData`). 
    - **Action:** When generating code, prefer `pathlib` or relative paths where possible, or clearly comment where user-specific paths need configuration.

## 🛠️ Development Workflow
1.  **Analysis Flow:** Data typically flows from raw files (TIFF/CSV) -> Processed Pickles (`.pkl`) -> Aggregated Analysis -> Figure Generation (`matplotlib`/`seaborn`).
2.  **Visualization:** Figures generated in Python are often intended for direct inclusion in the `Manuscript/` directory or export to vector formats. The `Glasbey` library consists of local files in `ExtraFunctions/` for color palette management.

## 🚀 Specific Instructions for Copilot
- **Imports:** When adding new functional logic, check `HabTrackFunctions.py` or `Ca2ImagingFns.py` first to see if a utility already exists.
- **Refactoring:** If modifying `jitted` functions (numba), ensure type stability and avoid Python object overhead inside the compiled loop.
- **Context Awareness:** When answering questions about "Figure X", refer to the `Manuscript/` files to understand the scientific claim being supported.
