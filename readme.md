# MRI-based Brain Age Prediction

This project implements a brain age prediction model based on the methodology described in the Nature Medicine paper:  
[*MRI-based multi-organ clocks for healthy aging and disease assessment*](https://www.nature.com/articles/s41591-025-03999-8).  
The focus is on the **brain MRIBAG (MRI-based Biological Age Gap)** using 119 gray matter volumes extracted from T1-weighted MRI data.

## Features
- Fully automated feature extraction from UK Biobank raw T1 MRI compressed files (`.zip`)
- Uses **AAL3 atlas** (170 regions, but can be filtered to 119) to compute regional gray matter volumes
- Supports multiple machine learning models:
  - LASSO regression
  - Support Vector Regressor (SVR)
  - ElasticNet
  - 1D Convolutional Neural Network (CNN) via PyTorch
- Handles age priority (`age_2` if available, otherwise `age_0`)
- Includes age‑bias correction as described in the literature
- Training/validation/test split with optional hyperparameter tuning

## Project Structure
```
.
├── data_process.py          # Extract features from raw .zip MRI files
├── run.py                    # Unified training & testing script
├── model.py                  # Model definitions (LASSO, SVR, ElasticNet, CNN)
├── data_loader.py            # Load and preprocess feature CSV
├── requirements.txt          # Python dependencies (see below)
└── README.md                 # This file
```

## Data Requirements
You need:
1. **Participant CSV** – contains at least the columns:  
   `eid`, `sex`, `age_0`, `age_2` (age at imaging visit).  
   Missing values are allowed and handled automatically.
2. **Raw MRI zip files** – named as `{eid}_20252_2_0.zip` (UK Biobank T1 structural MRI).  
   Each zip must contain a `T1/` folder with the FSL‑processed outputs (e.g., `T1_fast/`, `T1_first/`, `transforms/`).

## Installation
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction
Run `data_process.py` to generate the feature CSV.  
Edit the configuration variables at the top of the script:
```python
ZIP_DIR    = "/path/to/zip/files"
CSV_PATH   = "/path/to/participants.csv"
ATLAS_MNI  = "/path/to/AAL3v1_1mm.nii.gz"
ATLAS_TXT  = "/path/to/AAL3v1.nii.txt"
OUTPUT_CSV = "/path/to/output/brain_features.csv"
```
Then execute:
```bash
python data_process.py
```
This will process all available zip files (only those listed in the CSV) and create a CSV with columns:  
`eid`, `age`, `sex`, `atlas_1`, `atlas_2`, … (170 regions).  
Processing time depends on the number of participants; a progress bar is shown for each file.

### 2. Model Training & Testing
Use `run.py` to train and evaluate models. Basic usage:
```bash
python run.py --train --test --data_path brain_features.csv --model_type lasso
```
Available options:
- `--model_type`: `lasso`, `svr`, `elasticnet`, `cnn` (default: `lasso`)
- `--tune` : perform grid search for hyperparameters (valid for traditional models)
- `--test_size` : proportion for test set (default: 0.2)
- `--val_size` : proportion for validation set (default: 0.2)
- `--include_sex` : include sex as a feature (default: `True`)
- `--output_model` : file to save the trained model (default: `brain_age_model.pkl` for sklearn, `.pth` for CNN)
- `--output_metrics` : JSON file to save metrics

Example with CNN:
```bash
python run.py --train --test --data_path brain_features.csv --model_type cnn 
```
If you only want to test the model:
```bash
python run.py --test --model_type cnn
```

### 3. Model Evaluation
After training, the script prints:
- Training, validation, and test **MAE** and **R²**
- Test performance **after age‑bias correction** (linear correction fitted on training set)

All metrics are also saved in the specified JSON file.

## Notes on AAL3 and 119 Regions
- The extracted features currently use **all 170 AAL3 regions**.  
  If you need to strictly replicate the paper's **119 gray matter regions**, you can filter the columns in `data_loader.py` by passing a `selected_region_ids` list.  
  Refer to the paper's supplementary materials to obtain the exact list of region IDs.

## Troubleshooting
- **FSL commands not found**: The script uses a **pure‑Python** affine transformation (via `nibabel` and `scipy`), so no external tools are required.
- **Missing files in zip**: The script checks for required files (`T1_fast_pve_0.nii.gz`, `T1_to_MNI_linear.mat`) and skips participants if they are absent.
- **CUDA out of memory**: Reduce batch size in `model.py` for the CNN model.
- **All region volumes zero**: This indicates an error in the affine transformation. Ensure the `ATLAS_MNI` file matches the version (1mm) and that the `.mat` file contains a valid 4×4 matrix.

## Citation
If you use this code for your research, please cite the original paper:
```
Rao, S. et al. MRI-based multi-organ clocks for healthy aging and disease assessment. 
Nat Med 32, 82–92 (2026). https://doi.org/10.1038/s41591-025-03999-8
```

## License
```
This project is for research purposes only. The AAL3 atlas is distributed under the GNU General Public License.
```
