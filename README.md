# ROI Extraction Pipeline for Stroke Imaging

This Python pipeline extracts standardized **Regions of Interest (ROIs)** from 3D CT images (CTA, NCCT, and lesion masks) using a reference atlas and ROI definition. It performs robust affine registration (with SimpleITK + Elastix) and applies the transformed ROI to each patient scan. The final result is a cropped 3D volume for each modality centered around the anatomical ROI.

---

## 🧠 Use Case

This tool is ideal for:
- Extracting matched anatomical ROIs across different stroke patient scans
- Standardizing image crops for deep learning or radiomics analysis
- Leveraging an ROI drawn on an atlas for batch application to patient images

---

## 📁 Folder Structure
roi_extraction_pipeline/\
├── roi_extraction.py # Main pipeline script\
├── README.md # This guide\
├── requirements.txt # Dependencies\
├── atlas/\
│ ├── atlas_image.nii.gz # Your reference atlas image\
│ └── roi_box.json # ROI defined in 3D Slicer\
├── test_data/\
│ └── AIS_0001/\
│ ├── cta.nii.gz\
│ ├── ncct.nii.gz\
│ └── mask.nii.gz\
├── outputs/ # Automatically created results\
└── .gitignore\

---

## 📥 Inputs

For each patient, you need:
- `cta.nii.gz`: Co-registered CT Angiography
- `ncct.nii.gz`: Non-contrast CT image (used as base for registration)
- `mask.nii.gz`: (Optional) Lesion mask from segmentation

Also required:
- `atlas_image.nii.gz`: Atlas image aligned to patient space
- `roi_box.json`: 3D Slicer-style ROI, containing `"center"` and `"size"` in mm

---

## 📤 Outputs

For each patient:
- `roi_ncct.nii.gz`: Cropped NCCT volume at the registered ROI
- `roi_cta.nii.gz`: Cropped CTA volume
- `roi_lesion_mask.nii.gz`: Cropped lesion mask (if provided)
- `roi_mask.nii.gz`: The registered ROI mask in NCCT space
- `atlas_after_initial.nii.gz`: Atlas after initial Euler transform
- `atlas_registered_to_ncct.nii.gz`: Final registered atlas
- `ncct_top_half.nii.gz`: Cropped top-half NCCT for registration

---

## 🚀 How to Run

### Batch Mode:

```
python roi_extraction.py \
  --atlas atlas/atlas_image.nii.gz \
  --roi_json atlas/roi_box.json \
  --input_dir test_data \
  --output_dir outputs
```

Each subfolder in test_data/ must have:
- cta.nii.gz
- ncct.nii.gz
- Optional: mask.nii.gz

### Single Case Mode:
```
python roi_extraction.py \
  --atlas atlas/atlas_image.nii.gz \
  --roi_json atlas/roi_box.json \
  --cta test_data/AIS_0001/cta.nii.gz \
  --ncct test_data/AIS_0001/ncct.nii.gz \
  --mask test_data/AIS_0001/mask.nii.gz \
  --output_dir outputs/AIS_0001 \
  --visualize
```

### 📦 Installation
Create an environment (recommended):
```
python -m venv roi_env
source roi_env/bin/activate      # or roi_env\\Scripts\\activate on Windows
pip install -r requirements.txt
```

### 🔧 Dependencies
- SimpleITK>=2.2
- numpy
- scipy
- scikit-image
- matplotlib

### 📌 Notes
ROI registration uses a two-step process:
- Euler3D initialization with CenteredTransformInitializer
- Affine refinement via Elastix toolbox
Only the top half of the NCCT is used during registration for better robustness
Final ROI cropping uses morphological filtering to avoid overfitting to slanted masks

