# -*- coding: utf-8 -*-
"""

Author: Tatsat Rajendra Patel
06/02/2025

"""

import os
import argparse
import logging
import json
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_closing
from skimage.morphology import remove_small_objects

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def load_slicer_roi_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    roi = data['markups'][0]
    return roi['center'], roi['size']

def get_atlas_roi_mask(atlas_img, center_phys, size_phys):
    spacing = atlas_img.GetSpacing()
    center_index = atlas_img.TransformPhysicalPointToIndex(center_phys)
    size_voxels = [int(round(size_phys[i] / spacing[i])) for i in range(3)]

    start_index = [center_index[i] - size_voxels[i] // 2 for i in range(3)]
    end_index = [start_index[i] + size_voxels[i] for i in range(3)]

    img_size = atlas_img.GetSize()
    for i in range(3):
        start_index[i] = max(0, start_index[i])
        end_index[i] = min(img_size[i], end_index[i])

    roi_mask = sitk.Image(img_size, sitk.sitkUInt8)
    roi_mask.CopyInformation(atlas_img)
    for i in range(start_index[0], end_index[0]):
        for j in range(start_index[1], end_index[1]):
            for k in range(start_index[2], end_index[2]):
                roi_mask[i, j, k] = 1
    return roi_mask

def crop_top_half_z(image):
    size = list(image.GetSize())
    start = [0, 0, size[2] // 2]
    crop_size = [size[0], size[1], size[2] - start[2]]
    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetSize(crop_size)
    extractor.SetIndex(start)
    return extractor.Execute(image)

def apply_initial_transform(ref_image, mov_image, mask):
    transform = sitk.CenteredTransformInitializer(
        sitk.Cast(ref_image, mov_image.GetPixelID()),
        mov_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref_image)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)
    tr_image = resample.Execute(mov_image)
    tr_mask = resample.Execute(mask)
    return tr_image, transform, tr_mask

def apply_affine_transform(ref_image, mov_image, mask):
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(ref_image)
    elastix.SetMovingImage(mov_image)
    param_map = sitk.GetDefaultParameterMap("affine")
    param_map['MaximumNumberOfIterations'] = ['1024']
    param_map['MaximumNumberOfSamplingAttempts'] = ['32']
    param_map['NumberOfResolutions'] = ['4']
    param_map['NumberOfSpatialSamples'] = ['30000']
    param_map['NumberOfHistogramBins'] = ['128']
    elastix.SetParameterMap(param_map)
    elastix.Execute()
    result_img = elastix.GetResultImage()
    transform_map = elastix.GetTransformParameterMap()

    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transform_map)
    transformix.SetMovingImage(mask)
    transformix.Execute()
    result_mask = transformix.GetResultImage()
    return result_img, transform_map, result_mask

def extract_roi(image, roi_mask, threshold=0.5, padding=2, min_size=100):
    arr = sitk.GetArrayFromImage(roi_mask)  # (z, y, x)
    binary = arr > threshold

    binary = binary_closing(binary, structure=np.ones((3, 3, 3)))
    binary = remove_small_objects(binary, min_size=min_size)

    if not binary.any():
        raise ValueError("Refined ROI is empty after thresholding and filtering.")

    labeled, num = label(binary)
    largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    refined = (labeled == largest_label)

    indices = np.argwhere(refined)
    minz, miny, minx = indices.min(axis=0)
    maxz, maxy, maxx = indices.max(axis=0) + 1

    minx = max(int(minx - padding), 0)
    miny = max(int(miny - padding), 0)
    minz = max(int(minz - padding), 0)
    maxx = min(int(maxx + padding), image.GetSize()[0])
    maxy = min(int(maxy + padding), image.GetSize()[1])
    maxz = min(int(maxz + padding), image.GetSize()[2])

    start_index = [int(minx), int(miny), int(minz)]
    size = [int(maxx - minx), int(maxy - miny), int(maxz - minz)]

    roi_extractor = sitk.RegionOfInterestImageFilter()
    roi_extractor.SetIndex(start_index)
    roi_extractor.SetSize(size)

    return roi_extractor.Execute(image)

def visualize_mid_slice(image, mask, center_phys, title="Mid-slice overlay"):
    arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    z = image.TransformPhysicalPointToIndex(center_phys)[2]
    z = np.clip(z, 0, arr.shape[0] - 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(arr[z], cmap='gray')
    plt.imshow(mask_arr[z], cmap='Reds', alpha=0.4)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_single(atlas_path, roi_json_path, cta_path, ncct_path, mask_path, out_dir, visualize_flag):
    os.makedirs(out_dir, exist_ok=True)
    try:
        atlas = sitk.ReadImage(atlas_path, sitk.sitkFloat32)
        ncct = sitk.ReadImage(ncct_path, sitk.sitkFloat32)
        cta = sitk.ReadImage(cta_path, sitk.sitkFloat32)
        center_phys, size_phys = load_slicer_roi_json(roi_json_path)
        roi_mask_atlas = get_atlas_roi_mask(atlas, center_phys, size_phys)

        ncct_crop = crop_top_half_z(ncct)
        sitk.WriteImage(ncct_crop, os.path.join(out_dir, "ncct_top_half.nii.gz"))

        atlas_init, init_transform, roi_init = apply_initial_transform(ncct_crop, atlas, roi_mask_atlas)
        sitk.WriteImage(atlas_init, os.path.join(out_dir, "atlas_after_initial.nii.gz"))

        atlas_registered, transform_map, roi_mask_ncct = apply_affine_transform(ncct, atlas_init, roi_init)
        sitk.WriteImage(atlas_registered, os.path.join(out_dir, "atlas_registered_to_ncct.nii.gz"))
        sitk.WriteImage(roi_mask_ncct, os.path.join(out_dir, "roi_mask.nii.gz"))

        roi_ncct = extract_roi(ncct, roi_mask_ncct)
        sitk.WriteImage(roi_ncct, os.path.join(out_dir, "roi_ncct.nii.gz"))
        if visualize_flag:
            visualize_mid_slice(ncct, roi_mask_ncct, center_phys, "NCCT + ROI")

        roi_cta = extract_roi(cta, roi_mask_ncct)
        sitk.WriteImage(roi_cta, os.path.join(out_dir, "roi_cta.nii.gz"))
        if visualize_flag:
            visualize_mid_slice(cta, roi_mask_ncct, center_phys, "CTA + ROI")

        if mask_path and os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
            roi_masked = extract_roi(mask, roi_mask_ncct)
            sitk.WriteImage(roi_masked, os.path.join(out_dir, "roi_lesion_mask.nii.gz"))
            if visualize_flag:
                visualize_mid_slice(mask, roi_mask_ncct, center_phys, "Lesion Mask + ROI")

    except Exception as e:
        logging.error(f"Error during patient processing: {e}")

def discover_patients(input_dir):
    patients = []
    for sub in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, sub)
        if not os.path.isdir(path): continue
        cta = os.path.join(path, "cta.nii.gz")
        if not os.path.exists(cta): continue
        ncct = os.path.join(path, "ncct.nii.gz")
        mask = os.path.join(path, "mask.nii.gz")
        patients.append({"id": sub, "cta": cta, "ncct": ncct if os.path.exists(ncct) else "", "mask": mask if os.path.exists(mask) else ""})
    return patients

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True, help="Atlas NIfTI file")
    parser.add_argument("--roi_json", required=True, help="3D Slicer ROI JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Batch mode: dir with patient subfolders")
    group.add_argument("--cta", help="Single patient CTA image")
    parser.add_argument("--ncct", help="Single patient NCCT image")
    parser.add_argument("--mask", help="Single patient lesion mask")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show overlay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_dir:
        patients = discover_patients(args.input_dir)
        for p in patients:
            out_sub = os.path.join(args.output_dir, p["id"])
            logging.info(f"Processing {p['id']}")
            process_single(args.atlas, args.roi_json, p["cta"], p["ncct"], p["mask"], out_sub, args.visualize)
    else:
        process_single(args.atlas, args.roi_json, args.cta, args.ncct, args.mask, args.output_dir, args.visualize)

if __name__ == "__main__":
    main()
