# Add the project root (e.g., home/ws/Repo_name) to Python path
import sys
from pathlib import Path
import os

# Get the path of the directory containing the current script (generation/)
script_dir = os.path.dirname(__file__)

# Get the path of the project root (repo/)
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from PIL import Image
import cv2
from libs.DiffDeepDRR.Differentiable_DRRs import Differentiable_DRRs
from libs.DiffDeepDRR.vol.volume_Realistic import Volume_Realistc
from libs.DiffDeepDRR.drr_projectors.proj_zbc import Deepdrrbased_Projector
from glob import glob


def test_PatientCT_DRR(ct_volume_path):

    vol = Volume_Realistc.from_nifti(
        filepath=ct_volume_path,
        resample=True,
        resample_spacing=[2.0, 2.0, 2.0],
        HU_segments=[100, 300],  # HU_segments=[-800, 350],
        target_orient="RIA",
        spectrum="90KV_AL40",
        use_cache=False,
    )
    vol.Update()
    assert vol.check_ready(), f"please call vol.Update()"

    # Make the DRR Engine
    Proj = Deepdrrbased_Projector(vol, step=max(vol.get_spacing()))
    drr = Differentiable_DRRs(
        Vol=vol,
        Projector=Proj,
        Target_id=None,
        detector_center_x=216.0,
        detector_center_y=216.0,
        height=256,
        pixel_size=1.6875,
        normlized=False,
        bone_dark=True,
    )

    poses = [
        [0, 0, 0, 0, 0, 0],  # Pose 1: neutral
        [10, 10, 10, 10, 10, 10],  # Pose 2: shifted/rotated
        [-10, -10, -10, -10, -10, -10],  # Pose 3: different offsets
    ]

    results = []

    for i, (alpha, beta, gamma, tx, ty, tz) in enumerate(poses):
        print(f"\n--- Pose {i+1} ---")
        print("pose:", alpha, beta, gamma, tx, ty, tz)

        start_timer = time.time()

        img, _ = drr(alpha, beta, gamma, tx, ty, tz)
        DRR_img = np.squeeze(img[0, :, :].detach().cpu().numpy())

        end_timer = time.time()
        elapsed_time = end_timer - start_timer
        print(f"Image created in : {elapsed_time:.4f} seconds")

        # Normalize image to 0–255 for visualization
        DRR_img_uint8 = cv2.normalize(DRR_img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        DRR_img_color = cv2.cvtColor(DRR_img_uint8, cv2.COLOR_GRAY2BGR)

        # Add labels: pose params, and time

        cv2.putText(
            DRR_img_color,
            f"[{alpha}, {beta}, {gamma}, {tx}, {ty}, {tz}]",
            (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            DRR_img_color,
            f"Time: {elapsed_time:.4f}s",
            (10, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

        results.append(DRR_img_color)

    # Make a horizontal collage
    collage = cv2.hconcat(results)  # or cv2.vconcat(results) for vertical

    images_results_dir = Path(
        "/home/future-lab/fmarcantoni_ws/DiffDeepDRR/DRRGeneration_imgs"
    )

    output_filename = (
        images_results_dir / f"{Path(ct_volume_path).stem}_collagefile_100_300.png"
    )

    print("Output file_10_30")
    print(output_filename)

    cv2.imwrite(output_filename, collage)
    print(f"Saved collage to {output_filename}")

    # Show the collage
    cv2.imshow("DRR Collage", collage)
    print("Press any key to exit...")
    key = cv2.waitKey(0)
    print(f"Key pressed: {key}")
    cv2.destroyAllWindows()

    # plt.imshow(DRR_img, cmap="gray", vmax=1, vmin=0)
    # # plt.show()
    # plt.savefig("drr_output.png")


if __name__ == "__main__":
    base_dir = Path("/home/future-lab/fmarcantoni_ws/DiffDeepDRR/testdata")
    for i in range(1):
        absolute_filepath = base_dir / f"ct{i}.nii"
        test_PatientCT_DRR(absolute_filepath)
